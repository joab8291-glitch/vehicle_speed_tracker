"""
vehicle_speed_tracker.py

Config-driven vehicle speed tracker for recorded video, live RTSP CCTV, or a webcam.

Quick start (no config file, same as before):
    python vehicle_speed_tracker.py --source video.mp4

Recommended for real deployment (per-camera config file):
    cp config.example.yaml config.yaml   # edit calibration/thresholds
    python vehicle_speed_tracker.py --config config.yaml

CLI flags always override the config file's source/output/log/model/device/display.

Secrets: any string value in config.yaml can reference an environment variable with
${VAR_NAME} syntax, e.g.  source: "rtsp://${RTSP_USER}:${RTSP_PASS}@${RTSP_HOST}:554/..."
This keeps camera credentials out of the file you commit to GitHub.
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta

# ------------------------------------------------------------------------
# Ultralytics configuration
# ------------------------------------------------------------------------
# Render's /tmp directory is writable. This prevents:
# WARNING ⚠️ user config directory '/tmp/Ultralytics/Ultralytics'
# is not writable, using '/tmp/Ultralytics'.
#
# IMPORTANT: this must be set BEFORE importing ultralytics.
YOLO_CONFIG_DIR = "/tmp/Ultralytics"
os.environ["YOLO_CONFIG_DIR"] = YOLO_CONFIG_DIR
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
FALLBACK_FPS = 25.0
RECONNECT_DELAY_S = 3.0
LOG_FLUSH_EVERY_S = 30.0
ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# ------------------------------------------------------------------------
# Config handling
# ------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "source": "video.mp4",
    "output": None,
    "log": os.getenv("SPEED_LOG_PATH", "/app/data/speed_log.csv"),
    "model": "yolov8n.pt",
    "device": "cpu",
    "imgsz": 640,
    "process_every": 1,
    "display": False,
    "camera": {
        "lane_width_m": 3.75,
        "num_lanes_left": 5,
        "num_lanes_right": 3,
        "bev_scale": 18,
        "visible_length_m": 70,
        "src_road_l": [[155, 415], [600, 395], [845, 1079], [0, 1079]],
        "src_road_r": [[845, 395], [1140, 395], [1220, 1079], [900, 1079]],
        "undistort": {"enabled": False, "camera_matrix": None, "dist_coeffs": None},
    },
    "tracking": {
        "tracker_yaml": "bytetrack_custom.yaml",
        "min_track_frames": 8,
        "history_len": 25,
        "max_plausible_kph": 200,
        "min_plausible_kph": 2,
    },
    "speed_thresholds": {"green_kph": 60, "yellow_kph": 100},
    "counting": {
    "enabled": True,
    "csv_path": os.getenv("COUNTS_LOG_PATH", "/app/data/vehicle_counts.csv")
},
    "alerts": {
        "enabled": False,
        "speed_kph_threshold": 100,
        "webhook_url": None,
        "telegram_bot_token": None,
        "telegram_chat_id": None,
        "snapshot_dir": "snapshots",
        "save_full_frame": True,
        "log_path": "alerts.csv",
    },
    # 24/7 operation: writing one continuously-growing file fills the disk eventually.
    # This rotates the output video into fixed-length segments and deletes old ones.
    "recording": {
        "enabled": True,
        "output_dir": "recordings",
        "segment_minutes": 60,
        "retention_days": 7,
    },
}


def _resolve_env_vars(value):
    """Recursively replace ${VAR_NAME} in strings with os.environ values.

    Raises a clear error if a referenced env var isn't set, rather than silently
    embedding the literal '${VAR}' string into a URL and failing later with a
    confusing connection error.
    """
    if isinstance(value, str):
        def _sub(match):
            var_name = match.group(1)
            if var_name not in os.environ:
                sys.exit(
                    f"Config references ${{{var_name}}} but that environment "
                    f"variable is not set. Set it before running, e.g.\n"
                    f"  export {var_name}=... "
                )
            return os.environ[var_name]
        return ENV_VAR_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def deep_merge(base, override):
    """Merge `override` dict into `base` dict, recursively. Returns a new dict."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path):
    cfg = DEFAULT_CONFIG
    if path:
        if not HAVE_YAML:
            sys.exit("PyYAML is required for --config. Install with: pip install pyyaml")
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = deep_merge(DEFAULT_CONFIG, user_cfg)
    cfg = _resolve_env_vars(cfg)
    return cfg


# ------------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------------

def build_bev_transform(src_pts, road_width_m, visible_len_m, scale):
    """Return (M, Minv, bev_w, bev_h) for a perspective -> bird's-eye warp."""
    bev_w = int(road_width_m * scale)
    bev_h = int(visible_len_m * scale)
    dst = np.float32([
        [0, 0],
        [bev_w, 0],
        [bev_w, bev_h],
        [0, bev_h],
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst)
    Minv = cv2.getPerspectiveTransform(dst, src_pts)
    return M, Minv, bev_w, bev_h


def to_bev(M, pt):
    p = np.float32([[[pt[0], pt[1]]]])
    t = cv2.perspectiveTransform(p, M)
    return float(t[0, 0, 0]), float(t[0, 0, 1])


def fit_speed_kph(history, dt, scale):
    """
    Fit a straight line through the recent (x, y, frame_idx) ground-plane
    positions and return speed in km/h from the fitted velocity, instead of
    just using the first and last point (which is noisy - a single bad
    detection at either end used to skew the whole reading).

    Needs at least 3 points to be worth fitting; returns None otherwise.
    """
    if len(history) < 3:
        return None
    pts = list(history)
    t0 = pts[0][2]
    times = np.array([(p[2] - t0) * dt for p in pts], dtype=np.float64)
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    if times[-1] - times[0] <= 0:
        return None
    vx = np.polyfit(times, xs, 1)[0]  # px/s in BEV space
    vy = np.polyfit(times, ys, 1)[0]
    speed_px_s = math.hypot(vx, vy)
    speed_m_s = speed_px_s / scale
    return speed_m_s * 3.6


def speed_color(kph, green, yellow):
    if kph < green:
        return (0, 220, 0)
    elif kph < yellow:
        return (0, 200, 255)
    else:
        return (0, 50, 255)


def draw_label(frame, text, pos, color, font_scale=0.6, thickness=2):
    x, y = int(pos[0]), int(pos[1])
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    pad = 4
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), (20, 20, 20), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)


# ------------------------------------------------------------------------
# Source handling
# ------------------------------------------------------------------------

def parse_source(raw_source):
    raw_source = str(raw_source)
    if raw_source.isdigit():
        return int(raw_source)
    return raw_source


def is_live_source(raw_source):
    if isinstance(raw_source, int):
        return True
    lowered = raw_source.lower()
    return lowered.startswith(("rtsp://", "rtmp://", "http://", "https://"))


def open_capture(source):
    cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


# ------------------------------------------------------------------------
# Video rotation + retention (for 24/7 recording without filling the disk)
# ------------------------------------------------------------------------

class RotatingVideoWriter:
    """Writes output video in fixed-length segments and deletes segments older
    than the configured retention window. One continuously-growing file would
    otherwise fill the disk on a 24/7 live run; this caps disk usage at
    roughly (segment count within the retention window)."""

    def __init__(self, rec_cfg, fps, width, height, fallback_path):
        self.enabled = bool(rec_cfg.get("enabled", True))
        self.output_dir = rec_cfg.get("output_dir", "recordings")
        self.segment_seconds = max(60, int(rec_cfg.get("segment_minutes", 60) * 60))
        self.retention_days = rec_cfg.get("retention_days", 7)
        self.fps = fps
        self.width = width
        self.height = height
        self.fallback_path = fallback_path
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = None
        self.segment_start = 0.0
        self.current_path = None

        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            self._open_new_segment()
            self._purge_old_segments()
        else:
            # Single continuously-growing file, same as the original behaviour.
            self.writer = cv2.VideoWriter(fallback_path, self.fourcc, fps, (width, height))
            self.current_path = fallback_path

    def _open_new_segment(self):
        if self.writer is not None:
            self.writer.release()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_path = os.path.join(self.output_dir, f"segment_{ts}.mp4")
        self.writer = cv2.VideoWriter(self.current_path, self.fourcc, self.fps, (self.width, self.height))
        self.segment_start = time.time()
        print(f"  [recording] new segment -> {self.current_path}")

    def _purge_old_segments(self):
        if not self.retention_days:
            return
        cutoff = time.time() - self.retention_days * 86400
        for path in glob.glob(os.path.join(self.output_dir, "segment_*.mp4")):
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    print(f"  [recording] purged old segment -> {path}")
            except OSError:
                pass

    def write(self, frame):
        if self.enabled and (time.time() - self.segment_start) >= self.segment_seconds:
            self._open_new_segment()
            self._purge_old_segments()
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()


# ------------------------------------------------------------------------
# Alerts / snapshots
# ------------------------------------------------------------------------

def send_alert(cfg_alerts, event):
    """Best-effort webhook + Telegram notification. Never raises - a network
    hiccup must not crash a live 24/7 tracking run."""
    if cfg_alerts["webhook_url"]:
        try:
            if HAVE_REQUESTS:
                requests.post(cfg_alerts["webhook_url"], json=event, timeout=5)
            else:
                print("  [alert] 'requests' not installed - skipping webhook. pip install requests")
        except Exception as e:
            print(f"  [alert] webhook failed: {e}")

    token = cfg_alerts["telegram_bot_token"]
    chat_id = cfg_alerts["telegram_chat_id"]
    if token and chat_id:
        try:
            if HAVE_REQUESTS:
                text = (f"Speeding: {event['class']} #{event['track_id']} at "
                        f"{event['kph']:.0f} km/h ({event['timestamp']})")
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=5)
            else:
                print("  [alert] 'requests' not installed - skipping Telegram. pip install requests")
        except Exception as e:
            print(f"  [alert] telegram failed: {e}")


def save_snapshot(cfg_alerts, frame, box, event):
    snap_dir = cfg_alerts["snapshot_dir"]
    os.makedirs(snap_dir, exist_ok=True)
    ts = event["timestamp"].replace(":", "-").replace(" ", "_")
    base = f"{ts}_id{event['track_id']}_{event['kph']:.0f}kph"
    crop_name = None
    if cfg_alerts["save_full_frame"]:
        cv2.imwrite(os.path.join(snap_dir, f"{base}_full.jpg"), frame)
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    crop = frame[y1:y2, x1:x2]
    if crop.size > 0:
        crop_name = f"{base}_crop.jpg"
        cv2.imwrite(os.path.join(snap_dir, crop_name), crop)
    return crop_name


def log_alert(alerts_csv_path, event, crop_name):
    """Append one row per speeding alert. This is what the dashboard reads to
    show a live alert feed - webhook/Telegram are fire-and-forget, so without
    this there'd be no record to look back at."""
    file_exists = os.path.isfile(alerts_csv_path)
    with open(alerts_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "track_id", "class", "kph", "snapshot"])
        writer.writerow([event["timestamp"], event["track_id"], event["class"],
                          round(event["kph"], 1), crop_name or ""])


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Vehicle speed tracker (file, RTSP CCTV, or webcam).")
    p.add_argument("--config", default=None, help="Path to a YAML config file (see config.example.yaml).")
    p.add_argument("--source", default=None, help="Overrides config: file path, RTSP/HTTP URL, or webcam index.")
    p.add_argument("--output", default=None, help="Overrides config: output video path (non-rotating fallback).")
    p.add_argument("--log", default=None, help="Overrides config: CSV speed-log path.")
    p.add_argument("--model", default=None, help="Overrides config: YOLO weights path.")
    p.add_argument("--device", default=None, help="Overrides config: cpu / cuda:0 / mps.")
    p.add_argument("--display", action="store_true", help="Overrides config: show a live preview window.")
    p.add_argument("--max-reconnects", type=int, default=0,
                    help="For live sources: give up after N failed reconnects. 0 = retry forever.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.source is not None:
        cfg["source"] = args.source
    if args.output is not None:
        cfg["output"] = args.output
    if args.log is not None:
        cfg["log"] = args.log
    if args.model is not None:
        cfg["model"] = args.model
    if args.device is not None:
        cfg["device"] = args.device
    if args.display:
        cfg["display"] = True

    cam = cfg["camera"]
    trk = cfg["tracking"]
    thr = cfg["speed_thresholds"]
    cnt_cfg = cfg["counting"]
    alert_cfg = cfg["alerts"]
    rec_cfg = cfg["recording"]

    source = parse_source(cfg["source"])
    live = is_live_source(source)

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback_video_out = cfg["output"] or (f"output_speed_{timestamp_tag}.mp4" if live else "output_speed.mp4")

    cap = open_capture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source '{cfg['source']}'.")

    fps = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not live else 0

    print(f"Source: {cfg['source']} ({'LIVE' if live else 'recorded file'})")
    print(f"Video: {width}x{height} @ {fps:.1f} fps"
          + (f" | {total} frames ({total/fps:.1f} s)" if total > 0 else " | duration unknown (live)"))
    print(f"Device: {cfg['device']} | imgsz: {cfg['imgsz']} | process_every: {cfg['process_every']}")
    if live and rec_cfg.get("enabled", True):
        print(f"Recording: rotating every {rec_cfg['segment_minutes']} min, "
              f"keeping {rec_cfg['retention_days']} days -> {rec_cfg['output_dir']}/")

    # Lens undistortion setup
    undist = cam.get("undistort", {}) or {}
    do_undistort = bool(undist.get("enabled")) and undist.get("camera_matrix") and undist.get("dist_coeffs")
    if do_undistort:
        K = np.array(undist["camera_matrix"], dtype=np.float64)
        D = np.array(undist["dist_coeffs"], dtype=np.float64)
        print("Lens undistortion: ENABLED")
    else:
        K = D = None

    # Only live sources rotate; recorded-file runs keep the old single-file behaviour
    # since a finite input video doesn't need disk-usage management.
    effective_rec_cfg = dict(rec_cfg)
    if not live:
        effective_rec_cfg["enabled"] = False
    out = RotatingVideoWriter(effective_rec_cfg, fps, width, height, fallback_video_out)

    road_width_l_m = cam["lane_width_m"] * cam["num_lanes_left"]
    road_width_r_m = cam["lane_width_m"] * cam["num_lanes_right"]
    src_road_l = np.float32(cam["src_road_l"])
    src_road_r = np.float32(cam["src_road_r"])
    bev_scale = cam["bev_scale"]
    visible_len_m = cam["visible_length_m"]

    ML, MLinv, bev_wL, bev_hL = build_bev_transform(src_road_l, road_width_l_m, visible_len_m, bev_scale)
    MR, MRinv, bev_wR, bev_hR = build_bev_transform(src_road_r, road_width_r_m, visible_len_m, bev_scale)

    model = YOLO(cfg["model"])
    print(f"Model loaded: {cfg['model']}")

    history_len = trk["history_len"]
    min_track_frames = trk["min_track_frames"]
    max_plausible = trk["max_plausible_kph"]
    min_plausible = trk["min_plausible_kph"]
    green_kph = thr["green_kph"]
    yellow_kph = thr["yellow_kph"]

    bev_history = defaultdict(lambda: deque(maxlen=history_len))
    speed_smooth = defaultdict(lambda: deque(maxlen=history_len))
    track_frames = defaultdict(int)
    track_class = {}
    track_max_kph = defaultdict(float)
    track_kph_sum = defaultdict(float)
    track_kph_cnt = defaultdict(int)
    counted_ids = set()
    class_counts = defaultdict(int)
    alerted_ids = set()

    frame_idx = 0
    dt = 1.0 / fps
    reconnect_attempts = 0
    last_log_flush = time.time()

    def write_speed_log():
        with open(cfg["log"], "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "class", "frames_tracked", "avg_kph", "max_kph"])
            for tid in sorted(track_frames):
                if track_kph_cnt[tid] == 0:
                    continue
                avg_kph = track_kph_sum[tid] / track_kph_cnt[tid]
                writer.writerow([tid, track_class.get(tid, "Unknown"), track_frames[tid],
                                  round(avg_kph, 1), round(track_max_kph[tid], 1)])
        print(f"Speed log saved -> {cfg['log']}")

    def write_counts_log():
        if not cnt_cfg["enabled"]:
            return
        with open(cnt_cfg["csv_path"], "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "count"])
            for cls_name, c in sorted(class_counts.items()):
                writer.writerow([cls_name, c])
        print(f"Vehicle counts saved -> {cnt_cfg['csv_path']}")

    print("Processing frames... (Ctrl+C to stop)")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if live:
                    print("Stream read failed - attempting to reconnect...")
                    cap.release()
                    time.sleep(RECONNECT_DELAY_S)
                    cap = open_capture(source)
                    if not cap.isOpened():
                        reconnect_attempts += 1
                        print(f"Reconnect failed ({reconnect_attempts}).")
                        if args.max_reconnects and reconnect_attempts >= args.max_reconnects:
                            print("Max reconnect attempts reached, stopping.")
                            break
                        continue
                    else:
                        print("Reconnected.")
                        reconnect_attempts = 0
                        continue
                else:
                    break

            frame_idx += 1
            if do_undistort:
                frame = cv2.undistort(frame, K, D)

            if frame_idx % 30 == 0:
                if total > 0:
                    pct = frame_idx / total * 100
                    print(f"  {frame_idx}/{total} frames ({pct:.0f}%)")
                else:
                    print(f"  {frame_idx} frames processed (live)")

            # Skip detection on some frames for throughput; timestamps still
            # stay correct because frame_idx keeps counting through skips.
            if cfg["process_every"] > 1 and frame_idx % cfg["process_every"] != 0:
                out.write(frame)
                if cfg["display"]:
                    cv2.imshow("Vehicle Speed Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            results = model.track(
                frame,
                persist=True,
                tracker=trk["tracker_yaml"],
                classes=list(VEHICLE_CLASSES.keys()),
                conf=0.35,
                iou=0.45,
                imgsz=cfg["imgsz"],
                device=cfg["device"],
                verbose=False,
            )

            wall_clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if results[0].boxes is None or results[0].boxes.id is None:
                cv2.putText(frame, wall_clock, (12, height - 12), cv2.FONT_HERSHEY_DUPLEX,
                            0.55, (200, 200, 200), 1, cv2.LINE_AA)
                out.write(frame)
                if cfg["display"]:
                    cv2.imshow("Vehicle Speed Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, tid, cls_id, conf in zip(boxes, ids, clss, confs):
                tid = int(tid)
                cls_id = int(cls_id)
                cls_name = VEHICLE_CLASSES.get(cls_id, "Vehicle")
                track_class[tid] = cls_name
                track_frames[tid] += 1

                if cnt_cfg["enabled"] and tid not in counted_ids:
                    counted_ids.add(tid)
                    class_counts[cls_name] += 1

                x1, y1, x2, y2 = box
                # Ground-contact point (bottom-center of the box), NOT the box
                # center - the homography only maps the road plane correctly,
                # and the box center sits at roof height, causing parallax
                # error that grows with vehicle height and distance.
                cx = (x1 + x2) / 2
                cy = y2

                use_left = cx < (width / 2 + 80)
                M_use = ML if use_left else MR
                scale_use = bev_scale

                bx, by = to_bev(M_use, (cx, cy))
                bev_history[tid].append((bx, by, frame_idx))

                kph = None
                raw_kph = fit_speed_kph(bev_history[tid], dt, scale_use)
                if raw_kph is not None and min_plausible < raw_kph < max_plausible:
                    speed_smooth[tid].append(raw_kph)
                    kph = float(np.mean(speed_smooth[tid]))
                    track_kph_sum[tid] += kph
                    track_kph_cnt[tid] += 1
                    track_max_kph[tid] = max(track_max_kph[tid], kph)

                show_speed = kph is not None and track_frames[tid] >= min_track_frames
                color = speed_color(kph, green_kph, yellow_kph) if show_speed else (180, 180, 180)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{cls_name} ({kph:.0f} km/h)" if show_speed else cls_name
                draw_label(frame, label, (x1, y1 - 6), color)

                if show_speed:
                    bar_max = yellow_kph
                    bar_frac = min(kph / bar_max, 1.0)
                    bar_len = int((x2 - x1) * bar_frac)
                    bar_y = int(y2) + 4
                    cv2.rectangle(frame, (int(x1), bar_y), (int(x1) + bar_len, bar_y + 5), color, -1)

                    if (alert_cfg["enabled"] and kph >= alert_cfg["speed_kph_threshold"]
                            and tid not in alerted_ids):
                        alerted_ids.add(tid)
                        event = {"track_id": tid, "class": cls_name, "kph": kph, "timestamp": wall_clock}
                        print(f"  [ALERT] {cls_name} #{tid} at {kph:.0f} km/h")
                        send_alert(alert_cfg, event)
                        crop_name = save_snapshot(alert_cfg, frame, (x1, y1, x2, y2), event)
                        log_alert(alert_cfg.get("log_path", "alerts.csv"), event, crop_name)

            status = f"Frame {frame_idx}/{total}" if total > 0 else f"Frame {frame_idx} (LIVE)"
            cv2.putText(frame, f"{status} | {fps:.0f} fps", (12, 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, wall_clock, (12, height - 12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            for i, (label, colour) in enumerate([
                ("< 60 km/h", (0, 220, 0)),
                ("60-100 km/h", (0, 200, 255)),
                ("> 100 km/h", (0, 50, 255)),
            ]):
                y_leg = height - 80 + i * 26
                cv2.rectangle(frame, (12, y_leg - 14), (32, y_leg + 4), colour, -1)
                cv2.putText(frame, label, (38, y_leg), cv2.FONT_HERSHEY_DUPLEX,
                            0.55, (230, 230, 230), 1, cv2.LINE_AA)

            if cnt_cfg["enabled"]:
                count_text = " | ".join(f"{c}:{n}" for c, n in sorted(class_counts.items()))
                if count_text:
                    cv2.putText(frame, count_text, (width - 10 - 9 * len(count_text), 28),
                                cv2.FONT_HERSHEY_DUPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

            out.write(frame)
            if cfg["display"]:
                cv2.imshow("Vehicle Speed Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if live and (time.time() - last_log_flush) > LOG_FLUSH_EVERY_S:
                write_speed_log()
                write_counts_log()
                last_log_flush = time.time()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        out.release()
        if cfg["display"]:
            cv2.destroyAllWindows()

    print(f"\nDone! Output -> {out.current_path if out.enabled else fallback_video_out}")
    write_speed_log()
    write_counts_log()


if __name__ == "__main__":
    main()
