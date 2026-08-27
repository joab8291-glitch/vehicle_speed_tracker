"""
vehicle_speed_tracker.py

Config-driven vehicle speed tracker for:

    - Recorded video
    - Live RTSP CCTV
    - RTMP/HTTP streams
    - Webcam

Recorded video:
    python vehicle_speed_tracker.py --source video.mp4

Config:
    python vehicle_speed_tracker.py --config config.yaml

Environment variables can be referenced in config.yaml using:

    ${RTSP_USER}
    ${RTSP_PASS}
    ${RTSP_HOST}

Example:

    source: "rtsp://${RTSP_USER}:${RTSP_PASS}@${RTSP_HOST}:554/stream"

Important:
    YOLO_CONFIG_DIR is set BEFORE importing ultralytics so Render does not
    produce the unwritable Ultralytics configuration warning.
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
from datetime import datetime

# ============================================================
# Ultralytics configuration
# ============================================================

YOLO_CONFIG_DIR = os.getenv(
    "YOLO_CONFIG_DIR",
    "/tmp/Ultralytics"
)

os.environ["YOLO_CONFIG_DIR"] = YOLO_CONFIG_DIR
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)


# ============================================================
# Imports
# ============================================================

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


# ============================================================
# Constants
# ============================================================

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

FALLBACK_FPS = 25.0

RECONNECT_DELAY_S = 3.0

LOG_FLUSH_EVERY_S = 30.0

ENV_VAR_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}"
)


# ============================================================
# Default configuration
# ============================================================

DEFAULT_CONFIG = {

    "source": "video.mp4",

    # None means:
    #
    # recorded video -> output_speed.mp4
    # live source    -> timestamped output
    #
    "output": None,

    "log": os.getenv(
        "SPEED_LOG_PATH",
        "/app/data/speed_log.csv"
    ),

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

        "src_road_l": [
            [155, 415],
            [600, 395],
            [845, 1079],
            [0, 1079],
        ],

        "src_road_r": [
            [845, 395],
            [1140, 395],
            [1220, 1079],
            [900, 1079],
        ],

        "undistort": {
            "enabled": False,
            "camera_matrix": None,
            "dist_coeffs": None,
        },
    },

    "tracking": {

        "tracker_yaml": "bytetrack_custom.yaml",

        "min_track_frames": 8,

        "history_len": 25,

        "max_plausible_kph": 200,

        "min_plausible_kph": 2,
    },

    "speed_thresholds": {

        "green_kph": 60,

        "yellow_kph": 100,
    },

    "counting": {

        "enabled": True,

        "csv_path": os.getenv(
            "COUNTS_LOG_PATH",
            "/app/data/vehicle_counts.csv"
        ),
    },

    "alerts": {

        "enabled": False,

        "speed_kph_threshold": 100,

        "webhook_url": None,

        "telegram_bot_token": os.getenv(
            "TELEGRAM_BOT_TOKEN"
        ),

        "telegram_chat_id": os.getenv(
            "TELEGRAM_CHAT_ID"
        ),

        "snapshot_dir": os.getenv(
            "SNAPSHOT_DIR",
            "/app/data/snapshots"
        ),

        "save_full_frame": True,

        "log_path": os.getenv(
            "ALERTS_LOG_PATH",
            "/app/data/alerts.csv"
        ),
    },

    "recording": {

        "enabled": True,

        "output_dir": "recordings",

        "segment_minutes": 60,

        "retention_days": 7,
    },
}


# ============================================================
# Utility functions
# ============================================================

def ensure_parent_dir(path):
    """
    Make sure the parent directory of a file exists.
    """

    directory = os.path.dirname(os.path.abspath(path))

    if directory:
        os.makedirs(directory, exist_ok=True)


def _resolve_env_vars(value):
    """
    Replace ${VAR_NAME} with environment variables.
    """

    if isinstance(value, str):

        def _sub(match):

            var_name = match.group(1)

            if var_name not in os.environ:

                sys.exit(
                    f"Config references ${{{var_name}}} "
                    f"but environment variable "
                    f"{var_name} is not set."
                )

            return os.environ[var_name]

        return ENV_VAR_PATTERN.sub(_sub, value)

    if isinstance(value, dict):

        return {
            k: _resolve_env_vars(v)
            for k, v in value.items()
        }

    if isinstance(value, list):

        return [
            _resolve_env_vars(v)
            for v in value
        ]

    return value


def deep_merge(base, override):
    """
    Recursively merge dictionaries.
    """

    result = dict(base)

    for key, value in override.items():

        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):

            result[key] = deep_merge(
                result[key],
                value
            )

        else:

            result[key] = value

    return result


def load_config(path):

    cfg = DEFAULT_CONFIG

    if path:

        if not HAVE_YAML:

            sys.exit(
                "PyYAML is required for --config. "
                "Install it with: pip install pyyaml"
            )

        with open(path, "r", encoding="utf-8") as f:

            user_cfg = yaml.safe_load(f) or {}

        cfg = deep_merge(
            DEFAULT_CONFIG,
            user_cfg
        )

    cfg = _resolve_env_vars(cfg)

    return cfg


# ============================================================
# Geometry
# ============================================================

def build_bev_transform(
    src_pts,
    road_width_m,
    visible_len_m,
    scale
):

    bev_w = int(
        road_width_m * scale
    )

    bev_h = int(
        visible_len_m * scale
    )

    dst = np.float32([
        [0, 0],
        [bev_w, 0],
        [bev_w, bev_h],
        [0, bev_h],
    ])

    M = cv2.getPerspectiveTransform(
        src_pts,
        dst
    )

    Minv = cv2.getPerspectiveTransform(
        dst,
        src_pts
    )

    return (
        M,
        Minv,
        bev_w,
        bev_h
    )


def to_bev(M, pt):

    p = np.float32([
        [[
            pt[0],
            pt[1]
        ]]
    ])

    transformed = cv2.perspectiveTransform(
        p,
        M
    )

    return (
        float(transformed[0, 0, 0]),
        float(transformed[0, 0, 1])
    )


def fit_speed_kph(
    history,
    dt,
    scale
):

    if len(history) < 3:
        return None

    pts = list(history)

    t0 = pts[0][2]

    times = np.array(
        [
            (p[2] - t0) * dt
            for p in pts
        ],
        dtype=np.float64
    )

    xs = np.array(
        [
            p[0]
            for p in pts
        ],
        dtype=np.float64
    )

    ys = np.array(
        [
            p[1]
            for p in pts
        ],
        dtype=np.float64
    )

    if (
        times[-1] - times[0]
        <= 0
    ):
        return None

    vx = np.polyfit(
        times,
        xs,
        1
    )[0]

    vy = np.polyfit(
        times,
        ys,
        1
    )[0]

    speed_px_s = math.hypot(
        vx,
        vy
    )

    speed_m_s = (
        speed_px_s / scale
    )

    return speed_m_s * 3.6


def speed_color(
    kph,
    green,
    yellow
):

    if kph < green:

        return (
            0,
            220,
            0
        )

    if kph < yellow:

        return (
            0,
            200,
            255
        )

    return (
        0,
        50,
        255
    )


def draw_label(
    frame,
    text,
    pos,
    color,
    font_scale=0.6,
    thickness=2
):

    x, y = (
        int(pos[0]),
        int(pos[1])
    )

    (
        tw,
        th
    ), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        thickness
    )

    pad = 4

    cv2.rectangle(
        frame,
        (
            x - pad,
            y - th - pad
        ),
        (
            x + tw + pad,
            y + baseline + pad
        ),
        (
            20,
            20,
            20
        ),
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# Source handling
# ============================================================

def parse_source(raw_source):

    raw_source = str(raw_source)

    if raw_source.isdigit():

        return int(raw_source)

    return raw_source


def is_live_source(raw_source):

    if isinstance(raw_source, int):

        return True

    lowered = raw_source.lower()

    return lowered.startswith(
        (
            "rtsp://",
            "rtmp://",
            "http://",
            "https://",
        )
    )


def open_capture(source):

    cap = cv2.VideoCapture(source)

    try:

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

    except Exception:
        pass

    return cap


# ============================================================
# Video writer
# ============================================================

class RotatingVideoWriter:

    def __init__(
        self,
        rec_cfg,
        fps,
        width,
        height,
        fallback_path
    ):

        self.enabled = bool(
            rec_cfg.get(
                "enabled",
                True
            )
        )

        self.output_dir = rec_cfg.get(
            "output_dir",
            "recordings"
        )

        self.segment_seconds = max(
            60,
            int(
                rec_cfg.get(
                    "segment_minutes",
                    60
                ) * 60
            )
        )

        self.retention_days = rec_cfg.get(
            "retention_days",
            7
        )

        self.fps = fps

        self.width = width

        self.height = height

        self.fallback_path = fallback_path

        self.fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        self.writer = None

        self.segment_start = 0.0

        self.current_path = None

        if self.enabled:

            os.makedirs(
                self.output_dir,
                exist_ok=True
            )

            self._open_new_segment()

            self._purge_old_segments()

        else:

            ensure_parent_dir(
                fallback_path
            )

            self.writer = cv2.VideoWriter(
                fallback_path,
                self.fourcc,
                fps,
                (
                    width,
                    height
                )
            )

            self.current_path = fallback_path

    def _open_new_segment(self):

        if self.writer is not None:

            self.writer.release()

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.current_path = os.path.join(
            self.output_dir,
            f"segment_{timestamp}.mp4"
        )

        self.writer = cv2.VideoWriter(
            self.current_path,
            self.fourcc,
            self.fps,
            (
                self.width,
                self.height
            )
        )

        self.segment_start = time.time()

        print(
            f"[recording] new segment -> "
            f"{self.current_path}"
        )

    def _purge_old_segments(self):

        if not self.retention_days:
            return

        cutoff = (
            time.time()
            - self.retention_days * 86400
        )

        for path in glob.glob(
            os.path.join(
                self.output_dir,
                "segment_*.mp4"
            )
        ):

            try:

                if os.path.getmtime(path) < cutoff:

                    os.remove(path)

                    print(
                        f"[recording] purged -> "
                        f"{path}"
                    )

            except OSError:
                pass

    def write(self, frame):

        if self.enabled:

            if (
                time.time()
                - self.segment_start
                >= self.segment_seconds
            ):

                self._open_new_segment()

                self._purge_old_segments()

        self.writer.write(frame)

    def release(self):

        if self.writer is not None:

            self.writer.release()


# ============================================================
# Alerts
# ============================================================

def send_alert(
    cfg_alerts,
    event
):

    webhook_url = cfg_alerts.get(
        "webhook_url"
    )

    if webhook_url:

        try:

            if HAVE_REQUESTS:

                requests.post(
                    webhook_url,
                    json=event,
                    timeout=5
                )

        except Exception as e:

            print(
                f"[alert] webhook failed: {e}"
            )

    token = cfg_alerts.get(
        "telegram_bot_token"
    )

    chat_id = cfg_alerts.get(
        "telegram_chat_id"
    )

    if token and chat_id:

        try:

            if HAVE_REQUESTS:

                text = (
                    f"Speeding: "
                    f"{event['class']} "
                    f"#{event['track_id']} "
                    f"at "
                    f"{event['kph']:.0f} km/h "
                    f"({event['timestamp']})"
                )

                url = (
                    "https://api.telegram.org/"
                    f"bot{token}/sendMessage"
                )

                requests.post(
                    url,
                    data={
                        "chat_id": chat_id,
                        "text": text,
                    },
                    timeout=5
                )

        except Exception as e:

            print(
                f"[alert] telegram failed: {e}"
            )


def save_snapshot(
    cfg_alerts,
    frame,
    box,
    event
):

    snap_dir = cfg_alerts[
        "snapshot_dir"
    ]

    os.makedirs(
        snap_dir,
        exist_ok=True
    )

    ts = (
        event["timestamp"]
        .replace(":", "-")
        .replace(" ", "_")
    )

    base = (
        f"{ts}_"
        f"id{event['track_id']}_"
        f"{event['kph']:.0f}kph"
    )

    crop_name = None

    if cfg_alerts.get(
        "save_full_frame",
        True
    ):

        cv2.imwrite(
            os.path.join(
                snap_dir,
                f"{base}_full.jpg"
            ),
            frame
        )

    x1, y1, x2, y2 = [
        int(v)
        for v in box
    ]

    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        frame.shape[1],
        x2
    )

    y2 = min(
        frame.shape[0],
        y2
    )

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size > 0:

        crop_name = (
            f"{base}_crop.jpg"
        )

        cv2.imwrite(
            os.path.join(
                snap_dir,
                crop_name
            ),
            crop
        )

    return crop_name


def log_alert(
    alerts_csv_path,
    event,
    crop_name
):

    ensure_parent_dir(
        alerts_csv_path
    )

    file_exists = os.path.isfile(
        alerts_csv_path
    )

    with open(
        alerts_csv_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow([
                "timestamp",
                "track_id",
                "class",
                "kph",
                "snapshot",
            ])

        writer.writerow([
            event["timestamp"],
            event["track_id"],
            event["class"],
            round(
                event["kph"],
                1
            ),
            crop_name or "",
        ])


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Vehicle speed tracker "
            "(file, RTSP CCTV, "
            "or webcam)."
        )
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file."
    )

    parser.add_argument(
        "--source",
        default=None,
        help=(
            "File path, RTSP/HTTP URL, "
            "or webcam index."
        )
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output video path."
    )

    parser.add_argument(
        "--log",
        default=None,
        help="Speed CSV path."
    )

    parser.add_argument(
        "--model",
        default=None,
        help="YOLO model path."
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "cpu / cuda:0 / mps"
        )
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live preview."
    )

    parser.add_argument(
        "--max-reconnects",
        type=int,
        default=0,
        help=(
            "Live source reconnect attempts. "
            "0 = retry forever."
        )
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    cfg = load_config(
        args.config
    )

    # --------------------------------------------------------
    # CLI overrides
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Configuration sections
    # --------------------------------------------------------

    cam = cfg["camera"]

    trk = cfg["tracking"]

    thr = cfg["speed_thresholds"]

    cnt_cfg = cfg["counting"]

    alert_cfg = cfg["alerts"]

    rec_cfg = cfg["recording"]

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    source = parse_source(
        cfg["source"]
    )

    live = is_live_source(
        source
    )

    # ========================================================
    # IMPORTANT OUTPUT PATH LOGIC
    # ========================================================
    #
    # For recorded video.mp4:
    #
    #     output_speed.mp4
    #
    # For live CCTV:
    #
    #     output_speed_YYYYMMDD_HHMMSS.mp4
    #
    # If config.yaml explicitly supplies "output:",
    # that value wins.
    # ========================================================

    timestamp_tag = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    fallback_video_out = cfg["output"] or (
        f"output_speed_{timestamp_tag}.mp4"
        if live
        else "output_speed.mp4"
    )

    print(
        "============================================================"
    )

    print(
        "Webazi Vehicle Speed Tracker"
    )

    print(
        "============================================================"
    )

    print(
        f"Source: {cfg['source']}"
    )

    print(
        f"Mode: "
        f"{'LIVE' if live else 'RECORDED VIDEO'}"
    )

    print(
        f"Output: {fallback_video_out}"
    )

    print(
        f"Speed log: {cfg['log']}"
    )

    print(
        f"YOLO_CONFIG_DIR: "
        f"{YOLO_CONFIG_DIR}"
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = open_capture(
        source
    )

    if not cap.isOpened():

        raise FileNotFoundError(
            f"Cannot open source "
            f"'{cfg['source']}'."
        )

    fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or FALLBACK_FPS
    )

    if fps <= 0:

        fps = FALLBACK_FPS

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total = (
        int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )
        if not live
        else 0
    )

    print(
        f"Video: {width}x{height} "
        f"@ {fps:.2f} FPS"
    )

    if total > 0:

        print(
            f"Frames: {total}"
        )

        print(
            f"Duration: "
            f"{total / fps:.1f} seconds"
        )

    # --------------------------------------------------------
    # Ensure output directories
    # --------------------------------------------------------

    ensure_parent_dir(
        cfg["log"]
    )

    ensure_parent_dir(
        cnt_cfg["csv_path"]
    )

    ensure_parent_dir(
        alert_cfg["log_path"]
    )

    os.makedirs(
        alert_cfg["snapshot_dir"],
        exist_ok=True
    )

    # --------------------------------------------------------
    # Lens undistortion
    # --------------------------------------------------------

    undist = (
        cam.get(
            "undistort",
            {}
        )
        or {}
    )

    do_undistort = bool(
        undist.get("enabled")
        and undist.get("camera_matrix")
        and undist.get("dist_coeffs")
    )

    if do_undistort:

        K = np.array(
            undist["camera_matrix"],
            dtype=np.float64
        )

        D = np.array(
            undist["dist_coeffs"],
            dtype=np.float64
        )

        print(
            "Lens undistortion: ENABLED"
        )

    else:

        K = None

        D = None

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    effective_rec_cfg = dict(
        rec_cfg
    )

    # Recorded video is finite.
    # Therefore do NOT rotate it into recordings/.
    #
    # It uses:
    #
    #     output_speed.mp4
    #
    # Live CCTV uses rotating segments.

    if not live:

        effective_rec_cfg[
            "enabled"
        ] = False

    out = RotatingVideoWriter(
        effective_rec_cfg,
        fps,
        width,
        height,
        fallback_video_out
    )

    # --------------------------------------------------------
    # Bird's-eye-view geometry
    # --------------------------------------------------------

    road_width_l_m = (
        cam["lane_width_m"]
        * cam["num_lanes_left"]
    )

    road_width_r_m = (
        cam["lane_width_m"]
        * cam["num_lanes_right"]
    )

    src_road_l = np.float32(
        cam["src_road_l"]
    )

    src_road_r = np.float32(
        cam["src_road_r"]
    )

    bev_scale = float(
        cam["bev_scale"]
    )

    visible_len_m = float(
        cam["visible_length_m"]
    )

    (
        ML,
        MLinv,
        bev_wL,
        bev_hL
    ) = build_bev_transform(
        src_road_l,
        road_width_l_m,
        visible_len_m,
        bev_scale
    )

    (
        MR,
        MRinv,
        bev_wR,
        bev_hR
    ) = build_bev_transform(
        src_road_r,
        road_width_r_m,
        visible_len_m,
        bev_scale
    )

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print(
        f"Loading YOLO model: "
        f"{cfg['model']}"
    )

    model = YOLO(
        cfg["model"]
    )

    print(
        f"Model loaded: "
        f"{cfg['model']}"
    )

    # --------------------------------------------------------
    # Tracking settings
    # --------------------------------------------------------

    history_len = int(
        trk["history_len"]
    )

    min_track_frames = int(
        trk["min_track_frames"]
    )

    max_plausible = float(
        trk["max_plausible_kph"]
    )

    min_plausible = float(
        trk["min_plausible_kph"]
    )

    green_kph = float(
        thr["green_kph"]
    )

    yellow_kph = float(
        thr["yellow_kph"]
    )

    process_every = max(
        1,
        int(
            cfg.get(
                "process_every",
                1
            )
        )
    )

    tracker_yaml = trk.get(
        "tracker_yaml",
        "bytetrack_custom.yaml"
    )

    # --------------------------------------------------------
    # Tracking state
    # --------------------------------------------------------

    bev_history = defaultdict(
        lambda: deque(
            maxlen=history_len
        )
    )

    speed_smooth = defaultdict(
        lambda: deque(
            maxlen=history_len
        )
    )

    track_frames = defaultdict(
        int
    )

    track_class = {}

    track_max_kph = defaultdict(
        float
    )

    track_kph_sum = defaultdict(
        float
    )

    track_kph_cnt = defaultdict(
        int
    )

    counted_ids = set()

    class_counts = defaultdict(
        int
    )

    alerted_ids = set()

    frame_idx = 0

    dt = 1.0 / fps

    reconnect_attempts = 0

    last_log_flush = time.time()

    # --------------------------------------------------------
    # CSV writers
    # --------------------------------------------------------

    def write_speed_log():

        ensure_parent_dir(
            cfg["log"]
        )

        with open(
            cfg["log"],
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "track_id",
                "class",
                "frames_tracked",
                "avg_kph",
                "max_kph",
            ])

            for tid in sorted(
                track_frames
            ):

                if (
                    track_kph_cnt[tid]
                    == 0
                ):

                    continue

                avg_kph = (
                    track_kph_sum[tid]
                    / track_kph_cnt[tid]
                )

                writer.writerow([
                    tid,
                    track_class.get(
                        tid,
                        "Unknown"
                    ),
                    track_frames[tid],
                    round(
                        avg_kph,
                        1
                    ),
                    round(
                        track_max_kph[tid],
                        1
                    ),
                ])

        print(
            f"Speed log saved -> "
            f"{cfg['log']}"
        )

    def write_counts_log():

        if not cnt_cfg["enabled"]:

            return

        ensure_parent_dir(
            cnt_cfg["csv_path"]
        )

        with open(
            cnt_cfg["csv_path"],
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "class",
                "count"
            ])

            for (
                cls_name,
                count
            ) in sorted(
                class_counts.items()
            ):

                writer.writerow([
                    cls_name,
                    count
                ])

        print(
            f"Vehicle counts saved -> "
            f"{cnt_cfg['csv_path']}"
        )

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    print(
        "Starting video processing..."
    )

    if not live:

        print(
            "Recorded video mode:"
        )

        print(
            f"Input: "
            f"{cfg['source']}"
        )

        print(
            f"Processed output: "
            f"{fallback_video_out}"
        )

    try:

        while True:

            ret, frame = cap.read()

            # ------------------------------------------------
            # End of recorded video
            # ------------------------------------------------

            if not ret:

                if not live:

                    print(
                        "End of recorded video."
                    )

                    break

                # --------------------------------------------
                # Live source reconnect
                # --------------------------------------------

                reconnect_attempts += 1

                if (
                    args.max_reconnects > 0
                    and reconnect_attempts
                    > args.max_reconnects
                ):

                    print(
                        "Maximum reconnect attempts "
                        "reached."
                    )

                    break

                print(
                    f"Live source disconnected. "
                    f"Reconnect attempt "
                    f"{reconnect_attempts}..."
                )

                cap.release()

                time.sleep(
                    RECONNECT_DELAY_S
                )

                cap = open_capture(
                    source
                )

                continue

            reconnect_attempts = 0

            frame_idx += 1

            # ------------------------------------------------
            # Undistort
            # ------------------------------------------------

            if do_undistort:

                frame = cv2.undistort(
                    frame,
                    K,
                    D
                )

            # ------------------------------------------------
            # Process only selected frames
            # ------------------------------------------------

            if (
                frame_idx
                % process_every
                != 0
            ):

                out.write(
                    frame
                )

                continue

            # ------------------------------------------------
            # YOLO tracking
            # ------------------------------------------------

            try:

                results = model.track(
                    frame,
                    persist=True,
                    tracker=tracker_yaml,
                    classes=list(
                        VEHICLE_CLASSES.keys()
                    ),
                    device=cfg["device"],
                    imgsz=int(
                        cfg["imgsz"]
                    ),
                    verbose=False,
                )

            except Exception as e:

                print(
                    f"YOLO processing error: "
                    f"{e}"
                )

                out.write(
                    frame
                )

                continue

            if not results:

                out.write(
                    frame
                )

                continue

            result = results[0]

            boxes = result.boxes

            if boxes is None:

                out.write(
                    frame
                )

                continue

            # ------------------------------------------------
            # Extract detections
            # ------------------------------------------------

            if (
                boxes.id is None
                or len(boxes) == 0
            ):

                out.write(
                    frame
                )

                continue

            ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )

            classes = (
                boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            coords = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Process every tracked vehicle
            # ------------------------------------------------

            for (
                track_id,
                class_id,
                box
            ) in zip(
                ids,
                classes,
                coords
            ):

                if class_id not in VEHICLE_CLASSES:

                    continue

                track_id = int(
                    track_id
                )

                class_name = (
                    VEHICLE_CLASSES[
                        class_id
                    ]
                )

                track_class[
                    track_id
                ] = class_name

                track_frames[
                    track_id
                ] += 1

                x1, y1, x2, y2 = box

                # Bottom-center point
                center_x = (
                    x1 + x2
                ) / 2.0

                bottom_y = y2

                point = (
                    center_x,
                    bottom_y
                )

                # --------------------------------------------
                # Determine road side
                # --------------------------------------------

                inside_left = (
                    cv2.pointPolygonTest(
                        src_road_l,
                        (
                            float(center_x),
                            float(bottom_y)
                        ),
                        False
                    )
                    >= 0
                )

                inside_right = (
                    cv2.pointPolygonTest(
                        src_road_r,
                        (
                            float(center_x),
                            float(bottom_y)
                        ),
                        False
                    )
                    >= 0
                )

                # --------------------------------------------
                # Transform to BEV
                # --------------------------------------------

                if inside_left:

                    bev_x, bev_y = to_bev(
                        ML,
                        point
                    )

                elif inside_right:

                    bev_x, bev_y = to_bev(
                        MR,
                        point
                    )

                else:

                    # If detection is outside both
                    # configured road polygons, do not
                    # calculate speed from it.

                    draw_color = (
                        255,
                        255,
                        255
                    )

                    cv2.rectangle(
                        frame,
                        (
                            int(x1),
                            int(y1)
                        ),
                        (
                            int(x2),
                            int(y2)
                        ),
                        draw_color,
                        2
                    )

                    draw_label(
                        frame,
                        f"{class_name} #{track_id}",
                        (
                            x1,
                            max(
                                20,
                                y1
                            )
                        ),
                        draw_color
                    )

                    continue

                # --------------------------------------------
                # Store BEV history
                # --------------------------------------------

                bev_history[
                    track_id
                ].append(
                    (
                        bev_x,
                        bev_y,
                        frame_idx
                    )
                )

                # --------------------------------------------
                # Calculate speed
                # --------------------------------------------

                speed = fit_speed_kph(
                    bev_history[
                        track_id
                    ],
                    dt * process_every,
                    bev_scale
                )

                valid_speed = False

                if speed is not None:

                    if (
                        min_plausible
                        <= speed
                        <= max_plausible
                    ):

                        valid_speed = True

                        speed_smooth[
                            track_id
                        ].append(
                            speed
                        )

                if speed_smooth[
                    track_id
                ]:

                    current_speed = (
                        sum(
                            speed_smooth[
                                track_id
                            ]
                        )
                        / len(
                            speed_smooth[
                                track_id
                            ]
                        )
                    )

                else:

                    current_speed = 0.0

                # --------------------------------------------
                # Record speed statistics
                # --------------------------------------------

                if (
                    valid_speed
                    and track_frames[
                        track_id
                    ] >= min_track_frames
                ):

                    track_kph_sum[
                        track_id
                    ] += current_speed

                    track_kph_cnt[
                        track_id
                    ] += 1

                    if (
                        current_speed
                        > track_max_kph[
                            track_id
                        ]
                    ):

                        track_max_kph[
                            track_id
                        ] = current_speed

                # --------------------------------------------
                # Count vehicles
                # --------------------------------------------

                if (
                    cnt_cfg["enabled"]
                    and track_frames[
                        track_id
                    ] >= min_track_frames
                    and track_id
                    not in counted_ids
                ):

                    counted_ids.add(
                        track_id
                    )

                    class_counts[
                        class_name
                    ] += 1

                    print(
                        f"[count] "
                        f"{class_name} "
                        f"#{track_id}"
                    )

                # --------------------------------------------
                # Alert
                # --------------------------------------------

                if (
                    alert_cfg["enabled"]
                    and valid_speed
                    and current_speed
                    >= float(
                        alert_cfg[
                            "speed_kph_threshold"
                        ]
                    )
                    and track_id
                    not in alerted_ids
                    and track_frames[
                        track_id
                    ] >= min_track_frames
                ):

                    alerted_ids.add(
                        track_id
                    )

                    event = {

                        "timestamp":
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "track_id":
                            track_id,

                        "class":
                            class_name,

                        "kph":
                            current_speed,
                    }

                    crop_name = save_snapshot(
                        alert_cfg,
                        frame,
                        box,
                        event
                    )

                    log_alert(
                        alert_cfg[
                            "log_path"
                        ],
                        event,
                        crop_name
                    )

                    send_alert(
                        alert_cfg,
                        event
                    )

                    print(
                        f"[ALERT] "
                        f"{class_name} "
                        f"#{track_id} "
                        f"{current_speed:.1f} km/h"
                    )

                # --------------------------------------------
                # Draw vehicle
                # --------------------------------------------

                color = speed_color(
                    current_speed,
                    green_kph,
                    yellow_kph
                )

                cv2.rectangle(
                    frame,
                    (
                        int(x1),
                        int(y1)
                    ),
                    (
                        int(x2),
                        int(y2)
                    ),
                    color,
                    2
                )

                if current_speed > 0:

                    label = (
                        f"{class_name} "
                        f"#{track_id} "
                        f"{current_speed:.1f} km/h"
                    )

                else:

                    label = (
                        f"{class_name} "
                        f"#{track_id}"
                    )

                draw_label(
                    frame,
                    label,
                    (
                        x1,
                        max(
                            20,
                            y1
                        )
                    ),
                    color
                )

            # ------------------------------------------------
            # Draw processing information
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Frame: {frame_idx}",
                (
                    15,
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255
                ),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                (
                    f"Vehicles: "
                    f"{len(track_frames)}"
                ),
                (
                    15,
                    60
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255
                ),
                2,
                cv2.LINE_AA
            )

            # ------------------------------------------------
            # Write processed video
            # ------------------------------------------------

            out.write(
                frame
            )

            # ------------------------------------------------
            # Optional display
            # ------------------------------------------------

            if cfg.get(
                "display",
                False
            ):

                cv2.imshow(
                    "Webazi Vehicle Speed Tracker",
                    frame
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):

                    print(
                        "Stopped by user."
                    )

                    break

            # ------------------------------------------------
            # Periodic CSV flush
            # ------------------------------------------------

            if (
                time.time()
                - last_log_flush
                >= LOG_FLUSH_EVERY_S
            ):

                write_speed_log()

                write_counts_log()

                last_log_flush = time.time()

    finally:

        # ----------------------------------------------------
        # Final CSV write
        # ----------------------------------------------------

        print(
            "Writing final results..."
        )

        write_speed_log()

        write_counts_log()

        # ----------------------------------------------------
        # Release video
        # ----------------------------------------------------

        cap.release()

        out.release()

        cv2.destroyAllWindows()

        print(
            "============================================================"
        )

        print(
            "Vehicle tracking finished."
        )

        print(
            f"Processed video: "
            f"{fallback_video_out}"
        )

        print(
            f"Speed results: "
            f"{cfg['log']}"
        )

        print(
            f"Vehicle counts: "
            f"{cnt_cfg['csv_path']}"
        )

        print(
            f"Snapshots: "
            f"{alert_cfg['snapshot_dir']}"
        )

        print(
            "============================================================"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()