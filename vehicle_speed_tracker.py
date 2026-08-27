"""
Webazi Vehicle Speed Tracker

Supports:
- Recorded video: video.mp4
- RTSP / RTMP / HTTP live sources
- Webcam
- YOLO vehicle detection + ByteTrack
- Speed estimation
- Vehicle counting
- Speeding alerts
- Snapshots
- CSV logs
- Processed output video

For Render:
    Input:
        /app/video.mp4

    Processed output:
        /app/data/output_speed.mp4

    Logs:
        /app/data/speed_log.csv
        /app/data/vehicle_counts.csv
        /app/data/alerts.csv

Ultralytics configuration is set BEFORE importing YOLO.
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
# ULTRALYTICS CONFIGURATION
# ============================================================

YOLO_CONFIG_DIR = os.getenv(
    "YOLO_CONFIG_DIR",
    "/tmp/Ultralytics"
)

os.environ["YOLO_CONFIG_DIR"] = YOLO_CONFIG_DIR
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)

# ============================================================
# IMPORTS
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
# CONSTANTS
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
# DEFAULT CONFIG
# ============================================================

DEFAULT_CONFIG = {
    "source": "video.mp4",

    # IMPORTANT:
    # Processed video is stored on Render persistent disk.
    "output": "/app/data/output_speed.mp4",

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
            [0, 1079]
        ],

        "src_road_r": [
            [845, 395],
            [1140, 395],
            [1220, 1079],
            [900, 1079]
        ],

        "undistort": {
            "enabled": False,
            "camera_matrix": None,
            "dist_coeffs": None
        }
    },

    "tracking": {
        "tracker_yaml": "bytetrack_custom.yaml",
        "min_track_frames": 8,
        "history_len": 25,
        "max_plausible_kph": 200,
        "min_plausible_kph": 2
    },

    "speed_thresholds": {
        "green_kph": 60,
        "yellow_kph": 100
    },

    "counting": {
        "enabled": True,
        "csv_path": os.getenv(
            "COUNTS_LOG_PATH",
            "/app/data/vehicle_counts.csv"
        )
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
        )
    },

    "recording": {
        "enabled": True,
        "output_dir": "/app/data/recordings",
        "segment_minutes": 60,
        "retention_days": 7
    }
}


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

def _resolve_env_vars(value):

    if isinstance(value, str):

        def substitute(match):

            var_name = match.group(1)

            if var_name not in os.environ:
                sys.exit(
                    f"Config references ${{{var_name}}} "
                    f"but environment variable {var_name} "
                    f"is not set."
                )

            return os.environ[var_name]

        return ENV_VAR_PATTERN.sub(
            substitute,
            value
        )

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
                "PyYAML is required for --config."
            )

        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}

        cfg = deep_merge(
            DEFAULT_CONFIG,
            user_cfg
        )

    cfg = _resolve_env_vars(cfg)

    return cfg


# ============================================================
# GEOMETRY
# ============================================================

def build_bev_transform(
    src_pts,
    road_width_m,
    visible_len_m,
    scale
):

    bev_w = int(road_width_m * scale)
    bev_h = int(visible_len_m * scale)

    dst = np.float32([
        [0, 0],
        [bev_w, 0],
        [bev_w, bev_h],
        [0, bev_h]
    ])

    M = cv2.getPerspectiveTransform(
        src_pts,
        dst
    )

    Minv = cv2.getPerspectiveTransform(
        dst,
        src_pts
    )

    return M, Minv, bev_w, bev_h


def to_bev(M, pt):

    p = np.float32([
        [[pt[0], pt[1]]]
    ])

    transformed = cv2.perspectiveTransform(
        p,
        M
    )

    return (
        float(transformed[0, 0, 0]),
        float(transformed[0, 0, 1])
    )


# ============================================================
# SPEED
# ============================================================

def fit_speed_kph(
    history,
    dt,
    scale
):

    if len(history) < 3:
        return None

    points = list(history)

    t0 = points[0][2]

    times = np.array(
        [
            (p[2] - t0) * dt
            for p in points
        ],
        dtype=np.float64
    )

    xs = np.array(
        [p[0] for p in points],
        dtype=np.float64
    )

    ys = np.array(
        [p[1] for p in points],
        dtype=np.float64
    )

    if times[-1] - times[0] <= 0:
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

    speed_m_s = speed_px_s / scale

    return speed_m_s * 3.6


def speed_color(
    kph,
    green,
    yellow
):

    if kph < green:
        return (0, 220, 0)

    if kph < yellow:
        return (0, 200, 255)

    return (0, 50, 255)


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

    (tw, th), baseline = cv2.getTextSize(
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
        (20, 20, 20),
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
# SOURCE
# ============================================================

def parse_source(raw_source):

    raw_source = str(raw_source)

    if raw_source.isdigit():
        return int(raw_source)

    return raw_source


def is_live_source(source):

    if isinstance(source, int):
        return True

    lowered = str(source).lower()

    return lowered.startswith(
        (
            "rtsp://",
            "rtmp://",
            "http://",
            "https://"
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
# ROTATING VIDEO WRITER
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
            "/app/data/recordings"
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
        self.segment_start = 0
        self.current_path = None

        if self.enabled:

            os.makedirs(
                self.output_dir,
                exist_ok=True
            )

            self._open_new_segment()

        else:

            os.makedirs(
                os.path.dirname(
                    fallback_path
                ) or ".",
                exist_ok=True
            )

            self.writer = cv2.VideoWriter(
                fallback_path,
                self.fourcc,
                fps,
                (width, height)
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
            f"{self.current_path}",
            flush=True
        )

    def write(self, frame):

        if (
            self.enabled
            and time.time() - self.segment_start
            >= self.segment_seconds
        ):
            self._open_new_segment()

        self.writer.write(frame)

    def release(self):

        if self.writer is not None:
            self.writer.release()


# ============================================================
# ALERTS
# ============================================================

def send_alert(
    cfg_alerts,
    event
):

    webhook = cfg_alerts.get(
        "webhook_url"
    )

    if webhook and HAVE_REQUESTS:

        try:

            requests.post(
                webhook,
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

    if token and chat_id and HAVE_REQUESTS:

        try:

            text = (
                f"Speeding: "
                f"{event['class']} "
                f"#{event['track_id']} "
                f"at {event['kph']:.0f} km/h "
                f"({event['timestamp']})"
            )

            url = (
                f"https://api.telegram.org/"
                f"bot{token}/sendMessage"
            )

            requests.post(
                url,
                data={
                    "chat_id": chat_id,
                    "text": text
                },
                timeout=5
            )

        except Exception as e:

            print(
                f"[alert] Telegram failed: {e}"
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

    ts = event[
        "timestamp"
    ].replace(
        ":",
        "-"
    ).replace(
        " ",
        "_"
    )

    base = (
        f"{ts}_"
        f"id{event['track_id']}_"
        f"{event['kph']:.0f}kph"
    )

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

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    crop = frame[
        y1:y2,
        x1:x2
    ]

    crop_name = None

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

    os.makedirs(
        os.path.dirname(
            alerts_csv_path
        ) or ".",
        exist_ok=True
    )

    exists = os.path.isfile(
        alerts_csv_path
    )

    with open(
        alerts_csv_path,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        if not exists:

            writer.writerow([
                "timestamp",
                "track_id",
                "class",
                "kph",
                "snapshot"
            ])

        writer.writerow([
            event["timestamp"],
            event["track_id"],
            event["class"],
            round(event["kph"], 1),
            crop_name or ""
        ])


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Webazi Vehicle Speed Tracker"
    )

    parser.add_argument(
        "--config",
        default=None
    )

    parser.add_argument(
        "--source",
        default=None
    )

    parser.add_argument(
        "--output",
        default=None
    )

    parser.add_argument(
        "--log",
        default=None
    )

    parser.add_argument(
        "--model",
        default=None
    )

    parser.add_argument(
        "--device",
        default=None
    )

    parser.add_argument(
        "--display",
        action="store_true"
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    cfg = load_config(
        args.config
    )

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
    # CONFIG
    # --------------------------------------------------------

    cam = cfg["camera"]
    trk = cfg["tracking"]
    thr = cfg["speed_thresholds"]
    cnt_cfg = cfg["counting"]
    alert_cfg = cfg["alerts"]

    source = parse_source(
        cfg["source"]
    )

    live = is_live_source(
        source
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    timestamp_tag = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # IMPORTANT:
    # This is the fallback output requested.
    fallback_video_out = cfg["output"] or (
        f"/app/data/output_speed_{timestamp_tag}.mp4"
        if live
        else "/app/data/output_speed.mp4"
    )

    # Ensure output directory exists.

    os.makedirs(
        os.path.dirname(
            fallback_video_out
        ) or ".",
        exist_ok=True
    )

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    print("=" * 60, flush=True)
    print("WEBazi VEHICLE SPEED TRACKER", flush=True)
    print("=" * 60, flush=True)

    print(
        f"Input source: {cfg['source']}",
        flush=True
    )

    print(
        f"Output video: {fallback_video_out}",
        flush=True
    )

    print(
        f"YOLO_CONFIG_DIR: {YOLO_CONFIG_DIR}",
        flush=True
    )

    cap = open_capture(
        source
    )

    if not cap.isOpened():

        raise FileNotFoundError(
            f"Cannot open source: "
            f"{cfg['source']}"
        )

    fps = (
        cap.get(cv2.CAP_PROP_FPS)
        or FALLBACK_FPS
    )

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
        f"@ {fps:.2f} FPS",
        flush=True
    )

    if total:
        print(
            f"Frames: {total}",
            flush=True
        )

        print(
            f"Duration: "
            f"{total / fps:.2f} seconds",
            flush=True
        )

    # --------------------------------------------------------
    # OUTPUT WRITER
    # --------------------------------------------------------

    # Recorded video gets a single output file.
    # Live sources use rotating recordings.

    effective_rec_cfg = dict(
        cfg.get(
            "recording",
            {}
        )
    )

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
    # CAMERA CALIBRATION
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

    bev_scale = cam[
        "bev_scale"
    ]

    visible_len_m = cam[
        "visible_length_m"
    ]

    ML, MLinv, bev_wL, bev_hL = (
        build_bev_transform(
            src_road_l,
            road_width_l_m,
            visible_len_m,
            bev_scale
        )
    )

    MR, MRinv, bev_wR, bev_hR = (
        build_bev_transform(
            src_road_r,
            road_width_r_m,
            visible_len_m,
            bev_scale
        )
    )

    # --------------------------------------------------------
    # YOLO MODEL
    # --------------------------------------------------------

    print(
        f"Loading YOLO model: "
        f"{cfg['model']}",
        flush=True
    )

    model = YOLO(
        cfg["model"]
    )

    print(
        "YOLO model loaded successfully.",
        flush=True
    )

    # --------------------------------------------------------
    # TRACKING STATE
    # --------------------------------------------------------

    history_len = trk[
        "history_len"
    ]

    min_track_frames = trk[
        "min_track_frames"
    ]

    max_plausible = trk[
        "max_plausible_kph"
    ]

    min_plausible = trk[
        "min_plausible_kph"
    ]

    green_kph = thr[
        "green_kph"
    ]

    yellow_kph = thr[
        "yellow_kph"
    ]

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

    last_log_flush = time.time()

    # --------------------------------------------------------
    # LOG FUNCTIONS
    # --------------------------------------------------------

    def write_speed_log():

        os.makedirs(
            os.path.dirname(
                cfg["log"]
            ) or ".",
            exist_ok=True
        )

        with open(
            cfg["log"],
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "track_id",
                "class",
                "frames_tracked",
                "avg_kph",
                "max_kph"
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
                    /
                    track_kph_cnt[tid]
                )

                writer.writerow([
                    tid,
                    track_class.get(
                        tid,
                        "Unknown"
                    ),
                    track_frames[tid],
                    round(avg_kph, 1),
                    round(
                        track_max_kph[tid],
                        1
                    )
                ])

    def write_counts_log():

        if not cnt_cfg[
            "enabled"
        ]:
            return

        path = cnt_cfg[
            "csv_path"
        ]

        os.makedirs(
            os.path.dirname(path)
            or ".",
            exist_ok=True
        )

        with open(
            path,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "class",
                "count"
            ])

            for cls_name, count in sorted(
                class_counts.items()
            ):

                writer.writerow([
                    cls_name,
                    count
                ])

    # --------------------------------------------------------
    # PROCESS VIDEO
    # --------------------------------------------------------

    print(
        "Starting video processing...",
        flush=True
    )

    while True:

        ret, frame = cap.read()

        if not ret:

            if live:

                print(
                    "Live source disconnected. "
                    "Reconnecting...",
                    flush=True
                )

                cap.release()

                time.sleep(
                    RECONNECT_DELAY_S
                )

                cap = open_capture(
                    source
                )

                continue

            break

        frame_idx += 1

        original_frame = frame.copy()

        # ----------------------------------------------------
        # UNDISTORT
        # ----------------------------------------------------

        undist = cam.get(
            "undistort",
            {}
        ) or {}

        if (
            undist.get(
                "enabled"
            )
            and undist.get(
                "camera_matrix"
            )
            and undist.get(
                "dist_coeffs"
            )
        ):

            K = np.array(
                undist[
                    "camera_matrix"
                ],
                dtype=np.float64
            )

            D = np.array(
                undist[
                    "dist_coeffs"
                ],
                dtype=np.float64
            )

            frame = cv2.undistort(
                frame,
                K,
                D
            )

        # ----------------------------------------------------
        # YOLO TRACK
        # ----------------------------------------------------

        if (
            frame_idx
            % max(
                1,
                int(
                    cfg[
                        "process_every"
                    ]
                )
            )
            != 0
        ):

            out.write(frame)

            continue

        try:

            results = model.track(
                frame,
                persist=True,
                tracker=trk[
                    "tracker_yaml"
                ],
                classes=list(
                    VEHICLE_CLASSES.keys()
                ),
                imgsz=cfg[
                    "imgsz"
                ],
                device=cfg[
                    "device"
                ],
                verbose=False
            )

        except Exception as e:

            print(
                f"YOLO tracking error: {e}",
                flush=True
            )

            out.write(frame)

            continue

        result = results[0]

        boxes = result.boxes

        if (
            boxes is not None
            and len(boxes) > 0
        ):

            xyxy = boxes.xyxy.cpu().numpy()

            classes = (
                boxes.cls.cpu().numpy()
                if boxes.cls is not None
                else []
            )

            ids = (
                boxes.id.cpu().numpy()
                if boxes.id is not None
                else []
            )

            for i, box in enumerate(
                xyxy
            ):

                if len(ids) <= i:
                    continue

                track_id = int(
                    ids[i]
                )

                cls_id = int(
                    classes[i]
                )

                class_name = VEHICLE_CLASSES.get(
                    cls_id,
                    "Vehicle"
                )

                x1, y1, x2, y2 = box

                cx = (
                    x1 + x2
                ) / 2

                cy = (
                    y1 + y2
                ) / 2

                # ------------------------------------------------
                # Determine road side.
                # ------------------------------------------------

                road_side = (
                    "left"
                    if cx < width / 2
                    else "right"
                )

                if road_side == "left":

                    bx, by = to_bev(
                        ML,
                        (
                            cx,
                            cy
                        )
                    )

                else:

                    bx, by = to_bev(
                        MR,
                        (
                            cx,
                            cy
                        )
                    )

                # ------------------------------------------------
                # TRACK HISTORY
                # ------------------------------------------------

                bev_history[
                    track_id
                ].append(
                    (
                        bx,
                        by,
                        frame_idx
                    )
                )

                track_frames[
                    track_id
                ] += 1

                track_class[
                    track_id
                ] = class_name

                # ------------------------------------------------
                # COUNT VEHICLE
                # ------------------------------------------------

                if (
                    cnt_cfg[
                        "enabled"
                    ]
                    and track_id
                    not in counted_ids
                    and track_frames[
                        track_id
                    ] >= min_track_frames
                ):

                    counted_ids.add(
                        track_id
                    )

                    class_counts[
                        class_name
                    ] += 1

                # ------------------------------------------------
                # SPEED
                # ------------------------------------------------

                speed = fit_speed_kph(
                    bev_history[
                        track_id
                    ],
                    dt,
                    bev_scale
                )

                display_speed = None

                if speed is not None:

                    if (
                        min_plausible
                        <= speed
                        <= max_plausible
                    ):

                        speed_smooth[
                            track_id
                        ].append(
                            speed
                        )

                        display_speed = (
                            sum(
                                speed_smooth[
                                    track_id
                                ]
                            )
                            /
                            len(
                                speed_smooth[
                                    track_id
                                ]
                            )
                        )

                        track_kph_sum[
                            track_id
                        ] += display_speed

                        track_kph_cnt[
                            track_id
                        ] += 1

                        track_max_kph[
                            track_id
                        ] = max(
                            track_max_kph[
                                track_id
                            ],
                            display_speed
                        )

                # ------------------------------------------------
                # DRAW
                # ------------------------------------------------

                draw_color = (
                    speed_color(
                        display_speed,
                        green_kph,
                        yellow_kph
                    )
                    if display_speed is not None
                    else (255, 255, 255)
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

                if display_speed is not None:

                    label = (
                        f"{class_name} "
                        f"#{track_id} "
                        f"{display_speed:.1f} km/h"
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
                            25,
                            y1
                        )
                    ),
                    draw_color
                )

                # ------------------------------------------------
                # SPEEDING ALERT
                # ------------------------------------------------

                alert_threshold = alert_cfg.get(
                    "speed_kph_threshold",
                    100
                )

                if (
                    alert_cfg.get(
                        "enabled",
                        False
                    )
                    and display_speed is not None
                    and display_speed
                    >= alert_threshold
                    and track_id
                    not in alerted_ids
                ):

                    alerted_ids.add(
                        track_id
                    )

                    timestamp = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    event = {
                        "timestamp": timestamp,
                        "track_id": track_id,
                        "class": class_name,
                        "kph": display_speed
                    }

                    snapshot = save_snapshot(
                        alert_cfg,
                        original_frame,
                        box,
                        event
                    )

                    log_alert(
                        alert_cfg[
                            "log_path"
                        ],
                        event,
                        snapshot
                    )

                    send_alert(
                        alert_cfg,
                        event
                    )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Webazi Traffic Watch",
            (
                20,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Vehicles: {len(counted_ids)}",
            (
                20,
                70
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # WRITE OUTPUT
        # ----------------------------------------------------

        out.write(frame)

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        if cfg.get(
            "display",
            False
        ):

            cv2.imshow(
                "Webazi Traffic Watch",
                frame
            )

            if (
                cv2.waitKey(1)
                & 0xFF
            ) == ord("q"):

                break

        # ----------------------------------------------------
        # PERIODIC LOG FLUSH
        # ----------------------------------------------------

        if (
            time.time()
            - last_log_flush
            >= LOG_FLUSH_EVERY_S
        ):

            write_speed_log()
            write_counts_log()

            last_log_flush = time.time()

            print(
                f"Processed frame "
                f"{frame_idx}"
                + (
                    f"/{total}"
                    if total
                    else ""
                ),
                flush=True
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    out.release()

    cv2.destroyAllWindows()

    # Final logs

    write_speed_log()
    write_counts_log()

    print("=" * 60, flush=True)
    print(
        "VIDEO PROCESSING COMPLETE",
        flush=True
    )
    print(
        f"Processed video: "
        f"{fallback_video_out}",
        flush=True
    )
    print(
        f"Speed log: "
        f"{cfg['log']}",
        flush=True
    )
    print(
        f"Counts log: "
        f"{cnt_cfg['csv_path']}",
        flush=True
    )
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()