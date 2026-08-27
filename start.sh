#!/bin/bash

set -e

echo "============================================================"
echo "Starting Webazi Vehicle Speed Tracker"
echo "============================================================"


# ============================================================
# Persistent directories
# ============================================================

mkdir -p /app/data
mkdir -p /app/data/snapshots
mkdir -p /app/data/recordings


# ============================================================
# Ultralytics configuration
# ============================================================

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"

mkdir -p "$YOLO_CONFIG_DIR"

echo "YOLO_CONFIG_DIR=$YOLO_CONFIG_DIR"


# ============================================================
# Verify important files
# ============================================================

echo "Checking application files..."

if [ ! -f "/app/video.mp4" ]; then
    echo "ERROR: /app/video.mp4 does not exist."
    exit 1
fi

if [ ! -f "/app/vehicle_speed_tracker.py" ]; then
    echo "ERROR: /app/vehicle_speed_tracker.py does not exist."
    exit 1
fi

if [ ! -f "/app/yolov8n.pt" ]; then
    echo "ERROR: /app/yolov8n.pt does not exist."
    exit 1
fi

if [ ! -f "/app/config.yaml" ]; then
    echo "ERROR: /app/config.yaml does not exist."
    exit 1
fi

echo "Input video: /app/video.mp4"
echo "YOLO model: /app/yolov8n.pt"
echo "Config: /app/config.yaml"


# ============================================================
# Persistent paths
# ============================================================

export SPEED_LOG_PATH="${SPEED_LOG_PATH:-/app/data/speed_log.csv}"
export COUNTS_LOG_PATH="${COUNTS_LOG_PATH:-/app/data/vehicle_counts.csv}"
export ALERTS_LOG_PATH="${ALERTS_LOG_PATH:-/app/data/alerts.csv}"
export SNAPSHOT_DIR="${SNAPSHOT_DIR:-/app/data/snapshots}"

export PROCESSED_VIDEO_PATH="${PROCESSED_VIDEO_PATH:-/app/output_speed.mp4}"

echo "Speed log: $SPEED_LOG_PATH"
echo "Counts log: $COUNTS_LOG_PATH"
echo "Alerts log: $ALERTS_LOG_PATH"
echo "Snapshots: $SNAPSHOT_DIR"
echo "Processed video: $PROCESSED_VIDEO_PATH"


# ============================================================
# Start tracker
# ============================================================

echo "Starting vehicle_speed_tracker.py..."

python /app/vehicle_speed_tracker.py \
    --config /app/config.yaml &

TRACKER_PID=$!

echo "Vehicle tracker PID: $TRACKER_PID"


# ============================================================
# Start dashboard
# ============================================================

echo "Starting dashboard..."

exec gunicorn \
    --bind 0.0.0.0:${PORT:-10000} \
    --workers 2 \
    --timeout 120 \
    dashboard.app:app