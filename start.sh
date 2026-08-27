#!/bin/bash

set -u

echo "============================================================"
echo "Starting Webazi Vehicle Speed Tracker"
echo "============================================================"

# ============================================================
# DIRECTORIES
# ============================================================

mkdir -p /app/data
mkdir -p /app/data/snapshots
mkdir -p /app/data/recordings
mkdir -p /tmp/Ultralytics

# ============================================================
# ULTRALYTICS
# ============================================================

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"

echo "YOLO_CONFIG_DIR=$YOLO_CONFIG_DIR"

# ============================================================
# CHECK FILES
# ============================================================

echo "Checking application files..."

echo "Input video: /app/video.mp4"
echo "YOLO model: /app/yolov8n.pt"
echo "Config: /app/config.yaml"
echo "Speed log: /app/data/speed_log.csv"
echo "Counts log: /app/data/vehicle_counts.csv"
echo "Alerts log: /app/data/alerts.csv"
echo "Snapshots: /app/data/snapshots"
echo "Processed video: /app/data/output_speed.mp4"

if [ ! -f /app/video.mp4 ]; then
    echo "ERROR: /app/video.mp4 does not exist."
fi

if [ ! -f /app/yolov8n.pt ]; then
    echo "ERROR: /app/yolov8n.pt does not exist."
fi

if [ ! -f /app/config.yaml ]; then
    echo "ERROR: /app/config.yaml does not exist."
fi

# ============================================================
# START TRACKER
# ============================================================

echo "Starting vehicle_speed_tracker.py..."

python -u /app/vehicle_speed_tracker.py \
    --config /app/config.yaml &

TRACKER_PID=$!

echo "Vehicle tracker PID: $TRACKER_PID"

# ============================================================
# START DASHBOARD
# ============================================================

echo "Starting dashboard..."

gunicorn \
    --bind 0.0.0.0:${PORT:-10000} \
    --workers 2 \
    --timeout 120 \
    dashboard.app:app &

DASH_PID=$!

echo "Dashboard PID: $DASH_PID"
echo "Dashboard port: ${PORT:-10000}"

# ============================================================
# CLEANUP
# ============================================================

cleanup() {

    echo "Stopping services..."

    kill "$TRACKER_PID" 2>/dev/null || true
    kill "$DASH_PID" 2>/dev/null || true

    wait "$TRACKER_PID" 2>/dev/null || true
    wait "$DASH_PID" 2>/dev/null || true
}

trap cleanup SIGTERM SIGINT

# ============================================================
# IMPORTANT
#
# The tracker processes a finite video and is allowed to finish.
# DO NOT kill the dashboard when the tracker finishes.
#
# Render needs the dashboard to remain alive so you can watch
# the completed processed video.
# ============================================================

while true; do

    if ! kill -0 "$DASH_PID" 2>/dev/null; then

        echo "Dashboard stopped."

        cleanup

        exit 1
    fi

    if ! kill -0 "$TRACKER_PID" 2>/dev/null; then

        echo "Vehicle tracker has finished."

        echo "The dashboard will remain running."

        # Tracker has finished processing video.
        # Dashboard remains available.

        wait "$TRACKER_PID" 2>/dev/null || true

        TRACKER_PID=0
    fi

    sleep 5

done