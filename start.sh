#!/bin/bash

set -e

echo "============================================================"
echo "Starting Webazi Vehicle Speed Tracker"
echo "============================================================"

# ------------------------------------------------------------
# Create persistent directories
# ------------------------------------------------------------

mkdir -p /app/data
mkdir -p /app/data/snapshots
mkdir -p /app/data/recordings

# ------------------------------------------------------------
# Create Ultralytics configuration directory
# ------------------------------------------------------------

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"

mkdir -p "$YOLO_CONFIG_DIR"

echo "YOLO_CONFIG_DIR=$YOLO_CONFIG_DIR"

# ------------------------------------------------------------
# Start vehicle tracker
# ------------------------------------------------------------

echo "Starting vehicle_speed_tracker.py..."

python /app/vehicle_speed_tracker.py \
    --config /app/config.yaml &

TRACKER_PID=$!

echo "Vehicle tracker PID: $TRACKER_PID"

# ------------------------------------------------------------
# Start dashboard
# ------------------------------------------------------------

echo "Starting dashboard..."

gunicorn \
    --bind 0.0.0.0:${PORT:-10000} \
    --workers 2 \
    --timeout 120 \
    dashboard.app:app &

DASH_PID=$!

echo "Dashboard PID: $DASH_PID"
echo "Dashboard port: ${PORT:-10000}"

# ------------------------------------------------------------
# Monitor both processes
# ------------------------------------------------------------

cleanup() {
    echo "Stopping services..."

    kill "$TRACKER_PID" 2>/dev/null || true
    kill "$DASH_PID" 2>/dev/null || true

    wait "$TRACKER_PID" 2>/dev/null || true
    wait "$DASH_PID" 2>/dev/null || true
}

trap cleanup SIGTERM SIGINT

# If either process exits, stop the container.
wait -n "$TRACKER_PID" "$DASH_PID"

EXIT_CODE=$?

echo "A service exited with code $EXIT_CODE"

cleanup

exit "$EXIT_CODE"