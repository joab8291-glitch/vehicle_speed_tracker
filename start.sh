#!/bin/bash

set -e

echo "============================================================"
echo "Starting Webazi Vehicle Speed Tracker"
echo "============================================================"

mkdir -p /app/data
mkdir -p /app/data/snapshots
mkdir -p /app/data/recordings

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"

echo "YOLO_CONFIG_DIR=$YOLO_CONFIG_DIR"

echo "Starting vehicle_speed_tracker.py..."

python /app/vehicle_speed_tracker.py \
    --config /app/config.yaml &

TRACKER_PID=$!

echo "Vehicle tracker PID: $TRACKER_PID"

echo "Starting dashboard..."

exec gunicorn \
    --bind 0.0.0.0:${PORT:-10000} \
    --workers 2 \
    --timeout 120 \
    dashboard.app:app