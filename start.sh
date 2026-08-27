#!/bin/bash
set -e

echo "Starting vehicle_speed_tracker.py in the background..."
python vehicle_speed_tracker.py --config /app/config.yaml &
TRACKER_PID=$!

echo "Starting dashboard on port 10000..."
cd /app
gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 2 dashboard.app:app &
DASH_PID=$!

# Wait for either process to exit
wait $TRACKER_PID $DASH_PID