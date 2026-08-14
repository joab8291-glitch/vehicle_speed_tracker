#!/bin/bash
# start.sh — runs the tracker as a background process and the dashboard as
# the foreground web process, both reading/writing the same local disk.
# This is what lets one Render Web Service (or one Docker container) serve
# both: the tracker itself never receives HTTP traffic, only the dashboard
# does, so the heavy CV workload isn't gated behind request/response cycles.
set -e

echo "Starting vehicle_speed_tracker.py in the background..."
python vehicle_speed_tracker.py --config config.yaml &
TRACKER_PID=$!

cleanup() {
  echo "Stopping tracker (pid $TRACKER_PID)..."
  kill "$TRACKER_PID" 2>/dev/null || true
}
trap cleanup TERM INT

echo "Starting dashboard on port ${PORT:-8000}..."
gunicorn -b 0.0.0.0:${PORT:-8000} --chdir dashboard --workers 2 app:app &
DASHBOARD_PID=$!

wait -n "$TRACKER_PID" "$DASHBOARD_PID"
