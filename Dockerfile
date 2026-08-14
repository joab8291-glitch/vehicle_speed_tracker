FROM python:3.11-slim

# ffmpeg + libgl/libglib are needed by OpenCV for RTSP decoding, even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY dashboard/requirements.txt dashboard/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r dashboard/requirements.txt

COPY video.mp4 /app/video.mp4
COPY config.yaml /app/config.yaml
COPY vehicle_speed_tracker.py .
COPY calibrate_points.py .
COPY bytetrack_custom.yaml .
COPY dashboard/ dashboard/
COPY start.sh .
RUN chmod +x start.sh

# YOLO weights auto-download on first run if not present; bake them in instead
# to avoid a slow/flaky download on every container restart:
ADD https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt /app/yolov8n.pt

# Mount your real config.yaml (with real calibration + secrets resolved from env)
# at /app/config.yaml when running the container, e.g.:
#   docker run -v $(pwd)/config.yaml:/app/config.yaml --env-file .env -p 8000:8000 vehicle-speed-tracker
# .env must include DASHBOARD_USER and DASHBOARD_PASS - the dashboard refuses
# to start without them (see dashboard/app.py).
#
# start.sh runs the tracker (no HTTP) in the background and the dashboard
# (Flask/gunicorn, serves the UI) in the foreground on $PORT.
CMD ["./start.sh"]
