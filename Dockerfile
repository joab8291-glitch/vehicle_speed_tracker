FROM python:3.11-slim

# ============================================================
# System dependencies
# ============================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# Application directory
# ============================================================

WORKDIR /app


# ============================================================
# Python dependencies
# ============================================================

COPY requirements.txt .

COPY dashboard/requirements.txt dashboard/requirements.txt

RUN pip install --no-cache-dir \
    -r requirements.txt \
    -r dashboard/requirements.txt


# ============================================================
# Application files
# ============================================================

COPY video.mp4 /app/video.mp4

COPY config.yaml /app/config.yaml

COPY vehicle_speed_tracker.py /app/vehicle_speed_tracker.py

COPY calibrate_points.py /app/calibrate_points.py

COPY bytetrack_custom.yaml /app/bytetrack_custom.yaml

COPY dashboard/ /app/dashboard/

COPY start.sh /app/start.sh


# ============================================================
# Permissions
# ============================================================

RUN chmod +x /app/start.sh


# ============================================================
# YOLO model
# ============================================================

ADD https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt /app/yolov8n.pt


# ============================================================
# Ultralytics configuration
# ============================================================

ENV YOLO_CONFIG_DIR=/tmp/Ultralytics


# ============================================================
# Persistent directories
# ============================================================

RUN mkdir -p \
    /app/data \
    /app/data/snapshots \
    /app/data/recordings \
    /tmp/Ultralytics


# ============================================================
# Start application
# ============================================================

CMD ["/app/start.sh"]