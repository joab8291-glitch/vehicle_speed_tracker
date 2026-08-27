FROM python:3.11-slim

# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# APPLICATION
# ============================================================

WORKDIR /app


# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY requirements.txt .

COPY dashboard/requirements.txt dashboard/requirements.txt

RUN pip install --no-cache-dir \
    -r requirements.txt \
    -r dashboard/requirements.txt


# ============================================================
# APPLICATION FILES
# ============================================================

COPY video.mp4 /app/video.mp4

COPY config.yaml /app/config.yaml

COPY vehicle_speed_tracker.py \
    /app/vehicle_speed_tracker.py

COPY calibrate_points.py \
    /app/calibrate_points.py

COPY bytetrack_custom.yaml \
    /app/bytetrack_custom.yaml

COPY dashboard/ \
    /app/dashboard/

COPY start.sh \
    /app/start.sh


# ============================================================
# PERMISSIONS
# ============================================================

RUN chmod +x /app/start.sh


# ============================================================
# YOLO MODEL
# ============================================================

ADD https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt \
    /app/yolov8n.pt


# ============================================================
# ULTRALYTICS
# ============================================================

ENV YOLO_CONFIG_DIR=/tmp/Ultralytics


# ============================================================
# PERSISTENT DIRECTORIES
# ============================================================

RUN mkdir -p \
    /app/data \
    /app/data/snapshots \
    /app/data/recordings \
    /tmp/Ultralytics


# ============================================================
# START
# ============================================================

CMD ["/app/start.sh"]