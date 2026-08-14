FROM python:3.11-slim

# System libraries needed by OpenCV + curl for downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
COPY dashboard/requirements.txt dashboard/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r dashboard/requirements.txt

# Application files
COPY video.mp4 /app/video.mp4
COPY config.yaml /app/config.yaml
COPY vehicle_speed_tracker.py .
COPY calibrate_points.py .
COPY bytetrack_custom.yaml .
COPY dashboard/ dashboard/
COPY start.sh .
RUN chmod +x start.sh

# Bake YOLO weights
ADD https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt /app/yolov8n.pt

# Silence the Ultralytics config directory warning
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# Start both tracker + dashboard
CMD ["./start.sh"]