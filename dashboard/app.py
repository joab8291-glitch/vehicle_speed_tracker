"""
dashboard/app.py

Webazi Traffic Watch dashboard.

Displays:
- Processed YOLO video
- Vehicle statistics
- Vehicle counts
- Speeding alerts
- Alert snapshots

The dashboard reads the same files written by vehicle_speed_tracker.py.
"""

import csv
import hmac
import os
import time
from datetime import datetime
from functools import wraps

from flask import (
    Flask,
    jsonify,
    render_template,
    send_from_directory,
    abort,
    request,
    Response,
    send_file,
)

# ============================================================
# Paths
# ============================================================

SPEED_LOG_PATH = os.environ.get(
    "SPEED_LOG_PATH",
    "/app/data/speed_log.csv"
)

COUNTS_LOG_PATH = os.environ.get(
    "COUNTS_LOG_PATH",
    "/app/data/vehicle_counts.csv"
)

ALERTS_LOG_PATH = os.environ.get(
    "ALERTS_LOG_PATH",
    "/app/data/alerts.csv"
)

SNAPSHOT_DIR = os.environ.get(
    "SNAPSHOT_DIR",
    "/app/data/snapshots"
)

# Processed video created by vehicle_speed_tracker.py
PROCESSED_VIDEO_PATH = os.environ.get(
    "PROCESSED_VIDEO_PATH",
    "/app/output_speed.mp4"
)

MAX_ALERTS_SHOWN = 30


# ============================================================
# Dashboard authentication
# ============================================================

DASHBOARD_USER = os.environ.get(
    "DASHBOARD_USER",
    "admin"
)

DASHBOARD_PASS = os.environ.get(
    "DASHBOARD_PASS"
)

if not DASHBOARD_PASS:
    raise SystemExit(
        "DASHBOARD_PASS environment variable is not set. "
        "Refusing to start an unprotected dashboard."
    )


# ============================================================
# Flask
# ============================================================

START_TIME = time.time()

app = Flask(__name__)


# ============================================================
# Authentication
# ============================================================

def _check_auth(username, password):
    return (
        hmac.compare_digest(
            username or "",
            DASHBOARD_USER
        )
        and
        hmac.compare_digest(
            password or "",
            DASHBOARD_PASS
        )
    )


def require_auth(view):
    @wraps(view)
    def wrapped(*args, **kwargs):

        auth = request.authorization

        if not auth or not _check_auth(
            auth.username,
            auth.password
        ):
            return Response(
                "Authentication required.",
                401,
                {
                    "WWW-Authenticate":
                    'Basic realm="Webazi Traffic Watch"'
                },
            )

        return view(*args, **kwargs)

    return wrapped


# ============================================================
# CSV helpers
# ============================================================

def _read_csv(path):

    if not os.path.isfile(path):
        return []

    try:
        with open(
            path,
            newline="",
            encoding="utf-8"
        ) as f:
            return list(csv.DictReader(f))

    except Exception as e:
        print(f"CSV read error for {path}: {e}")
        return []


def _file_age_seconds(path):

    if not os.path.isfile(path):
        return None

    try:
        return time.time() - os.path.getmtime(path)

    except OSError:
        return None


# ============================================================
# Dashboard statistics
# ============================================================

def build_stats():

    speed_rows = _read_csv(SPEED_LOG_PATH)
    count_rows = _read_csv(COUNTS_LOG_PATH)
    alert_rows = _read_csv(ALERTS_LOG_PATH)

    avg_kphs = []

    max_kphs = []

    for row in speed_rows:

        try:
            if row.get("avg_kph"):
                avg_kphs.append(
                    float(row["avg_kph"])
                )

            if row.get("max_kph"):
                max_kphs.append(
                    float(row["max_kph"])
                )

        except (ValueError, TypeError):
            continue


    class_counts = {}

    for row in count_rows:

        try:

            class_name = row.get(
                "class",
                "Unknown"
            )

            class_counts[class_name] = int(
                row.get("count", 0)
            )

        except (ValueError, TypeError):
            continue


    total_vehicles = sum(
        class_counts.values()
    )


    alerts = sorted(
        alert_rows,
        key=lambda r: r.get(
            "timestamp",
            ""
        ),
        reverse=True
    )[:MAX_ALERTS_SHOWN]


    for alert in alerts:

        try:
            alert["kph"] = float(
                alert.get("kph", 0)
            )

        except (ValueError, TypeError):
            alert["kph"] = 0


    pulse = [
        float(alert["kph"])
        for alert in reversed(alerts)
        if alert.get("kph") is not None
    ]


    log_age = _file_age_seconds(
        SPEED_LOG_PATH
    )

    # Tracker writes logs every 30 seconds
    is_live = (
        log_age is not None
        and log_age < 90
    )


    uptime_s = int(
        time.time() - START_TIME
    )


    video_exists = os.path.isfile(
        PROCESSED_VIDEO_PATH
    )

    video_size = (
        os.path.getsize(
            PROCESSED_VIDEO_PATH
        )
        if video_exists
        else 0
    )


    video_mtime = (
        os.path.getmtime(
            PROCESSED_VIDEO_PATH
        )
        if video_exists
        else None
    )


    return {
        "live": is_live,

        "last_update": datetime.now().strftime(
            "%H:%M:%S"
        ),

        "uptime_s": uptime_s,

        "total_vehicles": total_vehicles,

        "class_counts": class_counts,

        "avg_kph": round(
            sum(avg_kphs) / len(avg_kphs),
            1
        ) if avg_kphs else 0,

        "max_kph": round(
            max(max_kphs),
            1
        ) if max_kphs else 0,

        "tracked_vehicles": len(
            speed_rows
        ),

        "alert_count": len(
            alert_rows
        ),

        "alerts": alerts,

        "pulse": pulse,

        "video_available": video_exists,

        "video_size": video_size,

        "video_modified": (
            datetime.fromtimestamp(
                video_mtime
            ).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if video_mtime
            else None
        ),
    }


# ============================================================
# Health check
# ============================================================

@app.route("/health")
def health():

    return "OK", 200


# ============================================================
# Dashboard
# ============================================================

@app.route("/")
@require_auth
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# Statistics API
# ============================================================

@app.route("/api/stats")
@require_auth
def api_stats():

    return jsonify(
        build_stats()
    )


# ============================================================
# Processed video
# ============================================================

@app.route("/processed-video")
@require_auth
def processed_video():

    if not os.path.isfile(
        PROCESSED_VIDEO_PATH
    ):
        abort(404)

    response = send_file(
        PROCESSED_VIDEO_PATH,
        mimetype="video/mp4",
        conditional=True,
        max_age=0,
    )

    response.headers[
        "Cache-Control"
    ] = "no-cache, no-store, must-revalidate"

    return response


# ============================================================
# Video status
# ============================================================

@app.route("/api/video-status")
@require_auth
def video_status():

    if not os.path.isfile(
        PROCESSED_VIDEO_PATH
    ):
        return jsonify({
            "available": False
        })

    stat = os.stat(
        PROCESSED_VIDEO_PATH
    )

    return jsonify({
        "available": True,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(
            stat.st_mtime
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    })


# ============================================================
# Snapshots
# ============================================================

@app.route("/snapshots/<path:filename>")
@require_auth
def snapshots(filename):

    if not os.path.isdir(
        SNAPSHOT_DIR
    ):
        abort(404)

    return send_from_directory(
        SNAPSHOT_DIR,
        filename
    )


# ============================================================
# Local development
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )