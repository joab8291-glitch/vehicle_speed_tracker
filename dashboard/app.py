"""
Webazi Traffic Watch Dashboard
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
    request,
    Response,
    abort,
)


# ============================================================
# PATHS
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

VIDEO_PATH = os.environ.get(
    "VIDEO_PATH",
    "/app/data/output_speed.mp4"
)

INPUT_VIDEO_PATH = os.environ.get(
    "INPUT_VIDEO_PATH",
    "/app/video.mp4"
)

MAX_ALERTS_SHOWN = 30


# ============================================================
# AUTH
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
        "DASHBOARD_PASS environment variable "
        "is not set."
    )


# ============================================================
# FLASK
# ============================================================

START_TIME = time.time()

app = Flask(
    __name__
)


# ============================================================
# AUTH FUNCTIONS
# ============================================================

def _check_auth(
    username,
    password
):

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
    def wrapped(
        *args,
        **kwargs
    ):

        auth = request.authorization

        if (
            not auth
            or not _check_auth(
                auth.username,
                auth.password
            )
        ):

            return Response(
                "Authentication required.",
                401,
                {
                    "WWW-Authenticate":
                    'Basic realm="Webazi Traffic Watch"'
                }
            )

        return view(
            *args,
            **kwargs
        )

    return wrapped


# ============================================================
# CSV
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

            return list(
                csv.DictReader(f)
            )

    except Exception:

        return []


def _file_age_seconds(path):

    if not os.path.isfile(path):
        return None

    try:
        return (
            time.time()
            - os.path.getmtime(path)
        )
    except Exception:
        return None


# ============================================================
# STATS
# ============================================================

def build_stats():

    speed_rows = _read_csv(
        SPEED_LOG_PATH
    )

    count_rows = _read_csv(
        COUNTS_LOG_PATH
    )

    alert_rows = _read_csv(
        ALERTS_LOG_PATH
    )

    avg_kphs = []

    max_kphs = []

    for row in speed_rows:

        try:

            if row.get("avg_kph"):
                avg_kphs.append(
                    float(
                        row["avg_kph"]
                    )
                )

            if row.get("max_kph"):
                max_kphs.append(
                    float(
                        row["max_kph"]
                    )
                )

        except (
            ValueError,
            TypeError
        ):
            pass

    class_counts = {}

    for row in count_rows:

        try:

            class_name = row.get(
                "class",
                "Unknown"
            )

            count = int(
                row.get(
                    "count",
                    0
                )
            )

            class_counts[
                class_name
            ] = count

        except (
            ValueError,
            TypeError
        ):
            pass

    total_vehicles = sum(
        class_counts.values()
    )

    alerts = sorted(
        alert_rows,
        key=lambda row:
        row.get(
            "timestamp",
            ""
        ),
        reverse=True
    )[
        :MAX_ALERTS_SHOWN
    ]

    for alert in alerts:

        try:

            alert["kph"] = float(
                alert.get(
                    "kph",
                    0
                )
            )

        except (
            ValueError,
            TypeError
        ):

            alert["kph"] = 0

    pulse = [
        float(
            row["kph"]
        )
        for row in reversed(
            alerts
        )
        if row.get("kph")
    ]

    log_age = _file_age_seconds(
        SPEED_LOG_PATH
    )

    is_live = (
        log_age is not None
        and log_age < 90
    )

    output_exists = os.path.isfile(
        VIDEO_PATH
    )

    input_exists = os.path.isfile(
        INPUT_VIDEO_PATH
    )

    uptime_s = int(
        time.time()
        - START_TIME
    )

    return {

        "live": is_live,

        "last_update":
            datetime.now().strftime(
                "%H:%M:%S"
            ),

        "uptime_s":
            uptime_s,

        "total_vehicles":
            total_vehicles,

        "class_counts":
            class_counts,

        "avg_kph":
            round(
                sum(avg_kphs)
                / len(avg_kphs),
                1
            )
            if avg_kphs
            else 0,

        "max_kph":
            round(
                max(max_kphs),
                1
            )
            if max_kphs
            else 0,

        "tracked_vehicles":
            len(speed_rows),

        "alert_count":
            len(alert_rows),

        "alerts":
            alerts,

        "pulse":
            pulse,

        "video_exists":
            output_exists,

        "input_video_exists":
            input_exists,

        "video_path":
            VIDEO_PATH,
    }


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health"
)
def health():

    return "OK", 200


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
@require_auth
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# API
# ============================================================

@app.route(
    "/api/stats"
)
@require_auth
def api_stats():

    return jsonify(
        build_stats()
    )


# ============================================================
# PROCESSED VIDEO
# ============================================================

@app.route(
    "/video"
)
@require_auth
def video():

    if not os.path.isfile(
        VIDEO_PATH
    ):

        return (
            "Processed video is not "
            "available yet.",
            404
        )

    directory = os.path.dirname(
        VIDEO_PATH
    )

    filename = os.path.basename(
        VIDEO_PATH
    )

    response = send_from_directory(
        directory,
        filename,
        mimetype="video/mp4",
        conditional=True
    )

    response.headers[
        "Cache-Control"
    ] = "no-cache"

    return response


# ============================================================
# ORIGINAL VIDEO
# ============================================================

@app.route(
    "/input-video"
)
@require_auth
def input_video():

    if not os.path.isfile(
        INPUT_VIDEO_PATH
    ):

        return (
            "Input video not found.",
            404
        )

    directory = os.path.dirname(
        INPUT_VIDEO_PATH
    )

    filename = os.path.basename(
        INPUT_VIDEO_PATH
    )

    return send_from_directory(
        directory,
        filename,
        mimetype="video/mp4",
        conditional=True
    )


# ============================================================
# SNAPSHOTS
# ============================================================

@app.route(
    "/snapshots/<path:filename>"
)
@require_auth
def snapshots(
    filename
):

    if not os.path.isdir(
        SNAPSHOT_DIR
    ):

        abort(404)

    return send_from_directory(
        SNAPSHOT_DIR,
        filename
    )


# ============================================================
# RUN
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