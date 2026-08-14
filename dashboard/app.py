"""
dashboard/app.py

Lightweight read-only dashboard for vehicle_speed_tracker.py. Reads the same
CSV files and snapshot images the tracker writes - no separate database.

Run it co-located with the tracker (same VPS, same container, same disk):
    pip install -r dashboard/requirements.txt
    python dashboard/app.py

Config is via environment variables so it can point at whatever paths your
config.yaml uses:
    SPEED_LOG_PATH    default: speed_log.csv
    COUNTS_LOG_PATH   default: vehicle_counts.csv
    ALERTS_LOG_PATH   default: alerts.csv
    SNAPSHOT_DIR      default: snapshots
    PORT              default: 8000

Auth (required - the app refuses to start without DASHBOARD_PASS set):
    DASHBOARD_USER    default: admin
    DASHBOARD_PASS    no default - must be set
"""

import csv
import hmac
import os
import time
from datetime import datetime
from functools import wraps

from flask import Flask, jsonify, render_template, send_from_directory, abort, request, Response

SPEED_LOG_PATH = os.environ.get("SPEED_LOG_PATH", "speed_log.csv")
COUNTS_LOG_PATH = os.environ.get("COUNTS_LOG_PATH", "vehicle_counts.csv")
ALERTS_LOG_PATH = os.environ.get("ALERTS_LOG_PATH", "alerts.csv")
SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "snapshots")
MAX_ALERTS_SHOWN = 30

DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "admin")
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASS")  # required - app refuses to start without it

if not DASHBOARD_PASS:
    # Checked at import time (not just in __main__) so this also fires when
    # gunicorn imports the app directly, e.g. via start.sh.
    raise SystemExit(
        "DASHBOARD_PASS environment variable is not set. Refusing to start "
        "an unprotected dashboard. Set DASHBOARD_USER and DASHBOARD_PASS "
        "before running (see the deployment docs)."
    )

START_TIME = time.time()

app = Flask(__name__)


def _check_auth(username, password):
    # hmac.compare_digest avoids leaking timing info that could help guess the password
    return (
        hmac.compare_digest(username, DASHBOARD_USER)
        and hmac.compare_digest(password, DASHBOARD_PASS)
    )


def require_auth(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        auth = request.authorization
        if not auth or not _check_auth(auth.username, auth.password):
            return Response(
                "Authentication required.", 401,
                {"WWW-Authenticate": 'Basic realm="Webazi Traffic Watch"'},
            )
        return view(*args, **kwargs)
    return wrapped


def _read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _file_age_seconds(path):
    if not os.path.isfile(path):
        return None
    return time.time() - os.path.getmtime(path)


def build_stats():
    speed_rows = _read_csv(SPEED_LOG_PATH)
    count_rows = _read_csv(COUNTS_LOG_PATH)
    alert_rows = _read_csv(ALERTS_LOG_PATH)

    avg_kphs = [float(r["avg_kph"]) for r in speed_rows if r.get("avg_kph")]
    max_kphs = [float(r["max_kph"]) for r in speed_rows if r.get("max_kph")]

    class_counts = {r["class"]: int(r["count"]) for r in count_rows}
    total_vehicles = sum(class_counts.values())

    alerts = sorted(alert_rows, key=lambda r: r.get("timestamp", ""), reverse=True)[:MAX_ALERTS_SHOWN]
    for a in alerts:
        a["kph"] = float(a["kph"]) if a.get("kph") else 0

    # Recent alert speeds, oldest -> newest, for the pulse strip (signature element)
    pulse = [float(r["kph"]) for r in reversed(alerts)] if alerts else []

    log_age = _file_age_seconds(SPEED_LOG_PATH)
    is_live = log_age is not None and log_age < 90  # tracker flushes every 30s

    uptime_s = int(time.time() - START_TIME)

    return {
        "live": is_live,
        "last_update": datetime.now().strftime("%H:%M:%S"),
        "uptime_s": uptime_s,
        "total_vehicles": total_vehicles,
        "class_counts": class_counts,
        "avg_kph": round(sum(avg_kphs) / len(avg_kphs), 1) if avg_kphs else 0,
        "max_kph": round(max(max_kphs), 1) if max_kphs else 0,
        "tracked_vehicles": len(speed_rows),
        "alert_count": len(alert_rows),
        "alerts": alerts,
        "pulse": pulse,
    }


@app.route("/")
@require_auth
def index():
    return render_template("index.html")


@app.route("/api/stats")
@require_auth
def api_stats():
    return jsonify(build_stats())


@app.route("/snapshots/<path:filename>")
@require_auth
def snapshots(filename):
    if not os.path.isdir(SNAPSHOT_DIR):
        abort(404)
    return send_from_directory(SNAPSHOT_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
