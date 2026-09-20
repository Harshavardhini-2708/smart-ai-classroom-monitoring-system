"""
web.py - AI Classroom Flask Server
====================================
All API routes, JSON persistence, history management.
Run with:  python web.py
"""

import os
import json
import threading
import webbrowser
import time
from datetime import datetime
from flask import Flask, jsonify, request, render_template, abort

import app as detection_engine

# Re-export persistence helpers so any legacy callers still work
from app import save_class_history, append_attendance_txt

# ─────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE     = os.path.join(BASE_DIR, "class_history.json")
TIMETABLE_FILE   = os.path.join(BASE_DIR, "timetable.json")
ATTENDANCE_FILE  = os.path.join(BASE_DIR, "attendance.txt")
STUDENT_FACES    = os.path.join(BASE_DIR, "student_faces")
TEACHER_FACES    = os.path.join(BASE_DIR, "teacher_faces")

# ─────────────────────────────────────────
#  Flask app
# ─────────────────────────────────────────
flask_app = Flask(__name__, template_folder="templates", static_folder="static")
flask_app.config["JSON_SORT_KEYS"] = False

_history_lock   = threading.Lock()
_timetable_lock = threading.Lock()


# ─────────────────────────────────────────
#  JSON helpers
# ─────────────────────────────────────────
def _load_json(path, default):
    """Load JSON file safely; return default if missing or corrupt."""
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _save_json(path, data):
    """Save data as pretty JSON atomically."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ─────────────────────────────────────────
#  Timetable helpers
# ─────────────────────────────────────────
def load_timetable():
    with _timetable_lock:
        return _load_json(TIMETABLE_FILE, {"periods": []})


def save_timetable(data):
    with _timetable_lock:
        _save_json(TIMETABLE_FILE, data)


def get_period_info(period_number):
    """Return timetable entry for given period number, or None."""
    tt = load_timetable()
    for p in tt.get("periods", []):
        if str(p.get("period")) == str(period_number):
            return p
    return None


def get_current_period():
    """
    Return the timetable period whose time window contains now.
    Falls back to the next upcoming period, then period 1.
    """
    tt = load_timetable()
    periods = tt.get("periods", [])
    if not periods:
        return None

    now = datetime.now()
    now_minutes = now.hour * 60 + now.minute

    # Find active period
    for p in periods:
        try:
            sh, sm = map(int, p["start_time"].split(":"))
            eh, em = map(int, p["end_time"].split(":"))
            start_m = sh * 60 + sm
            end_m   = eh * 60 + em
            if start_m <= now_minutes < end_m:
                return p
        except Exception:
            continue

    # Find next upcoming period
    upcoming = []
    for p in periods:
        try:
            sh, sm = map(int, p["start_time"].split(":"))
            start_m = sh * 60 + sm
            if start_m > now_minutes:
                upcoming.append((start_m, p))
        except Exception:
            continue

    if upcoming:
        upcoming.sort(key=lambda x: x[0])
        return upcoming[0][1]

    # Default to first period
    return periods[0]


# ─────────────────────────────────────────
#  History helpers
# ─────────────────────────────────────────
def load_history():
    with _history_lock:
        return _load_json(HISTORY_FILE, {"classes": []})


# save_class_history and append_attendance_txt live in app.py (no circular import)


# ─────────────────────────────────────────
#  Ensure required directories exist
# ─────────────────────────────────────────
for _d in [STUDENT_FACES, TEACHER_FACES,
           os.path.join(BASE_DIR, "templates"),
           os.path.join(BASE_DIR, "static")]:
    os.makedirs(_d, exist_ok=True)

# Ensure JSON files exist
if not os.path.isfile(HISTORY_FILE):
    _save_json(HISTORY_FILE, {"classes": []})
if not os.path.isfile(TIMETABLE_FILE):
    _save_json(TIMETABLE_FILE, {"periods": []})


# ═══════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════

# ── Main page ──────────────────────────────
@flask_app.route("/")
def index():
    return render_template("index.html")


# ── Live status ────────────────────────────
@flask_app.route("/status")
def status():
    snap = detection_engine.get_state_snapshot()

    # Augment with timetable info when not running
    if not snap["running"]:
        current = get_current_period()
        if current:
            snap["current_period_info"] = current

    return jsonify(snap)


# ── Start detection ────────────────────────
@flask_app.route("/start-detection", methods=["POST"])
def start_detection():
    body = request.get_json(silent=True) or {}
    period_number = body.get("period")

    if period_number is None:
        # Auto-detect from clock
        p_info = get_current_period()
    else:
        p_info = get_period_info(period_number)

    if not p_info:
        return jsonify({"success": False,
                        "error": "Could not determine period. Check timetable."}), 400

    # Ensure required fields
    if "teacher_display" not in p_info:
        p_info["teacher_display"] = p_info.get("teacher", "Unknown Teacher").replace("_", " ")

    duration = int(body.get("duration_seconds", 3600))
    ok, err = detection_engine.start_detection(p_info, duration)
    if not ok:
        return jsonify({"success": False, "error": err}), 409

    return jsonify({"success": True,
                    "message": f"Detection started for Period {p_info['period']} – {p_info['subject']}",
                    "period_info": p_info})


# ── Stop detection ─────────────────────────
@flask_app.route("/stop-detection", methods=["POST"])
def stop_detection():
    ok, err = detection_engine.stop_detection()
    if not ok:
        return jsonify({"success": False, "error": err}), 409
    return jsonify({"success": True, "message": "Detection stopped and class saved."})


# ── History list ───────────────────────────
@flask_app.route("/history")
def history():
    data = load_history()
    classes = data.get("classes", [])

    # Optional query filters
    date_f    = request.args.get("date")
    period_f  = request.args.get("period")
    subject_f = request.args.get("subject", "").strip().lower()
    teacher_f = request.args.get("teacher", "").strip().lower()

    if date_f:
        classes = [c for c in classes if c.get("date") == date_f]
    if period_f:
        classes = [c for c in classes if str(c.get("period")) == str(period_f)]
    if subject_f:
        classes = [c for c in classes if subject_f in c.get("subject", "").lower()]
    if teacher_f:
        classes = [c for c in classes if teacher_f in c.get("teacher", "").lower()]

    # Sort newest first
    classes = sorted(classes, key=lambda c: (c.get("date", ""), c.get("period", 0)),
                     reverse=True)

    return jsonify({"classes": classes})


# ── Clear history — MUST be defined before /history/<class_id> ──
@flask_app.route("/history/clear", methods=["POST"])
def clear_history():
    with _history_lock:
        _save_json(HISTORY_FILE, {"classes": []})
    return jsonify({"success": True, "message": "History cleared."})


# ── Single class detail ────────────────────
@flask_app.route("/history/<class_id>")
def history_detail(class_id):
    data = load_history()
    for c in data.get("classes", []):
        if c.get("class_id") == class_id:
            return jsonify(c)
    abort(404, description=f"Class '{class_id}' not found in history.")


# ── Students list ──────────────────────────
@flask_app.route("/students")
def students():
    result = detection_engine.get_registered_students()
    return jsonify({"students": result})


# ── Register student ───────────────────────
@flask_app.route("/students/register", methods=["POST"])
def register_student():
    body = request.get_json(silent=True) or {}
    name = body.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "error": "Student name is required."}), 400

    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
    if not safe_name:
        return jsonify({"success": False, "error": "Invalid student name."}), 400

    num_images = int(body.get("num_images", 5))
    # Start async job — returns immediately, camera opens in background
    job_id = detection_engine.start_registration_job(safe_name, "student", num_images)
    return jsonify({"success": True, "job_id": job_id,
                    "message": f"Camera opening for {safe_name}. Watch for the camera window on your screen."})


# ── Registration job status (poll from browser) ────
@flask_app.route("/registration-status/<job_id>")
def registration_status(job_id):
    s = detection_engine.get_registration_status(job_id)
    return jsonify(s)


# ── Delete student ─────────────────────────
@flask_app.route("/students/<student_name>", methods=["DELETE"])
def delete_student(student_name):
    import shutil
    folder = os.path.join(STUDENT_FACES, student_name)
    if not os.path.isdir(folder):
        return jsonify({"success": False, "error": "Student not found."}), 404
    try:
        shutil.rmtree(folder)
        return jsonify({"success": True, "message": f"Student '{student_name}' removed."})
    except OSError as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Teachers list ──────────────────────────
@flask_app.route("/teachers")
def teachers():
    result = detection_engine.get_registered_teachers()
    # Also include timetable teacher info
    tt = load_timetable()
    tt_teachers = {}
    for p in tt.get("periods", []):
        key = p.get("teacher", "")
        if key and key not in tt_teachers:
            tt_teachers[key] = p.get("teacher_display", key)

    enriched = []
    for t in result:
        enriched.append({
            "name":         t["name"],
            "display_name": tt_teachers.get(t["name"], t["name"].replace("_", " ")),
            "subject":      _teacher_subject(t["name"], tt),
            "samples":      t["samples"],
        })
    return jsonify({"teachers": enriched})


def _teacher_subject(teacher_key, tt):
    subjects = []
    for p in tt.get("periods", []):
        if p.get("teacher") == teacher_key:
            s = p.get("subject", "")
            if s not in subjects:
                subjects.append(s)
    return ", ".join(subjects) if subjects else "—"


# ── Register teacher ───────────────────────
@flask_app.route("/teachers/register", methods=["POST"])
def register_teacher():
    body = request.get_json(silent=True) or {}
    name = body.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "error": "Teacher name is required."}), 400

    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
    if not safe_name:
        return jsonify({"success": False, "error": "Invalid teacher name."}), 400

    num_images = int(body.get("num_images", 5))
    # Start async job — returns immediately
    job_id = detection_engine.start_registration_job(safe_name, "teacher", num_images)
    return jsonify({"success": True, "job_id": job_id,
                    "message": f"Camera opening for {safe_name}. Watch for the camera window on your screen."})


# ── Delete teacher ─────────────────────────
@flask_app.route("/teachers/<teacher_name>", methods=["DELETE"])
def delete_teacher(teacher_name):
    import shutil
    folder = os.path.join(TEACHER_FACES, teacher_name)
    if not os.path.isdir(folder):
        return jsonify({"success": False, "error": "Teacher not found."}), 404
    try:
        shutil.rmtree(folder)
        return jsonify({"success": True, "message": f"Teacher '{teacher_name}' removed."})
    except OSError as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Timetable GET ──────────────────────────
@flask_app.route("/timetable", methods=["GET"])
def get_timetable():
    return jsonify(load_timetable())


# ── Timetable POST (replace) ───────────────
@flask_app.route("/timetable", methods=["POST"])
def set_timetable():
    body = request.get_json(silent=True)
    if not body or "periods" not in body:
        return jsonify({"success": False, "error": "Invalid timetable data."}), 400

    periods = body["periods"]
    # Validate each entry
    validated = []
    for i, p in enumerate(periods):
        try:
            entry = {
                "period":          int(p["period"]),
                "subject":         str(p.get("subject", "")).strip(),
                "teacher":         str(p.get("teacher", "")).strip(),
                "teacher_display": str(p.get("teacher_display",
                                             p.get("teacher", ""))).strip(),
                "start_time":      str(p.get("start_time", "")).strip(),
                "end_time":        str(p.get("end_time", "")).strip(),
            }
            if not entry["subject"] or not entry["teacher"]:
                return jsonify({"success": False,
                                "error": f"Period {i+1} missing subject or teacher."}), 400
            validated.append(entry)
        except (KeyError, ValueError) as e:
            return jsonify({"success": False,
                            "error": f"Invalid period data at index {i}: {e}"}), 400

    validated.sort(key=lambda x: x["period"])
    save_timetable({"periods": validated})
    return jsonify({"success": True,
                    "message": f"Timetable saved with {len(validated)} periods."})


# ── Dashboard summary ──────────────────────
@flask_app.route("/dashboard-summary")
def dashboard_summary():
    today = datetime.now().strftime("%Y-%m-%d")
    history_data = load_history()
    today_classes = [c for c in history_data.get("classes", [])
                     if c.get("date") == today]
    completed    = [c for c in today_classes if c.get("status") == "COMPLETED"]
    total_present  = sum(c.get("present", 0)  for c in completed)
    total_absent   = sum(c.get("absent", 0)   for c in completed)
    total_sleeping = sum(c.get("sleeping", 0) for c in completed)
    total_eating   = sum(c.get("eating", 0)   for c in completed)

    snap = detection_engine.get_state_snapshot()

    return jsonify({
        "today":             today,
        "total_today":       len(today_classes),
        "completed_today":   len(completed),
        "total_present":     total_present,
        "total_absent":      total_absent,
        "total_sleeping":    total_sleeping,
        "total_eating":      total_eating,
        "live":              snap,
        "today_classes":     today_classes,
    })


# ── Alerts ─────────────────────────────────
@flask_app.route("/alerts")
def alerts():
    snap = detection_engine.get_state_snapshot()
    return jsonify({"alerts": snap.get("alerts", [])})


# (clear_history route defined above, before /history/<class_id>)


# ── Download attendance.txt ────────────────
@flask_app.route("/download/attendance")
def download_attendance():
    from flask import send_file
    path = os.path.join(BASE_DIR, "attendance.txt")
    if not os.path.isfile(path):
        return jsonify({"error": "attendance.txt not found."}), 404
    return send_file(path, as_attachment=True, download_name="attendance.txt",
                     mimetype="text/plain")


# ── Test registration (debug) ──────────────
@flask_app.route("/test-register")
def test_register():
    try:
        job_id = detection_engine.start_registration_job("TestPerson", "student", 3)
        return jsonify({"success": True, "job_id": job_id})
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e),
                        "traceback": traceback.format_exc()}), 500


# ── Health check ───────────────────────────
@flask_app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


# ─────────────────────────────────────────
#  Error handlers
# ─────────────────────────────────────────
@flask_app.errorhandler(404)
def not_found(e):
    return jsonify({"error": str(e)}), 404


@flask_app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error.", "detail": str(e)}), 500


# ─────────────────────────────────────────
#  Browser auto-open
# ─────────────────────────────────────────
def _open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:5000")


# ─────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  AI CLASSROOM MANAGEMENT SYSTEM")
    print("  Starting Flask server...")
    print("  URL: http://127.0.0.1:5000")
    print("=" * 55)

    threading.Thread(target=_open_browser, daemon=True).start()

    flask_app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,   # Must be False – camera thread safety
        threaded=True,
    )
