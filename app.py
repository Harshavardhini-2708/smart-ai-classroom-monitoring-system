

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import json
import time
import threading
import winsound
from datetime import datetime, timedelta

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# ---------------- DETECTION SETTINGS ----------------
# Confirm behavior over multiple frames so brief blinks/hand movements do not
# immediately trigger sleeping/eating. The camera itself remains live until
# the user presses Stop Detection.
EAR_THRESHOLD = 0.25
SLEEP_FRAMES = 20
EAT_FRAMES = 15
# Distance is measured relative to face height, not the whole image.
EAT_PROXIMITY = 0.55

# Camera resolution: much faster than 1280x720 while still usable.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Face recognition threshold.
RECOGNITION_THRESHOLD = 0.72

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_HISTORY_FILE = os.path.join(_BASE_DIR, "class_history.json")
_ATTENDANCE_FILE = os.path.join(_BASE_DIR, "attendance.txt")
_HISTORY_LOCK = threading.Lock()


# ---------------- PERSISTENCE ----------------
def _load_json_safe(path, default):
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def save_class_history(class_record):
    with _HISTORY_LOCK:
        data = _load_json_safe(_HISTORY_FILE, {"classes": []})
        classes = data.get("classes", [])
        replaced = False
        for i, c in enumerate(classes):
            if c.get("class_id") == class_record.get("class_id"):
                classes[i] = class_record
                replaced = True
                break
        if not replaced:
            classes.append(class_record)
        data["classes"] = classes
        _save_json_atomic(_HISTORY_FILE, data)


def append_attendance_txt(class_record):
    lines = [
        "", "=" * 60,
        f"Date   : {class_record.get('date', 'N/A')}",
        f"Period : {class_record.get('period', 'N/A')}",
        f"Subject: {class_record.get('subject', 'N/A')}",
        f"Teacher: {class_record.get('teacher', 'N/A')}",
        f"Time   : {class_record.get('start_time', '')} - {class_record.get('end_time', '')}",
        f"Present: {class_record.get('present', 0)} / Absent: {class_record.get('absent', 0)}",
        f"Sleeping: {class_record.get('sleeping', 0)}  Eating: {class_record.get('eating', 0)}",
        "-" * 40, "Student Attendance:",
    ]
    for sname, sinfo in sorted(class_record.get("students", {}).items()):
        if sname == "__unknown__":
            continue
        lines.append(
            f"  {sname:<25} {sinfo.get('status', 'ABSENT'):<10} "
            f"{sinfo.get('activity', 'ABSENT')}"
        )
    lines.extend(["=" * 60, ""])
    try:
        with open(_ATTENDANCE_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


# ---------------- STATE ----------------
class DetectionState:
    def __init__(self):
        self._lock = threading.Lock()
        self.running = False
        self.period = None
        self.subject = None
        self.teacher = None
        self.teacher_display = None
        self.date = None
        self.start_time = None
        self.end_time = None
        self.class_id = None
        self.end_timestamp = None
        self.teacher_verified = True
        self.students = {}
        self.total_registered = 0
        self.present_count = 0
        self.absent_count = 0
        self.sleeping_count = 0
        self.eating_count = 0
        self.unknown_count = 0
        self.alerts = []
        self.buzzer_cooldowns = {}
        self.student_encodings = {}
        self.teacher_encodings = {}

    def snapshot(self):
        with self._lock:
            remaining = 0
            if self.running and self.end_timestamp:
                remaining = max(
                    0, int((self.end_timestamp - datetime.now()).total_seconds())
                )
            return {
                "running": self.running,
                "period": self.period,
                "subject": self.subject,
                "teacher": self.teacher_display,
                "date": self.date,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "class_id": self.class_id,
                "remaining_seconds": remaining,
                "total_students": self.total_registered,
                "present": self.present_count,
                "absent": self.absent_count,
                "sleeping": self.sleeping_count,
                "eating": self.eating_count,
                "unknown": self.unknown_count,
                "teacher_verified": True,
                "students": {k: dict(v) for k, v in self.students.items()},
                "alerts": list(self.alerts[:20]),
            }

    def update_counts(self):
        present = sleeping = eating = unknown = 0
        for name, info in self.students.items():
            if name == "__unknown__":
                unknown += info.get("count", 1)
                continue
            if info.get("status") == "PRESENT":
                present += 1
            if info.get("activity") == "SLEEPING":
                sleeping += 1
            if info.get("activity") == "EATING":
                eating += 1
        self.present_count = present
        self.absent_count = max(0, self.total_registered - present)
        self.sleeping_count = sleeping
        self.eating_count = eating
        self.unknown_count = unknown


state = DetectionState()
_camera_thread = None


# ---------------- FACE RECOGNITION ----------------
_KEY_IDS = [
    1, 33, 61, 133, 152, 168, 197, 234, 263, 291,
    362, 389, 397, 454, 159, 145, 386, 374,
    4, 5, 6, 13, 14, 78, 308, 0, 17, 18,
    200, 50, 280, 330, 100, 70, 300, 336, 107,
    10, 338, 297, 332,
]


def _embedding_from_landmarks(lm):
    vec = []
    for idx in _KEY_IDS:
        if idx < len(lm):
            vec.extend([lm[idx].x, lm[idx].y, lm[idx].z])
    if not vec:
        return None
    arr = np.asarray(vec, dtype=np.float32)
    arr -= arr.mean()
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def _get_embedding(image_rgb, fm):
    res = fm.process(image_rgb)
    if not res.multi_face_landmarks:
        return None
    return _embedding_from_landmarks(res.multi_face_landmarks[0].landmark)


def _cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


def load_encodings_from_folder(base_folder):
    encodings = {}
    if not os.path.isdir(base_folder):
        return encodings

    fm = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
    )
    try:
        for person in os.listdir(base_folder):
            folder = os.path.join(base_folder, person)
            if not os.path.isdir(folder):
                continue
            embs = []
            for filename in os.listdir(folder):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(os.path.join(folder, filename))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                emb = _get_embedding(rgb, fm)
                if emb is not None:
                    embs.append(emb)
            if embs:
                encodings[person.strip()] = embs
    finally:
        fm.close()
    return encodings


def recognize_face(emb, encodings, threshold=RECOGNITION_THRESHOLD):
    best_name, best_score = "UNKNOWN", 0.0
    for name, refs in encodings.items():
        for ref in refs:
            score = _cosine(emb, ref)
            if score > best_score:
                best_score, best_name = score, name
    return (best_name, best_score) if best_score >= threshold else ("UNKNOWN", best_score)


# ---------------- FAST EYE/MOUTH/HAND DETECTION ----------------
def _pt(lm, idx, w, h):
    return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)


def _ear(lm, ids, w, h):
    p1 = _pt(lm, ids[0], w, h)
    p2 = _pt(lm, ids[1], w, h)
    p3 = _pt(lm, ids[2], w, h)
    p4 = _pt(lm, ids[3], w, h)
    p5 = _pt(lm, ids[4], w, h)
    p6 = _pt(lm, ids[5], w, h)
    a = np.linalg.norm(p2 - p6)
    b = np.linalg.norm(p3 - p5)
    c = np.linalg.norm(p1 - p4)
    return (a + b) / (2.0 * c) if c > 0 else 0.3


def get_ear(lm, w, h):
    left = _ear(lm, [263, 387, 385, 362, 380, 373], w, h)
    right = _ear(lm, [133, 160, 158, 33, 144, 153], w, h)
    return (left + right) / 2.0


# ---------------- BUZZER ----------------
def _buzz():
    try:
        winsound.Beep(1000, 350)
        time.sleep(0.08)
        winsound.Beep(1000, 350)
    except Exception:
        pass


def play_buzzer(name, activity):
    # This is deliberately NOT called every frame. It is called once when
    # the activity changes from PRESENT to SLEEPING or EATING. When the
    # student returns to PRESENT, the next detection will buzz again.
    if activity not in ("SLEEPING", "EATING"):
        return
    threading.Thread(target=_buzz, daemon=True).start()


# ---------------- DRAWING ----------------
COLORS = {
    "PRESENT": (0, 220, 80),
    "ABSENT": (100, 100, 100),
    "SLEEPING": (0, 165, 255),
    "EATING": (0, 80, 255),
    "UNKNOWN": (80, 80, 80),
}


def _label(frame, text, x, y, color):
    f = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, f, 0.5, 1)
    cv2.rectangle(frame, (x - 3, y - th - 5), (x + tw + 3, y + 3), color, cv2.FILLED)
    cv2.putText(frame, text, (x, y - 1), f, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_box(frame, x1, y1, x2, y2, name, activity, conf=None):
    color = COLORS.get(activity, COLORS["UNKNOWN"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{name} {conf:.0%}" if conf and name != "UNKNOWN" else name
    _label(frame, label, x1, max(y1 - 22, 20), color)
    _label(frame, activity, x1, max(y1 - 4, 36), color)


# ---------------- MAIN LOOP ----------------
def _run_detection(period_info, duration_seconds=3600):
    with state._lock:
        state.running = True
        state.period = str(period_info["period"])
        state.subject = period_info["subject"]
        state.teacher = period_info["teacher"]
        state.teacher_display = period_info["teacher_display"]
        state.date = datetime.now().strftime("%Y-%m-%d")
        state.start_time = datetime.now().strftime("%H:%M")
        # The live class may run for several hours, but history is saved
        # automatically as separate 1-hour class records.  A timestamp in the
        # id prevents one hour from overwriting the previous hour.
        segment_started = datetime.now()
        state.class_id = f"{state.date}_period_{state.period}_{segment_started.strftime('%H%M%S')}"
        # Do not automatically close the live detector after a few seconds or
        # after one hour. One hour is only the history-save interval.
        state.end_timestamp = None
        state.end_time = None
        state.teacher_verified = True
        state.students = {}
        state.alerts = []
        state.buzzer_cooldowns = {}
        state.total_registered = len(state.student_encodings)
        state.present_count = 0
        state.absent_count = state.total_registered
        state.sleeping_count = 0
        state.eating_count = 0
        state.unknown_count = 0

        for sname in state.student_encodings:
            state.students[sname] = {
                "status": "ABSENT",
                "activity": "ABSENT",
                "sf": 0,
                "ef": 0,
                "first_seen": None,
                "last_seen": None,
            }

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with state._lock:
            state.running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fm = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=8,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    hd = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=4,
        model_complexity=0,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    )

    # Per-student mouth movement history.
    previous_mouth_ratio = {}

    # IMPORTANT: save the class automatically every hour.  This does not
    # depend on the user opening/clicking Live Class again.
    next_auto_save = datetime.now() + timedelta(seconds=3600)
    segment_started = datetime.now()

    try:
        while True:
            now_dt = datetime.now()
            with state._lock:
                if not state.running:
                    break

            # Saving history every hour never stops the camera.
            if now_dt >= next_auto_save:
                _auto_save_hour()
                next_auto_save += timedelta(seconds=3600)
                segment_started = now_dt
                previous_mouth_ratio.clear()

            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # -------- HANDS: use only useful hand points (wrist/index/middle)
            hr = hd.process(rgb)
            hand_points = []
            if hr.multi_hand_landmarks:
                for hand in hr.multi_hand_landmarks:
                    for idx in (0, 5, 8, 9, 12):
                        p = hand.landmark[idx]
                        hand_points.append((p.x, p.y))

            # -------- FACES: one MediaPipe pass per frame
            fr = fm.process(rgb)

            elapsed = int(max(0, (now_dt - segment_started).total_seconds()))
            mins, secs = divmod(elapsed, 60)
            cv2.rectangle(frame, (0, 0), (w, 40), (10, 14, 30), cv2.FILLED)
            cv2.putText(
                frame,
                f"Period {state.period} | {state.subject} | LIVE {mins:02d}:{secs:02d}",
                (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180, 210, 255), 1, cv2.LINE_AA,
            )

            if not fr.multi_face_landmarks:
                cv2.imshow("AI Classroom", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            with state._lock:
                encodings = dict(state.student_encodings)

            multi = len(fr.multi_face_landmarks) > 1

            for face_lms in fr.multi_face_landmarks:
                lm = face_lms.landmark

                xs = [p.x * w for p in lm]
                ys = [p.y * h for p in lm]
                x1 = max(0, int(min(xs)) - 8)
                y1 = max(0, int(min(ys)) - 8)
                x2 = min(w, int(max(xs)) + 8)
                y2 = min(h, int(max(ys)) + 8)

                # IMPORTANT SPEED FIX:
                # Reuse the landmarks already calculated by FaceMesh.
                # Do NOT call FaceMesh a second time for recognition.
                emb = _embedding_from_landmarks(lm)
                identity, conf = "UNKNOWN", 0.0
                if emb is not None and encodings:
                    identity, conf = recognize_face(emb, encodings)

                # -------- SLEEPING
                ear = get_ear(lm, w, h)
                eyes_closed = ear < EAR_THRESHOLD

                # -------- EATING
                face_height = max(abs(lm[10].y - lm[152].y), 1e-6)
                mouth_ratio = abs(lm[MOUTH_TOP].y - lm[MOUTH_BOTTOM].y) / face_height
                mouth_x = (lm[MOUTH_LEFT].x + lm[MOUTH_RIGHT].x) / 2.0
                mouth_y = (lm[MOUTH_TOP].y + lm[MOUTH_BOTTOM].y) / 2.0

                # Only hand points near the mouth count. Distance is normalized
                # by face height, so it works at different camera distances.
                nearest_hand_distance = 999.0
                for hx, hy in hand_points:
                    d = float(np.hypot(hx - mouth_x, hy - mouth_y))
                    if d < nearest_hand_distance:
                        nearest_hand_distance = d

                # Face-relative threshold so the same rule works when a student
                # is closer to or farther from the camera.
                hand_near_mouth = nearest_hand_distance <= max(0.045, EAT_PROXIMITY * face_height)

                previous_ratio = previous_mouth_ratio.get(identity, mouth_ratio)
                mouth_motion = abs(mouth_ratio - previous_ratio) >= 0.006
                if identity != "UNKNOWN":
                    previous_mouth_ratio[identity] = mouth_ratio

                mouth_open = mouth_ratio >= 0.040
                eating_signal = hand_near_mouth and (mouth_open or mouth_motion)

                now_ts = datetime.now().strftime("%H:%M:%S")
                activity = "UNKNOWN"
                prev = "UNKNOWN"

                with state._lock:
                    if identity != "UNKNOWN":
                        if identity not in state.students:
                            state.students[identity] = {
                                "status": "ABSENT",
                                "activity": "ABSENT",
                                "sf": 0,
                                "ef": 0,
                                "first_seen": None,
                                "last_seen": None,
                            }
                        s = state.students[identity]

                        if s["status"] != "PRESENT":
                            s["status"] = "PRESENT"
                            s["first_seen"] = now_ts
                        s["last_seen"] = now_ts

                        # Slow confirmation: require sustained evidence and let the
                        # counters recover gradually when the signal disappears.
                        s["sf"] = min(SLEEP_FRAMES, s["sf"] + 1) if eyes_closed else max(0, s["sf"] - 1)
                        s["ef"] = min(EAT_FRAMES, s["ef"] + 1) if eating_signal else max(0, s["ef"] - 1)

                        prev = s["activity"]

                        if s["sf"] >= SLEEP_FRAMES:
                            activity = "SLEEPING"
                        elif s["ef"] >= EAT_FRAMES:
                            activity = "EATING"
                        else:
                            activity = "PRESENT"

                        # BUZZER: every NEW event, not every video frame.
                        if activity in ("SLEEPING", "EATING") and activity != prev:
                            state.alerts.insert(0, {
                                "type": activity,
                                "student": identity,
                                "time": datetime.now().strftime("%I:%M %p"),
                            })
                            threading.Thread(
                                target=play_buzzer,
                                args=(identity, activity),
                                daemon=True,
                            ).start()

                        s["activity"] = activity

                    else:
                        state.students.setdefault("__unknown__", {"count": 0})
                        state.students["__unknown__"]["count"] += 1
                        activity = "UNKNOWN"

                    state.update_counts()

                # Small diagnostics help verify the detector live.
                debug = f"EAR {ear:.2f} | M {mouth_ratio:.2f} | H {nearest_hand_distance:.2f}"
                cv2.putText(
                    frame, debug, (x1, min(h - 8, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1,
                    cv2.LINE_AA,
                )

                _draw_box(
                    frame, x1, y1, x2, y2,
                    identity, activity,
                    conf if identity != "UNKNOWN" else None,
                )

            cv2.imshow("AI Classroom", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        fm.close()
        hd.close()
        cv2.destroyAllWindows()
        _finish_class()


def _build_class_record_locked(end_time=None, status="COMPLETED"):
    """Build a history record from the current hour/segment.

    Caller must hold state._lock.
    """
    if not state.class_id:
        return None

    for sname, sinfo in state.students.items():
        if sname != "__unknown__" and sinfo.get("status") == "ABSENT":
            sinfo["activity"] = "ABSENT"

    state.update_counts()
    end_time = end_time or datetime.now().strftime("%H:%M")

    return {
        "class_id": state.class_id,
        "date": state.date,
        "period": state.period,
        "subject": state.subject,
        "teacher": state.teacher_display,
        "start_time": state.start_time,
        "end_time": end_time,
        "total_students": state.total_registered,
        "present": state.present_count,
        "absent": state.absent_count,
        "sleeping": state.sleeping_count,
        "eating": state.eating_count,
        "unknown": state.unknown_count,
        "teacher_verified": True,
        "students": {
            k: dict(v) for k, v in state.students.items()
            if k != "__unknown__"
        },
        "alerts": list(state.alerts),
        "status": status,
        "saved_automatically": True,
    }


def _reset_hour_locked(segment_started):
    """Start a new automatic one-hour history segment without stopping detection.

    Caller must hold state._lock.
    """
    state.date = segment_started.strftime("%Y-%m-%d")
    state.start_time = segment_started.strftime("%H:%M")
    state.class_id = (
        f"{state.date}_period_{state.period}_{segment_started.strftime('%H%M%S')}"
    )
    state.students = {}
    state.alerts = []
    state.buzzer_cooldowns = {}
    state.total_registered = len(state.student_encodings)
    state.present_count = 0
    state.absent_count = state.total_registered
    state.sleeping_count = 0
    state.eating_count = 0
    state.unknown_count = 0

    for sname in state.student_encodings:
        state.students[sname] = {
            "status": "ABSENT",
            "activity": "ABSENT",
            "sf": 0,
            "ef": 0,
            "first_seen": None,
            "last_seen": None,
        }


def _auto_save_hour():
    """Save the finished hour immediately, then continue with a new hour."""
    with state._lock:
        if not state.running or not state.class_id:
            return
        finished_at = datetime.now()
        record = _build_class_record_locked(
            finished_at.strftime("%H:%M"), status="COMPLETED"
        )

        # Prepare the next hour while the camera thread remains alive.
        _reset_hour_locked(finished_at)

    if record:
        save_class_history(record)
        append_attendance_txt(record)


def _finish_class():
    # Save the final hour/partial hour automatically when the live session ends.
    # This also makes stopping the camera safe: nothing is lost just because the
    # user did not open the Past Classes screen.
    with state._lock:
        if not state.class_id:
            state.running = False
            return
        record = _build_class_record_locked(
            datetime.now().strftime("%H:%M"), status="COMPLETED"
        )
        state.running = False

    if record:
        save_class_history(record)
        append_attendance_txt(record)


# ---------------- PUBLIC API ----------------
def start_detection(period_info, duration_seconds=0):
    global _camera_thread
    with state._lock:
        if state.running:
            return False, "A class is already running."

    student_dir = os.path.join(_BASE_DIR, "student_faces")
    with state._lock:
        state.student_encodings = load_encodings_from_folder(student_dir)
        state.teacher_encodings = {}

    _camera_thread = threading.Thread(
        target=_run_detection,
        args=(period_info, duration_seconds),
        daemon=True,
    )
    _camera_thread.start()
    return True, ""


def stop_detection():
    with state._lock:
        if not state.running:
            return False, "No class running."
        state.running = False
    return True, ""


def get_state_snapshot():
    return state.snapshot()


def get_registered_students():
    base = os.path.join(_BASE_DIR, "student_faces")
    if not os.path.isdir(base):
        return []
    result = []
    for name in sorted(os.listdir(base)):
        folder = os.path.join(base, name)
        if not os.path.isdir(folder):
            continue
        samples = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        result.append({"name": name, "samples": len(samples)})
    return result


def get_registered_teachers():
    return []


# ---------------- REGISTRATION ----------------
_reg_jobs = {}
_reg_lock = threading.Lock()


def start_registration_job(person_name, person_type="student", num_images=5):
    base_folder = "student_faces" if person_type == "student" else "teacher_faces"
    save_dir = os.path.join(_BASE_DIR, base_folder, person_name)
    os.makedirs(save_dir, exist_ok=True)
    job_id = f"{person_type}_{person_name}_{int(time.time())}"
    with _reg_lock:
        _reg_jobs[job_id] = {
            "status": "running",
            "saved": 0,
            "total": num_images,
            "error": None,
        }

    t = threading.Thread(
        target=_run_register_subprocess,
        args=(job_id, person_name, person_type, num_images, save_dir),
        daemon=True,
    )
    t.start()
    return job_id


def _run_register_subprocess(job_id, person_name, person_type, num_images, save_dir):
    import subprocess
    script = os.path.join(_BASE_DIR, "register_camera.py")
    try:
        proc = subprocess.Popen(
            [sys.executable, script, person_name, person_type, str(num_images), save_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            line = line.strip()
            if "Saved image" in line:
                try:
                    n = int(line.split()[2].split("/")[0])
                    with _reg_lock:
                        _reg_jobs[job_id]["saved"] = n
                except Exception:
                    pass
        proc.wait(timeout=120)
        saved = len([
            f for f in os.listdir(save_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        with _reg_lock:
            if saved > 0:
                _reg_jobs[job_id]["status"] = "done"
                _reg_jobs[job_id]["saved"] = saved
            else:
                _reg_jobs[job_id]["status"] = "error"
                _reg_jobs[job_id]["error"] = "No images captured."
    except Exception as e:
        with _reg_lock:
            _reg_jobs[job_id]["status"] = "error"
            _reg_jobs[job_id]["error"] = str(e)


def get_registration_status(job_id):
    with _reg_lock:
        return dict(_reg_jobs.get(job_id, {"status": "not_found"}))


def capture_registration_images(person_name, person_type="student", num_images=5):
    job_id = start_registration_job(person_name, person_type, num_images)
    deadline = time.time() + 90
    while time.time() < deadline:
        status = get_registration_status(job_id)
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    status = get_registration_status(job_id)
    return (
        True,
        status["saved"],
    ) if status["status"] == "done" else (
        False,
        status.get("error", "Timeout"),
    )