"""
AI Based Deepfake And NCII Detection Using Computer Vision
------------------------------------------------------------
Flask backend. Uses SQLite for persistence, Werkzeug for password hashing,
and real TensorFlow CNN for image deepfake detection.
"""

import os
import sqlite3
import secrets
import string
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, g, send_from_directory
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from PIL import Image

from models.image_detector import allowed_file as image_allowed
from models.video_detector import analyze_video, allowed_file as video_allowed
from models.audio_detector import analyze_audio, allowed_file as audio_allowed
from models.ncii_detector import analyze_ncii, allowed_file as ncii_allowed

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------------------------------------------------
# TensorFlow Deepfake Model Integration
# ----------------------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_cnn_224_repaired.keras")
deepfake_model = None

# Load model globally on startup if available
if os.path.exists(MODEL_PATH):
    try:
        deepfake_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[INFO] Successfully loaded deepfake model from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load deepfake model: {e}")
else:
    print(f"[WARNING] Model file '{MODEL_PATH}' not found. Please place deepfake_cnn_224_repaired.keras in root folder.")


def analyze_deepfake_image(file_path):
    """Predicts Real vs Fake for an image using the loaded Keras model.
    Class Mapping:
    - Raw Score > 0.5 -> FAKE
    - Raw Score <= 0.5 -> REAL
    """
    if deepfake_model is None:
        return {
            "prediction": "Unknown",
            "confidence": 0.0,
            "risk_level": "Unknown",
            "model_used": "Deepfake CNN (Not Loaded)",
            "explanation": "Model file deepfake_cnn_224_repaired.keras was not found on server.",
            "recommendation": "Please place deepfake_cnn_224_repaired.keras in the root folder."
        }

    try:
        # Load & Preprocess image (224x224)
        img = Image.open(file_path).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        # Get Raw Prediction Score
        raw_score = float(deepfake_model.predict(img_array, verbose=0)[0][0])
        print(f"[DEBUG] Raw prediction score: {raw_score:.6f}")
        
        if raw_score > 0.5:
            pred = "Fake"
            conf = round(raw_score * 100, 2)
            risk = "High" if conf > 85 else "Medium"
            expl = (
                f"The trained deepfake CNN classified this image as Deepfake "
                f"with a {conf:.2f}% probability."
            )
            recom = (
                "Review the image carefully and verify its original source "
                "before relying on or sharing it."
            )
        else:
            pred = "Real"
            conf = round((1 - raw_score) * 100, 2)
            risk = "Low"
            expl = (
                f"The trained deepfake CNN classified this image as Real "
                f"with a {conf:.2f}% probability."
            )
            recom = (
                "The image was classified as Real by the trained CNN. "
                "For important or sensitive content, independently verify the original source."
            )

        return {
            "prediction": pred,
            "confidence": conf,
            "risk_level": risk,
            "model_used": "Custom Deepfake CNN (224x224)",
            "explanation": expl,
            "recommendation": recom,
            "raw_score": round(raw_score, 4)
        }
    except Exception as e:
        return {
            "prediction": "Error",
            "confidence": 0.0,
            "risk_level": "High",
            "model_used": "Custom Deepfake CNN",
            "explanation": f"Failed to analyze image: {str(e)}",
            "recommendation": "Try uploading a valid JPG or PNG image."
        }


def _get_secret_key():
    env_key = os.environ.get("SENTRA_SECRET_KEY")
    if env_key:
        return env_key

    key_path = os.path.join(BASE_DIR, ".secret_key")
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            return f.read().strip()

    new_key = secrets.token_hex(32)
    with open(key_path, "w") as f:
        f.write(new_key)
    return new_key


app = Flask(__name__)
app.config["SECRET_KEY"] = _get_secret_key()
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


# ----------------------------------------------------------------------
# Database helpers
# ----------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            college TEXT DEFAULT '',
            department TEXT DEFAULT '',
            avatar TEXT DEFAULT '',
            dark_mode INTEGER DEFAULT 1,
            language TEXT DEFAULT 'English',
            notifications INTEGER DEFAULT 1,
            voice_assistant INTEGER DEFAULT 1,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,          -- image / video / audio / ncii
            prediction TEXT NOT NULL,          -- Real / Fake / Safe / Unsafe / Potential NCII
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            model_used TEXT NOT NULL,
            explanation TEXT,
            recommendation TEXT,
            extra_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS otp_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            code TEXT NOT NULL,
            purpose TEXT NOT NULL,             -- register / reset
            expires_at TEXT NOT NULL,
            used INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS qr_sessions (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'pending',  -- pending / scanned / authenticated / expired
            user_id INTEGER,
            created_at TEXT NOT NULL
        );
        """
    )
    db.commit()
    db.close()


# ----------------------------------------------------------------------
# Auth helpers
# ----------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def current_user():
    if not session.get("user_id"):
        return None
    db = get_db()
    return db.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()


def generate_otp():
    return "".join(secrets.choice(string.digits) for _ in range(6))


@app.context_processor
def inject_user():
    return {"logged_in_user": current_user()}


# ----------------------------------------------------------------------
# Public / Marketing Pages
# ----------------------------------------------------------------------
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/contact", methods=["POST"])
def contact_submit():
    flash("Thanks for reaching out — our team will get back to you shortly.", "success")
    return redirect(url_for("contact"))


# ----------------------------------------------------------------------
# Authentication
# ----------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        college = request.form.get("college", "").strip()
        department = request.form.get("department", "").strip()

        if not name or not email or not password:
            flash("Please fill in all required fields.", "danger")
            return redirect(url_for("register"))

        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            flash("An account with this email already exists.", "danger")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)
        db.execute(
            "INSERT INTO users (name, email, password_hash, college, department, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, email, password_hash, college, department, datetime.utcnow().isoformat()),
        )
        db.commit()

        otp = generate_otp()
        expires_at = (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        db.execute(
            "INSERT INTO otp_codes (email, code, purpose, expires_at) VALUES (?, ?, 'register', ?)",
            (email, otp, expires_at),
        )
        db.commit()

        session["pending_email"] = email
        flash(f"Account created! (Demo OTP: {otp}) Enter it below to verify your email.", "info")
        return redirect(url_for("verify_otp"))

    return render_template("register.html")


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    email = session.get("pending_email")
    if not email:
        return redirect(url_for("register"))

    if request.method == "POST":
        code = request.form.get("otp", "").strip()
        db = get_db()
        row = db.execute(
            "SELECT * FROM otp_codes WHERE email = ? AND purpose = 'register' "
            "ORDER BY id DESC LIMIT 1",
            (email,),
        ).fetchone()

        if row and row["code"] == code and row["used"] == 0 and row["expires_at"] > datetime.utcnow().isoformat():
            db.execute("UPDATE otp_codes SET used = 1 WHERE id = ?", (row["id"],))
            db.commit()
            session.pop("pending_email", None)
            flash("Email verified successfully! You can now log in.", "success")
            return redirect(url_for("login"))

        flash("Invalid or expired OTP. Please try again.", "danger")

    return render_template("verify_otp.html", email=email)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = request.form.get("remember")

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session.permanent = bool(remember)
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/login/google")
def login_google():
    flash("Google Login is a placeholder in this demo build.", "info")
    return redirect(url_for("login"))


@app.route("/login/microsoft")
def login_microsoft():
    flash("Microsoft Login is a placeholder in this demo build.", "info")
    return redirect(url_for("login"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user:
            otp = generate_otp()
            expires_at = (datetime.utcnow() + timedelta(minutes=5)).isoformat()
            db.execute(
                "INSERT INTO otp_codes (email, code, purpose, expires_at) VALUES (?, ?, 'reset', ?)",
                (email, otp, expires_at),
            )
            db.commit()
            session["reset_email"] = email
            flash(f"(Demo OTP: {otp}) Enter it below to reset your password.", "info")
            return redirect(url_for("reset_password"))
        flash("No account found with that email.", "danger")

    return render_template("forgot_password.html")


@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    email = session.get("reset_email")
    if not email:
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        code = request.form.get("otp", "").strip()
        new_password = request.form.get("password", "")
        db = get_db()
        row = db.execute(
            "SELECT * FROM otp_codes WHERE email = ? AND purpose = 'reset' ORDER BY id DESC LIMIT 1",
            (email,),
        ).fetchone()

        if row and row["code"] == code and row["used"] == 0 and row["expires_at"] > datetime.utcnow().isoformat():
            db.execute("UPDATE otp_codes SET used = 1 WHERE id = ?", (row["id"],))
            db.execute(
                "UPDATE users SET password_hash = ? WHERE email = ?",
                (generate_password_hash(new_password), email),
            )
            db.commit()
            session.pop("reset_email", None)
            flash("Password reset successfully. Please log in.", "success")
            return redirect(url_for("login"))

        flash("Invalid or expired OTP.", "danger")

    return render_template("reset_password.html", email=email)


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("landing"))


# ---------------- QR Code Login ----------------
@app.route("/qr-login")
def qr_login():
    return render_template("qr_login.html")


@app.route("/api/qr/create", methods=["POST"])
def api_qr_create():
    qr_id = secrets.token_urlsafe(16)
    db = get_db()
    db.execute(
        "INSERT INTO qr_sessions (id, status, created_at) VALUES (?, 'pending', ?)",
        (qr_id, datetime.utcnow().isoformat()),
    )
    db.commit()
    return jsonify({"qr_id": qr_id, "expires_in": 60})


@app.route("/api/qr/status/<qr_id>")
def api_qr_status(qr_id):
    db = get_db()
    row = db.execute("SELECT * FROM qr_sessions WHERE id = ?", (qr_id,)).fetchone()
    if not row:
        return jsonify({"status": "expired"}), 404
    return jsonify({"status": row["status"]})


@app.route("/api/qr/simulate-scan/<qr_id>", methods=["POST"])
def api_qr_simulate_scan(qr_id):
    db = get_db()
    db.execute("UPDATE qr_sessions SET status = 'scanned' WHERE id = ?", (qr_id,))
    db.commit()
    return jsonify({"ok": True})


@app.route("/api/qr/simulate-auth/<qr_id>", methods=["POST"])
def api_qr_simulate_auth(qr_id):
    db = get_db()
    user = db.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()
    if user:
        db.execute(
            "UPDATE qr_sessions SET status = 'authenticated', user_id = ? WHERE id = ?",
            (user["id"], qr_id),
        )
        db.commit()
    return jsonify({"ok": True})


@app.route("/api/qr/complete/<qr_id>", methods=["POST"])
def api_qr_complete(qr_id):
    db = get_db()
    row = db.execute("SELECT * FROM qr_sessions WHERE id = ?", (qr_id,)).fetchone()
    if row and row["status"] == "authenticated" and row["user_id"]:
        session["user_id"] = row["user_id"]
        return jsonify({"ok": True, "redirect": url_for("dashboard")})
    return jsonify({"ok": False}), 400


# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    user_id = session["user_id"]

    total_scans = db.execute("SELECT COUNT(*) c FROM scans WHERE user_id=?", (user_id,)).fetchone()["c"]
    image_scans = db.execute("SELECT COUNT(*) c FROM scans WHERE user_id=? AND file_type='image'", (user_id,)).fetchone()["c"]
    video_scans = db.execute("SELECT COUNT(*) c FROM scans WHERE user_id=? AND file_type='video'", (user_id,)).fetchone()["c"]
    audio_scans = db.execute("SELECT COUNT(*) c FROM scans WHERE user_id=? AND file_type='audio'", (user_id,)).fetchone()["c"]
    fake_detected = db.execute(
        "SELECT COUNT(*) c FROM scans WHERE user_id=? AND prediction='Fake'", (user_id,)
    ).fetchone()["c"]
    ncii_detected = db.execute(
        "SELECT COUNT(*) c FROM scans WHERE user_id=? AND file_type='ncii' AND prediction IN ('Unsafe','Potential NCII')",
        (user_id,),
    ).fetchone()["c"]
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    today_activity = db.execute(
        "SELECT COUNT(*) c FROM scans WHERE user_id=? AND created_at LIKE ?",
        (user_id, f"{today_str}%"),
    ).fetchone()["c"]
    avg_conf_row = db.execute("SELECT AVG(confidence) a FROM scans WHERE user_id=?", (user_id,)).fetchone()
    avg_confidence = round(avg_conf_row["a"], 1) if avg_conf_row["a"] else 96.4

    recent = db.execute(
        "SELECT * FROM scans WHERE user_id=? ORDER BY created_at DESC LIMIT 8", (user_id,)
    ).fetchall()

    weekly_labels = []
    weekly_counts = []
    for i in range(6, -1, -1):
        day = (datetime.utcnow() - timedelta(days=i))
        day_str = day.strftime("%Y-%m-%d")
        count = db.execute(
            "SELECT COUNT(*) c FROM scans WHERE user_id=? AND created_at LIKE ?",
            (user_id, f"{day_str}%"),
        ).fetchone()["c"]
        weekly_labels.append(day.strftime("%a"))
        weekly_counts.append(count)

    pie_data = [image_scans, video_scans, audio_scans, max(total_scans - image_scans - video_scans - audio_scans, 0)]

    return render_template(
        "dashboard.html",
        total_scans=total_scans,
        image_scans=image_scans,
        video_scans=video_scans,
        audio_scans=audio_scans,
        fake_detected=fake_detected,
        ncii_detected=ncii_detected,
        today_activity=today_activity,
        avg_confidence=avg_confidence,
        recent=recent,
        weekly_labels=weekly_labels,
        weekly_counts=weekly_counts,
        pie_data=pie_data,
    )


# ----------------------------------------------------------------------
# Detection helper: save scan to DB
# ----------------------------------------------------------------------
def save_scan(user_id, file_name, file_type, result, prediction_key="prediction"):
    db = get_db()
    import json
    extra = {k: v for k, v in result.items() if k not in (
        "prediction", "confidence", "risk_level", "explanation", "recommendation",
        "model_used", "category"
    )}
    db.execute(
        "INSERT INTO scans (user_id, file_name, file_type, prediction, confidence, risk_level, "
        "model_used, explanation, recommendation, extra_json, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            user_id, file_name, file_type,
            result.get(prediction_key, result.get("category")),
            result["confidence"], result["risk_level"], result["model_used"],
            result["explanation"], result["recommendation"], json.dumps(extra),
            datetime.utcnow().isoformat(),
        ),
    )
    db.commit()


def unique_upload_path(filename):
    safe_name = secure_filename(filename)
    unique_name = f"{secrets.token_hex(8)}_{safe_name}"
    return unique_name, os.path.join(app.config["UPLOAD_FOLDER"], unique_name)


# ----------------------------------------------------------------------
# Image Detection (Using Real TensorFlow Model)
# ----------------------------------------------------------------------
@app.route("/detection/image")
@login_required
def image_detection_page():
    return render_template("image_detection.html")


@app.route("/api/detect/image", methods=["POST"])
@login_required
def api_detect_image():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded."}), 400
    if not image_allowed(file.filename):
        return jsonify({"error": "Only JPG, JPEG, PNG files are allowed."}), 400

    unique_name, path = unique_upload_path(file.filename)
    file.save(path)

    # Use the real model prediction logic
    result = analyze_deepfake_image(path)
    save_scan(session["user_id"], file.filename, "image", result)
    result["file_name"] = file.filename
    
    # Cleanup uploaded file after processing
    if os.path.exists(path):
        os.remove(path)

    return jsonify(result)


# ----------------------------------------------------------------------
# Video Detection
# ----------------------------------------------------------------------
@app.route("/api/detect/video", methods=["POST"])
@login_required
def api_detect_video():
    file = request.files.get("file")

    if not file or file.filename == "":
        return jsonify({"error": "No video file uploaded."}), 400

    if not video_allowed(file.filename):
        return jsonify({
            "error": "Unsupported video format. Allowed: MP4, AVI, MOV, MKV."
        }), 400

    unique_name = None
    path = None

    try:
        unique_name, path = unique_upload_path(file.filename)
        file.save(path)

        print("[VIDEO API] File:", file.filename)
        print("[VIDEO API] Saved:", path)

        result = analyze_video(path)

        print("[VIDEO API] Model result:", result)

        try:
            save_scan(
                session["user_id"],
                file.filename,
                "video",
                result
            )
            print("[VIDEO API] Scan saved successfully.")
        except Exception as db_error:
            print("[VIDEO API] Database save error:", db_error)

        result["file_name"] = file.filename

        print("[VIDEO API] Returning successful response.")
        return jsonify(result), 200

    except Exception as e:
        import traceback
        print("[VIDEO API] VIDEO ANALYSIS ERROR")
        print(str(e))
        traceback.print_exc()

        return jsonify({
            "error": "Video analysis failed.",
            "details": str(e)
        }), 500

    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                print("[VIDEO API] Temporary video deleted.")
            except Exception as cleanup_error:
                print("[VIDEO API] Cleanup error:", cleanup_error)
# ----------------------------------------------------------------------
@app.route("/detection/video")
@login_required
def video_detection_page():
    return render_template("video_detection.html")


# Audio Detection
# ----------------------------------------------------------------------
@app.route("/detection/audio")
@login_required
def audio_detection_page():
    return render_template("audio_detection.html")


@app.route("/api/detect/audio", methods=["POST"])
@login_required
def api_detect_audio():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded."}), 400
    if not audio_allowed(file.filename):
        return jsonify({"error": "Only WAV, MP3 files are allowed."}), 400

    unique_name, path = unique_upload_path(file.filename)
    file.save(path)

    result = analyze_audio(path)
    save_scan(session["user_id"], file.filename, "audio", result)
    result["file_name"] = file.filename
    return jsonify(result)


# ----------------------------------------------------------------------
# NCII Detection
# ----------------------------------------------------------------------
@app.route("/detection/ncii")
@login_required
def ncii_detection_page():
    return render_template("ncii_detection.html")


@app.route("/api/detect/ncii", methods=["POST"])
@login_required
def api_detect_ncii():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded."}), 400
    if not ncii_allowed(file.filename):
        return jsonify({"error": "Only JPG, PNG, MP4, AVI, MOV files are allowed."}), 400

    unique_name, path = unique_upload_path(file.filename)
    file.save(path)

    result = analyze_ncii(path)
    save_scan(session["user_id"], file.filename, "ncii", result, prediction_key="category")
    result["file_name"] = file.filename
    return jsonify(result)


# ----------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------
@app.route("/reports")
@login_required
def reports():
    db = get_db()
    scans = db.execute(
        "SELECT * FROM scans WHERE user_id=? ORDER BY created_at DESC", (session["user_id"],)
    ).fetchall()
    return render_template("reports.html", scans=scans)


@app.route("/reports/<int:scan_id>/download")
@login_required
def download_report(scan_id):
    db = get_db()
    scan = db.execute(
        "SELECT * FROM scans WHERE id=? AND user_id=?", (scan_id, session["user_id"])
    ).fetchone()
    if not scan:
        flash("Report not found.", "danger")
        return redirect(url_for("reports"))
    return render_template("report_detail.html", scan=scan)


# ----------------------------------------------------------------------
# Scan History
# ----------------------------------------------------------------------
@app.route("/history")
@login_required
def history():
    db = get_db()
    scans = db.execute(
        "SELECT * FROM scans WHERE user_id=? ORDER BY created_at DESC", (session["user_id"],)
    ).fetchall()
    return render_template("history.html", scans=scans)


@app.route("/history/<int:scan_id>/delete", methods=["POST"])
@login_required
def delete_scan(scan_id):
    db = get_db()
    db.execute("DELETE FROM scans WHERE id=? AND user_id=?", (scan_id, session["user_id"]))
    db.commit()
    flash("Scan record deleted.", "success")
    return redirect(url_for("history"))


# ----------------------------------------------------------------------
# Profile & Settings
# ----------------------------------------------------------------------
@app.route("/profile")
@login_required
def profile():
    db = get_db()
    total_scans = db.execute(
        "SELECT COUNT(*) c FROM scans WHERE user_id=?", (session["user_id"],)
    ).fetchone()["c"]
    recent = db.execute(
        "SELECT * FROM scans WHERE user_id=? ORDER BY created_at DESC LIMIT 5", (session["user_id"],)
    ).fetchall()
    return render_template("profile.html", total_scans=total_scans, recent=recent)


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    db = get_db()
    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "profile":
            name = request.form.get("name", "").strip()
            college = request.form.get("college", "").strip()
            department = request.form.get("department", "").strip()
            db.execute(
                "UPDATE users SET name=?, college=?, department=? WHERE id=?",
                (name, college, department, session["user_id"]),
            )
            db.commit()
            flash("Profile updated successfully.", "success")

        elif form_type == "password":
            current_pw = request.form.get("current_password", "")
            new_pw = request.form.get("new_password", "")
            user = current_user()
            if not check_password_hash(user["password_hash"], current_pw):
                flash("Current password is incorrect.", "danger")
            else:
                db.execute(
                    "UPDATE users SET password_hash=? WHERE id=?",
                    (generate_password_hash(new_pw), session["user_id"]),
                )
                db.commit()
                flash("Password changed successfully.", "success")

        elif form_type == "preferences":
            dark_mode = 1 if request.form.get("dark_mode") else 0
            voice_assistant = 1 if request.form.get("voice_assistant") else 0
            notifications = 1 if request.form.get("notifications") else 0
            language = request.form.get("language", "English")
            db.execute(
                "UPDATE users SET dark_mode=?, voice_assistant=?, notifications=?, language=? WHERE id=?",
                (dark_mode, voice_assistant, notifications, language, session["user_id"]),
            )
            db.commit()
            flash("Preferences saved.", "success")

        return redirect(url_for("settings"))

    return render_template("settings.html")


# ----------------------------------------------------------------------
# AI Assistant
# ----------------------------------------------------------------------
ASSISTANT_KB = [
    (["deepfake", "what is deepfake"],
     "A deepfake is synthetic media where a person's face, body, or voice has been "
     "replaced or generated using AI, most often deep learning models like GANs or "
     "autoencoders. Our platform analyzes visual and audio artifacts to flag likely fakes."),
    (["ncii", "non-consensual"],
     "NCII stands for Non-Consensual Intimate Imagery — private or intimate images/videos "
     "shared without the subject's consent. Our NCII module flags such content, blurs unsafe "
     "previews automatically, and connects you to reporting and support resources."),
    (["upload", "how to upload", "how do i upload"],
     "Just open any Detection module from the sidebar, then drag & drop your file (or click "
     "to browse). Supported formats depend on the module — for example JPG/PNG for images, "
     "MP4/AVI/MOV for videos, and WAV/MP3 for audio."),
    (["confidence", "confidence score"],
     "The confidence score reflects how certain the AI model is about its prediction, shown "
     "as a percentage. Higher confidence means the model found stronger evidence for its "
     "Real/Fake or Safe/Unsafe classification."),
    (["safety", "cyber safety", "privacy"],
     "Great question! Always verify suspicious content before sharing it, avoid uploading "
     "sensitive files to untrusted sites, keep your account secured with a strong password, "
     "and use our Report Abuse tools if you encounter harmful content."),
    (["report abuse", "report"],
     "You can report harmful content directly from the NCII Detection page using the "
     "'Report Abuse' button, which guides you through documenting and escalating the issue."),
]

DEFAULT_ASSISTANT_REPLY = (
    "I can help you understand Deepfake Detection, NCII Detection, the upload process, "
    "confidence scores, and general cyber safety. Could you tell me a bit more about what "
    "you'd like to know?"
)


@app.route("/api/assistant/chat", methods=["POST"])
def assistant_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").lower().strip()

    reply = DEFAULT_ASSISTANT_REPLY
    for keywords, answer in ASSISTANT_KB:
        if any(k in message for k in keywords):
            reply = answer
            break

    return jsonify({"reply": reply})


# ----------------------------------------------------------------------
# Error handlers
# ----------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File is too large. Maximum upload size is 100 MB."}), 413


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        init_db()
    else:
        init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)