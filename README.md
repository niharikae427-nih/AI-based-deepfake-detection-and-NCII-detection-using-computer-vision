# SENTRA AI — Deepfake & NCII Detection Platform

**AI Based Deepfake And NCII Detection Using Computer Vision**

A full-stack demo platform (Flask + SQLite + Tailwind) for detecting deepfake
images, videos, and voices, plus a dedicated Non-Consensual Intimate Imagery
(NCII) safety module. Ships with realistic dummy AI predictions and clearly
marked integration points for real TensorFlow / OpenCV / Librosa models.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

The first run automatically creates `database.db` (SQLite) with all required
tables. Uploaded files are stored in `uploads/`.

## Project Structure

```
app.py                     Main Flask application (routes, auth, DB)
requirements.txt           Python dependencies
database.db                SQLite database (auto-created on first run)
uploads/                   Uploaded scan files
models/
  image_detector.py         Dummy image deepfake detector (TODO: XceptionNet)
  video_detector.py          Dummy video deepfake detector (TODO: EfficientNet+LSTM)
  audio_detector.py          Dummy voice deepfake detector (TODO: Librosa + CNN)
  ncii_detector.py           Dummy NCII safety classifier
templates/                 Jinja2 HTML templates
  includes/                  Shared partials (navbar, sidebar, topbar, AI assistant)
static/
  css/style.css              Design system (glassmorphism, neon, animations)
  js/main.js                  Shared JS (dropzones, charts glue, AI assistant, toasts)
```

## Demo Notes

- **OTP codes** (registration + password reset) are shown directly in the
  on-screen flash message instead of being emailed — this is a demo
  convenience, not a production pattern.
- **Google / Microsoft login** buttons are placeholders that redirect back to
  the login page with a notice.
- **QR Code Login** generates a real QR code and includes a "Simulate QR Scan
  on Phone" demo button, since a second physical device isn't available during
  a typical demo/grading session.
- **AI predictions** are deterministic-but-randomized dummy values (seeded by
  filename + file size) — see the `TODO` comments inside each file in
  `models/` for exactly where to plug in trained TensorFlow/OpenCV/Librosa
  models.
- **PDF Reports**: the `/reports/<id>/download` page is a print-optimized
  view — click "Download / Print PDF" to save it as a PDF via the browser's
  print dialog.

## Security Notes

- Passwords are hashed with Werkzeug's `generate_password_hash`.
- Sessions use Flask's signed cookie sessions with a secret key persisted to
  `.secret_key` (or read from the `SENTRA_SECRET_KEY` environment variable in
  production — recommended for real deployments).
- File uploads are validated by extension per module and capped at 100 MB.
- All authenticated routes are protected with a `login_required` decorator.

## Next Steps for Production

1. Swap each `models/*_detector.py` dummy function for real inference using
   the TODO comments as a guide.
2. Move the secret key, mail/SMS OTP delivery, and file storage to proper
   production services.
3. Add CSRF tokens (e.g. Flask-WTF) to all POST forms.
4. Replace the SQLite database with PostgreSQL/MySQL for multi-user scale.
