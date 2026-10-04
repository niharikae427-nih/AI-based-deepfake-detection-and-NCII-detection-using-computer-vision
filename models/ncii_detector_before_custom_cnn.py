"""
ncii_detector.py
----------------
NCII Safety Screening Module.

This module screens uploaded images for sensitive/NSFW visual content.

IMPORTANT:
This is NOT a consent detector and cannot prove that content is
non-consensually shared. The result is a safety-screening / risk
assessment signal only.
"""

import os
import time
from datetime import datetime, timezone

import cv2
from transformers import pipeline


# ============================================================
# CONFIGURATION
# ============================================================

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

NSFW_HIGH_THRESHOLD = 0.80
NSFW_MEDIUM_THRESHOLD = 0.40

MODEL_NAME = "Falconsai/nsfw_image_detection"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading NCII safety screening model...")

_ncii_classifier = pipeline(
    "image-classification",
    model=MODEL_NAME,
)

print("NCII safety screening model loaded successfully!")


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename: str) -> bool:
    """Check whether the uploaded file is a supported image."""

    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(filepath: str) -> int:
    """
    Detect faces using OpenCV Haar Cascade.

    This is only a supporting signal. Face detection does NOT
    determine identity, consent, or NCII.
    """

    try:
        image = cv2.imread(filepath)

        if image is None:
            return 0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        face_detector = cv2.CascadeClassifier(cascade_path)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        return len(faces)

    except Exception as exc:
        print(f"Face detection warning: {exc}")
        return 0


# ============================================================
# NCII SAFETY SCREENING
# ============================================================

def analyze_ncii(filepath: str) -> dict:
    """
    Analyze an image for sensitive/NSFW visual-content risk.

    Returns a structured result suitable for the Flask API.

    IMPORTANT:
    The result does not establish whether content was shared
    consensually or non-consensually.
    """

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    filename = os.path.basename(filepath)

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    predictions = _ncii_classifier(filepath)

    scores = {
        str(item["label"]).lower(): float(item["score"])
        for item in predictions
    }

    normal_score = scores.get("normal", 0.0)
    nsfw_score = scores.get("nsfw", 0.0)

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    face_count = detect_faces(filepath)

    # --------------------------------------------------------
    # RISK ASSESSMENT
    # --------------------------------------------------------

    if nsfw_score >= NSFW_HIGH_THRESHOLD:

        category = "Potential NCII"
        risk_level = "High"

        confidence = round(nsfw_score * 100, 2)

        blur_preview = True

        explanation = (
            "The AI safety-screening model detected a high probability "
            "of sexually explicit or sensitive visual content."
        )

        recommendation = (
            "Do not share or redistribute this content. "
            "This result does not establish consent or prove NCII. "
            "If non-consensual sharing is suspected, use an appropriate "
            "reporting or content-removal process."
        )

    elif nsfw_score >= NSFW_MEDIUM_THRESHOLD:

        category = "Potentially Sensitive"

        risk_level = "Medium"

        confidence = round(nsfw_score * 100, 2)

        blur_preview = True

        explanation = (
            "The AI safety-screening model detected potentially "
            "sensitive visual content."
        )

        recommendation = (
            "Avoid further sharing. Human review and consent information "
            "are required before determining whether the content "
            "constitutes NCII."
        )

    else:

        category = "Safe"

        risk_level = "Low"

        confidence = round(normal_score * 100, 2)

        blur_preview = False

        explanation = (
            "The AI safety-screening model did not detect a high "
            "probability of sexually explicit or sensitive visual content."
        )

        recommendation = (
            "No sensitive-content warning was triggered. "
            "This screening does not determine consent."
        )

    # --------------------------------------------------------
    # PROCESSING TIME
    # --------------------------------------------------------

    processing_time_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {
        "category": category,
        "confidence": confidence,
        "risk_level": risk_level,

        "explanation": explanation,
        "recommendation": recommendation,

        "blur_preview": blur_preview,

        "model_used": MODEL_NAME,

        "nsfw_score": round(nsfw_score * 100, 2),
        "normal_score": round(normal_score * 100, 2),

        "face_count": face_count,

        "processing_time_ms": processing_time_ms,

        "analyzed_at": datetime.now(timezone.utc).isoformat(),

        "file_name": filename,
    }