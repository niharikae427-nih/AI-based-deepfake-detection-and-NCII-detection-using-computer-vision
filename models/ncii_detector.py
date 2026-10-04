"""
ncii_detector.py
----------------
Custom CNN-based NCII Safety Screening Module.

The model was trained locally using the project's NCII dataset.

IMPORTANT:
This model performs sensitive-content screening.
It cannot determine consent or prove that an image is NCII.
"""

import os
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "ncii_model.keras"
)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

IMAGE_SIZE = (224, 224)

NSFW_HIGH_THRESHOLD = 0.80
NSFW_MEDIUM_THRESHOLD = 0.40


# ============================================================
# LOAD CUSTOM CNN MODEL
# ============================================================

print("=" * 60)
print("Loading custom NCII CNN model...")
print("=" * 60)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"NCII CNN model not found: {MODEL_PATH}"
    )

_ncii_model = tf.keras.models.load_model(MODEL_PATH)

print("Custom NCII CNN model loaded successfully!")
print(f"Model path: {MODEL_PATH}")


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename: str) -> bool:
    """Check whether the uploaded file is a supported image."""

    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(filepath: str) -> int:
    """
    Detect faces using OpenCV Haar Cascade.

    Face detection is only a supporting signal.
    It does not determine identity, consent, or NCII.
    """

    try:

        image = cv2.imread(filepath)

        if image is None:
            return 0

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        face_detector = cv2.CascadeClassifier(
            cascade_path
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        return len(faces)

    except Exception as exc:

        print(
            f"Face detection warning: {exc}"
        )

        return 0


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(filepath: str) -> np.ndarray:
    """
    Prepare image for the trained MobileNetV2-based CNN.

    Input:
        JPG / JPEG / PNG

    Output:
        Tensor shape: (1, 224, 224, 3)
    """

    image = cv2.imread(filepath)

    if image is None:
        raise ValueError(
            "Unable to read the uploaded image."
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# CNN PREDICTION
# ============================================================

def predict_nsfw(filepath: str):
    """
    Run the custom CNN model.

    The model was trained with class order:

        0 = normal
        1 = nsfw

    Returns:
        normal_score, nsfw_score
    """

    image = preprocess_image(filepath)

    prediction = _ncii_model.predict(
        image,
        verbose=0
    )

    nsfw_score = float(
        np.squeeze(prediction)
    )

    nsfw_score = max(
        0.0,
        min(
            1.0,
            nsfw_score
        )
    )

    normal_score = 1.0 - nsfw_score

    return normal_score, nsfw_score


# ============================================================
# NCII SAFETY SCREENING
# ============================================================

def analyze_ncii(filepath: str) -> dict:
    """
    Analyze an uploaded image using the custom CNN model.

    IMPORTANT:
    This is a sensitive-content screening system.
    It does NOT determine consent and does NOT prove NCII.
    """

    if not os.path.isfile(filepath):

        raise FileNotFoundError(
            f"File not found: {filepath}"
        )

    filename = os.path.basename(
        filepath
    )

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # CNN PREDICTION
    # --------------------------------------------------------

    normal_score, nsfw_score = predict_nsfw(
        filepath
    )


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    face_count = detect_faces(
        filepath
    )


    # --------------------------------------------------------
    # RISK ASSESSMENT
    # --------------------------------------------------------

    if nsfw_score >= NSFW_HIGH_THRESHOLD:

        category = "Potential NCII"

        risk_level = "High"

        confidence = round(
            nsfw_score * 100,
            2
        )

        blur_preview = True

        explanation = (
            "The custom CNN safety-screening model "
            "detected a high probability of sexually "
            "explicit or sensitive visual content."
        )

        recommendation = (
            "Do not share or redistribute this content. "
            "This result does not establish consent or "
            "prove NCII. Human review and contextual "
            "information are required."
        )


    elif nsfw_score >= NSFW_MEDIUM_THRESHOLD:

        category = "Potentially Sensitive"

        risk_level = "Medium"

        confidence = round(
            nsfw_score * 100,
            2
        )

        blur_preview = True

        explanation = (
            "The custom CNN safety-screening model "
            "detected potentially sensitive visual content."
        )

        recommendation = (
            "Avoid further sharing. Human review and "
            "consent information are required before "
            "determining whether the content constitutes NCII."
        )


    else:

        category = "Safe"

        risk_level = "Low"

        confidence = round(
            normal_score * 100,
            2
        )

        blur_preview = False

        explanation = (
            "The custom CNN safety-screening model "
            "did not detect a high probability of "
            "sexually explicit or sensitive visual content."
        )

        recommendation = (
            "No sensitive-content warning was triggered. "
            "This screening does not determine consent."
        )


    # --------------------------------------------------------
    # PROCESSING TIME
    # --------------------------------------------------------

    processing_time_ms = round(
        (
            time.perf_counter()
            - start_time
        ) * 1000,
        2
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

        "model_used": "Custom MobileNetV2 CNN",

        "model_path": MODEL_PATH,

        "nsfw_score": round(
            nsfw_score * 100,
            2
        ),

        "normal_score": round(
            normal_score * 100,
            2
        ),

        "face_count": face_count,

        "processing_time_ms":
            processing_time_ms,

        "analyzed_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "file_name": filename,
    }