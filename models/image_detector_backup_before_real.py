"""Real TensorFlow/Keras deepfake classifier for the existing Flask app.

The model is expected to be saved as a Keras `*.keras` file in
`models/weights/` and trained with the Kaggle-ready script:
`train_deepfake_cnn.py`.

If no trained model is present, the app falls back to the previous seeded-
random placeholder output so the UI remains usable during development.
"""

import os
import random
import hashlib
from datetime import datetime

import numpy as np

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

MODEL_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "weights", "deepfake_cnn_224.keras"),
    os.path.join(os.path.dirname(__file__), "weights", "best_model.keras"),
    os.path.join(os.path.dirname(__file__), "weights", "best_model.h5"),
    os.path.join(os.path.dirname(__file__), "weights", "model.keras"),
]

_model = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _seeded_random(seed_source: str) -> random.Random:
    digest = hashlib.sha256(seed_source.encode()).hexdigest()
    return random.Random(int(digest[:8], 16))


def _find_model_path():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _try_load_real_model():
    global _model
    if _model is not None:
        return True

    model_path = _find_model_path()
    if not model_path:
        return False

    try:
        import tensorflow as tf

        _model = tf.keras.models.load_model(model_path)
        return True
    except Exception as exc:
        print(f"[image_detector] Failed to load Keras model '{model_path}': {exc}")
        _model = None
        return False


def _dummy_result(filepath: str) -> dict:
    filename = os.path.basename(filepath)
    rng = _seeded_random(filename + str(os.path.getsize(filepath)))

    is_fake = rng.random() < 0.42
    confidence = round(rng.uniform(78, 99) if is_fake else rng.uniform(70, 98), 2)

    if is_fake:
        risk_level = "High" if confidence > 90 else "Medium"
        explanation = (
            "This result is a placeholder demonstration output. A trained deepfake model "
            "is not available yet; once model weights are placed in models/weights/, "
            "real inference will be used instead."
        )
        recommendation = (
            "Train the Kaggle CNN and save the model in models/weights/ to enable real "
            "deepfake detection."
        )
    else:
        risk_level = "Low"
        explanation = (
            "No significant manipulation artifacts were detected by the placeholder model. "
            "Place a real trained model in models/weights/ to switch to actual predictions."
        )
        recommendation = "No action required for this demo placeholder result."

    result = {
        "prediction": "Fake" if is_fake else "Real",
        "confidence": confidence,
        "risk_level": risk_level,
        "explanation": explanation,
        "recommendation": recommendation,
        "heatmap": "placeholder_heatmap.png",
        "model_used": "Keras CNN (not loaded yet)",
        "processing_time_ms": rng.randint(400, 1600),
        "analyzed_at": datetime.utcnow().isoformat(),
    }
    return result


def _preprocess_image(image_path: str):
    import cv2

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = image_rgb[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))
        return face.astype(np.float32) / 255.0

    image_rgb = cv2.resize(image_rgb, (224, 224))
    return image_rgb.astype(np.float32) / 255.0


def analyze_image(filepath: str) -> dict:
    """Analyze an uploaded image with the trained CNN if available."""
    started = datetime.utcnow()

    if not _try_load_real_model():
        return _dummy_result(filepath)

    try:
        image = _preprocess_image(filepath)
        x = np.expand_dims(image, axis=0)
        pred = _model.predict(x, verbose=0)

        if pred.ndim == 2 and pred.shape[1] == 2:
            fake_prob = float(pred[0][1])
        elif pred.ndim == 2 and pred.shape[1] == 1:
            fake_prob = float(pred[0][0])
        elif pred.ndim == 1:
            fake_prob = float(pred[0])
        else:
            fake_prob = float(np.asarray(pred).reshape(-1)[0])

        is_fake = fake_prob >= 0.5
        confidence = round((fake_prob if is_fake else 1.0 - fake_prob) * 100.0, 2)
        risk_level = "High" if is_fake and confidence > 90 else "Medium" if is_fake else "Low"

        explanation = (
            "The trained CNN analyzed the face region and found features consistent with a deepfake or manipulated image."
            if is_fake else
            "The trained CNN did not find strong manipulation cues in the uploaded image."
        )
        recommendation = (
            "Treat this image with caution and verify the source before sharing it."
            if is_fake else
            "No action required. This image appears authentic."
        )

        processing_time_ms = int((datetime.utcnow() - started).total_seconds() * 1000)

        return {
            "prediction": "Fake" if is_fake else "Real",
            "confidence": confidence,
            "risk_level": risk_level,
            "explanation": explanation,
            "recommendation": recommendation,
            "heatmap": "placeholder_heatmap.png",
            "model_used": "Trained Keras CNN (224x224)",
            "processing_time_ms": processing_time_ms,
            "analyzed_at": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        print(f"[image_detector] Inference failed: {exc}")
        return _dummy_result(filepath)
