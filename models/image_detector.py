import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "deepfake_cnn_224_repaired.keras"
)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

_model = None


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_model():
    global _model

    if _model is None:
        print("Loading image deepfake model...")
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("Image model loaded successfully.")
        print("Input shape:", _model.input_shape)
        print("Output shape:", _model.output_shape)

    return _model


def analyze_image(filepath):
    started = time.time()

    model = get_model()

    # Load image
    image = Image.open(filepath).convert("RGB")

    # Resize to model input
    image = image.resize((224, 224))

    # Convert to numpy
    img_array = np.asarray(image, dtype=np.float32)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)

    raw_probability = float(np.asarray(prediction).reshape(-1)[0])

    # Assume model output represents fake probability
    fake_probability = raw_probability

    # Keep probability in valid range
    fake_probability = max(0.0, min(1.0, fake_probability))

    real_probability = 1.0 - fake_probability

    if fake_probability >= 0.5:
        prediction_label = "Fake"
        confidence = fake_probability * 100

        if confidence >= 90:
            risk_level = "High"
        else:
            risk_level = "Medium"

        explanation = (
            "The trained image classifier detected pixel-level patterns "
            "that are consistent with possible digital manipulation."
        )

        recommendation = (
            "Treat this image with caution and verify the original source "
            "before sharing or relying on it."
        )

    else:
        prediction_label = "Real"
        confidence = real_probability * 100
        risk_level = "Low"

        explanation = (
            "The trained image classifier did not detect strong "
            "manipulation-consistent patterns in the image."
        )

        recommendation = (
            "The image appears likely authentic according to the model. "
            "For important content, verify the original source."
        )

    processing_time_ms = int((time.time() - started) * 1000)

    return {
        "prediction": prediction_label,
        "confidence": round(confidence, 2),
        "risk_level": risk_level,
        "explanation": explanation,
        "recommendation": recommendation,
        "model_used": "Deepfake CNN 224 (trained image classifier)",
        "processing_time_ms": processing_time_ms,
        "analyzed_at": datetime.utcnow().isoformat(),
        "fake_probability": round(fake_probability * 100, 2),
        "real_probability": round(real_probability * 100, 2),
        "heatmap": None,
    }