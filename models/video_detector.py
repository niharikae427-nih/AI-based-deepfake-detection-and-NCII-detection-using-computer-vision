import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "video_cnn_lstm_fixed.keras"
)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

_model = None

NUM_FRAMES = 8
IMG_SIZE = 224


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def load_model():
    global _model

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Video model not found: {MODEL_PATH}"
            )

        _model = tf.keras.models.load_model(MODEL_PATH)

        print(
            f"[INFO] Video CNN-LSTM loaded successfully: {MODEL_PATH}"
        )

        print(
            f"[INFO] Input shape: {_model.input_shape}"
        )

        print(
            f"[INFO] Output shape: {_model.output_shape}"
        )

    return _model


def extract_video_frames(video_path, num_frames=NUM_FRAMES):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            "Unable to open the uploaded video."
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise ValueError(
            "The video does not contain readable frames."
        )

    # Select exactly 8 frames
    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int
    )

    frames = []

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = cap.read()

        if not success:
            continue

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame_rgb)

        image = image.resize(
            (IMG_SIZE, IMG_SIZE)
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        frames.append(image_array)

    cap.release()

    if not frames:
        raise ValueError(
            "No readable frames could be extracted."
        )

    # Ensure exactly 8 frames
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1].copy())

    frames = frames[:NUM_FRAMES]

    return np.asarray(
        frames,
        dtype=np.float32
    )


def analyze_video(video_path):

    model = load_model()

    frames = extract_video_frames(
        video_path,
        NUM_FRAMES
    )

    # Add batch dimension
    video_input = np.expand_dims(
        frames,
        axis=0
    )

    print(
        "[INFO] Video input shape:",
        video_input.shape
    )

    # Expected:
    # (1, 8, 224, 224, 3)

    output = model.predict(
        video_input,
        verbose=0
    )

    probability = float(
        np.asarray(output).reshape(-1)[0]
    )

    probability = float(
        np.clip(probability, 0.0, 1.0)
    )

    # Assumption: model output is Deepfake probability
    deepfake_probability = probability
    real_probability = 1.0 - probability

    if deepfake_probability >= 0.5:

        prediction = "DEEPFAKE"
        confidence = deepfake_probability * 100
        risk = "High"

        explanation = (
            "The trained CNN-LSTM video model classified "
            "the analyzed video sequence as Deepfake."
        )

        recommendation = (
            "Verify the original source of the video before "
            "sharing or relying on it."
        )

    else:

        prediction = "REAL"
        confidence = real_probability * 100
        risk = "Low"

        explanation = (
            "The trained CNN-LSTM video model classified "
            "the analyzed video sequence as Real."
        )

        recommendation = (
            "The analyzed video was classified as Real by "
            "the trained model. For important content, "
            "independently verify the original source."
        )

    return {
        "prediction": prediction,

        "confidence": round(
            confidence,
            2
        ),

        "fake_probability": round(
            deepfake_probability * 100,
            2
        ),

        "real_probability": round(
            real_probability * 100,
            2
        ),

        "risk": risk,

        "model_used": (
            "CNN-LSTM Video Deepfake Model "
            "(8 Frames, 224x224)"
        ),

        "frames_analyzed": int(
            len(frames)
        ),

        "explanation": explanation,

        "recommendation": recommendation,
    }