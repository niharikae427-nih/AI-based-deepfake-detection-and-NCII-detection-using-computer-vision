"""
video_detector.py
------------------
Video Deepfake Detection Module.

Current state: DUMMY / PLACEHOLDER predictions for demo purposes.

TODO (Future AI Integration):
    - Use OpenCV (cv2.VideoCapture) to extract frames at a fixed interval
      (e.g. every 10th frame) instead of the simulated frame count below.
    - Run face detection per frame (MTCNN / RetinaFace) and crop faces.
    - Feed cropped face sequences into a CNN + temporal model
      (e.g. EfficientNet backbone + LSTM/Transformer head, or a 3D-CNN)
      trained on FaceForensics++ / DFDC style datasets.
    - Aggregate per-frame fake probabilities (mean / max pooling) into a
      single video-level prediction.
    - Track which frame indices exceeded the fake-probability threshold to
      populate `unsafe_frames` with real data instead of random samples.
    - Optionally run audio-visual sync checks (lip-sync mismatch) using the
      audio_detector module together with this one.
"""

import os
import random
import hashlib
from datetime import datetime


ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _seeded_random(seed_source: str) -> random.Random:
    digest = hashlib.sha256(seed_source.encode()).hexdigest()
    return random.Random(int(digest[:8], 16))


def analyze_video(filepath: str) -> dict:
    """
    Analyze an uploaded video and return a dummy deepfake-detection result.
    """
    filename = os.path.basename(filepath)
    rng = _seeded_random(filename + str(os.path.getsize(filepath)))

    total_frames = rng.randint(120, 480)
    is_fake = rng.random() < 0.38
    confidence = round(rng.uniform(75, 99) if is_fake else rng.uniform(72, 97), 2)

    unsafe_frames = []
    if is_fake:
        unsafe_count = rng.randint(3, 10)
        unsafe_frames = sorted(rng.sample(range(total_frames), min(unsafe_count, total_frames)))
        risk_level = "High" if confidence > 90 else "Medium"
        explanation = (
            "Temporal inconsistencies were detected across facial frames, "
            "including flickering facial boundaries and irregular blinking "
            "patterns typical of frame-by-frame face reenactment."
        )
        recommendation = (
            "This video shows strong signs of manipulation. Do not share it "
            "further and consider reporting it through the platform where "
            "you found it."
        )
    else:
        risk_level = "Low"
        explanation = (
            "Facial motion, blinking cadence, and frame-to-frame lighting "
            "remained consistent throughout the video, indicating no "
            "detectable manipulation."
        )
        recommendation = "No action required. This video appears authentic."

    result = {
        "prediction": "Fake" if is_fake else "Real",
        "confidence": confidence,
        "risk_level": risk_level,
        "explanation": explanation,
        "recommendation": recommendation,
        "total_frames": total_frames,
        "unsafe_frames": unsafe_frames,
        "model_used": "EfficientNet-B4 + LSTM (dummy)",
        "processing_time_ms": rng.randint(1800, 6000),
        "analyzed_at": datetime.utcnow().isoformat(),
    }
    return result
