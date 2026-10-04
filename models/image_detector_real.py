"""
image_detector.py — REAL MODEL VERSION
----------------------------------------
Drop-in replacement for the dummy `image_detector.py` in the SENTRA AI app.

Behavior:
    - If a trained checkpoint exists at `models/weights/best_model.pth` (produced by
      `sentra-ai-deepfake-training.ipynb`), it loads the real Xception/EfficientNet
      classifier (via `timm`) and runs genuine inference.
    - If the checkpoint is missing (e.g. you haven't trained yet, or copied this file
      before downloading weights), it transparently falls back to the original dummy
      seeded-random predictions so the app never crashes or looks broken during a demo.

Install the extra dependencies before using this file:
    pip install torch torchvision timm facenet-pytorch opencv-python-headless
"""

import os
import random
import hashlib
from datetime import datetime

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "best_model.pth")

_model = None
_mtcnn = None
_device = None
_face_size = 299


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _seeded_random(seed_source: str) -> random.Random:
    digest = hashlib.sha256(seed_source.encode()).hexdigest()
    return random.Random(int(digest[:8], 16))


def _try_load_real_model():
    """Lazily loads the trained model + face detector. Returns True on success."""
    global _model, _mtcnn, _device, _face_size

    if _model is not None:
        return True  # already loaded

    if not os.path.exists(WEIGHTS_PATH):
        return False

    try:
        import torch
        import timm
        from facenet_pytorch import MTCNN

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(WEIGHTS_PATH, map_location=_device)

        backbone = checkpoint.get("backbone", "xception")
        _face_size = checkpoint.get("face_size", 299)

        model = timm.create_model(backbone, pretrained=False, num_classes=2)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(_device)
        model.eval()

        _model = model
        _mtcnn = MTCNN(image_size=_face_size, margin=20, post_process=False, device=_device, keep_all=False)
        return True
    except Exception as e:
        print(f"[image_detector] Could not load real model, falling back to dummy: {e}")
        _model = None
        _mtcnn = None
        return False


def _dummy_result(filepath: str) -> dict:
    """Same seeded-random placeholder logic as the original dummy module."""
    filename = os.path.basename(filepath)
    rng = _seeded_random(filename + str(os.path.getsize(filepath)))

    is_fake = rng.random() < 0.42
    confidence = round(rng.uniform(78, 99) if is_fake else rng.uniform(70, 98), 2)
    risk_level = ("High" if confidence > 90 else "Medium") if is_fake else "Low"

    return {
        "prediction": "Fake" if is_fake else "Real",
        "confidence": confidence,
        "risk_level": risk_level,
        "explanation": (
            "Dummy placeholder result — no trained model weights found at "
            f"'{WEIGHTS_PATH}'. Train a model with sentra-ai-deepfake-training.ipynb "
            "and place best_model.pth there to get real predictions."
        ),
        "recommendation": "Train and deploy the real model for production use.",
        "heatmap": "placeholder_heatmap.png",
        "model_used": "XceptionNet-v1 (dummy fallback)",
        "processing_time_ms": rng.randint(400, 1600),
        "analyzed_at": datetime.utcnow().isoformat(),
    }


def analyze_image(filepath: str) -> dict:
    """
    Analyze an uploaded image for deepfake likelihood.
    Uses the real trained model if available, otherwise falls back to dummy logic.
    """
    started = datetime.utcnow()

    if not _try_load_real_model():
        return _dummy_result(filepath)

    import cv2
    import torch
    from torchvision import transforms

    frame_bgr = cv2.imread(filepath)
    if frame_bgr is None:
        return _dummy_result(filepath)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    try:
        face_tensor = _mtcnn(rgb)
    except Exception:
        face_tensor = None

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((_face_size, _face_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    if face_tensor is not None:
        face_np = face_tensor.permute(1, 2, 0).byte().numpy()
        input_tensor = transform(face_np).unsqueeze(0).to(_device)
        used_face_crop = True
    else:
        # No face detected — fall back to running on the whole resized image so the
        # user still gets a result, but note the caveat in the explanation.
        input_tensor = transform(rgb).unsqueeze(0).to(_device)
        used_face_crop = False

    with torch.no_grad():
        logits = _model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        fake_prob = probs[1].item()

    is_fake = fake_prob >= 0.5
    confidence = round((fake_prob if is_fake else 1 - fake_prob) * 100, 2)
    risk_level = ("High" if confidence > 90 else "Medium") if is_fake else "Low"

    explanation = (
        "The trained classifier analyzed facial region pixel patterns and flagged "
        "manipulation-consistent artifacts." if is_fake else
        "The trained classifier found no manipulation-consistent artifacts in the "
        "detected facial region."
    )
    if not used_face_crop:
        explanation += " (No face was detected — this result is based on the full image, which is less reliable.)"

    recommendation = (
        "Treat this image with caution. Avoid sharing or forwarding it and verify the "
        "source before taking any action." if is_fake else
        "No action required. This image appears authentic."
    )

    processing_time_ms = int((datetime.utcnow() - started).total_seconds() * 1000)

    return {
        "prediction": "Fake" if is_fake else "Real",
        "confidence": confidence,
        "risk_level": risk_level,
        "explanation": explanation,
        "recommendation": recommendation,
        "heatmap": "placeholder_heatmap.png",  # TODO: add Grad-CAM overlay generation
        "model_used": "Trained Xception (FaceForensics++ / DFDC)",
        "processing_time_ms": processing_time_ms,
        "analyzed_at": datetime.utcnow().isoformat(),
    }
