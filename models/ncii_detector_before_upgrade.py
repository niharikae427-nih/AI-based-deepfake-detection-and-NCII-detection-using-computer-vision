"""
ncii_detector.py
----------------
NCII Safety Screening Module.

Uses a local Hugging Face image-safety classifier to screen uploaded
images for normal / NSFW content.

IMPORTANT:
This is an NSFW safety classifier, not a consent detector.
An image classifier cannot determine whether content was shared
with or without consent.
"""

import os
from datetime import datetime

from transformers import pipeline


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


# Load the model once when Flask starts.
# This prevents downloading/loading the model for every request.
print("Loading NCII safety screening model...")

_ncii_classifier = pipeline(
    "image-classification",
    model="Falconsai/nsfw_image_detection",
)

print("NCII safety screening model loaded successfully!")


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def analyze_ncii(filepath: str) -> dict:
    """
    Screen an uploaded image for NSFW/sensitive-content risk.

    This does NOT determine consent or prove NCII.
    """

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    filename = os.path.basename(filepath)

    predictions = _ncii_classifier(filepath)

    scores = {
        item["label"].lower(): float(item["score"])
        for item in predictions
    }

    normal_score = scores.get("normal", 0.0)
    nsfw_score = scores.get("nsfw", 0.0)

    # Demo thresholds for safety screening.
    if nsfw_score >= 0.80:
        category = "Potential NCII"
        risk_level = "High"
        confidence = round(nsfw_score * 100, 2)
        blur_preview = True

        explanation = (
            "The safety screening model detected a high probability "
            "of sexually explicit or sensitive visual content. "
            "This result does not establish non-consensual sharing."
        )

        recommendation = (
            "Do not share or redistribute the content. "
            "If the content was shared without consent, use the "
            "appropriate reporting and content-removal process."
        )

    elif nsfw_score >= 0.40:
        category = "Potential NCII"
        risk_level = "Medium"
        confidence = round(nsfw_score * 100, 2)
        blur_preview = True

        explanation = (
            "The safety screening model detected potentially sensitive "
            "visual content. Human review and consent information are "
            "required before determining whether this is NCII."
        )

        recommendation = (
            "Avoid sharing the content further. Review the situation "
            "and use appropriate reporting or support resources if "
            "non-consensual distribution is suspected."
        )

    else:
        category = "Safe"
        risk_level = "Low"
        confidence = round(normal_score * 100, 2)
        blur_preview = False

        explanation = (
            "The safety screening model did not detect a high probability "
            "of sexually explicit visual content."
        )

        recommendation = (
            "No sensitive-content warning was triggered. "
            "This screening does not determine consent."
        )

    return {
        "category": category,
        "confidence": confidence,
        "risk_level": risk_level,
        "explanation": explanation,
        "recommendation": recommendation,
        "blur_preview": blur_preview,
        "model_used": "Falconsai NSFW Image Detection",
        "processing_time_ms": None,
        "analyzed_at": datetime.utcnow().isoformat(),
        "nsfw_score": round(nsfw_score * 100, 2),
        "normal_score": round(normal_score * 100, 2),
        "file_name": filename,
    }