"""
ncii_detector.py
----------------
Non-Consensual Intimate Image (NCII) Detection Module.

This module classifies uploaded media into Safe / Unsafe / Potential NCII
categories to help users identify and report harmful content. It does not
store, display, or redistribute flagged content in the clear -- unsafe
previews are always blurred by the front-end before rendering.

Current state: DUMMY / PLACEHOLDER predictions for demo purposes.

TODO (Future AI Integration):
    - Integrate a content-safety classifier (e.g. a fine-tuned CNN such as
      NudeNet-style architecture or a custom EfficientNet classifier)
      trained on labeled safety datasets, run fully on-device / on your own
      infrastructure with strict access controls.
    - Combine with the deepfake models (image_detector / video_detector) so
      content can be flagged as "synthetic + intimate" (a common NCII
      pattern) rather than running the checks independently.
    - Add perceptual hashing (e.g. PhotoDNA-style or pHash) to match against
      known-harmful hash databases (e.g. StopNCII.org hash lists) so
      previously reported content can be detected without re-analysis.
    - Ensure all flagged uploads are handled per platform policy: encrypted
      at rest, access-logged, and auto-deleted after the retention window.
"""

import os
import random
import hashlib
from datetime import datetime


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "mp4", "avi", "mov"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _seeded_random(seed_source: str) -> random.Random:
    digest = hashlib.sha256(seed_source.encode()).hexdigest()
    return random.Random(int(digest[:8], 16))


def analyze_ncii(filepath: str) -> dict:
    """
    Analyze an uploaded file for NCII risk and return a dummy result.

    Categories:
        "Safe"           - no concerning content detected
        "Unsafe"         - content flagged as intimate / non-consensual risk
        "Potential NCII"  - borderline / needs human review
    """
    filename = os.path.basename(filepath)
    rng = _seeded_random(filename + str(os.path.getsize(filepath)))

    roll = rng.random()
    if roll < 0.55:
        category = "Safe"
        risk_level = "Low"
        confidence = round(rng.uniform(85, 99), 2)
        explanation = (
            "The AI safety classifier did not detect intimate or exploitative "
            "content in this file. Standard deepfake checks are still "
            "recommended for authenticity."
        )
        recommendation = "No action required."
        blur_preview = False
    elif roll < 0.8:
        category = "Potential NCII"
        risk_level = "Medium"
        confidence = round(rng.uniform(60, 85), 2)
        explanation = (
            "The classifier detected borderline signals that may indicate "
            "non-consensual intimate content. This result requires human "
            "review before any conclusion is drawn."
        )
        recommendation = (
            "Do not share this file further. Consider using the Report Abuse "
            "and Victim Support options while a review is pending."
        )
        blur_preview = True
    else:
        category = "Unsafe"
        risk_level = "High"
        confidence = round(rng.uniform(85, 99), 2)
        explanation = (
            "The classifier flagged this content as a high-confidence match "
            "for non-consensual intimate imagery patterns."
        )
        recommendation = (
            "Do not share or download this file. Use the Report Abuse and "
            "Request Content Removal options immediately, and reach out to "
            "the Cybercrime Help resources provided below."
        )
        blur_preview = True

    result = {
        "category": category,
        "confidence": confidence,
        "risk_level": risk_level,
        "explanation": explanation,
        "recommendation": recommendation,
        "blur_preview": blur_preview,
        "model_used": "NCII-SafetyNet (dummy)",
        "processing_time_ms": rng.randint(500, 1800),
        "analyzed_at": datetime.utcnow().isoformat(),
    }
    return result
