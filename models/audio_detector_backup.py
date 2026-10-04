"""
audio_detector.py
------------------
Audio / Voice Deepfake Detection Module.

Current state: DUMMY / PLACEHOLDER predictions for demo purposes.

TODO (Future AI Integration):
    - Use Librosa to load audio and extract features:
        y, sr = librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    - Feed MFCC / mel-spectrogram features into a trained CNN
      (e.g. a lightweight ResNet or a custom architecture trained on
      ASVspoof-style synthetic vs. genuine voice datasets).
    - Generate a real spectrogram image (librosa.display) to replace the
      placeholder waveform data below.
    - Return real confidence scores from the model's softmax output.
"""

import os
import random
import hashlib
from datetime import datetime


ALLOWED_EXTENSIONS = {"wav", "mp3"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _seeded_random(seed_source: str) -> random.Random:
    digest = hashlib.sha256(seed_source.encode()).hexdigest()
    return random.Random(int(digest[:8], 16))


def analyze_audio(filepath: str) -> dict:
    """
    Analyze an uploaded audio file and return a dummy voice-deepfake result.
    """
    filename = os.path.basename(filepath)
    rng = _seeded_random(filename + str(os.path.getsize(filepath)))

    is_fake = rng.random() < 0.35
    confidence = round(rng.uniform(76, 98) if is_fake else rng.uniform(70, 96), 2)

    # Fake waveform amplitude samples for front-end visualization
    waveform = [round(rng.uniform(-1, 1), 3) for _ in range(80)]

    if is_fake:
        risk_level = "High" if confidence > 90 else "Medium"
        explanation = (
            "Spectral analysis revealed unnatural pitch consistency and "
            "missing micro-variations in formant frequencies, patterns "
            "commonly produced by voice-cloning / text-to-speech synthesis."
        )
        recommendation = (
            "This audio is likely synthetic. Avoid trusting instructions or "
            "claims made in this recording without independent verification."
        )
    else:
        risk_level = "Low"
        explanation = (
            "Natural pitch variation, breathing patterns, and formant "
            "transitions are consistent with a genuine human voice recording."
        )
        recommendation = "No action required. This audio appears authentic."

    result = {
        "prediction": "Fake" if is_fake else "Real",
        "confidence": confidence,
        "risk_level": risk_level,
        "explanation": explanation,
        "recommendation": recommendation,
        "waveform": waveform,
        "model_used": "CNN + MFCC Spectral Analysis (dummy)",
        "processing_time_ms": rng.randint(600, 2200),
        "analyzed_at": datetime.utcnow().isoformat(),
    }
    return result
