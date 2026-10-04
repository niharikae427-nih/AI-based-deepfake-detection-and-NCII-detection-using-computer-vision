import os
import json
import time
import numpy as np
import librosa
from scipy.fftpack import dct
from datetime import datetime
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Real trained model files
ARCH_PATH = os.path.join(BASE_DIR, "audio_deepfake_architecture.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "audio_deepfake.weights.h5")

# Audio settings used during Kaggle training
SR = 16000
DURATION = 4.0
N_FFT = 512
HOP_LENGTH = 160
TARGET_FRAMES = 128
N_LFCC = 60
N_MFCC = 60
N_FILTERS = 128
THRESHOLD = 0.74

_model = None


# ---------------------------------------------------------
# Remove unsupported Keras 3 configuration
# ---------------------------------------------------------
def clean_config(obj):
    if isinstance(obj, dict):
        return {
            key: clean_config(value)
            for key, value in obj.items()
            if key != "quantization_config"
        }

    if isinstance(obj, list):
        return [clean_config(item) for item in obj]

    return obj


# ---------------------------------------------------------
# Load model from JSON architecture + H5 weights
# ---------------------------------------------------------
def get_model():
    global _model

    if _model is not None:
        return _model

    if not os.path.exists(ARCH_PATH):
        raise FileNotFoundError(
            f"Audio model architecture not found: {ARCH_PATH}"
        )

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Audio model weights not found: {WEIGHTS_PATH}"
        )

    print("Loading audio model architecture...")

    with open(ARCH_PATH, "r", encoding="utf-8") as f:
        architecture = json.load(f)

    # Remove Keras settings unsupported by local version
    architecture = clean_config(architecture)

    architecture_json = json.dumps(architecture)

    _model = keras.models.model_from_json(architecture_json)

    print("Loading audio model weights...")

    _model.load_weights(WEIGHTS_PATH)

    print("Audio model loaded successfully.")
    print("Input shape:", _model.input_shape)
    print("Output shape:", _model.output_shape)

    return _model


# ---------------------------------------------------------
# Allowed audio files
# ---------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in {"wav", "mp3", "m4a"}
    )


# ---------------------------------------------------------
# LFCC extraction
# ---------------------------------------------------------
def compute_lfcc(audio):

    stft = (
        np.abs(
            librosa.stft(
                audio,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
        )
        ** 2
    )

    freqs = librosa.fft_frequencies(
        sr=SR,
        n_fft=N_FFT
    )

    lin_f = np.linspace(
        0,
        SR // 2,
        N_FILTERS + 2
    )

    fb = np.zeros(
        (N_FILTERS, len(freqs))
    )

    for m in range(1, N_FILTERS + 1):

        fl = lin_f[m - 1]
        fc = lin_f[m]
        fr = lin_f[m + 1]

        for k, f in enumerate(freqs):

            if fl <= f <= fc:

                fb[m - 1, k] = (
                    (f - fl)
                    / (fc - fl + 1e-8)
                )

            elif fc < f <= fr:

                fb[m - 1, k] = (
                    (fr - f)
                    / (fr - fc + 1e-8)
                )

    return dct(
        np.log(
            np.dot(fb, stft) + 1e-8
        ),
        type=2,
        axis=0,
        norm="ortho"
    )[:N_LFCC]


# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------
def extract_features(file_path):

    n = int(SR * DURATION)

    audio, _ = librosa.load(
        file_path,
        sr=SR,
        duration=DURATION,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError(
            "The uploaded audio file is empty."
        )

    # Pad / crop to exactly 4 seconds
    audio = np.pad(
        audio,
        (0, max(0, n - len(audio)))
    )[:n]

    # Pre-emphasis
    audio = np.append(
        audio[0],
        audio[1:] - 0.97 * audio[:-1]
    ).astype(np.float32)

    # LFCC
    lfcc = compute_lfcc(audio)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Delta LFCC
    delta_lfcc = librosa.feature.delta(lfcc)

    # Combine
    feat = np.concatenate(
        [
            lfcc,
            mfcc,
            delta_lfcc
        ],
        axis=0
    )

    # Force 128 frames
    T = feat.shape[1]

    feat = np.pad(
        feat,
        (
            (0, 0),
            (
                0,
                max(
                    0,
                    TARGET_FRAMES - T
                )
            )
        )
    )[:, :TARGET_FRAMES]

    # Normalization
    feat = (
        feat - feat.mean(
            1,
            keepdims=True
        )
    ) / (
        feat.std(
            1,
            keepdims=True
        ) + 1e-6
    )

    # Model expects:
    # (batch, 180, 128, 1)
    return feat.astype(
        np.float32
    )[np.newaxis, ..., np.newaxis]


# ---------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------
def analyze_audio(filepath: str) -> dict:

    start_time = time.perf_counter()

    filename = os.path.basename(filepath)

    try:

        model = get_model()

        print(
            f"Analyzing audio: {filename}"
        )

        features = extract_features(
            filepath
        )

        print(
            "Feature shape:",
            features.shape
        )

        probability = float(
            model.predict(
                features,
                verbose=0
            )[0][0]
        )

        probability = max(
            0.0,
            min(1.0, probability)
        )

        # Kaggle threshold
        is_fake = probability >= THRESHOLD

        fake_probability_percent = round(
            probability * 100,
            2
        )

        confidence = (
            probability
            if is_fake
            else 1 - probability
        )

        confidence_percent = round(
            confidence * 100,
            2
        )

        # Risk level
        if probability >= 0.90:

            risk_level = "High"

        elif probability >= THRESHOLD:

            risk_level = "Medium"

        else:

            risk_level = "Low"

        # Prediction
        if is_fake:

            prediction = "Fake"

            explanation = (
                f"The anti-spoofing model assigned a "
                f"{fake_probability_percent}% probability "
                f"that this audio is synthetic."
            )

            recommendation = (
                "Treat this result as potentially "
                "synthetic and independently verify "
                "important recordings."
            )

        else:

            prediction = "Real"

            explanation = (
                f"The anti-spoofing model assigned a "
                f"{fake_probability_percent}% probability "
                f"that this audio is synthetic. "
                f"The result is below the configured "
                f"detection threshold."
            )

            recommendation = (
                "This automated result indicates a "
                "likely human recording, but important "
                "recordings should still be independently "
                "verified."
            )

        # -------------------------------------------------
        # Waveform for UI
        # -------------------------------------------------
        audio, _ = librosa.load(
            filepath,
            sr=SR,
            duration=DURATION,
            mono=True
        )

        if len(audio) > 0:

            indices = np.linspace(
                0,
                len(audio) - 1,
                80
            ).astype(int)

            waveform = [
                round(
                    float(audio[i]),
                    3
                )
                for i in indices
            ]

            max_value = max(
                abs(x)
                for x in waveform
            )

            if max_value > 0:

                waveform = [
                    round(
                        x / max_value,
                        3
                    )
                    for x in waveform
                ]

        else:

            waveform = [0.0] * 80

        processing_time = int(
            (
                time.perf_counter()
                - start_time
            ) * 1000
        )

        return {

            "prediction": prediction,

            "confidence": confidence_percent,

            "fake_probability":
                fake_probability_percent,

            "risk_level": risk_level,

            "explanation": explanation,

            "recommendation":
                recommendation,

            "waveform": waveform,

            "model_used":
                "AntiSpoof_CNN_BiLSTM "
                "(LFCC + MFCC + Delta LFCC)",

            "processing_time_ms":
                processing_time,

            "analyzed_at":
                datetime.utcnow().isoformat(),

            "threshold": THRESHOLD
        }

    except Exception as e:

        print(
            f"Audio analysis error: "
            f"{type(e).__name__}: {e}"
        )

        raise