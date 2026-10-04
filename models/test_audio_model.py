import json
import os
import numpy as np
import librosa
from scipy.fftpack import dct
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARCH_PATH = os.path.join(BASE_DIR, "audio_deepfake_architecture.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "audio_deepfake.weights.h5")

SR = 16000
DURATION = 4.0
N_FFT = 512
HOP_LENGTH = 160
TARGET_FRAMES = 128
N_LFCC = 60
N_MFCC = 60
N_FILTERS = 128
THRESHOLD = 0.74


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


def load_audio_model():
    with open(ARCH_PATH, "r", encoding="utf-8") as f:
        architecture = json.load(f)

    architecture = clean_config(architecture)

    model = keras.models.model_from_json(
        json.dumps(architecture)
    )

    model.load_weights(WEIGHTS_PATH)

    return model


def compute_lfcc(audio):
    stft = np.abs(
        librosa.stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
    ) ** 2

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
                    (f - fl) /
                    (fc - fl + 1e-8)
                )

            elif fc < f <= fr:
                fb[m - 1, k] = (
                    (fr - f) /
                    (fr - fc + 1e-8)
                )

    return dct(
        np.log(
            np.dot(fb, stft) + 1e-8
        ),
        type=2,
        axis=0,
        norm="ortho"
    )[:N_LFCC]


def extract_features(file_path):

    n = int(SR * DURATION)

    audio, _ = librosa.load(
        file_path,
        sr=SR,
        duration=DURATION,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    audio = np.pad(
        audio,
        (0, max(0, n - len(audio)))
    )[:n]

    audio = np.append(
        audio[0],
        audio[1:] - 0.97 * audio[:-1]
    ).astype(np.float32)

    lfcc = compute_lfcc(audio)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    delta_lfcc = librosa.feature.delta(lfcc)

    feat = np.concatenate(
        [
            lfcc,
            mfcc,
            delta_lfcc
        ],
        axis=0
    )

    T = feat.shape[1]

    feat = np.pad(
        feat,
        (
            (0, 0),
            (0, max(0, TARGET_FRAMES - T))
        )
    )[:, :TARGET_FRAMES]

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

    return feat.astype(
        np.float32
    )[np.newaxis, ..., np.newaxis]


audio_path = input(
    "Enter full path of WAV/MP3 audio file: "
).strip().strip('"')


print("\nLoading model...")
model = load_audio_model()

print("Extracting features...")
features = extract_features(audio_path)

print("Feature shape:", features.shape)

print("Running prediction...")
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

print()
print("=" * 45)
print("AUDIO DEEPFAKE MODEL RESULT")
print("=" * 45)

print(
    f"Synthetic probability : {probability * 100:.2f}%"
)

print(
    f"Threshold             : {THRESHOLD * 100:.0f}%"
)

if probability >= THRESHOLD:
    print("Prediction            : FAKE / SYNTHETIC")
else:
    print("Prediction            : REAL / LIKELY HUMAN")

print("=" * 45)