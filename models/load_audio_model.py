import json
import os
import tensorflow as tf
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARCH_PATH = os.path.join(
    BASE_DIR,
    "audio_deepfake_architecture.json"
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "audio_deepfake.weights.h5"
)

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("Loading architecture...")

with open(ARCH_PATH, "r", encoding="utf-8") as f:
    architecture = json.load(f)


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


cleaned_architecture = clean_config(architecture)

architecture_json = json.dumps(cleaned_architecture)

print("Building model...")

model = keras.models.model_from_json(
    architecture_json
)

print("Architecture loaded successfully.")

print("Loading weights...")

model.load_weights(WEIGHTS_PATH)

print("Weights loaded successfully.")

print()
print("MODEL READY")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)
print("Parameters  :", model.count_params())