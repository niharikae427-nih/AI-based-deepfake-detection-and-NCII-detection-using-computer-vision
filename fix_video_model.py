import json
import zipfile
import tensorflow as tf

source = "models/best_video_cnn_lstm.keras"
architecture_file = "models/video_model_architecture_fixed.json"
weights_file = "models/video_model.weights.h5"
output_file = "models/video_cnn_lstm_fixed.keras"

def remove_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for value in obj.values():
            remove_quantization_config(value)
    elif isinstance(obj, list):
        for value in obj:
            remove_quantization_config(value)

print("Reading downloaded model...")
with zipfile.ZipFile(source, "r") as z:
    names = z.namelist()
    print("Model files:", names)

    config = json.loads(z.read("config.json"))
    remove_quantization_config(config)

    with open(architecture_file, "w", encoding="utf-8") as f:
        json.dump(config, f)

    weight_name = next(
        (name for name in names if name.endswith("model.weights.h5")),
        None
    )
    if weight_name is None:
        raise FileNotFoundError("model.weights.h5 not found inside model")

    with z.open(weight_name) as src, open(weights_file, "wb") as dst:
        dst.write(src.read())

print("Rebuilding model...")
with open(architecture_file, "r", encoding="utf-8") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights(weights_file)

model.save(output_file)

print("VIDEO MODEL REPAIRED SUCCESSFULLY")
print("Input:", model.input_shape)
print("Output:", model.output_shape)
print("Saved:", output_file)
