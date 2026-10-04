import zipfile
import json
import os
import shutil

SOURCE = r"C:\Users\Dell\Downloads\app\deepfake_cnn_224.keras"
REPAIRED = r"C:\Users\Dell\Downloads\app\deepfake_cnn_224_repaired.keras"
TEMP_DIR = r"C:\Users\Dell\Downloads\app\keras_repair_temp"

if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

os.makedirs(TEMP_DIR)

print("Opening model:")
print(SOURCE)

with zipfile.ZipFile(SOURCE, "r") as z:
    z.extractall(TEMP_DIR)

config_path = os.path.join(TEMP_DIR, "config.json")

if not os.path.exists(config_path):
    raise FileNotFoundError("config.json not found inside .keras file")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

removed = 0

def clean(obj):
    global removed

    if isinstance(obj, dict):
        if "quantization_config" in obj and obj["quantization_config"] is None:
            del obj["quantization_config"]
            removed += 1

        for value in obj.values():
            clean(value)

    elif isinstance(obj, list):
        for item in obj:
            clean(item)

clean(config)

print("Removed quantization_config entries:", removed)

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

with zipfile.ZipFile(REPAIRED, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(TEMP_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, TEMP_DIR)
            z.write(full_path, arcname)

shutil.rmtree(TEMP_DIR)

print()
print("REPAIRED MODEL CREATED:")
print(REPAIRED)