import os
import numpy as np
from PIL import Image
import tensorflow as tf

model_path = r".\deepfake_cnn_224_repaired.keras"
frames_dir = r".\video_test_frames"

model = tf.keras.models.load_model(model_path)

results = []

for filename in sorted(os.listdir(frames_dir)):
    if not filename.lower().endswith(".jpg"):
        continue

    path = os.path.join(frames_dir, filename)

    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))

    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    probability = float(model.predict(x, verbose=0)[0][0])

    results.append((filename, probability))

print("\nFRAME RESULTS")
print("-" * 55)

for filename, probability in results:
    label = "DEEPFAKE" if probability >= 0.5 else "REAL"
    print(
        f"{filename}: {label} | "
        f"Deepfake probability: {probability*100:.2f}%"
    )

avg_probability = sum(p for _, p in results) / len(results)

print("\n" + "=" * 55)
print(f"Average Deepfake Probability: {avg_probability*100:.2f}%")

if avg_probability >= 0.5:
    print("VIDEO RESULT: DEEPFAKE")
else:
    print("VIDEO RESULT: REAL")

print("=" * 55)
