import os
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np


try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - only for training environment
    raise SystemExit(f"TensorFlow is required for this training script. Install it in Kaggle: {exc}")


DATASET_ROOT = Path("/kaggle/input")
OUTPUT_DIR = Path("/kaggle/working")
MODEL_DIR = Path("models/weights")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def discover_dataset_root():
    if not DATASET_ROOT.exists():
        print(f"[WARN] /kaggle/input does not exist in this environment: {DATASET_ROOT}")
        return None

    candidates = []
    for root, dirs, files in os.walk(DATASET_ROOT):
        for name in ["deepfake-detection-challenge", "deepfake_detection_challenge", "faceforensicspp", "faceforensics++", "faceforensics", "ffpp"]:
            if name.lower() in [d.lower() for d in dirs]:
                candidates.append(Path(root) / name)

    if not candidates:
        print("[INFO] No common deepfake dataset folder names were found under /kaggle/input.")
        for root, dirs, files in os.walk(DATASET_ROOT):
            if root == str(DATASET_ROOT):
                print("Available top-level folders:")
                for d in sorted(dirs[:20]):
                    print(f" - {d}")
                break
        return None

    for candidate in candidates:
        print(f"[INFO] Candidate dataset root: {candidate}")
    return candidates[0]


def list_dataset_samples(dataset_root):
    if dataset_root is None:
        return []

    samples = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                samples.append(Path(root) / file)
    return sorted(samples)


def build_video_index(dataset_root):
    index = []
    metadata_json = dataset_root / "metadata.json"
    dfdc_root = None

    if "deepfake" in dataset_root.name.lower() or metadata_json.exists():
        dfdc_root = dataset_root
        if not metadata_json.exists():
            print(f"[WARN] DFDC metadata.json not found at {metadata_json}")
            return index
        with open(metadata_json, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        for video_id, info in metadata.items():
            label = info.get("label", "unknown").strip().lower()
            if label not in {"real", "fake"}:
                continue
            video_path = dataset_root / "train_sample_videos" / f"{video_id}.mp4"
            if not video_path.exists():
                video_path = dataset_root / f"{video_id}.mp4"
            if video_path.exists():
                index.append({
                    "video_path": str(video_path),
                    "label": "REAL" if label == "real" else "DEEPFAKE",
                    "source": "DFDC",
                    "manipulation_type": "DFDC",
                    "video_id": video_id,
                })
        return index

    real_dirs = [
        dataset_root / "original_sequences",
        dataset_root / "original",
        dataset_root / "real",
    ]
    fake_dirs = [
        dataset_root / "manipulated_sequences",
        dataset_root / "manipulated",
        dataset_root / "fake",
    ]

    for real_dir in real_dirs:
        if real_dir.exists():
            for video_path in real_dir.rglob("*.mp4"):
                index.append({
                    "video_path": str(video_path),
                    "label": "REAL",
                    "source": "FaceForensics++",
                    "manipulation_type": "Original",
                    "video_id": video_path.stem,
                })

    for fake_dir in fake_dirs:
        if fake_dir.exists():
            for video_path in fake_dir.rglob("*.mp4"):
                manip = video_path.parent.name
                index.append({
                    "video_path": str(video_path),
                    "label": "DEEPFAKE",
                    "source": "FaceForensics++",
                    "manipulation_type": manip,
                    "video_id": video_path.stem,
                })

    return index


def split_videos(video_index, seed=42):
    if not video_index:
        return {"train": [], "val": [], "test": []}

    by_video = {}
    for item in video_index:
        vid = item["video_id"]
        by_video.setdefault(vid, []).append(item)

    unique_videos = list(by_video.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_videos)

    total = len(unique_videos)
    train_end = max(1, int(total * 0.70))
    val_end = train_end + max(1, int(total * 0.15))

    train_ids = unique_videos[:train_end]
    val_ids = unique_videos[train_end:val_end]
    test_ids = unique_videos[val_end:]

    split = {"train": [], "val": [], "test": []}
    for vid, items in by_video.items():
        bucket = "train" if vid in train_ids else "val" if vid in val_ids else "test"
        split[bucket].extend(items)

    for bucket in split:
        random.Random(seed + 11).shuffle(split[bucket])
    return split


def detect_face_and_crop(frame_bgr, detector=None):
    if detector is not None:
        try:
            result = detector.detect(frame_bgr)
            if result and len(result) > 0:
                x1, y1, x2, y2 = result[0][0:4].astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_bgr.shape[1], x2)
                y2 = min(frame_bgr.shape[0], y2)
                face = frame_bgr[y1:y2, x1:x2]
                if face.size > 0:
                    return face
        except Exception:
            pass

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = frame_bgr[y:y + h, x:x + w]
        if face.size > 0:
            return face

    return frame_bgr


def extract_frames(video_path, output_root, max_frames=8, detector=None):
    output_root.mkdir(parents=True, exist_ok=True)
    vc = cv2.VideoCapture(str(video_path))
    if not vc.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return []

    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        vc.release()
        return []

    sample_indices = []
    if total_frames <= max_frames:
        sample_indices = list(range(total_frames))
    else:
        step = max(1, total_frames // max_frames)
        sample_indices = list(range(0, total_frames, step))[:max_frames]

    saved = []
    frame_counter = 0
    for idx in sample_indices:
        vc.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = vc.read()
        if not ok:
            continue

        crop = detect_face_and_crop(frame, detector=detector)
        resized = cv2.resize(crop, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        out_path = output_root / f"frame_{frame_counter:03d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        saved.append(out_path)
        frame_counter += 1

    vc.release()
    return saved


def build_dataset_from_videos(video_index, split_name, output_base, detector=None):
    split_dir = output_base / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    for item in video_index:
        video_label = item["label"]
        class_dir = split_dir / video_label.lower()
        class_dir.mkdir(parents=True, exist_ok=True)
        saved = extract_frames(Path(item["video_path"]), class_dir, max_frames=8, detector=detector)
        for i, path in enumerate(saved):
            meta_rows.append({
                "dataset_source": item["source"],
                "video_id": item["video_id"],
                "frame_id": i,
                "label": video_label,
                "manipulation_type": item["manipulation_type"],
                "split": split_name,
                "file_path": str(path),
            })

    meta_path = split_dir / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_source", "video_id", "frame_id", "label", "manipulation_type", "split", "file_path"])
        writer.writeheader()
        writer.writerows(meta_rows)

    return meta_rows


def create_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name="image_input")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="deepfake_prob")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="deepfake_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def build_tf_dataset(split_dir):
    image_paths = []
    labels = []
    for label_name in ["real", "deepfake"]:
        label_dir = split_dir / label_name
        if label_dir.exists():
            for image_file in sorted(label_dir.glob("**/*.png")):
                image_paths.append(str(image_file))
                labels.append(1 if label_name == "deepfake" else 0)

    if not image_paths:
        raise ValueError(f"No extracted frames were found in {split_dir}")

    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))
        images.append(rgb.astype("float32") / 255.0)

    if not images:
        raise ValueError(f"Images could not be decoded from {split_dir}")

    x = np.stack(images)
    y = np.asarray(labels[:len(x)], dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(32).prefetch(tf.data.AUTOTUNE)
    return ds, y


def train_model():
    dataset_root = discover_dataset_root()
    if dataset_root is None:
        raise FileNotFoundError("No authorized deepfake dataset was found under /kaggle/input.")

    print("[INFO] Dataset inspection:")
    for root, dirs, files in os.walk(dataset_root):
        if root.count(os.sep) - dataset_root.as_posix().count(os.sep) <= 2:
            print(f"{root}")
            if files:
                print("  sample:", files[:5])

    all_videos = build_video_index(dataset_root)
    print(f"[INFO] Total videos discovered: {len(all_videos)}")
    for label in ["REAL", "DEEPFAKE"]:
        count = sum(1 for item in all_videos if item["label"] == label)
        print(f"[INFO] {label}: {count}")

    split = split_videos(all_videos)
    output_base = OUTPUT_DIR / "deepfake_dataset"
    output_base.mkdir(parents=True, exist_ok=True)

    print("[INFO] Building training, validation, and test splits...")
    train_meta = build_dataset_from_videos(split["train"], "train", output_base)
    val_meta = build_dataset_from_videos(split["val"], "val", output_base)
    test_meta = build_dataset_from_videos(split["test"], "test", output_base)

    print(f"[INFO] Train frames: {len(train_meta)}")
    print(f"[INFO] Validation frames: {len(val_meta)}")
    print(f"[INFO] Test frames: {len(test_meta)}")

    train_ds, _ = build_tf_dataset(output_base / "train")
    val_ds, _ = build_tf_dataset(output_base / "val")
    test_ds, _ = build_tf_dataset(output_base / "test")

    model = create_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_DIR / "best_model.keras"), monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    print("[INFO] Training CNN on Kaggle GPU...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_ds, verbose=0)
    print("[INFO] Test loss:", test_loss)
    print("[INFO] Test accuracy:", test_acc)
    print("[INFO] Test precision:", test_precision)
    print("[INFO] Test recall:", test_recall)

    final_path = MODEL_DIR / "deepfake_cnn_224.keras"
    model.save(final_path)
    print(f"[INFO] Saved model: {final_path}")
    print("[INFO] TensorFlow model is ready for the Flask app image endpoint.")

    return history, final_path


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    train_model()
