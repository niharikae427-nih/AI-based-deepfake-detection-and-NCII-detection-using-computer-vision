from datasets import load_dataset
from pathlib import Path
import random

BASE = Path("ncii_dataset")

folders = [
    BASE / "train" / "normal",
    BASE / "train" / "nsfw",
    BASE / "validation" / "normal",
    BASE / "validation" / "nsfw",
]

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
dataset = load_dataset("theusamaaslam/nsfw-dataset")

print("\nAvailable splits:")
print(dataset)

TRAIN_PER_CLASS = 500
VAL_PER_CLASS = 100

random.seed(42)


def save_images(split, label_id, output_folder, limit):
    rows = list(dataset[split])
    random.shuffle(rows)

    saved = 0

    for row in rows:
        if saved >= limit:
            break

        if row["label"] != label_id:
            continue

        image = row["image"]

        if image is None:
            continue

        try:
            image = image.convert("RGB")

            output_file = output_folder / f"{split}_{saved:05d}.jpg"

            image.save(
                output_file,
                "JPEG",
                quality=90
            )

            saved += 1

        except Exception as e:
            print("Skipping image:", e)

    print(f"{output_folder} -> {saved} images")


print("\nPreparing training images...")

# Dataset labels:
# 0 = nsfw
# 1 = safe

save_images(
    "train",
    1,
    BASE / "train" / "normal",
    TRAIN_PER_CLASS
)

save_images(
    "train",
    0,
    BASE / "train" / "nsfw",
    TRAIN_PER_CLASS
)

print("\nPreparing validation images...")

save_images(
    "validation",
    1,
    BASE / "validation" / "normal",
    VAL_PER_CLASS
)

save_images(
    "validation",
    0,
    BASE / "validation" / "nsfw",
    VAL_PER_CLASS
)

print("\n================================")
print("Dataset preparation completed!")
print("================================")