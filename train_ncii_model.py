import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

# ============================================================
# NCII SAFETY-SCREENING MODEL
# Binary classification:
# 0 = normal
# 1 = nsfw / sensitive
#
# IMPORTANT:
# This model detects sensitive visual content.
# It cannot determine consent or prove NCII.
# ============================================================

DATASET_DIR = "ncii_dataset"
MODEL_PATH = "models/ncii_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")

print("=" * 60)
print("NCII SAFETY-SCREENING MODEL TRAINING")
print("=" * 60)

# Check dataset folders
for folder in [
    os.path.join(TRAIN_DIR, "normal"),
    os.path.join(TRAIN_DIR, "nsfw"),
    os.path.join(VAL_DIR, "normal"),
    os.path.join(VAL_DIR, "nsfw"),
]:
    os.makedirs(folder, exist_ok=True)

# Count images
def count_files(folder):
    return len([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ])

print("\nDataset counts:")
print("Train normal :", count_files(os.path.join(TRAIN_DIR, "normal")))
print("Train nsfw   :", count_files(os.path.join(TRAIN_DIR, "nsfw")))
print("Val normal   :", count_files(os.path.join(VAL_DIR, "normal")))
print("Val nsfw     :", count_files(os.path.join(VAL_DIR, "nsfw")))

train_count = (
    count_files(os.path.join(TRAIN_DIR, "normal")) +
    count_files(os.path.join(TRAIN_DIR, "nsfw"))
)

val_count = (
    count_files(os.path.join(VAL_DIR, "normal")) +
    count_files(os.path.join(VAL_DIR, "nsfw"))
)

if train_count == 0 or val_count == 0:
    print("\nERROR: Dataset is empty.")
    print("Add images to the normal/nsfw folders before training.")
    raise SystemExit(1)

# ============================================================
# LOAD DATA
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("\nClass names:", train_ds.class_names)

# Improve input pipeline
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
])

# ============================================================
# MODEL
# ============================================================

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))

x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="nsfw_probability"
)(x)

model = keras.Model(inputs, outputs)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

os.makedirs("models", exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
]

# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ============================================================
# FINAL SAVE
# ============================================================

model.save(MODEL_PATH)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("Model saved to:")
print(os.path.abspath(MODEL_PATH))

# ============================================================
# FINAL EVALUATION
# ============================================================

results = model.evaluate(val_ds, verbose=1)

print("\nValidation results:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

print("\nIMPORTANT:")
print("This model performs sensitive-content screening.")
print("It does NOT determine consent or prove NCII.")