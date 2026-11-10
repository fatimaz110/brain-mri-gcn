from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from hybrid_model import build_gcn_transformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\fatim\OneDrive\Documents\brain-mri-gcn"
)

TRAIN_DIR = BASE_DIR / "data" / "training_split" / "train"
VAL_DIR = BASE_DIR / "data" / "training_split" / "validation"

MODEL_PATH = BASE_DIR / "brain_tumor_model.keras"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)

BATCH_SIZE = 32

NUM_CLASSES = 4

EPOCHS = 40

SEED = 42


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=True,
    seed=SEED
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=False
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomRotation(
            0.05
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        layers.RandomContrast(
            0.10
        )
    ],
    name="mri_augmentation"
)


# ============================================================
# APPLY AUGMENTATION ONLY TO TRAINING DATA
# ============================================================

train_ds = train_ds.map(
    lambda x, y: (
        data_augmentation(x, training=True),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# PERFORMANCE
# ============================================================

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\n========================================")
print("BUILDING GCN + TRANSFORMER")
print("========================================")

model = build_gcn_transformer(
    input_shape=(128, 128, 3),
    patch_size=8,
    embedding_dim=96,
    num_heads=4,
    dropout=0.25,
    num_classes=NUM_CLASSES
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    mode="max",
    restore_best_weights=True,
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)


checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

best_train_accuracy = max(
    history.history["accuracy"]
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)

print(
    f"Best Training Accuracy: "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "\nModel saved at:"
)

print(MODEL_PATH)