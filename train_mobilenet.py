import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ==========================================
# SETTINGS
# ==========================================

DATA_DIR = "processed_data"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 15

SEED = 42


# ==========================================
# LOAD TRAINING DATA
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["no_tumor", "tumor"],
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ==========================================
# LOAD VALIDATION DATA
# ==========================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["no_tumor", "tumor"],
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


print("\n======================================")
print("DATASET INFORMATION")
print("======================================")

print("Classes:", train_dataset.class_names)


# ==========================================
# CALCULATE CLASS WEIGHTS
# ==========================================

labels = []

for images, batch_labels in train_dataset:
    labels.extend(batch_labels.numpy().flatten())

labels = np.array(labels)

class_0_count = np.sum(labels == 0)
class_1_count = np.sum(labels == 1)

total = len(labels)

weight_0 = total / (2 * class_0_count)
weight_1 = total / (2 * class_1_count)

class_weights = {
    0: weight_0,
    1: weight_1
}

print("\nClass counts:")
print("No tumor:", class_0_count)
print("Tumor:", class_1_count)

print("\nClass weights:")
print(class_weights)


# ==========================================
# PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)


# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10)
])


# ==========================================
# LOAD MOBILE NET V2
# ==========================================

print("\n======================================")
print("LOADING MOBILENETV2")
print("======================================")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


print("MobileNetV2 loaded successfully!")


# ==========================================
# BUILD MODEL
# ==========================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = layers.Rescaling(
    1.0 / 127.5,
    offset=-1
)(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# MODEL SUMMARY
# ==========================================

print("\n======================================")
print("MODEL SUMMARY")
print("======================================")

model.summary()


# ==========================================
# CALLBACKS
# ==========================================

checkpoint = ModelCheckpoint(
    "brain_tumor_model.keras",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=0.000001,
    verbose=1
)


# ==========================================
# PHASE 1
# ==========================================

print("\n======================================")
print("PHASE 1: TRANSFER LEARNING")
print("======================================")


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ==========================================
# PHASE 2: FINE TUNING
# ==========================================

print("\n======================================")
print("PHASE 2: FINE-TUNING")
print("======================================")


base_model.trainable = True


# Freeze early layers
for layer in base_model.layers[:-30]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ==========================================
# LOAD BEST MODEL
# ==========================================

print("\n======================================")
print("LOADING BEST MODEL")
print("======================================")


model = tf.keras.models.load_model(
    "brain_tumor_model.keras"
)


# ==========================================
# FINAL VALIDATION
# ==========================================

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)


print("\n======================================")
print("FINAL RESULT")
print("======================================")

print(
    f"Validation Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Validation Loss: {loss:.4f}"
)

print("\n======================================")
print("MODEL SAVED SUCCESSFULLY")
print("======================================")

print(
    "brain_tumor_model.keras"
)