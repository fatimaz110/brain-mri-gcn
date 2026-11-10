from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from hybrid_model import build_gcn_transformer


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "processed_data"
MODEL_PATH = ROOT_DIR / "brain_tumor_model.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42


def load_datasets():
    class_names = ["no_tumor", "tumor"]
    common_args = {
        "directory": str(DATA_DIR),
        "labels": "inferred",
        "label_mode": "binary",
        "class_names": class_names,
        "validation_split": 0.2,
        "seed": SEED,
        "image_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
    }
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        subset="training", shuffle=True, **common_args
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        subset="validation", shuffle=False, **common_args
    )
    return train_dataset, validation_dataset


def calculate_class_weights(dataset):
    labels = np.concatenate([batch_labels.numpy().ravel() for _, batch_labels in dataset])
    counts = np.bincount(labels.astype(np.int32), minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Both classes must contain samples; counts={counts.tolist()}")
    total = counts.sum()
    return {
        class_index: total / (2.0 * count)
        for class_index, count in enumerate(counts)
    }


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    tf.keras.utils.set_random_seed(SEED)
    train_dataset, validation_dataset = load_datasets()
    class_weights = calculate_class_weights(train_dataset)

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="mri_augmentation",
    )

    train_dataset = train_dataset.map(
        lambda images, labels: (augmentation(images, training=True), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    model = build_gcn_transformer(input_shape=(*IMG_SIZE, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    loss, accuracy, auc = model.evaluate(validation_dataset, verbose=1)
    model.save(str(MODEL_PATH))
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation loss: {loss:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
