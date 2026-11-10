import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model


# ==========================================
# SETTINGS
# ==========================================

DATA_DIR = "processed_data"
MODEL_PATH = "brain_tumor_model.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


# ==========================================
# LOAD MODEL
# ==========================================

model = load_model(MODEL_PATH)

print("\nModel loaded successfully.")
print("Model:", MODEL_PATH)


# ==========================================
# LOAD DATA
# ==========================================

dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["no_tumor", "tumor"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# MODEL EVALUATION
# ==========================================

loss, accuracy = model.evaluate(
    dataset,
    verbose=1
)

print("\n===================================")
print("MODEL EVALUATION")
print("===================================")

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Loss: {loss:.4f}")


# ==========================================
# PREDICTIONS
# ==========================================

all_predictions = []
all_labels = []

for images, labels in dataset:

    predictions = model.predict(
        images,
        verbose=0
    ).flatten()

    all_predictions.extend(predictions)
    all_labels.extend(labels.numpy().flatten())


all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)


# ==========================================
# CONFUSION COUNTS
# ==========================================

predicted_labels = (
    all_predictions >= 0.5
).astype(int)


true_labels = all_labels.astype(int)


true_negative = np.sum(
    (true_labels == 0) &
    (predicted_labels == 0)
)

false_positive = np.sum(
    (true_labels == 0) &
    (predicted_labels == 1)
)

true_positive = np.sum(
    (true_labels == 1) &
    (predicted_labels == 1)
)

false_negative = np.sum(
    (true_labels == 1) &
    (predicted_labels == 0)
)


# ==========================================
# RESULTS
# ==========================================

print("\n===================================")
print("CONFUSION COUNTS")
print("===================================")

print("True Negatives :", true_negative)
print("False Positives:", false_positive)
print("True Positives :", true_positive)
print("False Negatives:", false_negative)


# ==========================================
# CLASS PERFORMANCE
# ==========================================

no_tumor_total = true_negative + false_positive
tumor_total = true_positive + false_negative


if no_tumor_total > 0:

    no_tumor_accuracy = (
        true_negative / no_tumor_total
    ) * 100

    print(
        f"\nNo-tumor accuracy: "
        f"{no_tumor_accuracy:.2f}%"
    )


if tumor_total > 0:

    tumor_accuracy = (
        true_positive / tumor_total
    ) * 100

    print(
        f"Tumor accuracy: "
        f"{tumor_accuracy:.2f}%"
    )


# ==========================================
# SAMPLE PREDICTIONS
# ==========================================

print("\n===================================")
print("SAMPLE PREDICTIONS")
print("===================================")

file_paths = []

for class_name in ["no_tumor", "tumor"]:

    folder = Path(DATA_DIR) / class_name

    for file in folder.iterdir():

        if file.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:

            file_paths.append(
                (class_name, file)
            )


for true_class, file_path in file_paths[:10]:

    image = tf.keras.utils.load_img(
        file_path,
        target_size=IMG_SIZE
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probability = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )


    if probability >= 0.5:

        prediction = "tumor"

    else:

        prediction = "no_tumor"


    print(
        f"{file_path.name:25} | "
        f"Actual: {true_class:8} | "
        f"Predicted: {prediction:8} | "
        f"Probability: {probability:.4f}"
    )