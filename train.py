import csv
import hashlib
import json
import os
import random

import numpy as np
import tensorflow as tf

from hybrid_model import build_gcn_transformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "data",
    "Training"
)

TEST_DIR = os.environ.get(
    "BRAIN_MRI_TEST_DIR",
    os.path.join(BASE_DIR, "data", "Testing"),
)
BRATS_DIR = os.environ.get(
    "BRAIN_MRI_BRATS_DIR",
    os.path.join(BASE_DIR, "data", "brats_slices"),
)
BRATS_SPLIT_DIR = os.environ.get(
    "BRAIN_MRI_BRATS_SPLIT_DIR",
    os.path.join(BASE_DIR, "data", "brats_splits"),
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.environ.get(
    "BRAIN_MRI_MODEL_PATH",
    os.path.join(MODEL_DIR, "gcn_transformer_brain_tumor.keras"),
)

CLASS_NAMES_PATH = os.environ.get(
    "BRAIN_MRI_CLASSES_PATH",
    os.path.join(MODEL_DIR, "gcn_transformer_brain_tumor_classes.json"),
)

METRICS_PATH = os.environ.get(
    "BRAIN_MRI_METRICS_PATH",
    os.path.join(MODEL_DIR, "gcn_transformer_brain_tumor_test_metrics.json"),
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = int(os.environ.get("BRAIN_MRI_EPOCHS", "40"))
VALIDATION_SPLIT = 0.20
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
BRATS_MAX_TRAIN_SLICES_PER_PATIENT = 8
BRATS_MAX_EVAL_SLICES_PER_PATIENT = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset(directory, name):

    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"{name} folder not found:\n{directory}"
        )

    print(f"\n{name}:")
    print(directory)

    total = 0

    for class_name in CLASS_NAMES:

        class_dir = os.path.join(
            directory,
            class_name
        )

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Missing class folder:\n{class_dir}"
            )

        image_count = 0

        for filename in os.listdir(class_dir):

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".gif"
            }:
                image_count += 1

        print(
            f"  {class_name}: {image_count}"
        )

        total += image_count

    print(f"  TOTAL: {total}")

    return total


# ============================================================
# LOAD DATASET
# ============================================================

def load_datasets():
    """Load four-class images and patient-split BraTS glioma slices."""
    train_paths = []
    train_labels = []
    validation_paths = []
    validation_labels = []
    test_paths = []
    test_labels = []
    training_hash_labels = {}
    test_hash_labels = {}
    duplicate_test_exclusions = {name: 0 for name in CLASS_NAMES}
    random_generator = random.Random(SEED)

    def list_images(directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Dataset class folder not found: {directory}")
        return sorted(
            os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        )

    def file_digest(path):
        digest = hashlib.sha256()
        with open(path, "rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.digest()

    def read_brats_split(split_name, per_patient_limit, split_seed):
        manifest_path = os.path.join(BRATS_SPLIT_DIR, split_name + ".csv")
        images_dir = os.path.join(BRATS_DIR, "images")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"BraTS patient split manifest not found: {manifest_path}")

        patient_rows = {}
        with open(manifest_path, newline="", encoding="utf-8") as manifest_file:
            reader = csv.DictReader(manifest_file)
            required_columns = {"patient_id", "filename"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(
                    f"BraTS manifest must contain columns {sorted(required_columns)}: "
                    f"{manifest_path}"
                )
            for row in reader:
                patient_id = row["patient_id"].strip()
                filename = row["filename"].strip()
                if not patient_id or not filename:
                    raise ValueError(f"Invalid patient/image row in {manifest_path}.")
                patient_rows.setdefault(patient_id, []).append(filename)

        if not patient_rows:
            raise ValueError(f"No BraTS slices found in {manifest_path}.")

        split_rng = random.Random(split_seed)
        selected_paths = []
        for patient_id in sorted(patient_rows):
            filenames = sorted(set(patient_rows[patient_id]))
            split_rng.shuffle(filenames)
            selected_paths.extend(
                os.path.join(images_dir, filename)
                for filename in filenames[:per_patient_limit]
            )

        missing_paths = [path for path in selected_paths if not os.path.isfile(path)]
        if missing_paths:
            raise FileNotFoundError(
                f"BraTS image listed in {manifest_path} is missing: {missing_paths[0]}"
            )
        return selected_paths, set(patient_rows)

    brats_train_paths, brats_train_patients = read_brats_split(
        "train",
        BRATS_MAX_TRAIN_SLICES_PER_PATIENT,
        SEED + 1,
    )
    brats_validation_paths, brats_validation_patients = read_brats_split(
        "validation",
        BRATS_MAX_EVAL_SLICES_PER_PATIENT,
        SEED + 2,
    )
    brats_test_paths, brats_test_patients = read_brats_split(
        "test",
        BRATS_MAX_EVAL_SLICES_PER_PATIENT,
        SEED + 3,
    )
    if (
        brats_train_patients & brats_validation_patients
        or brats_train_patients & brats_test_patients
        or brats_validation_patients & brats_test_patients
    ):
        raise ValueError("BraTS patients overlap between train, validation, and test.")

    train_groups_by_class = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        training_class_dir = os.path.join(TRAIN_DIR, class_name)
        class_paths = list_images(training_class_dir)
        if len(class_paths) < 2:
            raise ValueError(
                f"Class {class_name!r} needs at least two training images."
            )

        class_groups = {}
        for path in class_paths:
            digest = file_digest(path)
            prior_label = training_hash_labels.setdefault(digest, class_index)
            if prior_label != class_index:
                raise ValueError(
                    f"An identical image appears under multiple training labels: {path}"
                )
            class_groups.setdefault(digest, []).append(path)
        train_groups_by_class[class_index] = list(class_groups.values())

    for class_index, class_name in enumerate(CLASS_NAMES):
        testing_class_dir = os.path.join(TEST_DIR, class_name)
        class_test_paths = list_images(testing_class_dir)
        if not class_test_paths:
            raise ValueError(f"No test images found for class {class_name!r}.")

        for path in class_test_paths:
            digest = file_digest(path)
            if digest in training_hash_labels:
                duplicate_test_exclusions[class_name] += 1
                continue
            prior_label = test_hash_labels.get(digest)
            if prior_label is not None:
                if prior_label != class_index:
                    raise ValueError(
                        f"An identical image appears under multiple test labels: {path}"
                    )
                continue
            test_hash_labels[digest] = class_index
            test_paths.append(path)
            test_labels.append(class_index)

    brats_training_paths = []
    for path in brats_train_paths:
        digest = file_digest(path)
        prior_label = training_hash_labels.get(digest)
        if prior_label is not None:
            if prior_label != 0:
                raise ValueError(
                    f"A BraTS glioma slice exactly matches a non-glioma training label: {path}"
                )
            continue
        training_hash_labels[digest] = 0
        brats_training_paths.append(path)

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_groups = train_groups_by_class[class_index]
        random_generator.shuffle(class_groups)
        target_validation_count = max(
            1,
            int(round(sum(len(group) for group in class_groups) * VALIDATION_SPLIT)),
        )
        validation_class_paths = []
        training_class_paths = []
        for group in class_groups:
            if len(validation_class_paths) < target_validation_count:
                validation_class_paths.extend(group)
            else:
                training_class_paths.extend(group)

        if not training_class_paths or not validation_class_paths:
            raise ValueError(
                f"Could not create train/validation groups for {class_name!r}."
            )

        train_paths.extend(training_class_paths)
        train_labels.extend([class_index] * len(training_class_paths))
        validation_paths.extend(validation_class_paths)
        validation_labels.extend([class_index] * len(validation_class_paths))

        if class_index == 0:
            train_paths.extend(brats_training_paths)
            train_labels.extend([class_index] * len(brats_training_paths))
            validation_paths.extend(brats_validation_paths)
            validation_labels.extend([class_index] * len(brats_validation_paths))

        print(
            f"{class_name}: train={len(training_class_paths)}, "
            f"validation={len(validation_class_paths)}, "
            f"test={sum(label == class_index for label in test_labels)}, "
            f"excluded exact train/test copies="
            f"{duplicate_test_exclusions[class_name]}"
        )

    print(
        "BraTS 2020 glioma slices: "
        f"train={len(brats_training_paths)} from {len(brats_train_patients)} patients, "
        f"validation={len(brats_validation_paths)} from "
        f"{len(brats_validation_patients)} patients, "
        f"test={len(brats_test_paths)} from {len(brats_test_patients)} patients."
    )

    if not test_paths:
        raise ValueError("No unique held-out test images remain after de-duplication.")

    def decode_and_resize(image_path, label):
        image_contents = tf.io.read_file(image_path)
        image = tf.image.decode_image(
            image_contents,
            channels=3,
            expand_animations=False,
        )
        image.set_shape((None, None, 3))
        image = tf.image.resize_with_pad(
            image,
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            method="bilinear",
            antialias=True,
        )
        return image, label

    def make_dataset(paths, labels, shuffle):
        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, np.asarray(labels, dtype=np.int32))
        )
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(paths),
                seed=SEED,
                reshuffle_each_iteration=True,
            )
        dataset = dataset.map(
            decode_and_resize,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return (
        make_dataset(train_paths, train_labels, shuffle=True),
        make_dataset(validation_paths, validation_labels, shuffle=False),
        make_dataset(test_paths, test_labels, shuffle=False),
        make_dataset(
            brats_test_paths,
            [0] * len(brats_test_paths),
            shuffle=False,
        ),
        duplicate_test_exclusions,
        {
            "train_patients": len(brats_train_patients),
            "validation_patients": len(brats_validation_patients),
            "test_patients": len(brats_test_patients),
            "train_slices": len(brats_training_paths),
            "validation_slices": len(brats_validation_paths),
            "test_slices": len(brats_test_paths),
            "max_train_slices_per_patient": BRATS_MAX_TRAIN_SLICES_PER_PATIENT,
            "max_eval_slices_per_patient": BRATS_MAX_EVAL_SLICES_PER_PATIENT,
        },
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

def create_augmentation():

    return tf.keras.Sequential(

        [

            tf.keras.layers.RandomFlip(
                "horizontal"
            ),

            tf.keras.layers.RandomRotation(
                0.04,
                fill_mode="constant"
            ),

            tf.keras.layers.RandomZoom(
                0.08,
                fill_mode="constant"
            ),

            tf.keras.layers.RandomTranslation(
                height_factor=0.04,
                width_factor=0.04
            ),

            tf.keras.layers.RandomContrast(
                0.12
            )

        ],

        name="mri_augmentation"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataset):

    labels = []

    for _, batch_labels in dataset:

        labels.extend(
            batch_labels.numpy().tolist()
        )

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    counts = np.bincount(
        labels,
        minlength=len(CLASS_NAMES)
    )

    if np.any(counts == 0):

        raise ValueError(
            f"One or more classes have no images: "
            f"{counts.tolist()}"
        )

    total = int(counts.sum())

    class_weights = {

        class_index:
            total /
            (
                len(CLASS_NAMES)
                * int(count)
            )

        for class_index, count
        in enumerate(counts)
    }

    return class_weights


# ============================================================
# TEST METRICS
# ============================================================

def evaluate_test_set(
    model,
    test_dataset
):

    print("\n")
    print("=" * 60)
    print("FINAL HELD-OUT TEST")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1
    )

    predictions = model.predict(
        test_dataset,
        verbose=1
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    true_labels = np.concatenate(
        [
            labels.numpy()
            for _, labels in test_dataset
        ]
    )

    true_labels = true_labels.astype(
        np.int64
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion = np.zeros(
        (
            len(CLASS_NAMES),
            len(CLASS_NAMES)
        ),
        dtype=np.int64
    )

    np.add.at(
        confusion,
        (
            true_labels,
            predicted_labels
        ),
        1
    )

    print("\nTest Accuracy:")
    print(
        f"{test_accuracy * 100:.2f}%"
    )

    print("\nConfusion Matrix:")
    print(confusion)

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    class_metrics = {}

    print("\nPer-Class Results:")

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        true_positive = int(
            confusion[
                class_index,
                class_index
            ]
        )

        false_positive = int(
            confusion[
                :,
                class_index
            ].sum()
            - true_positive
        )

        false_negative = int(
            confusion[
                class_index,
                :
            ].sum()
            - true_positive
        )

        precision = (

            true_positive /
            (
                true_positive
                + false_positive
            )

            if (
                true_positive
                + false_positive
            )

            else 0.0
        )

        recall = (

            true_positive /
            (
                true_positive
                + false_negative
            )

            if (
                true_positive
                + false_negative
            )

            else 0.0
        )

        f1 = (

            2 *
            precision *
            recall /
            (
                precision
                + recall
            )

            if (
                precision
                + recall
            )

            else 0.0
        )

        class_metrics[class_name] = {

            "precision": precision,

            "recall": recall,

            "f1_score": f1
        }

        print(
            f"{class_name}: "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}"
        )

    macro_f1 = float(
        np.mean(
            [
                metrics["f1_score"]
                for metrics
                in class_metrics.values()
            ]
        )
    )

    print(
        f"\nMacro F1: {macro_f1:.4f}"
    )

    return {

        "test_loss": float(
            test_loss
        ),

        "test_accuracy": float(
            test_accuracy
        ),

        "macro_f1": macro_f1,

        "class_names": CLASS_NAMES,

        "confusion_matrix":
            confusion.tolist(),

        "per_class":
            class_metrics
    }


def evaluate_brats_glioma_slices(model, brats_test_dataset):
    """Report BraTS performance separately as sensitivity on glioma-positive slices."""
    predictions = model.predict(brats_test_dataset, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)
    glioma_probabilities = predictions[:, 0]
    glioma_recall = float(np.mean(predicted_labels == 0))
    mean_glioma_probability = float(np.mean(glioma_probabilities))
    predicted_counts = np.bincount(predicted_labels, minlength=len(CLASS_NAMES))

    print("\nBraTS 2020 patient-held-out glioma-positive slices:")
    print(f"Top-1 glioma recall: {glioma_recall * 100:.2f}%")
    print(f"Mean glioma probability: {mean_glioma_probability * 100:.2f}%")
    print(
        "Predicted classes on BraTS test slices:",
        {
            class_name: int(predicted_counts[index])
            for index, class_name in enumerate(CLASS_NAMES)
        },
    )

    return {
        "task": "top-1 glioma classification on glioma-positive FLAIR slices",
        "slice_count": int(len(predicted_labels)),
        "glioma_top1_recall": glioma_recall,
        "mean_glioma_probability": mean_glioma_probability,
        "predicted_class_counts": {
            class_name: int(predicted_counts[index])
            for index, class_name in enumerate(CLASS_NAMES)
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("BRAIN TUMOR GCN + TRANSFORMER TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    tf.keras.utils.set_random_seed(
        SEED
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading duplicate-safe, inference-matched datasets...")

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        brats_test_dataset,
        duplicate_test_exclusions,
        brats_split_summary,
    ) = load_datasets()

    print("\nExact train/test duplicate exclusions:")
    print(duplicate_test_exclusions)

    print("\nClasses:")
    print(CLASS_NAMES)

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_dataset
    )

    print("\nClass weights:")
    print(class_weights)

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------

    augmentation = create_augmentation()

    train_dataset = train_dataset.map(

        lambda images, labels:
        (
            augmentation(
                images,
                training=True
            ),
            labels
        ),

        num_parallel_calls=tf.data.AUTOTUNE
    )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    train_dataset = train_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    validation_dataset = validation_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    test_dataset = test_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("BUILDING GCN + TRANSFORMER")
    print("=" * 60)

    model = build_gcn_transformer(

        input_shape=(128, 128, 3),

        patch_size=8,

        embedding_dim=96,

        num_heads=4,

        dropout=0.15,

        num_classes=4
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    model.summary()

    # --------------------------------------------------------
    # Create model directory
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    checkpoint = tf.keras.callbacks.ModelCheckpoint(

        MODEL_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=8,

        restore_best_weights=True,

        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=1e-7,

        verbose=1
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    history = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=EPOCHS,

        class_weight=class_weights,

        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr
        ]
    )

    # --------------------------------------------------------
    # Best training / validation accuracy
    # --------------------------------------------------------

    best_train_accuracy = max(
        history.history["accuracy"]
    )

    best_validation_accuracy = max(
        history.history["val_accuracy"]
    )

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"\nBest Training Accuracy: "
        f"{best_train_accuracy * 100:.2f}%"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Save class names
    # --------------------------------------------------------

    with open(
        CLASS_NAMES_PATH,
        "w",
        encoding="utf-8"
    ) as class_file:

        json.dump(
            CLASS_NAMES,
            class_file,
            indent=2
        )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    metrics = evaluate_test_set(
        model,
        test_dataset
    )
    metrics["brats2020_test"] = evaluate_brats_glioma_slices(
        model,
        brats_test_dataset,
    )
    metrics["brats2020_split"] = brats_split_summary

    metrics[
        "training_accuracy"
    ] = float(
        best_train_accuracy
    )

    metrics[
        "validation_accuracy"
    ] = float(
        best_validation_accuracy
    )

    metrics[
        "test_dataset"
    ] = os.path.abspath(
        TEST_DIR
    )
    metrics["duplicate_test_exclusions"] = duplicate_test_exclusions
    metrics["preprocessing"] = (
        "RGB, bilinear resize_with_pad to 128x128; identical to backend geometry"
    )
    metrics["training_data_note"] = (
        "BraTS 2020 provides glioma-positive FLAIR slices only; its test recall is "
        "reported separately and is not included in the four-class accuracy."
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as metrics_file:

        json.dump(
            metrics,
            metrics_file,
            indent=2
        )

    print("\n")
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(
        f"\nModel:\n{MODEL_PATH}"
    )

    print(
        f"\nClasses:\n{CLASS_NAMES_PATH}"
    )

    print(
        f"\nMetrics:\n{METRICS_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()