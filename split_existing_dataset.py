from pathlib import Path
import random
import shutil


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(
    r"C:\Users\fatim\OneDrive\Documents\brain-mri-gcn"
)

SOURCE_DIR = BASE_DIR / "data" / "Training"

OUTPUT_DIR = BASE_DIR / "data" / "training_split"

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "validation"


# =========================================================
# SETTINGS
# =========================================================

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

VALIDATION_RATIO = 0.20

random.seed(42)


# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================

for class_name in CLASSES:

    (TRAIN_DIR / class_name).mkdir(
        parents=True,
        exist_ok=True
    )

    (VAL_DIR / class_name).mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# IMAGE EXTENSIONS
# =========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# =========================================================
# SPLIT EACH CLASS
# =========================================================

print("========================================")
print("EXISTING DATASET TRAIN / VALIDATION SPLIT")
print("========================================")

total_train = 0
total_val = 0


for class_name in CLASSES:

    source_class = SOURCE_DIR / class_name

    images = [
        file
        for file in source_class.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    val_count = int(
        len(images) * VALIDATION_RATIO
    )

    val_images = images[:val_count]
    train_images = images[val_count:]

    print(f"\nClass: {class_name}")
    print("Original:", len(images))
    print("Training:", len(train_images))
    print("Validation:", len(val_images))

    # -----------------------------------------------------
    # COPY TRAINING IMAGES
    # -----------------------------------------------------

    for image in train_images:

        destination = TRAIN_DIR / class_name / image.name

        shutil.copy2(
            image,
            destination
        )

    # -----------------------------------------------------
    # COPY VALIDATION IMAGES
    # -----------------------------------------------------

    for image in val_images:

        destination = VAL_DIR / class_name / image.name

        shutil.copy2(
            image,
            destination
        )

    total_train += len(train_images)
    total_val += len(val_images)


# =========================================================
# FINAL RESULT
# =========================================================

print("\n========================================")
print("SPLIT COMPLETE")
print("========================================")

print("Total training images:", total_train)
print("Total validation images:", total_val)

print("\nTraining folder:")
print(TRAIN_DIR)

print("\nValidation folder:")
print(VAL_DIR)

print("\nIMPORTANT:")
print("Original data/Training was NOT changed.")
print("Original data/Testing was NOT changed.")