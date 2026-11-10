from pathlib import Path


BASE_DIR = Path(
    r"C:\Users\fatim\OneDrive\Documents\brain-mri-gcn"
)

TRAIN_DIR = BASE_DIR / "data" / "Training"
TEST_DIR = BASE_DIR / "data" / "Testing"

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


def count_images(folder):
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in extensions
    )


print("========================================")
print("EXISTING DATASET CHECK")
print("========================================")


print("\nTRAINING DATA")
print("----------------------------------------")

total_train = 0

for class_name in CLASSES:

    folder = TRAIN_DIR / class_name

    count = count_images(folder)

    total_train += count

    print(f"{class_name}: {count}")


print("Total training images:", total_train)


print("\nTESTING DATA")
print("----------------------------------------")

total_test = 0

for class_name in CLASSES:

    folder = TEST_DIR / class_name

    count = count_images(folder)

    total_test += count

    print(f"{class_name}: {count}")


print("Total testing images:", total_test)


print("\n========================================")
print("DATASET CHECK COMPLETE")
print("========================================")