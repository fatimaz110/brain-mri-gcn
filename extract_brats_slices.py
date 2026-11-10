from pathlib import Path
import csv

import nibabel as nib
import numpy as np
from PIL import Image


# =========================================================
# PATHS
# =========================================================

BRATS_ROOT = Path(
    r"C:\Users\fatim\OneDrive\Documents\BraTS2020"
)

OUTPUT_ROOT = Path(
    r"C:\Users\fatim\OneDrive\Documents\brain-mri-gcn\data\brats_slices"
)

IMAGE_DIR = OUTPUT_ROOT / "images"
MASK_DIR = OUTPUT_ROOT / "masks"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = (128, 128)

# Very tiny tumor regions are ignored
MIN_TUMOR_PIXELS = 20


# =========================================================
# FIND ALL FLAIR FILES
# =========================================================

flair_files = list(BRATS_ROOT.rglob("*_flair.nii")) + \
              list(BRATS_ROOT.rglob("*_flair.nii.gz"))

print("BraTS FLAIR files found:", len(flair_files))

if len(flair_files) == 0:
    print("ERROR: No FLAIR files found.")
    raise SystemExit


# =========================================================
# PROCESS PATIENTS
# =========================================================

metadata = []

total_saved = 0

for patient_number, flair_path in enumerate(flair_files, start=1):

    print("\n========================================")
    print(f"Patient {patient_number}/{len(flair_files)}")
    print(flair_path.parent.name)
    print("========================================")

    patient_folder = flair_path.parent

    patient_id = flair_path.name.replace("_flair.nii.gz", "")
    patient_id = patient_id.replace("_flair.nii", "")

    # Find segmentation file
    seg_path = patient_folder / f"{patient_id}_seg.nii"

    if not seg_path.exists():
        seg_path = patient_folder / f"{patient_id}_seg.nii.gz"

    if not seg_path.exists():
        print("Segmentation file not found. Skipping.")
        continue

    # -----------------------------------------------------
    # LOAD MRI + SEGMENTATION
    # -----------------------------------------------------

    flair = nib.load(str(flair_path)).get_fdata()
    seg = nib.load(str(seg_path)).get_fdata()

    print("MRI shape:", flair.shape)
    print("SEG shape:", seg.shape)

    # -----------------------------------------------------
    # FIND TUMOR SLICES
    # -----------------------------------------------------

    tumor_slices = np.where(
        np.any(seg > 0, axis=(0, 1))
    )[0]

    print("Tumor-containing slices:", len(tumor_slices))

    # -----------------------------------------------------
    # EXTRACT EACH TUMOR SLICE
    # -----------------------------------------------------

    for slice_index in tumor_slices:

        mri_slice = flair[:, :, slice_index]
        mask_slice = seg[:, :, slice_index]

        # Tumor mask
        tumor_mask = mask_slice > 0

        tumor_pixels = int(np.sum(tumor_mask))

        if tumor_pixels < MIN_TUMOR_PIXELS:
            continue

        # -------------------------------------------------
        # NORMALIZE MRI
        # -------------------------------------------------

        nonzero = mri_slice[mri_slice > 0]

        if len(nonzero) == 0:
            continue

        low = np.percentile(nonzero, 1)
        high = np.percentile(nonzero, 99)

        if high <= low:
            continue

        normalized = np.clip(
            (mri_slice - low) / (high - low),
            0,
            1
        )

        image_uint8 = (normalized * 255).astype(np.uint8)

        # -------------------------------------------------
        # RESIZE IMAGE
        # -------------------------------------------------

        image = Image.fromarray(image_uint8)

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )

        # Convert grayscale to RGB
        image = image.convert("RGB")

        # -------------------------------------------------
        # RESIZE MASK
        # -------------------------------------------------

        mask_uint8 = tumor_mask.astype(np.uint8) * 255

        mask_image = Image.fromarray(mask_uint8)

        mask_image = mask_image.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST
        )

        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        filename = f"{patient_id}_slice_{slice_index:03d}.png"

        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / filename

        image.save(image_path)
        mask_image.save(mask_path)

        # -------------------------------------------------
        # METADATA
        # -------------------------------------------------

        metadata.append([
            patient_id,
            slice_index,
            tumor_pixels,
            filename
        ])

        total_saved += 1


# =========================================================
# SAVE METADATA CSV
# =========================================================

csv_path = OUTPUT_ROOT / "metadata.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "patient_id",
        "slice_index",
        "original_tumor_pixels",
        "filename"
    ])

    writer.writerows(metadata)


# =========================================================
# FINAL RESULT
# =========================================================

print("\n========================================")
print("EXTRACTION COMPLETE")
print("========================================")

print("Total tumor slices saved:", total_saved)
print("Images folder:", IMAGE_DIR)
print("Masks folder:", MASK_DIR)
print("Metadata:", csv_path)