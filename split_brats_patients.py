from pathlib import Path
import pandas as pd
import random


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(
    r"C:\Users\fatim\OneDrive\Documents\brain-mri-gcn"
)

CSV_PATH = BASE_DIR / "data" / "brats_slices" / "metadata.csv"

OUTPUT_DIR = BASE_DIR / "data" / "brats_splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD METADATA
# =========================================================

df = pd.read_csv(CSV_PATH)

patients = sorted(df["patient_id"].unique())

print("========================================")
print("BraTS Patient-wise Split")
print("========================================")

print("Total patients:", len(patients))
print("Total slices:", len(df))


# =========================================================
# SHUFFLE PATIENTS
# =========================================================

random.seed(42)

random.shuffle(patients)


# =========================================================
# 70% TRAIN / 15% VALIDATION / 15% TEST
# =========================================================

total_patients = len(patients)

train_end = int(total_patients * 0.70)
val_end = train_end + int(total_patients * 0.15)

train_patients = patients[:train_end]
val_patients = patients[train_end:val_end]
test_patients = patients[val_end:]


# =========================================================
# CREATE DATAFRAMES
# =========================================================

train_df = df[df["patient_id"].isin(train_patients)].copy()

val_df = df[df["patient_id"].isin(val_patients)].copy()

test_df = df[df["patient_id"].isin(test_patients)].copy()


# =========================================================
# SAVE CSV FILES
# =========================================================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\nPatient split:")
print("----------------------------")

print("Training patients:", len(train_patients))
print("Validation patients:", len(val_patients))
print("Testing patients:", len(test_patients))


print("\nSlice split:")
print("----------------------------")

print("Training slices:", len(train_df))
print("Validation slices:", len(val_df))
print("Testing slices:", len(test_df))


# =========================================================
# VERIFY NO PATIENT OVERLAP
# =========================================================

train_set = set(train_patients)
val_set = set(val_patients)
test_set = set(test_patients)

print("\nPatient overlap check:")
print("----------------------------")

print(
    "Train ∩ Validation:",
    len(train_set & val_set)
)

print(
    "Train ∩ Test:",
    len(train_set & test_set)
)

print(
    "Validation ∩ Test:",
    len(val_set & test_set)
)


# =========================================================
# FINAL
# =========================================================

print("\n========================================")
print("SPLIT COMPLETE")
print("========================================")

print("Files created:")

print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "validation.csv")
print(OUTPUT_DIR / "test.csv")