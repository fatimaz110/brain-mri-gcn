from pathlib import Path
import pandas as pd

CSV_PATH = Path(
    r"C:\Users\fatim\OneDrive\Documents\brain-mri-gcn\data\brats_slices\metadata.csv"
)

df = pd.read_csv(CSV_PATH)

print("\n==============================")
print("BraTS Extraction Check")
print("==============================")

print("Total extracted slices:", len(df))

print("Unique patients:", df["patient_id"].nunique())

print("\nSlices per patient:")
print(df.groupby("patient_id").size().describe())

print("\nFirst 10 patients:")
print(df["patient_id"].unique()[:10])

print("\nMaximum slices from one patient:")
print(df.groupby("patient_id").size().max())

print("\nMinimum slices from one patient:")
print(df.groupby("patient_id").size().min())