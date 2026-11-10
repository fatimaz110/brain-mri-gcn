from pathlib import Path
import nibabel as nib
import numpy as np

case_path = Path(
    r"C:\Users\fatim\OneDrive\Documents\BraTS2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001"
)

files = {
    "FLAIR": case_path / "BraTS20_Training_001_flair.nii",
    "T1": case_path / "BraTS20_Training_001_t1.nii",
    "T1ce": case_path / "BraTS20_Training_001_t1ce.nii",
    "T2": case_path / "BraTS20_Training_001_t2.nii",
    "SEG": case_path / "BraTS20_Training_001_seg.nii",
}

for name, file_path in files.items():

    print("\n==============================")
    print(name)
    print("==============================")

    img = nib.load(str(file_path))
    data = img.get_fdata()

    print("Shape:", data.shape)
    print("Min:", np.min(data))
    print("Max:", np.max(data))

    if name == "SEG":
        print("Segmentation labels:", np.unique(data))