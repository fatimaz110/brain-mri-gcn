import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

case_path = Path(
    r"C:\Users\fatim\OneDrive\Documents\BraTS2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001"
)

flair_path = case_path / "BraTS20_Training_001_flair.nii"
seg_path = case_path / "BraTS20_Training_001_seg.nii"

# Load MRI
flair = nib.load(str(flair_path)).get_fdata()

# Load segmentation
seg = nib.load(str(seg_path)).get_fdata()

# Find slices that actually contain tumor
tumor_slices = np.where(np.any(seg > 0, axis=(0, 1)))[0]

print("Total slices:", flair.shape[2])
print("Tumor slices:", len(tumor_slices))

# Middle tumor-containing slice
slice_index = tumor_slices[len(tumor_slices) // 2]

print("Selected slice:", slice_index)

# Get slice
mri_slice = flair[:, :, slice_index]
seg_slice = seg[:, :, slice_index]

# Display
plt.figure(figsize=(8, 8))

plt.imshow(mri_slice.T, cmap="gray")
plt.imshow(
    np.ma.masked_where(seg_slice.T == 0, seg_slice.T),
    alpha=0.4
)

plt.title(f"BraTS20_Training_001 - Slice {slice_index}")
plt.axis("off")

plt.show()