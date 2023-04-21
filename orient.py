
import nibabel as nib
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
import numpy as np 
t1_img = nib.load("742/3Hz/3HzfMri.nii")

t1_hdr = t1_img.header
print(t1_hdr)
print("\n")
print(t1_hdr.keys())
print("\n")
t1_data = t1_img.get_fdata()
mask = nib.load('742/3Hz/3HzMask.nii')
mask_data = mask.get_fdata()
print("Masked Data Dimensions")
print(mask_data.shape)
print("\n")
print("fMRI Data Dimensions")
print(t1_data.shape)
mask_data = np.tile(mask_data[..., np.newaxis], [1, 1, 1, t1_data.shape[-1]])
masked_data = np.multiply(t1_data, mask_data)
masked_nii = nib.Nifti1Image(masked_data,t1_img.affine, t1_img.header)
#nib.save(masked_nii, 'output.nii')
print(t1_data)
print("\n")
print(t1_data.dtype)
print("\n")
#print(np.min(t1_data))
#print(np.max(t1_data))
#print(t1_data.shape)
output_with_mask=nib.load("save_images/masked_1.nii.gz")
print(output_with_mask.shape)
data = output_with_mask.get_fdata()
print(data.shape)
# Take the average across the 4th dimension to create a 3D volume
avg_data = np.mean(data, axis=3)

# Display the 3D volume as a 2D image
plt.imshow(avg_data[:, :, 10], cmap='gray')
plt.show()


