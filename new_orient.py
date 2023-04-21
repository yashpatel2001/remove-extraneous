import nibabel as nib
import nibabel.processing
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
import numpy as np 
t1_img = nib.load("742/3Hz/3HzfMri.nii")
t1_hdr = t1_img.header
t1_data = t1_img.get_fdata()
mask = nib.load('742/Anatomical/AnatomicalMask.nii')
# Find the indices of the non-zero elements in the mask
nonzero_indices = np.nonzero(mask)
time_point =5
img_3d = t1_data[:, :, :, time_point]
print(img_3d.shape)
print(mask.shape)