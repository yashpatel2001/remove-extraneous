import nibabel as nib
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
import numpy as np 
t1_img = nib.load("742/5Hz/5HzfMri.nii")
t1_hdr = t1_img.header
print(t1_hdr)
print("\n")
print(t1_hdr.keys())
print("\n")
t1_data = t1_img.get_fdata()
print(t1_data)
print("\n")
print(t1_data.dtype)
print("\n")
print(np.min(t1_data))
print(np.max(t1_data))
print(t1_data.shape)
#plt.imshow((t1_data[:,:,20,50]),'gray')
#imshow for images show


#Split images into slices 

number_of_slices = 3
number_of_frames = 4

fig, ax = plt.subplots(number_of_frames, number_of_slices)
#-------------------------------------------------------------------------------
for slice in range(number_of_slices):
    for frame in range(number_of_frames):
        ax[frame, slice].imshow(t1_data[:,:,slice,frame],cmap='gray', interpolation=None)
        ax[frame, slice].set_title("layer {} / frame {}".format(slice, frame))
        ax[frame, slice].axis('off')

#plt.show()         

#Plot with NiLearn 
from nilearn import plotting, image
print(image.load_img("742/3Hz/3HzfMri.nii").shape)
first_rsn = image.index_img("742/3Hz/3HzfMri.nii", 0)
print(first_rsn.shape)
#plotting.plot_stat_map(first_rsn)
#plt.show()

#Gausian smoothing 
from nilearn import image

fwhm =0.05

brain_vol_smth = image.smooth_img(first_rsn, fwhm)
plotting.plot_img(brain_vol_smth, cmap='gray', cut_coords=(5, 10, 0))
plt.show()


