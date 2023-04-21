import os
import numpy as np
import nibabel as nib
from sklearn.linear_model import LinearRegression

# Define the paths to the training and test folders
train_folder = '/path/to/training/folder'
test_folder = '/path/to/test/folder'

# Load the mask file
mask_file = nib.load('/path/to/mask/file.nii.gz')
mask_data = mask_file.get_fdata()

# Reshape the mask to match the dimensions of the NIfTI files
mask_data_4d = np.repeat(np.expand_dims(mask_data, axis=3), len(os.listdir(train_folder)) + len(os.listdir(test_folder)), axis=3)

# Apply the mask to the NIfTI data and reshape it into a 2D array
def apply_mask(nifti_file):
    nifti_data = nifti_file.get_fdata()
    masked_data = nifti_data * mask_data_4d
    return masked_data.reshape(-1, masked_data.shape[-1])

# Load the training data
X_train_list = []
y_train_list = []
for filename in os.listdir(train_folder):
    nifti_file = nib.load(os.path.join(train_folder, filename))
    X_train_list.append(apply_mask(nifti_file))
    y_train_list.append(np.loadtxt(os.path.join(train_folder, filename[:-7] + '_label.txt')))

# Concatenate the training data into a single array
X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

# Train an OLS regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Load the test data
X_test_list = []
for filename in os.listdir(test_folder):
    nifti_file = nib.load(os.path.join(test_folder, filename))
    X_test_list.append(apply_mask(nifti_file))

# Concatenate the test data into a single array
X_test = np.concatenate(X_test_list, axis=0)

# Apply the model to the test data
predicted_data_list = []
for i in range(X_test.shape[0]):
    predicted_data = model.predict(X_test[i:i+1, :])
    predicted_data_list.append(predicted_data.reshape(mask_data.shape))
predicted_data_4d = np.stack(predicted_data_list, axis=3)

# Save the predicted data as a new NIfTI file for each test file
for i, filename in enumerate(os.listdir(test_folder)):
    test_nifti = nib.load(os.path.join(test_folder, filename))
    predicted_nifti = nib.Nifti1Image(predicted_data_4d[:, :, :, i], test_nifti.affine, test_nifti.header)
    nib.save(predicted_nifti, os.path.join(test_folder, filename[:-7] + '_predicted.nii.gz'))
