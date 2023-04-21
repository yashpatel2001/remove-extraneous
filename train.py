import os
import numpy as np
from nilearn import image
from sklearn.linear_model import LinearRegression

# Define the folder paths for the training and test NIfTI files and the mask file
train_folder = '/path/to/train/folder'
test_folder = '/path/to/test/folder'
mask_file = '/path/to/mask/file.nii.gz'

# Load the mask file
mask = image.load_img(mask_file)

# Initialize empty arrays for the predictor variables (X_train and X_test) and target variables (y_train and y_test)
X_train = np.empty((0, np.prod(mask.shape)))
y_train = np.empty((0,))
X_test = np.empty((0, np.prod(mask.shape)))
y_test = np.empty((0,))

# Loop through the training NIfTI files in the folder and apply the mask to each one
for file in os.listdir(train_folder):
    if file.endswith('.nii.gz'):
        # Load the NIfTI file
        train_file = os.path.join(train_folder, file)
        img = image.load_img(train_file)

        # Apply the mask to the NIfTI file
        masked_data = image.math_img('img * mask', img=img, mask=mask)

        # Get the data from the masked NIfTI file
        masked_data_array = masked_data.get_fdata()

        # Reshape the data to be 2D
        n_voxels = np.prod(masked_data_array.shape[:-1])
        masked_data_array = np.reshape(masked_data_array, (n_voxels, masked_data_array.shape[-1]))

        # Add the data to the X_train and y_train arrays
        X_train = np.vstack((X_train, masked_data_array[:, :-1]))
        y_train = np.concatenate((y_train, masked_data_array[:, -1]))

# Loop through the test NIfTI files in the folder and apply the mask to each one
for file in os.listdir(test_folder):
    if file.endswith('.nii.gz'):
        # Load the NIfTI file
        test_file = os.path.join(test_folder, file)
        img = image.load_img(test_file)

        # Apply the mask to the NIfTI file
        masked_data = image.math_img('img * mask', img=img, mask=mask)

        # Get the data from the masked NIfTI file
        masked_data_array = masked_data.get_fdata()

        # Reshape the data to be 2D
        n_voxels = np.prod(masked_data_array.shape[:-1])
        masked_data_array = np.reshape(masked_data_array, (n_voxels, masked_data_array.shape[-1]))

        # Add the data to the X_test and y_test arrays
        X_test = np.vstack((X_test, masked_data_array[:, :-1]))
        y_test = np.concatenate((y_test, masked_data_array[:, -1]))

# Train the OLS regression model
model = LinearRegression().fit(X_train, y_train)

# Get the R-squared value for the model using the training data
r2_train = model.score(X_train, y_train)

# Get the R-squared value for the model using the test data
r2_test = model.score(X_test, y_test)

# Get the coefficients and intercept for the model
coefficients = model.coef_
intercept = model.intercept_

# Print the results
print("Training R-squared:", r2_train)
