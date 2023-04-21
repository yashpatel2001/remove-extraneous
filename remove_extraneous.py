import numpy as np
from nilearn import image, masking
from sklearn.linear_model import LinearRegression

# Define the file paths for the NIfTI files and the masks for different frequency ranges
train_files = ['/path/to/train/nifti/file1.nii.gz', '/path/to/train/nifti/file2.nii.gz', ...]
test_files = ['/path/to/test/nifti/file1.nii.gz', '/path/to/test/nifti/file2.nii.gz', ...]
masks = ['/path/to/mask/frequency1.nii.gz', '/path/to/mask/frequency2.nii.gz', ...]

# Initialize empty arrays for the predictor variables (X_train and X_test) and target variables (y_train and y_test)
X_train = np.empty((0, 0))
y_train = np.empty((0,))
X_test = np.empty((0, 0))
y_test = np.empty((0,))

# Loop through the training NIfTI files and apply the corresponding mask to each one
for i, train_file in enumerate(train_files):
    # Load the NIfTI file
    img = image.load_img(train_file)

    # Load the mask file for the current frequency range
    mask = image.load_img(masks[i])

    # Loop through each time index in the 4D image and apply the mask to the corresponding 3D image
    for t in range(img.shape[-1]):
        # Extract the 3D image corresponding to the current time index
        img_3d = image.index_img(img, t)

        # Apply the mask to the 3D image
        masked_data = masking.apply_mask(img_3d, mask)
        masked_file = f"new_training/masked_{i}_t{t}.nii.gz"
        image.new_img_like(img_3d, masked_data).to_filename(masked_file)
        # Add the data to the X_train and y_train arrays
        X_train = np.vstack((X_train, masked_data))
        y_train = np.concatenate((y_train, np.array([t])))

# Loop through the test NIfTI files and apply the corresponding mask to each one
for i, test_file in enumerate(test_files):
    # Load the NIfTI file
    img = image.load_img(test_file)

    # Load the mask file for the current frequency range
    mask = image.load_img(masks[i])

    # Loop through each time index in the 4D image and apply the mask to the corresponding 3D image
    for t in range(img.shape[-1]):
        # Extract the 3D image corresponding to the current time index
        img_3d = image.index_img(img, t)

        # Apply the mask to the 3D image
        masked_data = masking.apply_mask(img_3d, mask)
        masked_file = f"new_test/masked_{i}_t{t}.nii.gz"
        image.new_img_like(img_3d, masked_data).to_filename(masked_file)
        # Add the data to the X_test and y_test arrays
        X_test = np.vstack((X_test, masked_data))
        y_test = np.concatenate((y_test, np.array([t])))

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
print("Test R-squared:", r2_test)
print("Coefficients:", coefficients)
print("Intercept:", intercept)
