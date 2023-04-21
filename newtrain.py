import numpy as np
import nibabel as nib
from nilearn import image
from skimage.filters import sobel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Define the file paths for the NIfTI files and the masks for different frequency ranges
test_files = ['743/20Hz/20HzfMRI.nii','742/20Hz/20HzfMRI.nii']
masks = ['743/20Hz/20HzMask.nii','742/20Hz/20HzMask.nii']

# Define the path for saving the masked data
output_dir = 'save_images'

# Initialize empty lists for the predictor variables (X) and target variables (y)
X = None
y = []

# Loop through the NIfTI files and apply the corresponding mask to each one
for i, test_file in enumerate(test_files):
    # Load the NIfTI file
    t1_img = image.load_img(test_file)
    t1_data = t1_img.get_fdata()

    # Load the mask file for the current frequency range
    mask_img = image.load_img(masks[i])
    mask_data = mask_img.get_fdata()

    # Apply Sobel edge detection to the mask
    #mask_data_edge = sobel(mask_data)
    #mask_data[mask_data_edge < 0.1] = 0

    # Tile the mask to have the same dimensions as the 4D image
    mask_data = np.tile(mask_data[..., np.newaxis], [1, 1, 1, t1_data.shape[-1]])

    masked_data = np.multiply(t1_data, mask_data)

    # Append the data to the X and y lists
    if X is None:
        X = [masked_data.reshape(-1, masked_data.shape[-1]).T]
    else:
        X.append(masked_data.reshape(-1, masked_data.shape[-1]).T)
    y.append(np.arange(masked_data.shape[-1]))

    # Save the masked data to a new NIfTI file
    masked_nii = nib.Nifti1Image(masked_data, t1_img.affine, t1_img.header)
    nib.save(masked_nii, output_dir + '/masked_' + str(i+1) + '.nii.gz')

# Convert the X and y lists to numpy arrays
X = np.concatenate(X)
y = np.concatenate(y)

# Reshape the X array to have a shape of (num_samples, num_features)
X = X.reshape((X.shape[0], -1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model on the training set
model = LinearRegression().fit(X_train, y_train)

# Get the R-squared value for the model using the test data
r2_test = model.score(X_test, y_test)

# Get the coefficients and intercept for the model

# Get the coefficients and intercept for the model
coefficients = model.coef_
intercept = model.intercept_

# Print the results
print("Test R-squared:", r2_test)
print("Coefficients:", coefficients)
print("Intercept:", intercept)