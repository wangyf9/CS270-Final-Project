import numpy as np
import tifffile as tiff
import os

def dice_score(image1, image2):
    """
    Compute the DICE score between two binary images.

    Parameters:
    image1 (numpy.ndarray): First binary image.
    image2 (numpy.ndarray): Second binary image.

    Returns:
    float: DICE score.
    """
    intersection = np.sum(image1 & image2)
    union = np.sum(image1) + np.sum(image2)
    dice = (2 * intersection) / union
    return dice



# Path to the base directory containing the results
base_path = './RESULT'

# RBFs and their corresponding directories
RBFs = {
    "original": ["ORI_1000"],
    "DS": ["Result_100_DS", "Result_250_DS", "Result_500_DS"],
    "RBF GA": ["Result_100_GA(zero)", "Result_250_GA(zero)", "Result_500_GA(zero)"],
    "RBF MQ": ["Result_100_MQ", "Result_250_MQ", "Result_500_MQ"],
    "RBF IMQ": ["Result_100_IMQ", "Result_250_IMQ", "Result_500_IMQ"]
}

# Function to read TIFF images
def read_tif_image(path):
    return tiff.imread(path)

# Read the original image from ORI_1000
original_image_path = os.path.join(base_path, RBFs["original"][0])
original_image = None
for file in os.listdir(original_image_path):
    if file.endswith('.tif'):
        original_image = read_tif_image(os.path.join(original_image_path, file))
        break

# Initialize result dictionary
results = {RBF: {"TEULM": []} for RBF in RBFs if RBF != "original"}

# Compute DICE scores for each RBF against the original image
for RBF, folders in RBFs.items():
    if RBF == "original":
        continue

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        image = None

        # Find the TIFF files in the directory
        for file in os.listdir(folder_path):
            if file.endswith('.tif'):
                image = read_tif_image(os.path.join(folder_path, file))
                break
        
        if image is not None:
            # Compute the DICE score against the original image
            score = dice_score(original_image, image)
            results[RBF]["TEULM"].append(score)

# Print the results
for RBF, scores in results.items():
    print(f"{RBF}:")
    print(f"  TEULM DICE scores: {scores['TEULM']}")