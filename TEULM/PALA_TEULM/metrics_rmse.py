import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import os



# Helper function to compute RMSE (used in visualize_rmse_heatmap)
def compute_rmse(img1, img2):
    squared_diff = np.square(img1 - img2)
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    return rmse


# Path to the base directory containing the results
base_path = './RESULT'

# RBFs and their corresponding directories
RBFs = {
    "original": ["ORI_1000"],
    # "DS": ["Result_100_DS", "Result_250_DS", "Result_500_DS"],
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
            score = compute_rmse(original_image, image)
            results[RBF]["TEULM"].append(score)

all_scores = []
# Print the results
for RBF, scores in results.items():
    print(f"{RBF}:")
    print(f"  TEULM RMSE scores: {scores['TEULM']}")
    all_scores.append(scores['TEULM'])
    
# Function to normalize the scores
def normalize(scores):
    min_score = 0
    max_score = np.max(scores)
    return (scores - min_score) / (max_score - min_score)


print(f"Normalized scores: {normalize(np.array(all_scores))}")

all_scores = normalize(np.array(all_scores))
# Create the heatmap
fig, ax = plt.subplots()
cax = ax.matshow(all_scores, cmap='YlOrRd')
fig.colorbar(cax)

# Set ticks and labels
ax.set_xticks(range(all_scores.shape[1]))
ax.set_yticks(range(all_scores.shape[0]))
ax.set_xticklabels(['100', '250', '500'])
ax.set_yticklabels(['RBF GA', 'RBF MQ', 'RBF IMQ'])

ax.xaxis.set_ticks_position('bottom')

ax.set_title("Normalized TEULM RMSE Scores Heatmap")
plt.show()