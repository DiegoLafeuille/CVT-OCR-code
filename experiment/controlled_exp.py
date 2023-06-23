import os
import random
import pandas as pd


# Import ground truths into dataframe
gt_df = pd.read_csv("experiment/ground_truths.csv", dtype=str)

# Get the list of image names in the folder
image_names = os.listdir("experiment/slides")

# Shuffle the image names in a random order
random.shuffle(image_names)

# Print the shuffled image names
for image_name in image_names:
    img_code = image_name[:-4]
    gt_row = gt_df.loc[gt_df["Code"] == img_code]
    ground_truth = gt_row['Ground truth'].values[0]
    print(f"Code '{img_code}' -> {ground_truth}")