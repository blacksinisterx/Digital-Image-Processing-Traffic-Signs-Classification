# Updated [dataset.py] with new class combinations and resolved merge conflicts
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.express as px
import PIL
from PIL import Image

# Define the path to the Meta folder
meta_path = "E:/FAST'22BS-AI/Semester6/Digital Image Processing (DIP)/DIP_Project/archive/Meta"
selected_classes = [ 14, 12, 17, 27, 6, 40]

# Number of rows and columns for the grid
rows, cols = 2, 4
fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
axes = axes.flatten()

# Iterate through the selected classes
for idx, class_id in enumerate(selected_classes):
    image_filename = f'{class_id}.png'
    image_path = os.path.join(meta_path, image_filename)

    # Read the image using OpenCV
    image = cv2.imread(image_path)

    if image is not None:
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image in the grid
        axes[idx].imshow(image_rgb)
        axes[idx].set_title(f'Class {class_id}')
        axes[idx].axis('off')
    else:
        axes[idx].set_title(f'Class {class_id} not found')
        axes[idx].axis('off')

# Hide any unused subplots
for ax in axes[len(selected_classes):]:
    ax.axis('off')

plt.tight_layout()
plt.show()


# Define the path to your dataset
dataset_path = "archive/"
train_path = os.path.join(dataset_path, "test")

# Read the CSV files

train_df = pd.read_csv(os.path.join( "E:/FAST'22BS-AI/Semester6/Digital Image Processing (DIP)/DIP_Project/archive/Test.csv"))

# Define the selected class IDs (based on our previous discussion)
selected_classes = [14, 12, 17, 27, 6, 40]

# Filter the train DataFrame to include only the selected classes
filtered_train_df = train_df[train_df["ClassId"].isin(selected_classes)]

# Create a dictionary to store the loaded images and their class IDs
loaded_images = {}

# Load images for the selected classes
for index, row in filtered_train_df.iterrows():
    image_path = os.path.join(train_path, row["Path"])
    class_id = row["ClassId"]
    
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Store the image and class ID in the dictionary
    if class_id not in loaded_images:
        loaded_images[class_id] = []
    loaded_images[class_id].append(image)

print(train_df.head())

# Create separate DataFrames for the 8 selected classes
train_class_12_df = train_df[train_df['ClassId'] == 12]
# train_class_13_df = train_df[train_df['ClassId'] == 13]
train_class_14_df = train_df[train_df['ClassId'] == 14]
train_class_6_df = train_df[train_df['ClassId'] == 6]
train_class_17_df = train_df[train_df['ClassId'] == 17]
train_class_27_df = train_df[train_df['ClassId'] == 27]
train_class_40_df = train_df[train_df['ClassId'] == 40]

# Example: Display the first few rows of one of the DataFrames
print('Class 0 DataFrame:')
train_class_12_df.tail()

# Create separate DataFrames for the 8 selected classes
# Combine all class-specific DataFrames into one DataFrame
combined_df = pd.concat([
    train_df[train_df['ClassId'] == 12],
    train_df[train_df['ClassId'] == 14],
    # train_df[train_df['ClassId'] == 13],
    train_df[train_df['ClassId'] == 17],
    train_df[train_df['ClassId'] == 6],
    train_df[train_df['ClassId'] == 27],
    train_df[train_df['ClassId'] == 40]
], ignore_index=True)

# Example: Display the first few rows of the combined DataFrame

combined_df.to_csv('testdf.csv', index=False)