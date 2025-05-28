import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.express as px
import PIL
from PIL import Image
from preprocessing import unsharp_masking
from postprocessing import process_traffic_sign
from conversions import rgb_to_hsv
from edge_detection import canny
from feature_extraction import extract_features
from geometric_norm import compute_homography, warp_perspective, resize_image_numpy, rotate_image_numpy
from tqdm import tqdm


train_df = pd.read_csv("combined_df.csv")
#test_df = pd.read_csv('test_data.csv')
#meta_df = pd.read_csv('meta_data.csv')

img_shape, img_color = None, None

predicted_classes = []

shapes = {1: "Circle", 0: "Triangle", 2: "Diamond", 3: "Octagon"}

def prediction(color, shape):
    cls = 0
    if color == 0: # Red
        if shape == 0:
            cls = 27
        elif shape == 1:
            cls = 17
        elif shape == 3:
            cls = 14
        elif shape == 4:
            cls = 13
    elif color == 1: #Blue
        cls = 40
    elif color == 3: #Yellow
        cls = 40
    elif shape == 2:
        cls = 40
    return cls

for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing images"):
# for i in range(1):
 
    # row = train_df.iloc[7144] #class 40
    # row = train_df.iloc[1880] #class 12
    # row = train_df.iloc[2126] #class 14
    # row = train_df.iloc[2931] #class 13

    image_folder = "archive/"
    image_path = os.path.join(image_folder, row["Path"])
    img = np.array(Image.open(image_path).convert('RGB'))
    
    if img is None:
        print(f"Could not read image {row['image_path']}")
        continue

    unsharp = unsharp_masking(img, sigma=1.0, strength=1.0)
    hsv = process_traffic_sign(img)  # Not used later unless needed elsewhere
    edges = canny(unsharp)
    
    # Convert edges to binary mask in range [0, 1]
    binary_mask = (edges / 255).astype(np.uint8)
    
    rotated = rotate_image_numpy(unsharp, angle_deg=0)
    resized = resize_image_numpy(rotated, new_size=(200, 200))

    x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
    src_pts = np.array([
        [x1, y1],  # Top-left
        [x2, y1],  # Top-right
        [x2, y2],  # Bottom-right
        [x1, y2]   # Bottom-left
    ], dtype=np.float32)

    dest_pts = np.array([
        [0, 0],
        [199, 0],
        [199, 199],
        [0, 199]
    ], dtype=np.float32)

    H = compute_homography(src_pts, dest_pts)
    normalized = warp_perspective(resized, H, output_size=(200, 200))

    # Ensure binary_mask is the same size as normalized
    binary_mask_resized = cv2.resize(binary_mask, (200, 200), interpolation=cv2.INTER_NEAREST)

    # Extract features
    features = extract_features(normalized, binary_mask_resized)
    # print(features)
    if features['corner_count'] < 2:
        shape = 1 # Circle
    elif features['corner_count'] < 4: 
        shape = 0 # Triangle
    elif features['corner_count'] == 4:
        shape = 2 #Diamond
    elif features['corner_count'] > 4:
        shape = 3 #Octagon

    predicted_classes.append(prediction(hsv, shape))


(train_df['predicted_class'] == predicted_classes).sum() / len(train_df) * 100
print("Predicted classes: ", predicted_classes)   
pd.to_csv("predictions.csv", index=False)


