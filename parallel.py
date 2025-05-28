import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from preprocessing import unsharp_masking
from postprocessing import process_traffic_sign
from conversions import rgb_to_hsv
from edge_detection import canny
from feature_extraction import extract_features
from geometric_norm import compute_homography, warp_perspective, resize_image_numpy, rotate_image_numpy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

train_df = pd.read_csv("combined_df.csv")
img_shape, img_color = None, None
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
    elif color == 1: #Blue
        cls = 40
        
    elif color == 2: #Yellow
        cls = 12
    else: #white
        cls = 6
    return cls

def process_row(row):
    try:
        image_folder = "archive/"
        image_path = os.path.join(image_folder, row["Path"])
        img = np.array(Image.open(image_path).convert('RGB'))
        if img is None:
            return None

        unsharp = unsharp_masking(img, sigma=1.0, strength=1.0)
        hsv = process_traffic_sign(img)
        edges = canny(unsharp)
        binary_mask = (edges / 255).astype(np.uint8)
        rotated = rotate_image_numpy(unsharp, angle_deg=0)
        resized = resize_image_numpy(rotated, new_size=(200, 200))

        x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
        src_pts = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32)
        dest_pts = np.array([
            [0, 0], [199, 0], [199, 199], [0, 199]
        ], dtype=np.float32)
        H = compute_homography(src_pts, dest_pts)
        normalized = warp_perspective(resized, H, output_size=(200, 200))
        binary_mask_resized = cv2.resize(binary_mask, (200, 200), interpolation=cv2.INTER_NEAREST)
        features = extract_features(normalized, binary_mask_resized)

        if features['corner_count'] < 2:
            shape = 1 # Circle
        elif features['corner_count'] < 4: 
            shape = 0 # Triangle
        elif features['corner_count'] == 4:
            shape = 2 #Diamond
        elif features['corner_count'] > 4:
            shape = 3 #Octagon
        else:
            shape = 1 # fallback

        pred_class = prediction(hsv, shape)
        return pred_class
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

predicted_classes = []
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_row, row) for _, row in train_df.iterrows()]
    for f in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
        result = f.result()
        predicted_classes.append(result)

# Ensure predicted_classes aligns with train_df
train_df['predicted_class'] = predicted_classes

print("Predicted classes: ", predicted_classes)
train_df.to_csv("predictions.csv", index=False)
