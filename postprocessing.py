import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from PIL import Image
import os
from preprocessing import adaptive_median_filter, unsharp_masking
# Define a custom structuring element (e.g., cross-shaped)
cross_structuring_element = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])

def process_traffic_sign(image):
    """
    Process a single traffic sign image: preprocess, segment blue/red in HSV, post-process manually, and visualize.
    
    Args:
        image (numpy.ndarray): Input RGB image (H, W, 3) with values in [0, 255]
        skip_preprocessing (bool): If True, skip adaptive median filter and unsharp masking
    
    Returns:
        int: 0 if red segment is dominant, 1 if blue segment is dominant
    """
    processed_img = adaptive_median_filter(image)
    processed_img = unsharp_masking(processed_img, sigma=1.0, strength=1.0)
    
    # Manual RGB to HSV conversion
    def rgb_to_hsv_manual(rgb_img):
        rgb_img = rgb_img.astype(float) / 255.0
        r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
        hsv_img = np.zeros_like(rgb_img)
        v = np.max(rgb_img, axis=2)
        min_val = np.min(rgb_img, axis=2)
        delta = v - min_val
        s = np.zeros_like(v)
        mask = v != 0
        s[mask] = delta[mask] / v[mask]
        h = np.zeros_like(v)
        mask_delta = delta != 0
        mask_r = (v == r) & mask_delta
        h[mask_r] = 60 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 360
        mask_g = (v == g) & mask_delta
        h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)
        mask_b = (v == b) & mask_delta
        h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)
        hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2] = h, s, v
        return hsv_img
    
    hsv_img = rgb_to_hsv_manual(processed_img)
    
    # Sample HSV values from a small central region (likely blue)
    h, w = hsv_img.shape[:2]
    center_region = hsv_img[h//2-h//10:h//2+h//10, w//2-w//10:w//2+w//10]  # Central 10%
    hues = center_region[:, :, 0].flatten()
    sats = center_region[:, :, 1].flatten()
    vals = center_region[:, :, 2].flatten()
    # print("Center region HSV statistics:")
    # print(f"Hue range: {np.min(hues):.1f}–{np.max(hues):.1f}, mean: {np.mean(hues):.1f}")
    # print(f"Saturation range: {np.min(sats):.3f}–{np.max(sats):.3f}, mean: {np.mean(sats):.3f}")
    # print(f"Value range: {np.min(vals):.3f}–{np.max(vals):.3f}, mean: {np.mean(vals):.3f}")
    
#     # Visualize HSV channels
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     plt.title("Hue")
#     plt.imshow(hsv_img[:, :, 0], cmap='hsv')
#     plt.colorbar(label='Degrees')
#     # Draw rectangle for sampled region
#     rect = plt.Rectangle((w//2-w//10, h//2-h//10), 2*w//10, 2*h//10, linewidth=2, edgecolor='white', facecolor='none')
#     plt.gca().add_patch(rect)
#     plt.axis('off')
#     plt.subplot(1, 3, 2)
#     plt.title("Saturation")
#     plt.imshow(hsv_img[:, :, 1], cmap='gray')
#     plt.colorbar()
#     plt.axis('off')
#     plt.subplot(1, 3, 3)
#     plt.title("Value")
#     plt.imshow(hsv_img[:, :, 2], cmap='gray')
#     plt.colorbar()
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
    
    # Segment blue and red regions
    def segment_blue_red(hsv_img):
        h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
        # Blue: Hue [200–250], S ≥ 0.4, V ≥ 0.2
        blue_lower = np.array([200, 0.4, 0.2])
        blue_upper = np.array([250, 1.0, 1.0])
        # Red: Hue [0–40] or [330–360], S ≥ 0.3, V ≥ 0.2
        red_lower1 = np.array([0, 0.3, 0.2])
        red_upper1 = np.array([15, 1.0, 1.0])
        red_lower2 = np.array([330, 0.3, 0.2])
        red_upper2 = np.array([360, 1.0, 1.0])
        # Yellow: Hue [40–70], S ≥ 0.3, V ≥ 0.4
        yellow_lower = np.array([20, 0.2, 0.3])  # Lower hue for more flexibility, lower saturation
        yellow_upper = np.array([80, 1.0, 1.0])  # Broader hue range, high saturation and value

        yellow_mask = (h >= yellow_lower[0]) & (h <= yellow_upper[0]) & \
                  (s >= yellow_lower[1]) & (s <= yellow_upper[1]) & \
                  (v >= yellow_lower[2]) & (v <= yellow_upper[2])

        blue_mask = (h >= blue_lower[0]) & (h <= blue_upper[0]) & \
                    (s >= blue_lower[1]) & (s <= blue_upper[1]) & \
                    (v >= blue_lower[2]) & (v <= blue_upper[2])
        red_mask1 = (h >= red_lower1[0]) & (h <= red_upper1[0]) & \
                    (s >= red_lower1[1]) & (s <= red_upper1[1]) & \
                    (v >= red_lower1[2]) & (v <= red_upper1[2])
        red_mask2 = (h >= red_lower2[0]) & (h <= red_upper2[0]) & \
                    (s >= red_lower2[1]) & (s <= red_upper2[1]) & \
                    (v >= red_lower2[2]) & (v <= red_upper2[2])
        red_mask = red_mask1 | red_mask2
        return blue_mask, red_mask,yellow_mask
    
    blue_mask, red_mask,yellow_mask = segment_blue_red(hsv_img)
    
    # Visualize blue and red masks for debugging
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 3, 1)
#     plt.imshow(blue_mask, cmap='gray')
#     plt.title("Blue Mask (Pre)")
#     plt.axis('off')
#     plt.subplot(1, 3, 2)
#     plt.imshow(yellow_mask, cmap='gray')
#     plt.title("Yellow Mask (Pre)")
#     plt.axis('off')
#     plt.subplot(1, 3, 3)
#     plt.imshow(red_mask, cmap='gray')
#     plt.title("Red Mask (Pre)")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
    
    # Manual Post-Processing
    def manual_post_processing(mask, visualize_steps=False):
        binary_mask = mask.astype(np.uint8)
        h, w = binary_mask.shape

        def get_neighborhood(img, i, j, size=3):
            half = size // 2
            return img[max(0, i - half):min(h, i + half + 1), max(0, j - half):min(w, j + half + 1)]

        # Step 1: Erosion
        eroded_mask = np.zeros_like(binary_mask)
        for i in range(h):
            for j in range(w):
                if np.sum(get_neighborhood(binary_mask, i, j)) >= 5:
                    eroded_mask[i, j] = 1
#        if visualize_steps:
#             plt.imshow(eroded_mask, cmap='gray')
#             plt.title("Eroded Mask")
#             plt.show()

        # Step 2: Dilation using custom structuring element
        def custom_dilation(mask, structuring_element):
            h, w = mask.shape
            dilated_mask = np.zeros_like(mask)
            se_h, se_w = structuring_element.shape
            half_se_h, half_se_w = se_h // 2, se_w // 2

            for i in range(h):
                for j in range(w):
                    should_dilate = False
                    for ni in range(se_h):
                        for nj in range(se_w):
                            ni_img = i + ni - half_se_h
                            nj_img = j + nj - half_se_w
                            if 0 <= ni_img < h and 0 <= nj_img < w:
                                if mask[ni_img, nj_img] == 1 and structuring_element[ni, nj] == 1:
                                    should_dilate = True
                                    break
                        if should_dilate:
                            break
                    if should_dilate:
                        dilated_mask[i, j] = 1
            return dilated_mask

        dilated_mask = custom_dilation(eroded_mask, cross_structuring_element)
#       if visualize_steps:
#             plt.imshow(dilated_mask, cmap='gray')
#             plt.title("Dilated Mask")
#             plt.show()

        # Step 3: Opening
        opened_mask = np.zeros_like(dilated_mask)
        for i in range(h):
            for j in range(w):
                if np.sum(get_neighborhood(dilated_mask, i, j)) >= 5:
                    opened_mask[i, j] = 1
#         if visualize_steps:
#             plt.imshow(opened_mask, cmap='gray')
#             plt.title("Opened Mask")
#             plt.show()

        # Step 4: Remove small connected components
        labeled_array, num_features = label(opened_mask)
        areas = np.bincount(labeled_array.ravel())[1:]
        filtered_mask = np.zeros_like(opened_mask)
        for i in range(1, num_features + 1):
            if areas[i - 1] >= 5:
                filtered_mask[labeled_array == i] = 1
#         if visualize_steps:
#             plt.imshow(filtered_mask, cmap='gray')
#             plt.title("After Area Filtering")
#             plt.show()

        # Step 5: Hole filling
        filled_mask = np.zeros_like(filtered_mask)
        for i in range(h):
            for j in range(w):
                if np.sum(get_neighborhood(filtered_mask, i, j)) >= 3:
                    filled_mask[i, j] = 1
        for i in range(h):
            for j in range(w):
                if np.sum(get_neighborhood(filled_mask, i, j)) >= 5:
                    filled_mask[i, j] = 1
        
#         if visualize_steps:
#             plt.imshow(filled_mask, cmap='gray')
#             plt.title("After Hole Filling")
#             plt.show()
        return filled_mask
    
    blue_mask_post = manual_post_processing(blue_mask, visualize_steps=True)
    red_mask_post = manual_post_processing(red_mask, visualize_steps=True)
    yellow_mask_post = manual_post_processing(yellow_mask, visualize_steps=True)
    # Visualize post-processed masks
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 3, 1)
#     plt.imshow(blue_mask_post, cmap='gray')
#     plt.title("Blue Mask (Post)")
#     plt.axis('off')
#     plt.subplot(1, 3, 2)
#     plt.imshow(red_mask_post, cmap='gray')
#     plt.title("Red Mask (Post)")
#     plt.axis('off')
#     plt.subplot(1, 3, 3)
#     plt.imshow(yellow_mask_post, cmap='gray')
#     plt.title("Yellow Mask (Post)")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()

    # Determine dominant color by comparing post-processed mask areas
    blue_area = np.sum(blue_mask_post)
    red_area = np.sum(red_mask_post)
    yellow_area = np.sum(yellow_mask_post)
    # print(f"Blue area (post-processed): {blue_area}")
    # print(f"Red area (post-processed): {red_area}")
    # print(f"Yellow area (post-processed): {yellow_area}")

    # Otherwise determine dominant color based on area
    min_color_area = 500  # Tune this threshold if needed
    if blue_area < min_color_area and red_area < min_color_area and yellow_area < min_color_area:
        # print("Dominant color: White (color areas below threshold)")
        return 3  # White
    if blue_area > red_area and blue_area > yellow_area:
        dominant_color = 1  # Blue dominant
        # print(f"Dominant color: Blue")
    elif red_area > blue_area and red_area > yellow_area:
        dominant_color = 0  # Red dominant
        # print(f"Dominant color: Red")
    else:
        dominant_color = 2  # Yellow dominant (if yellow has the largest area)
        # print(f"Dominant color: Yellow")
    # Visualization
    segmented_blue = np.zeros_like(image)
    segmented_red = np.zeros_like(image)
    segmented_yellow = np.zeros_like(image)
    segmented_blue_post = np.zeros_like(image)
    segmented_red_post = np.zeros_like(image)
    segmented_yellow_post = np.zeros_like(image)

    # Ensure masks are boolean
    blue_mask = blue_mask.astype(bool)
    red_mask = red_mask.astype(bool)
    yellow_mask = yellow_mask.astype(bool)
    blue_mask_post = blue_mask_post.astype(bool)
    red_mask_post = red_mask_post.astype(bool)
    yellow_mask_post = yellow_mask_post.astype(bool)
    # Apply masks
    segmented_blue[blue_mask] = image[blue_mask]
    segmented_red[red_mask] = image[red_mask]
    segmented_yellow[yellow_mask] = image[yellow_mask]
    segmented_blue_post[blue_mask_post] = image[blue_mask_post]
    segmented_red_post[red_mask_post] = image[red_mask_post]
    segmented_yellow_post[yellow_mask_post] = image[yellow_mask_post]
    # Debugging: Check mask sums
#     print("Blue mask pre sum (white pixels):", np.sum(blue_mask))
#     print("Blue mask post sum (white pixels):", np.sum(blue_mask_post))
#     print("Red mask pre sum (white pixels):", np.sum(red_mask))
#     print("Red mask post sum (white pixels):", np.sum(red_mask_post))
#     print("Segmented blue post max value:", np.max(segmented_blue_post))
#     print("Segmented red post max value:", np.max(segmented_red_post))

    # Visualize all images in one figure
    # plt.figure(figsize=(15, 10))

    # Original image
    # plt.subplot(2, 4, 1)
    # plt.title("Original")
    # plt.imshow(image)
    # plt.axis('off')

    # Blue masks (pre and post)
#     plt.subplot(2, 4, 2)
#     plt.title("Blue (Pre)")
#     plt.imshow(segmented_blue)
#     plt.axis('off')

#     plt.subplot(2, 4, 3)
#     plt.title("Blue (Post)")
#     plt.imshow(segmented_blue_post)
#     plt.axis('off')

#     # Yellow masks (pre and post)
#     plt.subplot(2, 4, 4)
#     plt.title("Yellow (Pre)")
#     plt.imshow(segmented_yellow)
#     plt.axis('off')

#     plt.subplot(2, 4, 5)
#     plt.title("Yellow (Post)")
#     plt.imshow(segmented_yellow_post)
#     plt.axis('off')
    
#     # Red masks (pre and post)
#     plt.subplot(2, 4, 6)
#     plt.title("Red (Pre)")
#     plt.imshow(segmented_red)
#     plt.axis('off')
#     plt.subplot(2, 4, 7)
#     plt.title("Red (Post)")
#     plt.imshow(segmented_red_post)
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

    return dominant_color

# Load and process the image
# image_path = "C:/Users/user/Desktop/dip_project/archive/Train/6/00006_00011_00025.png"
# try:
#     img = np.array(Image.open(image_path).convert('RGB'))
# except Exception as e:
#     print(f"Error loading {image_path}: {e}")
#     exit()

# # Call the function
# result = process_traffic_sign(img)
# print(f"Processed: 00033_00000_00027 (Class 33)")
# print(f"Result (0 for red dominant, 1 for blue dominant,2 for yellow dominant, 3 for white dominant): {result}")