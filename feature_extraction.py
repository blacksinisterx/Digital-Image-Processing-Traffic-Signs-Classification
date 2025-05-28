import numpy as np


def harris_corner_detection_numpy(image, window_size=3, k=0.04, threshold=0.01):
  
    # Convert to grayscale if the image is in RGB
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # Compute gradients using Sobel operator
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Ix = convolve2d(gray, sobel_x)
    Iy = convolve2d(gray, sobel_y)

    # Compute products of derivatives
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # Apply Gaussian filter to smooth the products
    gaussian_kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    Sxx = convolve2d(Ixx, gaussian_kernel)
    Syy = convolve2d(Iyy, gaussian_kernel)
    Sxy = convolve2d(Ixy, gaussian_kernel)

    # Compute Harris response
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M ** 2)

    # Threshold the response
    max_response = np.max(R)
    corners = R > threshold * max_response

    return corners

def convolve2d(image, kernel):
 
    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
    output = np.zeros_like(image, dtype=np.float32)
    padded_image = np.pad(image, ((kernel.shape[0] // 2, ), (kernel.shape[1] // 2, )), mode='constant')

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            output[i, j] = np.sum(region * kernel)

    return output

def calculate_circularity(binary_mask):
    # Calculate the area (number of pixels in the object)
    area = np.sum(binary_mask)

    # Calculate the perimeter using a simple 8-connected neighborhood
    padded_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    perimeter = 0
    for i in range(1, padded_mask.shape[0] - 1):
        for j in range(1, padded_mask.shape[1] - 1):
            if padded_mask[i, j] == 1:
                # Check the 8 neighbors
                neighbors = padded_mask[i-1:i+2, j-1:j+2]
                perimeter += 8 - np.sum(neighbors)

    # Calculate circularity
    if perimeter == 0:  # Avoid division by zero
        return 0
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity

def calculate_aspect_ratio(binary_mask):
    """
    Calculate the aspect ratio (width/height) of the bounding box of an object in a binary mask.

    Parameters:
    - binary_mask: A binary image (numpy array) where the object is represented by 1s and the background by 0s.

    Returns:
    - aspect_ratio: The aspect ratio (float).
    """
    # Find the coordinates of the non-zero pixels
    rows, cols = np.where(binary_mask == 1)

    # Calculate the bounding box dimensions
    width = cols.max() - cols.min() + 1
    height = rows.max() - rows.min() + 1

    # Calculate aspect ratio
    aspect_ratio = width / height
    return aspect_ratio

def calculate_extent(binary_mask):
    """
    Calculate the extent (ratio of region area to bounding box area) of an object in a binary mask.

    Parameters:
    - binary_mask: A binary image (numpy array) where the object is represented by 1s and the background by 0s.

    Returns:
    - extent: The extent value (float).
    """
    # Calculate the area of the object
    area = np.sum(binary_mask)

    # Find the coordinates of the non-zero pixels
    rows, cols = np.where(binary_mask == 1)

    # Calculate the bounding box dimensions
    width = cols.max() - cols.min() + 1
    height = rows.max() - rows.min() + 1

    # Calculate the bounding box area
    bounding_box_area = width * height

    # Calculate extent
    extent = area / bounding_box_area
    return extent

def calculate_average_hue(image, binary_mask):
    """
    Calculate the average hue of an object in an image using a binary mask.

    Parameters:
    - image: An RGB image (numpy array).
    - binary_mask: A binary image (numpy array) where the object is represented by 1s and the background by 0s.

    Returns:
    - average_hue: The average hue value (float).
    """
    # Convert the image to HSV
    hsv_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j] / 255.0
            c_max = max(r, g, b)
            c_min = min(r, g, b)
            delta = c_max - c_min

            # Hue calculation
            if delta == 0:
                h = 0
            elif c_max == r:
                h = (60 * ((g - b) / delta) + 360) % 360
            elif c_max == g:
                h = (60 * ((b - r) / delta) + 120) % 360
            else:
                h = (60 * ((r - g) / delta) + 240) % 360

            hsv_image[i, j, 0] = h

def extract_features(image, binary_mask):
    """
    Extract features from an image and its binary mask.

    Parameters:
    - image: An RGB image (numpy array of shape (height, width, 3)).
    - binary_mask: A binary image (numpy array of shape (height, width)) where the object is 1s and background is 0s.

    Returns:
    - features: A dictionary containing extracted features.
    """
    # Calculate aspect ratio
    rows, cols = np.where(binary_mask == 1)
    if len(rows) == 0 or len(cols) == 0:  # Handle empty mask
        return {'color': 'unknown', 'corner_count': 0, 'circularity': 0, 'aspect_ratio': 0, 'extent': 0, 'avg_hue': 0}
    width = cols.max() - cols.min() + 1
    height = rows.max() - rows.min() + 1
    aspect_ratio = width / height if height != 0 else 0

    # Calculate extent
    area = np.sum(binary_mask)
    bounding_box_area = width * height
    extent = area / bounding_box_area if bounding_box_area != 0 else 0

    # Calculate circularity
    padded_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    perimeter = 0
    for i in range(1, padded_mask.shape[0] - 1):
        for j in range(1, padded_mask.shape[1] - 1):
            if padded_mask[i, j] == 1:
                # Check 4-connectivity (up, down, left, right)
                if padded_mask[i-1, j] == 0 or padded_mask[i+1, j] == 0 or padded_mask[i, j-1] == 0 or padded_mask[i, j+1] == 0:
                    perimeter += 1
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

    # Calculate average hue
    hsv_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j] / 255.0
            c_max = max(r, g, b)
            c_min = min(r, g, b)
            delta = c_max - c_min
            if delta == 0:
                h = 0
            elif c_max == r:
                h = (60 * ((g - b) / delta) + 360) % 360
            elif c_max == g:
                h = (60 * ((b - r) / delta) + 120) % 360
            else:
                h = (60 * ((r - g) / delta) + 240) % 360
            hsv_image[i, j, 0] = h
    hue_values = hsv_image[:, :, 0][binary_mask == 1]
    avg_hue = np.mean(hue_values) if len(hue_values) > 0 else 0


    def count_clusters(corner_mask):
        # Simple connected component labeling for 8-connectivity using only numpy
        visited = np.zeros_like(corner_mask, dtype=bool)
        count = 0
        for i in range(corner_mask.shape[0]):
            for j in range(corner_mask.shape[1]):
                if corner_mask[i, j] and not visited[i, j]:
                    # Start a new cluster
                    stack = [(i, j)]
                    while stack:
                        x, y = stack.pop()
                        if (0 <= x < corner_mask.shape[0] and
                            0 <= y < corner_mask.shape[1] and
                            corner_mask[x, y] and not visited[x, y]):
                            visited[x, y] = True
                            # Add all 8 neighbors
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    if dx != 0 or dy != 0:
                                        nx, ny = x + dx, y + dy
                                        if (0 <= nx < corner_mask.shape[0] and
                                            0 <= ny < corner_mask.shape[1] and
                                            corner_mask[nx, ny] and not visited[nx, ny]):
                                            stack.append((nx, ny))
                    count += 1
        return count

    def harris_corner_detection_numpy(img, mask=None, window_size=3, k=0.04, threshold=0.01):
        # Convert to grayscale
        if img.ndim == 3:
            img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        else:
            img_gray = img.astype(np.float32)

        # Compute gradients using Sobel operator
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        Ix = convolve2d(img_gray, sobel_x)
        Iy = convolve2d(img_gray, sobel_y)

        # Compute products of derivatives
        Ixx = Ix ** 2
        Iyy = Iy ** 2
        Ixy = Ix * Iy

        # Apply Gaussian filter to smooth the products
        gaussian_kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        Sxx = convolve2d(Ixx, gaussian_kernel)
        Syy = convolve2d(Iyy, gaussian_kernel)
        Sxy = convolve2d(Ixy, gaussian_kernel)

        # Compute Harris response
        det_M = (Sxx * Syy) - (Sxy ** 2)
        trace_M = Sxx + Syy
        R = det_M - k * (trace_M ** 2)

        # Threshold the response
        max_response = np.max(R) if np.max(R) > 0 else 1
        R_thresh = R > threshold * max_response

        # Non-maximum suppression using only numpy
        pad = 1
        padded_R = np.pad(R, pad, mode='constant', constant_values=0)
        corners = np.zeros_like(R, dtype=bool)
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                local_patch = padded_R[i:i+3, j:j+3]
                if R[i, j] == np.max(local_patch) and R_thresh[i, j]:
                    corners[i, j] = True

        # Mask corners to only count those inside the object
        if mask is not None:
            corners = corners & (mask.astype(bool))

        return corners

    corners = harris_corner_detection_numpy(image, mask=binary_mask)

    # corners = harris_corner_detection_numpy(image, mask=binary_mask)
    corner_count = count_clusters(corners)
    # corner_count = np.sum(corners) 

    return {
        'corner_count': corner_count,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'extent': extent,
        'avg_hue': avg_hue
    }
