import numpy as np


def mean_filter(image):
    # Convert image to float for accurate calculations
    image = image.astype(np.float32) / 255.0
    
    # Define a 3x3 mean kernel
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    
    # Pad the image to handle borders
    padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    
    # Apply the mean filter
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                filtered_image[i, j, c] = np.sum(padded_image[i:i+3, j:j+3, c] * kernel)
    
    # Convert back to uint8
    return (filtered_image * 255).astype(np.uint8)

def gaussian_filter(image, sigma=1.0):
    # Convert image to float
    image = image.astype(np.float32) / 255.0
    
    # Define a 3x3 Gaussian kernel
    size = 3
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    
    # Pad the image
    padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    
    # Apply the Gaussian filter
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                filtered_image[i, j, c] = np.sum(padded_image[i:i+3, j:j+3, c] * kernel)
    
    # Convert back to uint8
    return (filtered_image * 255).astype(np.uint8)


def median_filter(image):
    # Pad the image
    padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    
    # Apply the median filter
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                patch = padded_image[i:i+3, j:j+3, c]
                filtered_image[i, j, c] = np.median(patch)
    
    return filtered_image.astype(np.uint8)


def adaptive_median_filter(image, max_size=7):
    # Pad the image with max_size//2 to handle the largest window
    pad_size = max_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    
    filtered_image = np.copy(image).astype(np.uint8)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                # Start with a small window and increase up to max_size
                for window_size in range(3, max_size + 1, 2):
                    half_size = window_size // 2
                    patch = padded_image[i:i+window_size, j:j+window_size, c]
                    
                    # Compute median, min, and max of the patch
                    med = np.median(patch)
                    mn = np.min(patch)
                    mx = np.max(patch)
                    
                    # If the center pixel is an impulse noise, replace it with the median
                    if not (mn < padded_image[i+half_size, j+half_size, c] < mx):
                        filtered_image[i, j, c] = med
                        break  # No need to check larger windows
    
    return filtered_image


def unsharp_masking(image, sigma=1.0, strength=1.0):
    # Convert image to float
    image = image.astype(np.float32) / 255.0
    
    # Apply Gaussian blur
    blurred = gaussian_filter((image * 255).astype(np.uint8), sigma).astype(np.float32) / 255.0
    
    # Compute the high-pass image
    high_pass = image - blurred
    
    # Combine the original image with the high-pass image
    sharpened = image + strength * high_pass
    
    # Clip values to [0, 1] and convert back to uint8
    sharpened = np.clip(sharpened, 0, 1)
    return (sharpened * 255).astype(np.uint8)



