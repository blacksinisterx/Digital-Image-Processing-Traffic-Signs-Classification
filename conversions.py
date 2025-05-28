import numpy as np

def rgb_to_hsv(rgb):
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    delta = c_max - c_min

    # Hue
    if delta == 0:
        h = 0
    elif c_max == r:
        h = (60 * ((g - b) / delta) + 360) % 360
    elif c_max == g:
        h = (60 * ((b - r) / delta) + 120) % 360
    else:
        h = (60 * ((r - g) / delta) + 240) % 360

    # Saturation
    s = 0 if c_max == 0 else (delta / c_max)

    # Value
    v = c_max

    return np.array([h, s * 100, v * 100])

def segment_hsv(image):
    height, width, _ = image.shape
    red_segment = np.zeros_like(image)
    blue_segment = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            hsv_pixel = rgb_to_hsv(image[i, j])
            h, s, v = hsv_pixel

            s = round(s)
            v = round(v)

            #if ((0 <= h <= 15 or 165 <= h <= 180) and s == 100 and v == 80):
            #    red_segment[i, j] = [255, 0, 0]  # RGB Red
            #elif (100 <= h <= 130 and s == 100 and v == 80):
            #    blue_segment[i, j] = [0, 0, 255]  # RGB Blue
            # Use ranges instead of exact values
            if ((0 <= h <= 20 or 160 <= h <= 180) and s > 50  and v > 40):
                red_segment[i, j] = [255, 0, 0]  # Red in RGB
            elif ((200 <= h <= 240) or (100 <= h <= 130)) and (s >= 50) and (v >= 40):
                blue_segment[i, j] = [0, 0, 255]  # Blue in RGB

    return red_segment, blue_segment


