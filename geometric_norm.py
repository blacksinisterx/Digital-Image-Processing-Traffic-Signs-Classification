import numpy as np

def rotate_image_numpy(image, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    H, W = image.shape[:2]
    cx, cy = W / 2, H / 2

    # Output image same size as input
    if len(image.shape) == 2:
        rotated = np.zeros((H, W), dtype=np.uint8)
    else:
        rotated = np.zeros((H, W, image.shape[2]), dtype=np.uint8)

    # Inverse mapping: target → source
    for y in range(H):
        for x in range(W):
            # Translate to center, rotate, then back
            xt = x - cx
            yt = y - cy
            src_x = cos_a * xt + sin_a * yt + cx
            src_y = -sin_a * xt + cos_a * yt + cy

            src_x_int = int(round(src_x))
            src_y_int = int(round(src_y))

            if 0 <= src_x_int < W and 0 <= src_y_int < H:
                rotated[y, x] = image[src_y_int, src_x_int]

    return rotated

def resize_image_numpy(image, new_size=(200, 200)):
    new_h, new_w = new_size
    old_h, old_w = image.shape[:2]

    scale_y = old_h / new_h
    scale_x = old_w / new_w

    if len(image.shape) == 2:
        resized = np.zeros((new_h, new_w), dtype=np.uint8)
    else:
        resized = np.zeros((new_h, new_w, image.shape[2]), dtype=np.uint8)

    for y in range(new_h):
        for x in range(new_w):
            src_y = int(y * scale_y)
            src_x = int(x * scale_x)
            resized[y, x] = image[src_y, src_x]

    return resized

def compute_homography(src_pts, dst_pts):
    A = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    A = np.array(A)

    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape((3, 3))
    return H / H[2, 2]  

def warp_perspective(image, H, output_size=(200, 200)):
    H_inv = np.linalg.inv(H)
    H_out, W_out = output_size

    if len(image.shape) == 2:
        warped = np.zeros((H_out, W_out), dtype=np.uint8)
    else:
        warped = np.zeros((H_out, W_out, image.shape[2]), dtype=np.uint8)

    for y in range(H_out):
        for x in range(W_out):
            p = np.array([x, y, 1])
            p_src = H_inv @ p
            p_src /= p_src[2]
            sx, sy = p_src[0], p_src[1]
            sx, sy = int(round(sx)), int(round(sy))

            if 0 <= sx < image.shape[1] and 0 <= sy < image.shape[0]:
                warped[y, x] = image[sy, sx]

    return warped