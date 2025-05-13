import numpy as np


def extract_hog_features(img, cell_size=(4, 4), num_bins=9):
    """
    Compute Histogram of Oriented Gradients (HOG) features
    for a grayscale image.

    Args:
        img (np.ndarray): 2D image
        cell_size (tuple): Size of each cell (height, width)
        num_bins (int): Number of orientation bins

    Returns:
        np.ndarray: 1D HOG feature vector
    """
    H, W = img.shape
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)

    # Compute gradients using a [1 -1] filter
    gx[:, :-1] = np.diff(img, n=1, axis=1)
    gy[:-1, :] = np.diff(img, n=1, axis=0)

    # Compute gradient magnitude and angle
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.rad2deg(np.arctan2(gy, gx)) % 180

    # Compute number of cells
    cell_h, cell_w = cell_size
    n_cells_y = H // cell_h
    n_cells_x = W // cell_w

    hog = np.zeros((n_cells_y, n_cells_x, num_bins))

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_mag = magnitude[i*cell_h: (i+1)*cell_h,
                                 j*cell_w: (j+1)*cell_w]
            cell_ori = orientation[i*cell_h: (i+1)*cell_h,
                                   j*cell_w: (j+1)*cell_w]

            # Build histogram
            hist = np.zeros(num_bins)
            bin_width = 180 / num_bins
            for x in range(cell_h):
                for y in range(cell_w):
                    bin_idx = int(cell_ori[x, y] // bin_width) % num_bins
                    hist[bin_idx] += cell_mag[x, y]

            hog[i, j] = hist

    return hog.ravel()  # return flattened
