import numpy as np


def extract_zoning_features(img, grid_size=(4, 4), binarize=True):
    """
    Compute Zoning features for an image.

    Args:
        img (np.ndarray): 2D image
        grid_size (tuple): Size of each cell (height, width)
        binarize (boolean): Indicator to binarize the image before zoning

    Returns:
        np.ndarray: 1D Zoning feature vector
    """
    if binarize:
        img = (img > 0.5).astype(float)

    # Extract dimensions
    h, w = img.shape
    gh, gw = grid_size
    zone_h, zone_w = h // gh, w // gw

    # Get mean for each cell
    features = []
    for i in range(gh):
        for j in range(gw):
            zone = img[i*zone_h: (i+1)*zone_h, j*zone_w: (j+1)*zone_w]
            zone_mean = np.mean(zone)
            features.append(zone_mean)
    return np.array(features)
