import numpy as np


def extract_zoning_features(img, grid_size=(4, 4), binarize=True):
    if binarize:
        img = (img > 0.5).astype(float)

    h, w = img.shape
    gh, gw = grid_size
    zone_h, zone_w = h // gh, w // gw

    features = []
    for i in range(gh):
        for j in range(gw):
            zone = img[i*zone_h: (i+1)*zone_h, j*zone_w: (j+1)*zone_w]
            zone_mean = np.mean(zone)
            features.append(zone_mean)
    return np.array(features)
