import numpy as np

def angleDiff(x1, x2):
    return np.arctan2(np.sin(x1 - x2), np.cos(x1 - x2))

