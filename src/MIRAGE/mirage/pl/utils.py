import numpy as np

def get_overlay(ref, img):
    """Overlay a reference and another image"""
    overlay = np.concatenate([ref[:, :, None], img[:, :, None], img[:, :, None]], axis=2)
    return overlay

def generate_grid(img, lines_every=15):
    """
    Generate a grid with white background and black lines.
    """
    s = img.shape
    grid = np.ones(s)
    for i in range(0, s[0], lines_every):
        grid[i, :] = 0
    for i in range(0, s[1], lines_every):
        grid[:, i] = 0
    return grid
