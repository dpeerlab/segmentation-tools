# TODO: Consider switching to skimage.io.imread which supports 'as_gray' arguments

import os
import cv2
import importlib.resources
import numpy as np
import mirage


def get_data(filename=""):
    """
    Load data from mirage/data.
    """
    # Get the path to the data directory
    basepath = str(importlib.resources.files(mirage).joinpath('data'))
    filepath = os.path.join(basepath, filename)
    
    # Assert that the file exists
    if filename is None or not os.path.isfile(filepath):
        msg = f"File {filename} not found. Available files are:\n"
        for f in os.listdir(basepath):
            msg += f"    - {f}\n"
        raise FileNotFoundError(msg)
    
    # Read tiff files
    if filename.endswith(".tiff") or filename.endswith(".tif"):
        img = cv2.imread(filepath)
        if img is None:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    elif filename.endswith(".npy"):
        img = np.load(filepath)
    else:
        supported_extensions = [".tiff", ".tif", ".npy"]
        extension = os.path.splitext(filename)[-1]
        raise ValueError(f"File extension {extension} not supported. Supported extensions are: {supported_extensions}")
    
    # Convert to grayscale and normalize
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / img.max()

    return img
