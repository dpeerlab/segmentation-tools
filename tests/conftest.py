import importlib

import pytest
import numpy as np
import tifffile


def import_process(filename_stem: str):
    """Import a process module by its filename stem (without .py).

    Process files are named with numeric prefixes (e.g. ``007_warp_all_channels_and_downsample``)
    which aren't valid Python identifiers, so we use importlib.
    """
    return importlib.import_module(
        f"segmentation_tools.processes.{filename_stem}"
    )


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_grayscale_image():
    """A small synthetic grayscale image with structure (not constant)."""
    np.random.seed(42)
    img = np.random.rand(512, 512).astype(np.float32)
    # Add some bright blobs to ensure multi-Otsu can find modes
    for y, x in [(100, 100), (200, 300), (400, 150)]:
        img[y - 20 : y + 20, x - 20 : x + 20] = 0.9
    return img


@pytest.fixture
def sample_uint8_image():
    """A small synthetic uint8 grayscale image."""
    np.random.seed(42)
    img = (np.random.rand(256, 256) * 255).astype(np.uint8)
    for y, x in [(50, 50), (150, 200)]:
        img[y - 10 : y + 10, x - 10 : x + 10] = 230
    return img


@pytest.fixture
def sample_multichannel_image():
    """A small synthetic multi-channel image (C, H, W) stored as (H, W, C)."""
    np.random.seed(42)
    return np.random.randint(0, 65535, (128, 128, 3), dtype=np.uint16)


@pytest.fixture
def identity_transform_2x():
    """An identity dense coordinate map (2, H, W) that maps each pixel to itself."""
    h, w = 128, 128
    rows, cols = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    return np.stack([rows, cols], axis=0)


@pytest.fixture
def identity_matrix():
    """A 3x3 identity homography matrix."""
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def sample_tiff_path(tmp_path):
    """Create a small multi-channel pyramidal TIFF and return its path."""
    path = tmp_path / "sample.tiff"
    # 2 channels, 64x64
    data = np.random.randint(0, 255, (2, 64, 64), dtype=np.uint8)
    tifffile.imwrite(str(path), data, photometric="minisblack")
    return path


@pytest.fixture
def sample_labeled_mask():
    """A small labeled mask image with a few cells."""
    mask = np.zeros((64, 64), dtype=np.int32)
    # Cell 1: small square
    mask[10:20, 10:20] = 1
    # Cell 2: another square
    mask[30:45, 30:45] = 2
    # Cell 3: rectangle
    mask[5:15, 40:55] = 3
    return mask
