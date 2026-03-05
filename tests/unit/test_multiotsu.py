import numpy as np
import pytest

from segmentation_tools.utils.utils import get_multiotsu_threshold


class TestGetMultiotsuThreshold:
    def test_returns_float(self):
        np.random.seed(42)
        # Create image with clear bimodal distribution
        img = np.zeros((600, 600), dtype=np.float32)
        img[:300, :] = 0.2 + np.random.rand(300, 600).astype(np.float32) * 0.1
        img[300:, :] = 0.8 + np.random.rand(300, 600).astype(np.float32) * 0.1
        threshold = get_multiotsu_threshold(img, n_samples=5)
        assert np.issubdtype(type(threshold), np.floating)

    def test_threshold_in_valid_range(self):
        np.random.seed(42)
        img = np.zeros((600, 600), dtype=np.float32)
        img[:300, :] = 0.1 + np.random.rand(300, 600).astype(np.float32) * 0.05
        img[300:, :] = 0.9 + np.random.rand(300, 600).astype(np.float32) * 0.05
        threshold = get_multiotsu_threshold(img, n_samples=5)
        assert 0.0 < threshold < 1.0

    def test_raises_on_constant_image(self):
        img = np.ones((600, 600), dtype=np.float32) * 0.5
        with pytest.raises(ValueError, match="Could not find any valid"):
            get_multiotsu_threshold(img, n_samples=5)

    def test_small_image_uses_full_image(self):
        np.random.seed(42)
        # Image smaller than minimum glimpse size of 100
        img = np.zeros((80, 80), dtype=np.float32)
        img[:40, :] = 0.1 + np.random.rand(40, 80).astype(np.float32) * 0.05
        img[40:, :] = 0.9 + np.random.rand(40, 80).astype(np.float32) * 0.05
        threshold = get_multiotsu_threshold(img, n_samples=3)
        assert np.issubdtype(type(threshold), np.floating)
