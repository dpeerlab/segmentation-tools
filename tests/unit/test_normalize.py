import numpy as np
import pytest

from segmentation_tools.utils.utils import normalize


class TestNormalize:
    def test_output_range(self, sample_grayscale_image):
        result = normalize(sample_grayscale_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype(self, sample_grayscale_image):
        result = normalize(sample_grayscale_image)
        assert result.dtype == np.float32

    def test_output_shape_unchanged(self, sample_grayscale_image):
        result = normalize(sample_grayscale_image)
        assert result.shape == sample_grayscale_image.shape

    def test_constant_image_returns_constant(self):
        img = np.ones((100, 100), dtype=np.float32) * 0.5
        result = normalize(img)
        # A constant image produces zeros after quantile clipping,
        # but CLAHE may produce a small constant offset
        assert result.std() < 1e-6

    def test_rgb_image_normalizes_per_channel(self):
        np.random.seed(0)
        img = np.random.rand(100, 100, 3).astype(np.float32)
        result = normalize(img)
        assert result.shape == (100, 100, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_uint16_input(self):
        img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        result = normalize(img)
        assert result.dtype == np.float32
        assert result.shape == (100, 100)

    def test_custom_quantiles(self, sample_grayscale_image):
        result = normalize(sample_grayscale_image, quantiles=[0.05, 0.95])
        assert result.min() >= 0.0
        assert result.max() <= 1.0
