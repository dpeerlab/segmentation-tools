import numpy as np

from segmentation_tools.utils.utils import create_rgb_overlay


class TestCreateRgbOverlay:
    def test_output_shape(self):
        fixed = np.random.rand(100, 100).astype(np.float32)
        moving = np.random.rand(100, 100).astype(np.float32)
        result = create_rgb_overlay(fixed, moving)
        assert result.shape == (100, 100, 3)

    def test_output_range(self):
        fixed = np.random.rand(50, 50).astype(np.float32)
        moving = np.random.rand(50, 50).astype(np.float32)
        result = create_rgb_overlay(fixed, moving)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dtype(self):
        fixed = np.ones((10, 10), dtype=np.float32)
        moving = np.ones((10, 10), dtype=np.float32)
        result = create_rgb_overlay(fixed, moving)
        assert result.dtype == np.float32

    def test_zero_images(self):
        fixed = np.zeros((10, 10), dtype=np.float32)
        moving = np.zeros((10, 10), dtype=np.float32)
        result = create_rgb_overlay(fixed, moving)
        assert np.allclose(result, 0.0)

    def test_red_channel_is_fixed(self):
        fixed = np.ones((10, 10), dtype=np.float32)
        moving = np.zeros((10, 10), dtype=np.float32)
        result = create_rgb_overlay(fixed, moving)
        assert np.all(result[..., 0] == 1.0)
        assert np.all(result[..., 1] == 0.0)
        assert np.all(result[..., 2] == 0.0)

    def test_green_blue_channels_are_moving(self):
        fixed = np.zeros((10, 10), dtype=np.float32)
        moving = np.ones((10, 10), dtype=np.float32)
        result = create_rgb_overlay(fixed, moving)
        assert np.all(result[..., 0] == 0.0)
        assert np.all(result[..., 1] == 1.0)
        assert np.all(result[..., 2] == 1.0)
