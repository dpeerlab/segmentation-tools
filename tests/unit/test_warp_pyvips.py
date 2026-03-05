import numpy as np
import pytest

pyvips = pytest.importorskip("pyvips")

from tests.conftest import import_process

_mod = import_process("007_warp_all_channels_and_downsample")
_warp_channel_pyvips = _mod._warp_channel_pyvips
combine_transforms = _mod.combine_transforms


class TestWarpChannelPyvips:
    def test_identity_warp_preserves_image(self, sample_uint8_image, identity_transform_2x):
        h, w = sample_uint8_image.shape
        # Resize identity transform to match image dims
        rows, cols = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )
        identity = np.stack([rows, cols], axis=0)

        result = _warp_channel_pyvips(
            sample_uint8_image, identity, tile_height=64, interpolation="nearest"
        )
        assert result.shape == (h, w)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        # With nearest interpolation and identity map, values should match closely
        expected = sample_uint8_image.astype(np.float32) / 255.0
        np.testing.assert_allclose(result, expected, atol=1.0 / 255.0)

    def test_output_shape_matches_transform(self, sample_uint8_image):
        h, w = sample_uint8_image.shape
        out_h, out_w = 100, 120
        rows, cols = np.meshgrid(
            np.linspace(0, h - 1, out_h, dtype=np.float32),
            np.linspace(0, w - 1, out_w, dtype=np.float32),
            indexing="ij",
        )
        transform = np.stack([rows, cols], axis=0)

        result = _warp_channel_pyvips(sample_uint8_image, transform, tile_height=32)
        assert result.shape == (out_h, out_w)

    def test_tiling_produces_same_result(self, sample_uint8_image):
        h, w = sample_uint8_image.shape
        rows, cols = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )
        identity = np.stack([rows, cols], axis=0)

        result_small_tiles = _warp_channel_pyvips(
            sample_uint8_image, identity, tile_height=32, interpolation="nearest"
        )
        result_large_tiles = _warp_channel_pyvips(
            sample_uint8_image, identity, tile_height=1024, interpolation="nearest"
        )
        np.testing.assert_array_equal(result_small_tiles, result_large_tiles)


class TestCombineTransforms:
    def test_identity_combination(self):
        h, w = 64, 64
        rows, cols = np.meshgrid(
            np.arange(h, dtype=np.float64),
            np.arange(w, dtype=np.float64),
            indexing="ij",
        )
        mirage_warp = np.stack([rows, cols], axis=-1)  # (H, W, 2)
        linear_transform = np.eye(3, dtype=np.float64)

        result = combine_transforms(mirage_warp, linear_transform)
        assert result.shape == (2, h, w)
        # With identity linear transform, output should match input
        np.testing.assert_allclose(result[0], rows, atol=1e-6)
        np.testing.assert_allclose(result[1], cols, atol=1e-6)

    def test_output_shape(self):
        h, w = 32, 48
        rows, cols = np.meshgrid(
            np.arange(h, dtype=np.float64),
            np.arange(w, dtype=np.float64),
            indexing="ij",
        )
        mirage_warp = np.stack([rows, cols], axis=-1)
        linear_transform = np.eye(3, dtype=np.float64)

        result = combine_transforms(mirage_warp, linear_transform)
        assert result.shape == (2, h, w)
