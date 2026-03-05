"""Integration test: SIFT alignment -> warp with transform."""

import numpy as np
import pytest
from pathlib import Path

from tests.conftest import import_process

sift_mod = import_process("004_find_sift_alignment_transform")
warp_mod = import_process("005_warp_image_with_sift")


class TestSiftThenWarp:
    def test_sift_homography_then_warp(self, tmp_path):
        np.random.seed(42)
        # Create a textured image with features
        img = np.random.randint(30, 220, (200, 200), dtype=np.uint8).astype(np.float32) / 255.0

        # Compute SIFT on same image (should give near-identity)
        from skimage.util import img_as_ubyte
        img_ubyte = img_as_ubyte(img)
        transform = sift_mod.get_SIFT_homography(img_ubyte, img_ubyte)

        # Save moving image and transform
        moving_path = tmp_path / "moving.npy"
        transform_path = tmp_path / "transform.npy"
        np.save(moving_path, img)
        np.save(transform_path, transform.params)

        # Warp
        result = warp_mod.main(
            moving_file_path=str(moving_path),
            transform_file_path=str(transform_path),
            checkpoint_dir=tmp_path,
            fixed_shape=(200, 200),
        )
        assert result == 0

        warped = np.load(tmp_path / "moving_dapi_linear_warped.npy")
        assert warped.shape == (200, 200)
        # Since transform is near-identity, warped should be close to original
        np.testing.assert_allclose(warped, img, atol=0.15)
