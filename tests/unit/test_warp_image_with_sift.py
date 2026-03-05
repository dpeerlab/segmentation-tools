import numpy as np
import tifffile
from pathlib import Path

from tests.conftest import import_process

_mod = import_process("005_warp_image_with_sift")
main = _mod.main
get_shape_at_level = _mod.get_shape_at_level


class TestGetShapeAtLevel:
    def test_returns_height_width(self, sample_tiff_path):
        shape = get_shape_at_level(sample_tiff_path, level=0)
        assert len(shape) == 2
        assert shape == (64, 64)


class TestWarpWithSift:
    def test_identity_warp(self, tmp_path):
        # Create a simple image and identity transform
        img = np.random.rand(64, 64).astype(np.float32)
        transform = np.eye(3)

        img_path = tmp_path / "moving.npy"
        transform_path = tmp_path / "transform.npy"
        np.save(img_path, img)
        np.save(transform_path, transform)

        result = main(
            moving_file_path=str(img_path),
            transform_file_path=str(transform_path),
            checkpoint_dir=tmp_path,
            fixed_shape=(64, 64),
        )
        assert result == 0

        warped_path = tmp_path / "moving_dapi_linear_warped.npy"
        assert warped_path.exists()
        warped = np.load(warped_path)
        assert warped.shape == (64, 64)
