"""Integration test: combine transforms -> warp channels -> pyramidal TIFF."""

import numpy as np
import tifffile
import pytest
from pathlib import Path

pyvips = pytest.importorskip("pyvips")

from tests.conftest import import_process

warp_mod = import_process("007_warp_all_channels_and_downsample")


class TestWarpAndPyramid:
    def test_combine_transforms_then_save_pyramid(self, tmp_path):
        h, w = 64, 64
        n_channels = 2

        # Create identity mirage warp (H, W, 2) in row, col order
        rows, cols = np.meshgrid(
            np.arange(h, dtype=np.float64),
            np.arange(w, dtype=np.float64),
            indexing="ij",
        )
        mirage_warp = np.stack([rows, cols], axis=-1)

        # Identity linear transform
        linear_transform = np.eye(3, dtype=np.float64)

        # Combine
        combined = warp_mod.combine_transforms(mirage_warp, linear_transform)
        assert combined.shape == (2, h, w)

        # Create a multi-channel image (H, W, C) with bimodal data for Otsu
        moving_image = np.zeros((h, w, n_channels), dtype=np.uint16)
        for c in range(n_channels):
            bg = np.random.randint(100, 500, (h, w), dtype=np.uint16)
            bg[10:30, 10:30] = 50000  # bright blob
            moving_image[:, :, c] = bg

        output_path = tmp_path / "output.ome.tiff"
        warp_mod.warp_and_save_pyramidal_tiff(
            moving_image=moving_image,
            combined_transform=combined,
            output_file_path=output_path,
            n_levels=2,
            dtype_out=np.uint16,
            description="test",
        )

        assert output_path.exists()

        # Verify the TIFF is readable and has expected structure
        with tifffile.TiffFile(str(output_path)) as tif:
            base = tif.pages[0]
            assert base.shape[-1] == w or base.shape[-2] == w  # sanity check dims
