"""Integration test: setup directories -> convert/write TIFF -> preprocess."""

import numpy as np
import tifffile
import pytest
from pathlib import Path

from tests.conftest import import_process
from segmentation_tools.utils.config import CHECKPOINT_DIR_NAME, RESULTS_DIR_NAME

setup_dirs = import_process("000_setup_directories")
preprocess = import_process("003_preprocess_images")


class TestSetupAndPreprocess:
    def test_setup_then_preprocess(self, tmp_path):
        # Step 0: set up directories
        output_dir, results_dir, checkpoint_dir = setup_dirs.main(
            output_dir_root=tmp_path, job_title="integration_test"
        )
        assert checkpoint_dir.exists()
        assert results_dir.exists()

        # Create a synthetic 2-channel TIFF in the checkpoint dir
        h, w = 256, 256
        ch0 = np.random.randint(0, 65535, (h, w), dtype=np.uint16)
        ch1 = np.random.randint(0, 65535, (h, w), dtype=np.uint16)
        data = np.stack([ch0, ch1], axis=0)
        tiff_path = checkpoint_dir / "moving.tiff"
        tifffile.imwrite(str(tiff_path), data, photometric="minisblack")

        # Step 3: preprocess — load channel 0 at level 0, normalize (no filter)
        output_npy = checkpoint_dir / "ds_moving_dapi_level_0.npy"
        preprocess.main(
            input_file_path=str(tiff_path),
            dapi_channel_moving=0,
            level=0,
            output_file_path=str(output_npy),
            filter=False,
        )
        assert output_npy.exists()
        result = np.load(output_npy)
        assert result.shape == (h, w)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_with_filter(self, tmp_path):
        # Set up dirs + TIFF with bimodal data
        output_dir, results_dir, checkpoint_dir = setup_dirs.main(
            output_dir_root=tmp_path, job_title="filter_test"
        )

        h, w = 512, 512
        # Create bimodal image: background + bright blobs
        ch = np.random.randint(100, 500, (h, w), dtype=np.uint16)
        for y, x in [(100, 100), (200, 300), (400, 150)]:
            ch[y - 30 : y + 30, x - 30 : x + 30] = 60000
        data = ch[np.newaxis, ...]  # (1, H, W)
        tiff_path = checkpoint_dir / "fixed.tiff"
        tifffile.imwrite(str(tiff_path), data, photometric="minisblack")

        output_npy = checkpoint_dir / "fixed_filtered.npy"
        preprocess.main(
            input_file_path=str(tiff_path),
            dapi_channel_moving=0,
            level=0,
            output_file_path=str(output_npy),
            filter=True,
        )
        assert output_npy.exists()
        result = np.load(output_npy)
        assert result.shape == (h, w)
        # Filtered image should have zeros where below threshold
        assert (result == 0).any()
