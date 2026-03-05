"""Integration test: labeled mask -> contours -> polygons -> join masks."""

import numpy as np
import geopandas as gpd
import pytest

from tests.conftest import import_process


def _has_pyarrow():
    try:
        import pyarrow.parquet  # noqa: F401
        return True
    except ImportError:
        return False

masks_mod = import_process("011_convert_masks_to_gpd")
join_mod = import_process("012_combine_combined_and_nuclei_masks")


class TestMasksPipeline:
    def test_masks_to_polygons_to_join(self):
        # Create two separate labeled masks (simulating combined + nuclei segmentation)
        combined_mask = np.zeros((100, 100), dtype=np.int32)
        combined_mask[10:30, 10:30] = 1
        combined_mask[50:80, 50:80] = 2

        nuclei_mask = np.zeros((100, 100), dtype=np.int32)
        nuclei_mask[15:25, 15:25] = 1   # overlaps with combined cell 1
        nuclei_mask[70:90, 10:30] = 2   # does not overlap

        # Convert both to GeoDataFrames
        combined_contours = masks_mod.masks_to_contours(combined_mask)
        combined_gdf = masks_mod.contours_to_polygons(
            combined_contours[:, 0], combined_contours[:, 1], combined_contours[:, 2]
        )

        nuclei_contours = masks_mod.masks_to_contours(nuclei_mask)
        nuclei_gdf = masks_mod.contours_to_polygons(
            nuclei_contours[:, 0], nuclei_contours[:, 1], nuclei_contours[:, 2]
        )

        assert len(combined_gdf) == 2
        assert len(nuclei_gdf) == 2

        # Join: overlapping nuclei excluded, non-overlapping included
        final_gdf = join_mod.join_masks(combined_gdf, nuclei_gdf)
        assert isinstance(final_gdf, gpd.GeoDataFrame)
        # 2 combined + 1 non-overlapping nuclei = 3
        assert len(final_gdf) == 3

    @pytest.mark.skipif(
        not _has_pyarrow(), reason="pyarrow not installed"
    )
    def test_parquet_roundtrip(self, tmp_path):
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[5:20, 5:20] = 1
        mask[30:50, 30:50] = 2

        contours = masks_mod.masks_to_contours(mask)
        gdf = masks_mod.contours_to_polygons(
            contours[:, 0], contours[:, 1], contours[:, 2]
        )

        # Save and reload
        parquet_path = tmp_path / "masks.parquet"
        gdf.to_parquet(parquet_path)
        gdf_loaded = gpd.read_parquet(parquet_path)

        assert len(gdf_loaded) == len(gdf)
        assert gdf_loaded.geometry.is_valid.all()
