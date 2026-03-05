import geopandas as gpd
import numpy as np
from shapely.geometry import box

from tests.conftest import import_process

_mod = import_process("012_combine_combined_and_nuclei_masks")
join_masks = _mod.join_masks


class TestJoinMasks:
    def test_non_overlapping_nuclei_added(self):
        combined = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)])
        nuclei = gpd.GeoDataFrame(geometry=[box(50, 50, 60, 60)])

        result = join_masks(combined, nuclei)
        # Should contain both: 1 combined + 1 non-overlapping nuclei
        assert len(result) == 2

    def test_overlapping_nuclei_excluded(self):
        combined = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)])
        nuclei = gpd.GeoDataFrame(geometry=[box(5, 5, 15, 15)])  # overlaps

        result = join_masks(combined, nuclei)
        # Overlapping nuclei should be excluded, only combined remains
        assert len(result) == 1

    def test_mixed_overlap(self):
        combined = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)])
        nuclei = gpd.GeoDataFrame(
            geometry=[
                box(5, 5, 15, 15),    # overlaps -> excluded
                box(50, 50, 60, 60),  # no overlap -> included
            ]
        )

        result = join_masks(combined, nuclei)
        assert len(result) == 2  # 1 combined + 1 non-overlapping nuclei

    def test_empty_nuclei(self):
        combined = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)])
        nuclei = gpd.GeoDataFrame(geometry=[])

        result = join_masks(combined, nuclei)
        assert len(result) == 1

    def test_empty_combined(self):
        combined = gpd.GeoDataFrame(geometry=[])
        nuclei = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)])

        result = join_masks(combined, nuclei)
        assert len(result) == 1
