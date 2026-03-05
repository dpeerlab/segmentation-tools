import numpy as np
import pytest
import geopandas as gpd

from tests.conftest import import_process

_mod = import_process("011_convert_masks_to_gpd")
masks_to_contours = _mod.masks_to_contours
contours_to_polygons = _mod.contours_to_polygons


class TestMasksToContours:
    def test_returns_array(self, sample_labeled_mask):
        contours = masks_to_contours(sample_labeled_mask)
        assert isinstance(contours, np.ndarray)

    def test_contour_columns(self, sample_labeled_mask):
        contours = masks_to_contours(sample_labeled_mask)
        # Each row should have 3 columns: x, y, cell_id
        assert contours.shape[1] == 3

    def test_cell_ids_present(self, sample_labeled_mask):
        contours = masks_to_contours(sample_labeled_mask)
        cell_ids = np.unique(contours[:, 2])
        # All 3 labeled cells should appear
        assert len(cell_ids) == 3
        for label in [1, 2, 3]:
            assert label in cell_ids

    def test_empty_mask_raises(self):
        mask = np.zeros((64, 64), dtype=np.int32)
        with pytest.raises((ValueError, IndexError)):
            masks_to_contours(mask)


class TestContoursToPolygons:
    def test_returns_geodataframe(self, sample_labeled_mask):
        contours = masks_to_contours(sample_labeled_mask)
        gdf = contours_to_polygons(contours[:, 0], contours[:, 1], contours[:, 2])
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_polygon_count_matches_cells(self, sample_labeled_mask):
        contours = masks_to_contours(sample_labeled_mask)
        gdf = contours_to_polygons(contours[:, 0], contours[:, 1], contours[:, 2])
        assert len(gdf) == 3

    def test_geometries_are_valid(self, sample_labeled_mask):
        contours = masks_to_contours(sample_labeled_mask)
        gdf = contours_to_polygons(contours[:, 0], contours[:, 1], contours[:, 2])
        assert gdf.geometry.is_valid.all()
