import pytest
import numpy as np
import tempfile
import tifffile
import h5py
import json

from segmentation_tools.utils import load_feature_metadata, crop_visium_region


@pytest.fixture
def dummy_metadata_path():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as f:
            f.attrs["metadata"] = json.dumps(
                {
                    "ncols": 10,
                    "nrows": 10,
                    "transform_matrices": {
                        "spot_colrow_to_microscope_colrow": [
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                        ]
                    },
                }
            )
        yield tmp.name


def test_load_feature_metadata(dummy_metadata_path):
    meta = load_feature_metadata(dummy_metadata_path)
    assert isinstance(meta, dict)
    assert meta["ncols"] == 10
    assert "transform_matrices" in meta


def test_crop_visium_region(dummy_metadata_path):
    img = np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_img:
        tifffile.imwrite(tmp_img.name, img)
        cropped = crop_visium_region(tmp_img.name, dummy_metadata_path)
        assert isinstance(cropped, np.ndarray)
        assert cropped.ndim == 2
