import numpy as np
import pytest
from skimage.transform import ProjectiveTransform, AffineTransform

from tests.conftest import import_process

_mod = import_process("004_find_sift_alignment_transform")
get_SIFT_homography = _mod.get_SIFT_homography
_generate_image_variants = _mod._generate_image_variants


class TestGenerateImageVariants:
    def test_returns_8_variants(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        variants = _generate_image_variants(img)
        assert len(variants) == 8

    def test_variant_shapes(self):
        img = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
        variants = _generate_image_variants(img)
        for variant_img, name, matrix in variants:
            assert variant_img.ndim == 2
            assert matrix.shape == (3, 3)

    def test_original_variant_unchanged(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        variants = _generate_image_variants(img)
        original_img, name, matrix = variants[0]
        assert name == "original"
        np.testing.assert_array_equal(original_img, img)
        np.testing.assert_array_equal(matrix, np.eye(3))


class TestGetSIFTHomography:
    def test_identical_images_return_near_identity(self):
        np.random.seed(42)
        # Create a textured image with enough features for SIFT
        img = np.random.randint(50, 200, (300, 300), dtype=np.uint8)
        result = get_SIFT_homography(img, img)
        assert isinstance(result, ProjectiveTransform)
        # The transform should be close to identity
        np.testing.assert_allclose(result.params, np.eye(3), atol=0.1)

    def test_raises_on_featureless_images(self):
        fixed = np.ones((100, 100), dtype=np.uint8) * 128
        moving = np.ones((100, 100), dtype=np.uint8) * 128
        with pytest.raises(Exception):
            get_SIFT_homography(fixed, moving)
