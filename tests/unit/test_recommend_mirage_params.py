import numpy as np
import pytest

from tests.conftest import import_process

recommend_mod = import_process("005b_recommend_mirage_params")


class TestSampleTissueCrops:
    def test_returns_requested_number_of_crops(self):
        np.random.seed(0)
        img = np.random.rand(2000, 2000).astype(np.float32)
        crops = recommend_mod.sample_tissue_crops(img, crop_size=500, n_crops=3)
        assert len(crops) == 3
        assert crops[0]["crop"].shape == (500, 500)

    def test_small_image_returns_full(self):
        img = np.random.rand(200, 200).astype(np.float32)
        crops = recommend_mod.sample_tissue_crops(img, crop_size=500, n_crops=3)
        assert len(crops) == 1
        assert crops[0]["crop"].shape == (200, 200)

    def test_blank_image_falls_back_to_centre(self):
        img = np.zeros((2000, 2000), dtype=np.float32)
        crops = recommend_mod.sample_tissue_crops(img, crop_size=500, n_crops=3)
        assert len(crops) == 1  # fallback

    def test_tissue_filter_enforced(self):
        # Image with a small tissue region in one corner
        img = np.zeros((2000, 2000), dtype=np.float32)
        img[0:600, 0:600] = np.random.rand(600, 600).astype(np.float32)
        crops = recommend_mod.sample_tissue_crops(
            img, crop_size=500, n_crops=10, min_tissue_fraction=0.5
        )
        # All returned crops should have enough tissue
        for c in crops:
            frac = np.count_nonzero(c["crop"]) / c["crop"].size
            assert frac >= 0.5


class TestGetCentroids:
    def test_detects_blobs(self):
        img = np.zeros((200, 200), dtype=np.float32)
        # Two blobs
        img[20:40, 20:40] = 1.0
        img[100:130, 100:130] = 1.0
        centroids = recommend_mod.get_centroids(img, min_area=10)
        assert len(centroids) == 2

    def test_filters_small_components(self):
        img = np.zeros((200, 200), dtype=np.float32)
        img[10:12, 10:12] = 1.0  # 4 pixels
        img[50:70, 50:70] = 1.0  # 400 pixels
        centroids = recommend_mod.get_centroids(img, min_area=20)
        assert len(centroids) == 1


class TestMatchCentroids:
    def test_perfect_match(self):
        src = np.array([[10, 10], [50, 50], [90, 90]], dtype=np.float64)
        dst = src.copy()
        dists, src_m, dst_m = recommend_mod.match_centroids(src, dst, max_dist=5)
        assert len(dists) == 3
        np.testing.assert_allclose(dists, 0, atol=1e-10)

    def test_filters_distant_matches(self):
        src = np.array([[10, 10], [50, 50]], dtype=np.float64)
        dst = np.array([[10, 10], [200, 200]], dtype=np.float64)
        dists, _, _ = recommend_mod.match_centroids(src, dst, max_dist=50)
        assert len(dists) == 1

    def test_empty_input(self):
        dists, _, _ = recommend_mod.match_centroids(np.empty((0, 2)), np.array([[1, 1]]))
        assert len(dists) == 0


class TestRecommendMirageParams:
    def test_with_shifted_blobs(self):
        np.random.seed(42)
        fixed = np.zeros((500, 500), dtype=np.float32)
        moving = np.zeros((500, 500), dtype=np.float32)

        # Create matching blobs with a known shift
        shift_r, shift_c = 10, 5
        for r, c in [(50, 50), (100, 200), (200, 100), (300, 300), (400, 400)]:
            fixed[r - 15 : r + 15, c - 15 : c + 15] = 1.0
            mr, mc = r + shift_r, c + shift_c
            moving[mr - 15 : mr + 15, mc - 15 : mc + 15] = 1.0

        params = recommend_mod.recommend_mirage_params(fixed, moving)
        assert params["num_matched"] >= 3
        assert params["offset"] >= 10
        assert params["pad"] >= params["offset"]
        assert params["smoothness_radius"] >= 30

    def test_too_few_nuclei_returns_defaults(self):
        fixed = np.zeros((500, 500), dtype=np.float32)
        moving = np.zeros((500, 500), dtype=np.float32)
        fixed[10:30, 10:30] = 1.0  # only 1 nucleus

        params = recommend_mod.recommend_mirage_params(fixed, moving)
        assert params["offset"] == 30
        assert params["num_matched"] == 0


class TestAggregateRecommendations:
    def test_takes_median(self):
        results = [
            {"offset": 10, "pad": 15, "smoothness_radius": 40,
             "pos_encoding_L": 6, "dissim_sigma": 20, "num_matched": 5},
            {"offset": 20, "pad": 25, "smoothness_radius": 60,
             "pos_encoding_L": 8, "dissim_sigma": 30, "num_matched": 8},
            {"offset": 30, "pad": 35, "smoothness_radius": 80,
             "pos_encoding_L": 4, "dissim_sigma": 40, "num_matched": 3},
        ]
        for r in results:
            r["displacement_magnitude"] = 10.0

        agg = recommend_mod.aggregate_recommendations(results)
        assert agg["offset"] == 20  # median of [10, 20, 30]
        assert agg["num_crops_with_matches"] == 3
