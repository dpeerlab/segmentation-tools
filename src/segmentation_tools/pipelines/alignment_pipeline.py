import gc
import os
import shutil
from pathlib import Path

import numpy as np
import skimage
import tifffile
from pydantic import BaseModel, Field, PrivateAttr, validator

import segmentation_tools.utils.alignment_utils as alignment_utils
import segmentation_tools.utils.image_utils as image_utils
import segmentation_tools.utils.convert_image_utils as convert_utils
import segmentation_tools.utils.transform_utils as transform_utils

from segmentation_tools.logger import logger


class AlignmentPipeline(BaseModel):
    ### PIPELINE CONFIGURATION ###
    moving_file: Path = Field(..., description="Path to the input moving image")
    fixed_file: Path = Field(default=None, description="Path to the fixed image file")
    output_dir: Path = Field(..., description="Path to store outputs")
    job_title: str = Field(
        default="output", description="Title for the job, used in output naming"
    )

    nuclei_channel_moving: int = Field(
        1, description="DAPI channel index for moving image (default: 1)"
    )

    series_moving: int = Field(0, description="Series index for moving image (default: 0)")
    series_fixed: int = Field(0, description="Series index for fixed image (default: 0)")

    high_res_level: int = Field(
        0, description="High resolution level for alignment (default: 0)"
    )

    _processed_tiff_dir: Path = PrivateAttr()
    _poor_regions_dir: Path = PrivateAttr()
    _rotations_dir: Path = PrivateAttr()

    def model_post_init(self, __context) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_dirs()
        (self.output_dir / ".alignment_done").unlink(missing_ok=True)

        if self.xenium_file is None:
            self.xenium_file = (
                self.xenium_dir / "morphology_focus" / "morphology_focus_0000.ome.tif"
            )

    @validator("moving_file")
    def check_moving_file_exists(cls, v: Path):
        if not v.exists():
            raise FileNotFoundError(f"Input file does not exist: {v}")
        return v

    @validator("xenium_dir")
    def check_xenium_dir_exists(cls, v: Path):
        if not v.exists():
            raise FileNotFoundError(f"Xenium output directory does not exist: {v}")
        return v

    def _setup_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.output_dir / self.job_title

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._processed_tiff_dir = self.output_dir / "processed_tiffs"
        self._processed_tiff_dir.mkdir(parents=True, exist_ok=True)

        self._poor_regions_dir = self.output_dir / "poorly_aligned_regions"
        self._poor_regions_dir.mkdir(parents=True, exist_ok=True)

        self._rotations_dir = self.output_dir / "rotations"
        self._rotations_dir.mkdir(parents=True, exist_ok=True)

        return

    ### ALIGNMENT HELPER FUNCTIONS ###

    def _determine_alignment_levels(self):
        return alignment_utils.get_best_common_level(
            self.moving_file,
            self.xenium_file,
            min_size=1500,
        )
    
    def _load_matching_images(self, level_moving, level_fixed):
        # Load and normalize images at downsampled levels
        dapi_img_moving= tifffile.imread(
            self.moving_file,
            series=self.series_moving,
            key=self.nuclei_channel_moving,
            level=level_moving,
        )
        # Read and normalize Xenium image at the specified levels
        dapi_img_xenium = tifffile.imread(
            self.xenium_file, series=self.series_xenium, level=level_fixed
        )

        # Normalize images
        normalized_moving_dapi = image_utils.normalize(
            dapi_img_moving, return_float=True
        )
        normalized_xenium_dapi = image_utils.normalize(
            dapi_img_xenium, return_float=True
        )

        matched_moving_dapi, matched_xenium_dapi = (
            image_utils.match_image_histograms(
                normalized_moving_dapi,
                normalized_xenium_dapi,
            )
        )

        self._garbage_collect_objects(
            [
                dapi_img_moving,
                dapi_img_xenium,
                normalized_moving_dapi,
                normalized_xenium_dapi,
            ]
        )

        return matched_moving_dapi, matched_xenium_dapi

    def _get_sift_transform_and_warp(self, matched_moving_dapi_ds, matched_xenium_dapi_ds):
        # Find SIFT transform between downsampled images
        tm_sift = alignment_utils.find_best_sift(
            mvg_img=matched_moving_dapi_ds,
            fxd_img=matched_xenium_dapi_ds,
            save_img_dir=self._rotations_dir,
            draw_matches=False,
        )

        # Warp downsampled
        logger.info("Warping moving DAPI at downsampled resolution")
        matched_warped_moving_dapi_ds = skimage.transform.warp(
            matched_moving_dapi_ds,
            tm_sift,
            output_shape=matched_xenium_dapi_ds.shape,
            preserve_range=True,
        )

        # Save warped warped stacked image to tiff
        image_utils.save_image(
            image=matched_warped_moving_dapi_ds,
            output_file_path=self._processed_tiff_dir / f"matched_warped_moving_dapi_ds.tiff",
            description="Matched downsample warped moving DAPI",
        )

        # Save warped warped stacked image to tiff
        image_utils.save_image(
            image=matched_xenium_dapi_ds,
            output_file_path=self._processed_tiff_dir / "matched_xenium_dapi_ds.tiff",
            description="Matched downsampled Xenium DAPI",
        )

        poorly_aligned_regions = alignment_utils.find_poorly_aligned_regions(
            fixed_img=matched_xenium_dapi_ds,
            moving_img=matched_warped_moving_dapi_ds,
            win_size=11,
            min_brightness_factor=0.15,
            min_area_factor=5e-5,
            ssim_bounds=(0.0, 0.7),
            output_file_path=self._poor_regions_dir / "poorly_aligned_regions_mask",
        )

        image_utils.save_full_overlay(
            image_fixed=matched_xenium_dapi_ds,
            image_moving=matched_warped_moving_dapi_ds,
            boxes=poorly_aligned_regions,
            output_file_path=self.output_dir / "dapi_downsampled_overlay",
            plot_axis=True,
        )

        return (
            tm_sift,
            poorly_aligned_regions,
        )

    def _garbage_collect_objects(self, *objects):
        for obj in objects:
            del obj
        gc.collect()


    ### RUN FUNCTIONS ###
    def run(self):
        logger.info(
            f"Starting {self.job_title} alignment pipeline at high resolution level {self.high_res_level}"
        )

        logger.info(f"Input Moving file: {self.moving_file}\nXenium file: {self.xenium_file}")

        self.moving_file = convert_utils.convert_to_tiff_if_needed(
            input_file_path=self.moving_file,
            intermediates_dir=self._processed_tiff_dir,
        )

        # Get downsampled levels for alignment
        level_fixed, level_moving = self._determine_alignment_levels()
        logger.info(
            f"Using levels for downsampled alignment - Fixed: {level_fixed}, Moving: {level_moving}"
        )

        matched_moving_dapi_ds, matched_xenium_dapi_ds = self._load_matching_images(
            level_moving=level_moving, level_fixed=level_fixed
        )

        tm_sift, poorly_aligned_regions = self._get_sift_transform_and_warp(
            matched_moving_dapi_ds, matched_xenium_dapi_ds
        )


        # Free memory
        self._garbage_collect_objects(
            [
                matched_moving_dapi_ds,
                matched_xenium_dapi_ds,
            ]
        )

        # Get transform for high resolution
        tm_combined = (
            transform_utils.get_level_transform(
                self.xenium_file, level_to=level_moving, level_from=self.high_res_level
            ).params
            @ tm_sift.params
            @ transform_utils.get_level_transform(
                self.xenium_file, level_to=self.high_res_level, level_from=level_fixed
            ).params
        )

        # Warp DAPI image at high resolution
        matched_dapi_img_if_high_res, matched_dapi_img_xenium_high_res = self._load_matching_images(self.high_res_level, self.high_res_level)

        matched_dapi_img_if_warped_high_res = skimage.transform.warp(
            matched_dapi_img_if_high_res,
            tm_combined,
            output_shape=matched_dapi_img_xenium_high_res.shape,
            preserve_range=True,
        )

        transformed_poorly_aligned_regions = transform_utils.transform_polygons_to_high_res(
            polygons=poorly_aligned_regions,
            transform=transform_utils.get_level_transform(
                self.xenium_file, level_to=level_fixed, level_from=self.high_res_level
            ),
        )

        image_utils.save_poorly_aligned_cropped_overlays(
            image_fixed=matched_dapi_img_xenium_high_res,
            image_moving=matched_dapi_img_if_warped_high_res,
            boxes=transformed_poorly_aligned_regions,
            output_file_path=self._poor_regions_dir / "dapi_high_res",
        )

        image_utils.save_good_region_control_overlay(
            image_fixed=matched_dapi_img_xenium_high_res,
            image_moving=matched_dapi_img_if_warped_high_res,
            boxes=transformed_poorly_aligned_regions,
            output_file_path=self._poor_regions_dir / "dapi_high_res",
            ssim_threshold=0.8,
            max_attempts=100,
        )

        with tifffile.TiffFile(self.if_file) as tif:
            if_num_levels = len(tif.series[self.series_if].levels) - self.high_res_level

        logger.info(f"Saving high-res DAPI IF and Xenium overlay with {if_num_levels} levels")
        image_utils.save_pyramidal_tiff_two_channel(
            img1=matched_dapi_img_xenium_high_res,
            img2=matched_dapi_img_if_warped_high_res,
            output_file_path=self._processed_tiff_dir / "dapi_if_warped_xenium_high_res_overlay.tiff",
            description="High-res warped DAPI IF and Xenium overlay",
            n_levels=if_num_levels
        )

        exit()


        membrane_high_res = tifffile.imread(
            self.if_file,
            series=self.series_if,
            key=self.membrane_channel_if,
            level=self.high_res_level,
        )

        normalized_if_membrane_high_res = image_utils.normalize(
            membrane_high_res, return_float=True
        )

        matched_if_membrane_high_res, _ = image_utils.match_image_histograms(
            normalized_if_membrane_high_res,
            matched_dapi_img_if_warped_high_res,
        )

        # Warp membrane image at high resolution
        matched_membrane_if_warped_high_res = skimage.transform.warp(
            matched_if_membrane_high_res,
            tm_combined,
            output_shape=matched_dapi_img_if_warped_high_res.shape,
            preserve_range=True,
        )

        # Stack warped DAPI and membrane images to tiff
        if_warped_stacked_high_res = np.stack(
            [matched_dapi_img_if_high_res, matched_membrane_if_warped_high_res], axis=-1
        )


        # Save warped warped stacked image
        image_utils.save_image(
            image=if_warped_stacked_high_res,
            output_file_path=self._processed_tiff_dir / "stacked_if_warped_dapi_membrane.tiff",
            description="High-res warped IF stacked",
        )

        # Free memory
        self._garbage_collect_objects(
            [
                matched_dapi_img_if_warped_high_res,
                matched_membrane_if_warped_high_res,
                if_warped_stacked_high_res,
            ]
        )
