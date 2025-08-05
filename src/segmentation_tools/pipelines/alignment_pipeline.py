import gc
import os
import shutil
from pathlib import Path

import numpy as np
import skimage
import tifffile
from pydantic import BaseModel, Field, PrivateAttr, validator
from pprint import pprint

import segmentation_tools.utils.sift_alignment_utils as sift_alignment_utils
import segmentation_tools.utils.image_utils as image_utils
import segmentation_tools.utils.convert_image_utils as convert_utils
import segmentation_tools.utils.transform_utils as transform_utils
import segmentation_tools.utils.poor_alignment_utils as poor_alignment_utils

from segmentation_tools.logger import logger


class AlignmentPipeline(BaseModel):
    ### PIPELINE CONFIGURATION ###
    fixed_file: Path = Field(..., description="Path to the fixed image file")
    moving_file: Path = Field(..., description="Path to the input moving image")
    output_dir: Path = Field(..., description="Path to store outputs")

    job_title: str = Field(
        default="output", description="Title for the job, used in output naming"
    )

    series_moving: int = Field(0, description="Series index for moving image (default: 0)")
    series_fixed: int = Field(0, description="Series index for fixed image (default: 0)")

    nuclei_channel_moving: int = Field(
        ..., description="DAPI channel index for moving image (default: 1)"
    )

    high_res_level: int = Field(
        0, description="High resolution level for alignment (default: 0)"
    )
    
    find_poorly_aligned_regions: bool = Field(
        default=False, description="Whether to find poorly aligned regions"
    )

    apply_mirage_correction: bool = Field(
        default=False, description="Whether to apply mirage correction to SIFT aligned images"
    )
    save_intermediate_outputs: bool = Field(
        default=False, description="Whether to save intermediate outputs"
    )

    _processed_tiff_dir: Path = PrivateAttr()
    _poor_regions_dir: Path = PrivateAttr()
    _rotations_dir: Path = PrivateAttr()


    def model_post_init(self, __context) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_dirs()
        (self.output_dir / ".alignment_done").unlink(missing_ok=True)

        if self.fixed_file is None:
            self.fixed_file = (
                self.fixed_dir / "morphology_focus" / "morphology_focus_0000.ome.tif"
            )

    @validator("moving_file")
    def check_moving_file_exists(cls, v: Path):
        if not v.exists():
            raise FileNotFoundError(f"Input file does not exist: {v}")
        return v

    @validator("fixed_file")
    def check_fixed_file_exists(cls, v: Path):
        if not v.exists():
            raise FileNotFoundError(f"Input file does not exist: {v}")
        return v
    
    def _setup_dirs(self):
        """Set up output directories."""
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
        """ Determine appropriate downsampled levels for alignment."""
        return sift_alignment_utils.get_best_common_level(
            self.moving_file,
            self.fixed_file,
            min_size=1500,
        )
    
    def _load_matching_images(self, level_moving, level_fixed, channel_moving):
        # Load and normalize images at downsampled levels
        dapi_img_moving= tifffile.imread(
            self.moving_file,
            series=self.series_moving,
            key=channel_moving,
            level=level_moving,
        )
        # Read and normalize fixed image at the specified levels
        dapi_img_fixed = tifffile.imread(
            self.fixed_file, series=self.series_fixed, level=level_fixed
        )

        # Normalize images
        normalized_moving_dapi = image_utils.normalize(
            dapi_img_moving, return_float=True
        )
        normalized_fixed_dapi = image_utils.normalize(
            dapi_img_fixed, return_float=True
        )

        matched_moving_dapi, matched_fixed_dapi = (
            image_utils.match_image_histograms(
                normalized_moving_dapi,
                normalized_fixed_dapi,
            )
        )

        self._garbage_collect_objects(
            [
                dapi_img_moving,
                dapi_img_fixed,
                normalized_moving_dapi,
                normalized_fixed_dapi,
            ]
        )

        return matched_moving_dapi, matched_fixed_dapi
    
    def _get_sift_transform_and_warp(self, matched_moving_dapi_ds, matched_fixed_dapi_ds):
        """ Get SIFT features and warp moving image at downsampled resolution."""
        # Find SIFT transform between downsampled images
        tm_sift = sift_alignment_utils.find_best_sift(
            mvg_img=matched_moving_dapi_ds,
            fxd_img=matched_fixed_dapi_ds,
            save_img_dir=self._rotations_dir, ## TODO: Save only if save_intermediate_outputs is True
            draw_matches=False,
        )

        # Warp downsampled
        logger.info("Warping moving DAPI at downsampled resolution")
        matched_warped_moving_dapi_ds = skimage.transform.warp(
            matched_moving_dapi_ds,
            tm_sift,
            output_shape=matched_fixed_dapi_ds.shape,
            preserve_range=True,
        )

    
        image_utils.save_image( ## TODO: Save only if save_intermediate_outputs is True
            image=matched_warped_moving_dapi_ds,
            output_file_path=self._processed_tiff_dir / f"matched_warped_moving_dapi_ds.tiff",
            description="Matched downsample warped moving DAPI",
        )

        return tm_sift, matched_warped_moving_dapi_ds
    
    def _get_combined_transform(self, tm_sift, sift_level_fixed, sift_level_moving):
        """ Get combined transform for high resolution alignment."""
        tm_combined = (
            transform_utils.get_level_transform(
                self.moving_file, level_to=sift_level_moving, level_from=sift_level_moving,
            ).params
            @ tm_sift.params
            @ transform_utils.get_level_transform(
                self.fixed_file, level_to=sift_level_fixed, level_from=sift_level_fixed,
            ).params
        )
        return tm_combined
    
    def _find_poorly_aligned_regions(self, fixed_img, moving_img, sift_level_fixed):
        poorly_aligned_regions = poor_alignment_utils.find_poorly_aligned_regions(
            fixed_img=fixed_img,
            moving_img=moving_img,
            win_size=11,
            min_brightness_factor=0.15,
            min_area_factor=5e-5,
            ssim_bounds=(0.0, 0.7),
            output_file_path=self._poor_regions_dir / "poorly_aligned_regions_mask",
        )

        transformed_poorly_aligned_regions = transform_utils.transform_polygons_to_high_res(
            polygons=poorly_aligned_regions,
            transform=transform_utils.get_level_transform(
                self.fixed_file, level_to=sift_level_fixed, level_from=self.high_res_level
            ),
        )

        return poorly_aligned_regions, transformed_poorly_aligned_regions

    def _garbage_collect_objects(self, *objects):
        for obj in objects:
            del obj
        gc.collect()


    ### RUN FUNCTIONS ###
    def run(self):
        logger.info(
            f"Starting alignment pipeline at high resolution level {self.high_res_level} for job {self.job_title}."
        )

        logger.info(f"Input Moving file: {self.moving_file}\nfixed file: {self.fixed_file}")

        # Step 1: Convert moving file to TIFF if needed
        self.moving_file = convert_utils.convert_to_tiff_if_needed(
            input_file_path=self.moving_file,
            intermediates_dir=self._processed_tiff_dir,
        )

        # Step 2: Convert fixed file to TIFF if needed
        sift_level_fixed, sift_level_moving = self._determine_alignment_levels()
        logger.info(
            f"Using levels for downsampled alignment - Fixed: {sift_level_fixed}, Moving: {sift_level_moving}"
        )

        # Step 3: Load and normalize images at downsampled levels
        matched_moving_dapi_ds, matched_fixed_dapi_ds = self._load_matching_images(
            sift_level_moving=sift_level_moving, sift_level_fixed=sift_level_fixed, channel_moving=self.nuclei_channel_moving
        )

        # Step 4: Get SIFT transform and warp moving image at downsampled resolution
        tm_sift, matched_warped_moving_dapi_ds = self._get_sift_transform_and_warp(
            matched_moving_dapi_ds, matched_fixed_dapi_ds
        )

        # Step 5: Get combined transform for high resolution alignment
        tm_combined = self._get_combined_transforms(
            sift_level_fixed=sift_level_fixed,
            sift_level_moving=sift_level_moving,
            tm_sift=tm_sift,
        )




        # # Step 6: (Optional) Find poorly aligned regions
        # poorly_aligned_regions = None
        # if self.find_poorly_aligned_regions:
        #     logger.info("Finding poorly aligned regions")
        #     poorly_aligned_regions, transformed_poorly_aligned_regions = self._find_poorly_aligned_regions(
        #         fixed_img=matched_fixed_dapi_ds,
        #         moving_img=matched_warped_moving_dapi_ds,
        #         sift_level_fixed=sift_level_fixed,
        #     )
    
        # image_utils.save_full_overlay( ### TODO: Save only if save_intermediate_outputs is True
        #     image_fixed=matched_fixed_dapi_ds,
        #     image_moving=matched_warped_moving_dapi_ds,
        #     boxes=poorly_aligned_regions,
        #     output_file_path=self.output_dir / "dapi_downsampled_overlay",
        #     plot_axis=True,
        # )

        # Free memory
        self._garbage_collect_objects(
            [
                matched_moving_dapi_ds,
                matched_fixed_dapi_ds,
                matched_warped_moving_dapi_ds
            ]
        )


        # Step 7: Load high resolution images
        matched_dapi_img_moving_high_res, matched_dapi_img_fixed_high_res = self._load_matching_images(self.high_res_level, self.high_res_level)

        # Step 10: Save the high resolution warped images and xenium 
        with tifffile.TiffFile(self.moving_file) as tif:
            moving_num_levels = len(tif.series[self.series_moving].levels) - self.high_res_level

        logger.info(f"Saving high-res DAPI moving and fixed overlay with {moving_num_levels} levels")

        # Step 8: Warp moving image at high resolution
        matched_moving_image_warped = skimage.transform.warp(
            matched_dapi_img_moving_high_res,
            tm_combined,
            output_shape=matched_dapi_img_fixed_high_res.shape,
            preserve_range=True,
        )

        image_utils.save_pyramidal_tiff_from_high_res(
            image=matched_dapi_img_moving_warped_high_res,
            output_file_path=self._processed_tiff_dir / "dapi_moving_warped_fixed_high_res_overlay.tiff",
            description="High-res warped DAPI moving and fixed overlay",
            n_levels=moving_num_levels
        )




        # Step 9: (Optional) Save poorly aligned regions overlays at high resolution
        if self.find_poorly_aligned_regions:
            image_utils.save_poorly_aligned_cropped_overlays(
                image_fixed=matched_dapi_img_fixed_high_res,
                image_moving=matched_dapi_img_moving_warped_high_res,
                boxes=transformed_poorly_aligned_regions,
                output_file_path=self._poor_regions_dir / "dapi_high_res",
            )

            image_utils.save_good_region_control_overlay(
                image_fixed=matched_dapi_img_fixed_high_res,
                image_moving=matched_dapi_img_moving_warped_high_res,
                boxes=transformed_poorly_aligned_regions,
                output_file_path=self._poor_regions_dir / "dapi_high_res",
                ssim_threshold=0.8,
                max_attempts=100,
            )



        # Free memory
        self._garbage_collect_objects(
            [
                matched_dapi_img_if_warped_high_res,
                matched_membrane_if_warped_high_res,
                if_warped_stacked_high_res,
            ]
        )
