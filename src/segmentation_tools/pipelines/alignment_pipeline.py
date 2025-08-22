import gc
import os
import sys
import shutil
from pathlib import Path

import numpy as np
import skimage
import tifffile
from pydantic import BaseModel, Field, PrivateAttr, validator
from pprint import pprint
import cupy as cp
import cucim.skimage as cs


import segmentation_tools.utils.sift_alignment_utils as sift_alignment_utils
import segmentation_tools.utils.image_utils as image_utils
import segmentation_tools.utils.convert_image_utils as convert_utils
import segmentation_tools.utils.transform_utils as transform_utils
import segmentation_tools.utils.poor_alignment_utils as poor_alignment_utils
import segmentation_tools.utils.mirage_utils as mirage_utils

from segmentation_tools.logger import logger


class AlignmentPipeline(BaseModel):
    ### PIPELINE CONFIGURATION ###
    fixed_file: Path = Field(..., description="Path to the fixed image file")
    moving_file: Path = Field(..., description="Path to the input moving image")
    output_dir: Path = Field(..., description="Path to store outputs")

    job_title: str = Field(
        default="output", description="Title for the job, used in output naming"
    )

    series_moving: int = Field(
        0, description="Series index for moving image (default: 0)"
    )
    series_fixed: int = Field(
        0, description="Series index for fixed image (default: 0)"
    )

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
        default=False,
        description="Whether to apply mirage correction to SIFT aligned images",
    )
    save_intermediate_outputs: bool = Field(
        default=False, description="Whether to save intermediate outputs"
    )

    _processed_tiff_dir: Path = PrivateAttr()
    _poor_regions_dir: Path = PrivateAttr()
    _rotations_dir: Path = PrivateAttr()
    _visualizations_dir: Path = PrivateAttr()

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

        self._visualizations_dir = self.output_dir / "visualizations"
        self._visualizations_dir.mkdir(parents=True, exist_ok=True)
        return

    ### ALIGNMENT HELPER FUNCTIONS ###

    def _load_normalized_dapi_images(
        self,
        moving_file,
        fixed_file,
        series_moving,
        series_fixed,
        level_moving,
        level_fixed,
        channel_moving,
    ):
        # Load and normalize images at downsampled levels
        dapi_moving_image = tifffile.imread(
            moving_file,
            series=series_moving,
            key=channel_moving,
            level=level_moving,
        )
        # Read and normalize fixed image at the specified levels
        dapi_fixed_image = tifffile.imread(
            fixed_file, series=series_fixed, level=level_fixed
        )

        logger.info("Loaded in images")
        logger.info(
            f"Moving shape {dapi_moving_image.shape}, Fixed shape {dapi_fixed_image.shape}"
        )

        dapi_moving_image_gpu = cp.asarray(dapi_moving_image)
        dapi_fixed_image_gpu = cp.asarray(dapi_fixed_image)

        # Normalize images
        normalized_moving_dapi_gpu = image_utils.normalize(dapi_moving_image_gpu)

        logger.info(f"Normalized moving, {normalized_moving_dapi_gpu.max()=}")
        normalized_fixed_dapi_gpu = image_utils.normalize(dapi_fixed_image_gpu)

        logger.info(f"Normalized fixed, {normalized_fixed_dapi_gpu.max()=}")

        self._garbage_collect_objects(
            dapi_moving_image,
            dapi_fixed_image,
            normalized_moving_dapi_gpu,
            normalized_fixed_dapi_gpu,
        )

        return cp.asnumpy(normalized_moving_dapi_gpu), cp.asnumpy(
            normalized_fixed_dapi_gpu
        )

    def _get_sift_transform_and_warp(
        self,
        normalized_moving_dapi_ds: np.ndarray,
        normalized_fixed_dapi_ds: np.ndarray,
        rotations_dir: os.PathLike = None,
        save_tiff_dir: os.PathLike = None,
        draw_matches: bool = False,
    ):
        """Get SIFT features and warp moving image at downsampled resolution."""
        # Find SIFT transform between downsampled images
        tm_sift = sift_alignment_utils.find_best_sift(
            moving_image=normalized_moving_dapi_ds,
            fixed_image=normalized_fixed_dapi_ds,
            save_img_dir=rotations_dir,
            draw_matches=draw_matches,
        )
        # Warp downsampled
        logger.info("Warping moving DAPI at downsampled resolution")
        normalized_warped_dapi_ds = skimage.transform.warp(
            normalized_moving_dapi_ds,
            tm_sift,
            output_shape=normalized_fixed_dapi_ds.shape,
            preserve_range=True,
        )

        if save_tiff_dir is not None:
            image_utils.save_image(  ## TODO: Save only if save_intermediate_outputs is True
                image=normalized_warped_dapi_ds,
                output_file_path=self._processed_tiff_dir
                / f"normalized_warped_moving_dapi_ds.tiff",
                description="Matched downsample warped moving DAPI",
            )

            image_utils.save_full_overlay(
                fixed_image=normalized_fixed_dapi_ds,
                moving_image=normalized_warped_dapi_ds,
                output_file_path=self._visualizations_dir / f"warped_overlay_ds.png",
                title="Normalized Warped Overlay Downsampled",
            )

        return tm_sift

    def _get_combined_transform(
        self,
        moving_file,
        fixed_file,
        tm_sift,
        sift_level_fixed,
        sift_level_moving,
        high_res_level,
    ):
        """Get combined transform for high resolution alignment."""
        tm_combined = (
            transform_utils.get_level_transform(
                moving_file,
                level_to=sift_level_moving,
                level_from=high_res_level,
            ).params
            @ tm_sift.params
            @ transform_utils.get_level_transform(
                fixed_file,
                level_to=high_res_level,
                level_from=sift_level_fixed,
            ).params
        )
        return tm_combined

    def _warp_high_res_image_all_channels(
        self,
        moving_file,
        fixed_file,
        series_moving,
        series_fixed,
        high_res_level,
        tm_combined,
        nuclei_channel_moving,
        save_img_dir=None,
        apply_mirage_correction=False,
    ):
        num_moving_channels = image_utils.get_num_channels(
            tiff_file=moving_file, series=series_moving, level=high_res_level
        )

        warped_channels = []
        for channel_idx in range(num_moving_channels):
            moving_image = tifffile.imread(
                moving_file,
                series=series_moving,
                level=high_res_level,
                key=channel_idx,
            )

            fixed_dapi_shape = transform_utils.get_shape_at_level(
                tiff_path=fixed_file, level=high_res_level, series=series_fixed
            )

            # moving_image = cp.asarray(moving_image)
            self._garbage_collect_objects(moving_image)

            # moving_image_gpu = image_utils.normalize(moving_image)
            logger.info(
                f"Loaded normalized image high res for warping channel {channel_idx}"
            )

            warped = skimage.transform.warp(
                moving_image,
                tm_combined,
                output_shape=fixed_dapi_shape,
                preserve_range=True,
                order=5,
            )

            # self._garbage_collect_objects(moving_image)

            # warped = cp.asnumpy(warped_gpu)

            # self._garbage_collect_objects(warped)

            logger.info(f"Warped channel {channel_idx} at high resolution")

            if channel_idx == nuclei_channel_moving:
                image_utils.save_image(
                    image=warped,
                    output_file_path=self._processed_tiff_dir
                    / f"matched_warped_moving_dapi_high_res.tiff",
                    description="Matched high-res warped moving DAPI",
                )

                if apply_mirage_correction:
                    normalized_fixed_dapi = image_utils.normalize(
                        cp.asarray(
                            tifffile.imread(
                                fixed_file,
                                series=series_fixed,
                                level=high_res_level,
                            )
                        )
                    )

                    poor_alignment_utils.compute_ssim_tile_heatmap(
                        fixed_image = cp.asnumpy(normalized_fixed_dapi),
                        moving_image = warped,
                        save_img_dir=self._poor_regions_dir,
                        prefix = "pre_mirage"
                    )

                    warped = mirage_utils.run_mirage(
                        moving_image=warped,
                        fixed_image=normalized_fixed_dapi,
                        save_img_dir=save_img_dir,
                    )

                    poor_alignment_utils.compute_ssim_tile_heatmap(
                        fixed_image = cp.asnumpy(normalized_fixed_dapi),
                        moving_image = warped,
                        save_img_dir=self._poor_regions_dir,
                        prefix = "post_mirage"
                    )

            warped_channels.append(warped)

        warped_moving_stack = np.stack(warped_channels, axis=0)  # (C, H, W)
        return warped_moving_stack

    def _garbage_collect_objects(self, *objects):
        for obj in objects:
            del obj
        gc.collect()

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        return

    ### RUN FUNCTIONS ###
    def run(self):
        logger.info(
            f"Starting alignment pipeline at high resolution level {self.high_res_level} for job {self.job_title}."
        )

        logger.info(
            f"Input Moving file: {self.moving_file}\nfixed file: {self.fixed_file}"
        )

        # Step 1: Convert moving file to TIFF if needed
        self.moving_file = convert_utils.convert_to_tiff_if_needed(
            input_file_path=self.moving_file,
            intermediates_dir=self._processed_tiff_dir,
        )

        shutil.copy(
            self.fixed_file,
            self._processed_tiff_dir / "fixed_image.ome.tiff",
        )

        # Step 2: Determine downsampled levels for alignment
        sift_level_moving, sift_level_fixed = (
            sift_alignment_utils.get_best_common_level(
                moving_file=self.moving_file,
                fixed_file=self.fixed_file,
                min_size=1500,  # Minimum size for downsampled images
            )
        )

        logger.info(
            f"Determined downsampled levels for alignment: moving={sift_level_moving}, fixed={sift_level_fixed}"
        )

        # Step 3: Load and normalize images at downsampled levels
        normalized_moving_dapi_ds_cpu, normalized_fixed_dapi_ds_cpu = (
            self._load_normalized_dapi_images(
                moving_file=self.moving_file,
                fixed_file=self.fixed_file,
                series_moving=self.series_moving,
                series_fixed=self.series_fixed,
                level_moving=sift_level_moving,
                level_fixed=sift_level_fixed,
                channel_moving=self.nuclei_channel_moving,
            )
        )

        # Step 4: Get SIFT transform and warp moving image at downsampled resolution
        tm_sift = self._get_sift_transform_and_warp(
            rotations_dir=self._rotations_dir,
            normalized_moving_dapi_ds=normalized_moving_dapi_ds_cpu,
            normalized_fixed_dapi_ds=normalized_fixed_dapi_ds_cpu,
            save_tiff_dir=self._processed_tiff_dir,
            draw_matches=False,
        )

        # Step 5: Get combined transform for high resolution alignment
        tm_combined = self._get_combined_transform(
            moving_file=self.moving_file,
            fixed_file=self.fixed_file,
            tm_sift=tm_sift,
            sift_level_fixed=sift_level_fixed,
            sift_level_moving=sift_level_moving,
            high_res_level=self.high_res_level,
        )

        self._garbage_collect_objects(
            normalized_moving_dapi_ds_cpu, normalized_fixed_dapi_ds_cpu, tm_sift
        )
        logger.info("Starting warping of all channels")

        # Step 8: Warp moving image at high resolution
        warped_moving_stack = self._warp_high_res_image_all_channels(
            moving_file=self.moving_file,
            fixed_file=self.fixed_file,
            series_moving=self.series_moving,
            series_fixed=self.series_fixed,
            high_res_level=self.high_res_level,
            tm_combined=tm_combined,
            nuclei_channel_moving=self.nuclei_channel_moving,
            apply_mirage_correction=self.apply_mirage_correction,
            save_img_dir=self._processed_tiff_dir,
        )

        # Save warped moving stack
        moving_num_levels = image_utils.get_num_levels(
            tiff_file=self.moving_file,
            series=self.series_moving,
            level=self.high_res_level,
        )

        levels_to_save = moving_num_levels - self.high_res_level

        image_utils.save_pyramidal_tiff_multi_channel(
            image_stacked=warped_moving_stack,
            output_file_path=self._processed_tiff_dir
            / "all_channels_moving_warped.ome.tiff",
            description="High-res all_channels moving and fixed",
            num_levels=levels_to_save,
        )

        sys.exit(0)
