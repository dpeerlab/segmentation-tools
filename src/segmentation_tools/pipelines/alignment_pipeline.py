from pathlib import Path
from pydantic import BaseModel, Field, validator, PrivateAttr
import tifffile
from segmentation_tools.logger import logger
import segmentation_tools.utils as utils
import numpy as np
import imageio.v3 as iio
from skimage.metrics import structural_similarity as ssim
from skimage.transform import warp
from skimage import img_as_uint
from skimage.exposure import rescale_intensity

class AlignmentPipeline(BaseModel):
    if_file: Path = Field(..., description="Path to the input IF image")
    xenium_dir: Path = Field(..., description="Path to the Xenium output directory")
    output_dir: Path = Field(..., description="Path to store outputs")
    dapi_channel: int = Field(0, description="DAPI channel index (default: 0)")

    _intermediates_dir: Path = PrivateAttr()
    _if_dapi_tiff: tifffile.TiffFile = PrivateAttr()
    _xenium_dapi_tiff: tifffile.TiffFile = PrivateAttr()

    _if_dapi_img: np.ndarray = PrivateAttr()
    _xenium_dapi_img: np.ndarray = PrivateAttr()

    _if_dapi_img_aligned: np.ndarray = PrivateAttr()
    _resolution_level: int = PrivateAttr()

    def model_post_init(self, __context) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._intermediates_dir = self.output_dir / f"intermediates"
        self._intermediates_dir.mkdir(parents=True, exist_ok=True)

        self._setup_dirs()
        (self.output_dir / ".alignment_done").unlink(missing_ok = True)


    @validator("if_file")
    def check_if_file_exists(cls, v: Path):
        if not v.exists():
            raise FileNotFoundError(f"Input file does not exist: {v}")
        return v

    @validator("xenium_dir")
    def check_xenium_dir_exists(cls, v: Path):
        if not v.exists():
            raise FileNotFoundError(f"Xenium output directory does not exist: {v}")
        return v

    def __del__(self):
        if hasattr(self, "_if_dapi_tiff"):
            try:
                self._if_dapi_tiff.close()
                logger.debug("Closed IF DAPI TIFF")
            except Exception:
                pass

        if hasattr(self, "_xenium_dapi_tiff"):
            try:
                self._xenium_dapi_tiff.close()
                logger.debug("Closed Xenium DAPI TIFF")
            except Exception:
                pass

    def _setup_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._intermediates_dir = self.output_dir / "intermediates"
        self._intermediates_dir.mkdir(parents=True, exist_ok=True)

    def _convert_if_needed(self):
        if not utils.is_tiff_file(self.if_file):
            logger.info(f"Converting {self.if_file} to TIFF format")
            utils.convert_to_tiff(
                input_path=self.if_file, output_root=self._intermediates_dir
            )
            logger.success(
                f"Converted {self.if_file.name} to TIFF in {self._intermediates_dir}"
            )
            self.if_file = self._intermediates_dir / f"{self.if_file.stem}.tiff"
        else:
            logger.info(f"Input file is already a TIFF: {self.if_file}")
        return

    def _find_best_level(self):
        levels = self._if_dapi_tiff.series[0].levels  # list of TiffPageSeries
        best_level = 0  # start with highest resolution by default
        for i in reversed(range(len(levels))):  # start from lowest resolution
            shape = levels[i].shape
            if shape[-2] > 1024 and shape[-1] > 1024:
                best_level = i
                break
        return best_level


    def _load_tiff_objects(self):
        try:
            self._if_dapi_tiff = tifffile.TiffFile(self.if_file)
            logger.info(f"Loaded IF TIFF file: {self.if_file}")
        except:
            raise FileNotFoundError(f"Failed to load IF TIFF file: {self.if_file}")
        
        self._resolution_level = self._find_best_level()
        logger.info(f"Using level {self._resolution_level} for alignment")

        xenium_dapi_path = self.xenium_dir / "morphology.ome.tif"
        if not xenium_dapi_path.exists():
            raise FileNotFoundError(
                f"Xenium Morphology tiff file not found: {xenium_dapi_path}"
            )

        try:
            self._xenium_dapi_tiff = tifffile.TiffFile(xenium_dapi_path)
            logger.info(f"Loaded Xenium DAPI TIFF file: {xenium_dapi_path}")
        except:
            raise FileNotFoundError(
                f"Failed to load Xenium DAPI TIFF file: {xenium_dapi_path}"
            )
    
        self._if_dapi_img = utils.normalize(
            self._if_dapi_tiff.series[0]
            .levels[self._resolution_level]
            .asarray()[self.dapi_channel, :, :]
        )
        logger.info("Normalized IF DAPI image")


        output_path = self._intermediates_dir / "normalized_if_dapi.png"
        iio.imwrite(
            output_path, self._if_dapi_img
        )

        self._xenium_dapi_img = utils.normalize(
            self._xenium_dapi_tiff.series[0]
            .levels[self._resolution_level]
            .asarray()[self.dapi_channel, :, :]
        )

        output_path = self._intermediates_dir / "normalized_xenium_dapi.png"
        iio.imwrite(
            output_path, self._xenium_dapi_img
        )
        logger.info("Normalized Xenium DAPI image")
        
        return
    
    def _get_transformation_matrix(self):
        """
        Find the best alignment between IF and Xenium DAPI images using SIFT.
        Returns the aligned image, transform name, and score.
        """
        logger.info("Finding best SIFT between downsampled IF and Xenium DAPI images")
        tm_sift_if_to_xenium = utils.find_best_sift(
            mvg_img=self._if_dapi_img,
            fxd_img=self._xenium_dapi_img
        )

        if_level_tm_ds = utils.get_level_transform(
            tiff_object = self._if_dapi_tiff, level_to = self._resolution_level
        )

        xenium_level_tm_ds = utils.get_level_transform(
            tiff_object = self._xenium_dapi_tiff, level_to = self._resolution_level
        )

        tm_xenium_to_if = (
            xenium_level_tm_ds + \
            tm_sift_if_to_xenium.inverse + \
            if_level_tm_ds.inverse
        )

        return tm_xenium_to_if
    
### RUN FUNCTIONS ###

    def run(self):
        logger.info("Starting segmentation pipeline")
        self._convert_if_needed()
        self._load_tiff_objects()

        tm_xenium_to_if = self._get_transformation_matrix()
        self._if_dapi_img_aligned = warp(
            self._if_dapi_img,
            inverse_map=tm_xenium_to_if.inverse,
            output_shape=self._xenium_dapi_img.shape,
            preserve_range=True,
        )

        
        rescaled = rescale_intensity(self._if_dapi_img_aligned, in_range='image', out_range=(0, 1))
        output_path = self.output_dir / "if_dapi_aligned.tiff"
        iio.imwrite(output_path, rescaled)
        logger.success(f"Aligned IF DAPI image saved to {output_path}")

        # Save as NPY
        npy_path = self.output_dir / "if_dapi_aligned.npy"
        np.save(npy_path, rescaled)
        logger.success(f"Aligned IF DAPI image also saved as NPY to {npy_path}")

        # Normalize and save as PNG
        img_png = rescale_intensity(self._if_dapi_img_aligned, out_range=(0, 255)).astype(np.uint8)
        png_path = self.output_dir / "if_dapi_aligned.png"
        iio.imwrite(png_path, img_png)
        logger.success(f"Aligned IF DAPI image also saved as PNG to {png_path}")
        (self.output_dir / ".alignment_done").touch()
