from pathlib import Path
from pydantic import BaseModel, Field, validator, PrivateAttr
import tifffile
from segmentation_tools.logger import logger
from segmentation_tools.utils import convert_to_tiff, is_tiff_file
import numpy as np

LEVEL = 7


class SegmentationPipeline(BaseModel):
    if_file: Path = Field(..., description="Path to the input IF image")
    xenium_dir: Path = Field(..., description="Path to the Xenium output directory")
    output_dir: Path = Field(..., description="Path to store outputs")
    dapi_channel: int = Field(0, description="DAPI channel index (default: 0)")

    _intermediates_dir: Path = PrivateAttr()
    _if_dapi_tiff: tifffile.Tifffile = PrivateAttr()
    _xenium_dapi_tiff: tifffile.Tifffile = PrivateAttr()

    _if_dapi_img: np.ndarray = PrivateAttr()
    _xenium_dapi_img: np.ndarray = PrivateAttr()

    def model_post_init(self, __context) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._intermediates_dir = self.output_dir / "intermediates"
        self._intermediates_dir.mkdir(parents=True, exist_ok=True)

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

    def __post_init_post_parse__(self):
        self._setup_dirs()

    def _setup_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._intermediates_dir = self.output_dir / "intermediates"
        self._intermediates_dir.mkdir(parents=True, exist_ok=True)

    def convert_if_needed(self):
        if not is_tiff_file(self.if_file):
            logger.info(f"Converting {self.if_file} to TIFF format")
            convert_to_tiff(
                input_path=self.if_file, output_root=self._intermediates_dir
            )
            logger.success(
                f"Converted {self.if_file.name} to TIFF in {self._intermediates_dir}"
            )
            self.if_file = self._intermediates_dir / f"{self.if_file.stem}.tiff"
        else:
            logger.info(f"Input file is already a TIFF: {self.if_file}")

    def load_tiff_objects(self):
        with open(self.if_file, "rb") as f:
            self.if_file = tifffile.TiffFile(f)
            logger.info(f"Loaded IF TIFF file: {self.if_file}")

        xenium_dapi_path = self.xenium_dir / "morphology.ome.tiff"
        if not xenium_dapi_path.exists():
            raise FileNotFoundError(
                f"Xenium Morphology tiff file not found: {xenium_dapi_path}"
            )

        with open(xenium_dapi_path, "rb") as f:
            self._xenium_dapi_tiff = tifffile.TiffFile(f)
            logger.info(f"Loaded Xenium DAPI TIFF file: {xenium_dapi_path}")

        self._if_dapi_img = (
            self.if_dapi_tiff.series[0].levels[LEVEL].asarray()[self.dapi_channel, :, :]
        )
        self._xenium_dapi_img = (
            self._xenium_dapi_tiff.series[0]
            .levels[LEVEL]
            .asarray()[self.dapi_channel, :, :]
        )

        return

    def run(self):
        logger.info("Starting segmentation pipeline")
        self.convert_if_needed()
