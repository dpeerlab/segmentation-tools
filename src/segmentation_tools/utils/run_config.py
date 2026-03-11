"""Load and validate pipeline run configuration from YAML or CLI args."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sys

from loguru import logger


@dataclass
class SlurmConfig:
    conda_env: str = "contamination"
    partition: str = "peerd"
    time: str = "23:00:00"
    mem: str = "500G"
    gpus: int = 1
    cpus: int = 2


@dataclass
class RunConfig:
    # Required
    job_title: str = ""
    fixed_file: Path = Path()
    moving_file: Path = Path()
    output_root: Path = Path()

    # Channels
    fixed_dapi_channel: int = 0
    moving_dapi_channel: int = 1
    cellpose_dapi_channel: int = 1
    cellpose_membrane_channel: int = 0

    # MIRAGE (None = use auto-recommendation from step 5b)
    mirage_batch_size: Optional[int] = None
    mirage_learning_rate: Optional[float] = None
    mirage_num_steps: Optional[int] = None

    # Step range
    start_step: int = 1
    end_step: int = 9

    # SLURM
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_root / self.job_title / ".checkpoints"

    @property
    def results_dir(self) -> Path:
        return self.output_root / self.job_title / "results"


def load_config(config_path: Path) -> RunConfig:
    """Load a RunConfig from a YAML file."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML is required for config files: pip install pyyaml")
        sys.exit(1)

    with open(config_path) as f:
        data = yaml.safe_load(f)

    cfg = RunConfig()
    cfg.job_title = str(data["job_title"])
    cfg.fixed_file = Path(data["fixed_file"])
    cfg.moving_file = Path(data["moving_file"])
    cfg.output_root = Path(data["output_root"])

    cfg.fixed_dapi_channel = int(data.get("fixed_dapi_channel", 0))
    cfg.moving_dapi_channel = int(data.get("moving_dapi_channel", 1))
    cfg.cellpose_dapi_channel = int(data.get("cellpose_dapi_channel", 1))
    cfg.cellpose_membrane_channel = int(data.get("cellpose_membrane_channel", 0))
    cfg.mirage_batch_size = data.get("mirage_batch_size")
    cfg.mirage_learning_rate = data.get("mirage_learning_rate")
    cfg.mirage_num_steps = data.get("mirage_num_steps")

    cfg.start_step = int(data.get("start_step", 1))
    cfg.end_step = int(data.get("end_step", 9))

    slurm_data = data.get("slurm", {})
    cfg.slurm = SlurmConfig(
        conda_env=slurm_data.get("conda_env", "contamination"),
        partition=slurm_data.get("partition", "peerd"),
        time=slurm_data.get("time", "23:00:00"),
        mem=slurm_data.get("mem", "500G"),
        gpus=int(slurm_data.get("gpus", 1)),
        cpus=int(slurm_data.get("cpus", 2)),
    )

    return cfg


def validate_config(cfg: RunConfig) -> bool:
    """Validate inputs upfront before submitting. Returns True if all checks pass."""
    errors = []
    warnings = []

    # Required fields
    if not cfg.job_title:
        errors.append("job_title is required")
    if not cfg.fixed_file or str(cfg.fixed_file) == ".":
        errors.append("fixed_file is required")
    if not cfg.moving_file or str(cfg.moving_file) == ".":
        errors.append("moving_file is required")
    if not cfg.output_root or str(cfg.output_root) == ".":
        errors.append("output_root is required")

    # File existence
    if cfg.fixed_file and not cfg.fixed_file.exists():
        errors.append(f"fixed_file not found: {cfg.fixed_file}")
    if cfg.moving_file and not cfg.moving_file.exists():
        errors.append(f"moving_file not found: {cfg.moving_file}")

    # Output root writable
    if cfg.output_root:
        if cfg.output_root.exists() and not cfg.output_root.is_dir():
            errors.append(f"output_root exists but is not a directory: {cfg.output_root}")

    # Check disk space (warn if < 200 GB free)
    if cfg.output_root and cfg.output_root.parent.exists():
        import shutil
        free_gb = shutil.disk_usage(cfg.output_root.parent).free / (1024 ** 3)
        if free_gb < 200:
            warnings.append(f"Low disk space: {free_gb:.0f} GB free at {cfg.output_root.parent}")

    # Check GPU availability
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            warnings.append("No GPU detected (nvidia-smi failed) — MIRAGE requires a GPU")
        else:
            gpus = [g.strip() for g in result.stdout.strip().splitlines() if g.strip()]
            logger.info(f"  GPU(s) available: {', '.join(gpus)}")
    except Exception:
        warnings.append("Could not check GPU availability")

    # Channel range checks (best-effort — needs tifffile to read metadata)
    if cfg.start_step <= 1 and cfg.fixed_file and cfg.fixed_file.exists():
        try:
            import tifffile
            with tifffile.TiffFile(str(cfg.fixed_file)) as tf:
                series = tf.series[0]
                axes = series.axes
                if "C" in axes:
                    n_channels = series.shape[axes.index("C")]
                    if cfg.fixed_dapi_channel >= n_channels:
                        errors.append(
                            f"fixed_dapi_channel={cfg.fixed_dapi_channel} but fixed image "
                            f"only has {n_channels} channels"
                        )
        except Exception:
            pass  # Don't block on metadata read failure

    # Print results
    for w in warnings:
        logger.warning(f"  {w}")

    if errors:
        for e in errors:
            logger.error(f"  {e}")
        return False

    return True
