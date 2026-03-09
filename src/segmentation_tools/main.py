"""CLI entry point for segmentation-tools.

Usage examples
--------------
Run from a config file (recommended):
    segmentation-tools run --config my_sample.yaml
    segmentation-tools submit --config my_sample.yaml

Run from CLI flags:
    segmentation-tools run -f fixed.tiff -m moving.nd2 -o /output -j my_sample

Resume from a specific step:
    segmentation-tools run --config my_sample.yaml --start-step 4
    segmentation-tools submit --config my_sample.yaml --start-step 4

Generate QC report after a run:
    segmentation-tools view -o /output -j my_sample

Print a sample config to stdout:
    segmentation-tools init-config

Step numbers
------------
  0   Setup directories
  1   Convert to TIFF
  3   Preprocess high-res images (DAPI normalization + Otsu)
  4   VALIS alignment (rigid/affine)
  5   Warp moving DAPI with linear transform
  51  Recommend MIRAGE parameters
  6   MIRAGE non-linear registration
  61  Evaluate MIRAGE alignment
  7   Warp all channels + build pyramid
  8   CellPose segmentation
  10  SSIM quality checks
  11  Convert masks to GeoDataFrame
  12  Combine masks
"""

from pathlib import Path
import subprocess
import webbrowser
from typing import Optional

import typer
from loguru import logger

from segmentation_tools.utils.run_config import RunConfig, load_config, validate_config

app = typer.Typer(
    name="segmentation-tools",
    help="Align and segment multiplexed microscopy images.",
    add_completion=False,
)

PIPELINE_SCRIPT = Path(__file__).parent / "pipeline.sh"
CONFIG_EXAMPLE = Path(__file__).parent / "config_example.yaml"


def _build_config(
    config: Optional[Path],
    fixed_file: Optional[Path],
    moving_file: Optional[Path],
    output_root: Optional[Path],
    job_title: Optional[str],
    start_step: int,
    fixed_dapi_channel: int,
    moving_dapi_channel: int,
) -> RunConfig:
    """Merge config file (if given) with explicit CLI overrides."""
    cfg = load_config(config) if config else RunConfig()

    if fixed_file:
        cfg.fixed_file = fixed_file
    if moving_file:
        cfg.moving_file = moving_file
    if output_root:
        cfg.output_root = output_root
    if job_title:
        cfg.job_title = job_title
    if start_step != 0:
        cfg.start_step = start_step
    if fixed_dapi_channel != 0:
        cfg.fixed_dapi_channel = fixed_dapi_channel
    if moving_dapi_channel != 1:
        cfg.moving_dapi_channel = moving_dapi_channel

    return cfg


@app.command("run")
def run(
    config: Optional[Path] = typer.Option(None, "--config", "-c",
        help="Path to YAML config file. CLI flags override config values."),
    fixed_file: Optional[Path] = typer.Option(None, "-f", "--fixed-file",
        help="Path to fixed image (Xenium OME-TIFF)."),
    moving_file: Optional[Path] = typer.Option(None, "-m", "--moving-file",
        help="Path to moving image (IF, e.g. .nd2 or .tiff)."),
    output_root: Optional[Path] = typer.Option(None, "-o", "--output-root",
        help="Root output directory."),
    job_title: Optional[str] = typer.Option(None, "-j", "--job-title",
        help="Job name; results written to output_root/job_title/."),
    start_step: int = typer.Option(0, "--start-step",
        help="Step number to resume from (0 = run all)."),
    fixed_dapi_channel: int = typer.Option(0, "--fixed-dapi-channel"),
    moving_dapi_channel: int = typer.Option(1, "--moving-dapi-channel"),
    skip_validation: bool = typer.Option(False, "--skip-validation",
        help="Skip input validation checks."),
):
    """Run the alignment + segmentation pipeline locally."""
    cfg = _build_config(config, fixed_file, moving_file, output_root, job_title,
                        start_step, fixed_dapi_channel, moving_dapi_channel)

    if not skip_validation:
        logger.info("Validating inputs...")
        if not validate_config(cfg):
            logger.error("Validation failed. Fix the errors above or use --skip-validation.")
            raise typer.Exit(1)
        logger.success("Validation passed.")

    if not PIPELINE_SCRIPT.exists():
        logger.error(f"Pipeline script not found: {PIPELINE_SCRIPT}")
        raise typer.Exit(1)

    cmd = [
        "bash", str(PIPELINE_SCRIPT),
        cfg.job_title,
        str(cfg.fixed_file),
        str(cfg.moving_file),
        str(cfg.output_root),
        str(cfg.start_step),
    ]

    logger.info(f"Starting pipeline: {cfg.job_title} (start_step={cfg.start_step})")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@app.command("submit")
def submit(
    config: Optional[Path] = typer.Option(None, "--config", "-c",
        help="Path to YAML config file. CLI flags override config values."),
    fixed_file: Optional[Path] = typer.Option(None, "-f", "--fixed-file"),
    moving_file: Optional[Path] = typer.Option(None, "-m", "--moving-file"),
    output_root: Optional[Path] = typer.Option(None, "-o", "--output-root"),
    job_title: Optional[str] = typer.Option(None, "-j", "--job-title"),
    start_step: int = typer.Option(0, "--start-step",
        help="Step number to resume from (0 = run all)."),
    fixed_dapi_channel: int = typer.Option(0, "--fixed-dapi-channel"),
    moving_dapi_channel: int = typer.Option(1, "--moving-dapi-channel"),
    conda_env: Optional[str] = typer.Option(None, "--conda-env"),
    partition: Optional[str] = typer.Option(None, "--partition"),
    time: Optional[str] = typer.Option(None, "--time"),
    mem: Optional[str] = typer.Option(None, "--mem"),
    gpus: Optional[int] = typer.Option(None, "--gpus"),
    cpus: Optional[int] = typer.Option(None, "--cpus"),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir"),
    skip_validation: bool = typer.Option(False, "--skip-validation"),
    dry_run: bool = typer.Option(False, "--dry-run",
        help="Print the sbatch command without submitting."),
):
    """Submit the pipeline as a SLURM job via sbatch."""
    cfg = _build_config(config, fixed_file, moving_file, output_root, job_title,
                        start_step, fixed_dapi_channel, moving_dapi_channel)

    # SLURM CLI overrides
    if conda_env:
        cfg.slurm.conda_env = conda_env
    if partition:
        cfg.slurm.partition = partition
    if time:
        cfg.slurm.time = time
    if mem:
        cfg.slurm.mem = mem
    if gpus is not None:
        cfg.slurm.gpus = gpus
    if cpus is not None:
        cfg.slurm.cpus = cpus

    if not skip_validation:
        logger.info("Validating inputs...")
        if not validate_config(cfg):
            logger.error("Validation failed. Fix errors above or use --skip-validation.")
            raise typer.Exit(1)
        logger.success("Validation passed.")

    env_path = Path(f"/usersoftware/peerd/ghoshr/.conda/envs/{cfg.slurm.conda_env}")
    if not env_path.exists():
        logger.error(f"Conda env not found: {env_path}")
        raise typer.Exit(1)

    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pipeline_cmd = (
        f"export PATH={env_path}/bin:$PATH && "
        f"export LD_LIBRARY_PATH={env_path}/lib:$LD_LIBRARY_PATH && "
        f"bash {PIPELINE_SCRIPT} "
        f'"{cfg.job_title}" "{cfg.fixed_file}" "{cfg.moving_file}" '
        f'"{cfg.output_root}" {cfg.start_step}'
    )

    cmd = [
        "sbatch",
        f"--job-name=seg_{cfg.job_title}",
        f"--output={log_dir}/%x_%j.out",
        f"--error={log_dir}/%x_%j.err",
        f"--partition={cfg.slurm.partition}",
        f"--time={cfg.slurm.time}",
        f"--mem={cfg.slurm.mem}",
        f"--gres=gpu:{cfg.slurm.gpus}",
        f"--cpus-per-task={cfg.slurm.cpus}",
        f"--wrap={pipeline_cmd}",
    ]

    if dry_run:
        logger.info("Dry run — sbatch command:")
        print(" \\\n  ".join(cmd))
        return

    logger.info(f"Submitting: {cfg.job_title} (start_step={cfg.start_step})")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed:\n{result.stderr}")
        raise typer.Exit(result.returncode)
    logger.success(result.stdout.strip())


@app.command("view")
def view(
    output_root: Path = typer.Option(..., "-o", "--output-root"),
    job_title: str = typer.Option(..., "-j", "--job-title"),
    no_browser: bool = typer.Option(False, "--no-browser",
        help="Write report but don't open browser."),
):
    """Generate and open the QC HTML report for a completed run."""
    from segmentation_tools.view_results import generate_report
    report_path = generate_report(output_root, job_title)
    if not no_browser:
        try:
            webbrowser.open(f"file://{report_path.resolve()}")
        except Exception:
            pass
    print(f"Report: {report_path}")


@app.command("init-config")
def init_config(
    output: Optional[Path] = typer.Option(None, "--output", "-o",
        help="Write config to this file instead of stdout."),
):
    """Print a sample config YAML to stdout or write to a file."""
    content = CONFIG_EXAMPLE.read_text()
    if output:
        output.write_text(content)
        logger.success(f"Config written to {output}")
    else:
        print(content)


def main():
    app()


if __name__ == "__main__":
    main()
