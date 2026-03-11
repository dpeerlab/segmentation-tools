"""CLI entry point for segmentation-tools.

Usage examples
--------------
Run from a config file (recommended):
    segmentation-tools run --config my_sample.yaml
    segmentation-tools submit --config my_sample.yaml

Run from CLI flags:
    segmentation-tools run -f fixed.tiff -m moving.nd2 -o /output -j my_sample

Run a specific step range:
    segmentation-tools run --config my_sample.yaml --start-step 2 --end-step 2
    segmentation-tools run --config my_sample.yaml --start-step 6 --end-step 9

Pick steps interactively:
    segmentation-tools run --config my_sample.yaml --pick-steps

Generate QC report:
    segmentation-tools view -o /output -j my_sample

Print a sample config:
    segmentation-tools init-config

Pipeline steps
--------------
  1  Setup directories + convert to OME-TIFF
  2  VALIS rigid/affine alignment
  3  Preprocess high-res DAPI (normalize + Otsu filter)
  4  Apply linear transform to moving DAPI
  5  Recommend MIRAGE hyperparameters
  6  MIRAGE non-linear registration
  7  Evaluate MIRAGE alignment quality
  8  Warp all channels + build pyramid OME-TIFF
  9  CellPose segmentation + masks to GeoDataFrame parquet
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

STEPS = {
    1: "Setup directories + convert to OME-TIFF",
    2: "VALIS rigid/affine alignment",
    3: "Preprocess high-res DAPI (normalize + Otsu filter)",
    4: "Apply linear transform to moving DAPI",
    5: "Recommend MIRAGE hyperparameters",
    6: "MIRAGE non-linear registration",
    7: "Evaluate MIRAGE alignment quality",
    8: "Warp all channels + build pyramid OME-TIFF",
    9: "CellPose segmentation + masks to GeoDataFrame parquet",
}


def _print_steps():
    print("\nAvailable pipeline steps:")
    print("─" * 52)
    for num, name in STEPS.items():
        print(f"  {num}  {name}")
    print("─" * 52)


def _pick_steps_interactively() -> tuple[int, int]:
    """Print steps and ask user to enter a start and end step."""
    _print_steps()
    print()
    while True:
        try:
            start = int(input("  Start step [1-9]: ").strip())
            end = int(input("  End step   [1-9]: ").strip())
            if 1 <= start <= 9 and 1 <= end <= 9 and start <= end:
                return start, end
            print("  Invalid range. Start must be <= end, both between 1-9.")
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            raise typer.Exit(0)


def _build_config(
    config: Optional[Path],
    fixed_file: Optional[Path],
    moving_file: Optional[Path],
    output_root: Optional[Path],
    job_title: Optional[str],
    start_step: int,
    end_step: int,
    fixed_dapi_channel: int,
    moving_dapi_channel: int,
) -> RunConfig:
    cfg = load_config(config) if config else RunConfig()
    if fixed_file:
        cfg.fixed_file = fixed_file
    if moving_file:
        cfg.moving_file = moving_file
    if output_root:
        cfg.output_root = output_root
    if job_title:
        cfg.job_title = job_title
    if start_step != 1:
        cfg.start_step = start_step
    if end_step != 9:
        cfg.end_step = end_step
    if fixed_dapi_channel != 0:
        cfg.fixed_dapi_channel = fixed_dapi_channel
    if moving_dapi_channel != 1:
        cfg.moving_dapi_channel = moving_dapi_channel
    return cfg


@app.command("run")
def run(
    config: Optional[Path] = typer.Option(None, "--config", "-c",
        help="Path to YAML config file. CLI flags override config values."),
    fixed_file: Optional[Path] = typer.Option(None, "-f", "--fixed-file"),
    moving_file: Optional[Path] = typer.Option(None, "-m", "--moving-file"),
    output_root: Optional[Path] = typer.Option(None, "-o", "--output-root"),
    job_title: Optional[str] = typer.Option(None, "-j", "--job-title"),
    start_step: int = typer.Option(1, "--start-step",
        help="First step to run (1-9, default: 1)."),
    end_step: int = typer.Option(9, "--end-step",
        help="Last step to run (1-9, default: 9). Use with --start-step to run a single step."),
    pick_steps: bool = typer.Option(False, "--pick-steps",
        help="Interactively select which steps to run."),
    fixed_dapi_channel: int = typer.Option(0, "--fixed-dapi-channel"),
    moving_dapi_channel: int = typer.Option(1, "--moving-dapi-channel"),
    skip_validation: bool = typer.Option(False, "--skip-validation"),
):
    """Run the alignment + segmentation pipeline locally."""
    cfg = _build_config(config, fixed_file, moving_file, output_root, job_title,
                        start_step, end_step, fixed_dapi_channel, moving_dapi_channel)

    if pick_steps:
        cfg.start_step, cfg.end_step = _pick_steps_interactively()
        print(f"\n  Running steps {cfg.start_step}–{cfg.end_step}:")
        for n in range(cfg.start_step, cfg.end_step + 1):
            print(f"    {n}  {STEPS[n]}")
        print()

    if not skip_validation:
        logger.info("Validating inputs...")
        if not validate_config(cfg):
            logger.error("Validation failed. Fix errors above or use --skip-validation.")
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
        str(cfg.end_step),
    ]

    logger.info(f"Pipeline: {cfg.job_title} | steps {cfg.start_step}–{cfg.end_step}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@app.command("submit")
def submit(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    fixed_file: Optional[Path] = typer.Option(None, "-f", "--fixed-file"),
    moving_file: Optional[Path] = typer.Option(None, "-m", "--moving-file"),
    output_root: Optional[Path] = typer.Option(None, "-o", "--output-root"),
    job_title: Optional[str] = typer.Option(None, "-j", "--job-title"),
    start_step: int = typer.Option(1, "--start-step",
        help="First step to run (default: 1)."),
    end_step: int = typer.Option(9, "--end-step",
        help="Last step to run (default: 9)."),
    pick_steps: bool = typer.Option(False, "--pick-steps",
        help="Interactively select which steps to run before submitting."),
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
                        start_step, end_step, fixed_dapi_channel, moving_dapi_channel)

    if pick_steps:
        cfg.start_step, cfg.end_step = _pick_steps_interactively()
        print(f"\n  Submitting steps {cfg.start_step}–{cfg.end_step}:")
        for n in range(cfg.start_step, cfg.end_step + 1):
            print(f"    {n}  {STEPS[n]}")
        print()

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
        f'"{cfg.output_root}" {cfg.start_step} {cfg.end_step}'
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

    logger.info(f"Submitting: {cfg.job_title} | steps {cfg.start_step}–{cfg.end_step}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed:\n{result.stderr}")
        raise typer.Exit(result.returncode)
    logger.success(result.stdout.strip())


@app.command("view")
def view(
    output_root: Path = typer.Option(..., "-o", "--output-root"),
    job_title: str = typer.Option(..., "-j", "--job-title"),
    no_browser: bool = typer.Option(False, "--no-browser"),
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


@app.command("steps")
def steps():
    """Print all pipeline steps and their numbers."""
    _print_steps()


@app.command("init-config")
def init_config(
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
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
