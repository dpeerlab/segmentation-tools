"""Profiling utilities for pipeline steps.

Provides a context manager and decorator for timing and memory tracking.
Uses psutil for RSS memory measurement and loguru for output.
"""

import time
import functools
from contextlib import contextmanager

import numpy as np
import psutil
from loguru import logger


def _get_memory_mb():
    """Return current process RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _format_duration(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def _format_size_mb(size_mb):
    """Format MB into human-readable string."""
    if size_mb < 1024:
        return f"{size_mb:.1f} MB"
    return f"{size_mb / 1024:.2f} GB"


@contextmanager
def profile_block(name):
    """Context manager that logs wall time and memory delta for a block.

    Usage::

        with profile_block("Load images"):
            img = np.load(path)
    """
    mem_before = _get_memory_mb()
    t0 = time.perf_counter()
    logger.info(f"[START] {name} | mem={_format_size_mb(mem_before)}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        mem_after = _get_memory_mb()
        mem_delta = mem_after - mem_before
        sign = "+" if mem_delta >= 0 else ""
        logger.info(
            f"[DONE]  {name} | "
            f"time={_format_duration(elapsed)} | "
            f"mem={_format_size_mb(mem_after)} ({sign}{_format_size_mb(mem_delta)})"
        )


def profile_step(step_name):
    """Decorator that wraps an entire pipeline step with profiling.

    Usage::

        @profile_step("006 MIRAGE")
        def main(warped_path, fixed_path, ...):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"{'='*60}")
            logger.info(f"STEP: {step_name}")
            logger.info(f"{'='*60}")
            mem_start = _get_memory_mb()
            t_start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - t_start
                mem_end = _get_memory_mb()
                peak = mem_end  # RSS at exit (best proxy without tracemalloc)
                logger.info(f"{'-'*60}")
                logger.info(
                    f"STEP COMPLETE: {step_name} | "
                    f"total={_format_duration(elapsed)} | "
                    f"mem: {_format_size_mb(mem_start)} -> {_format_size_mb(mem_end)}"
                )
                logger.info(f"{'='*60}")
        return wrapper
    return decorator


def log_array(name, arr):
    """Log shape, dtype, and memory footprint of a numpy array."""
    if isinstance(arr, np.ndarray):
        size_mb = arr.nbytes / (1024 * 1024)
        logger.info(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, size={_format_size_mb(size_mb)}")
    else:
        logger.info(f"  {name}: type={type(arr).__name__}")
