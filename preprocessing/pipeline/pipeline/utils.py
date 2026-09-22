import os
import sys
import yaml
import shutil
import time
from functools import wraps

import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger

import spikeinterface as si


def format_duration(seconds: float) -> str:
    """Convert seconds to human-readable format."""
    if seconds < 0:
        return f"-{format_duration(-seconds)}"
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f}µs"
    if seconds < 1.0:
        return f"{seconds * 1000:.2f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"

def format_size(size_bytes: float) -> str:
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def format_unit(value: float, unit: str) -> str:
    """Format a value with SI prefix"""
    prefixes = [('G', 1e9), ('M', 1e6), ('k', 1e3)]
    for prefix, scale in prefixes:
        if abs(value) >= scale:
            scaled = value / scale
            if scaled == int(scaled):
                return f"{int(scaled)} {prefix}{unit}"
            return f"{scaled:.2f} {prefix}{unit}"
    if value == int(value):
        return f"{int(value)} {unit}"
    return f"{value:.2f} {unit}"


def format_ttl(ttl: np.ndarray, mode: str = 'timestamp') -> str:
    if mode == 'timestamp':
        return f"[{ttl[0]:.3f}s → {ttl[-1]:.3f}s] ({len(ttl)} edges)"
    elif mode == 'sample_number':
        return f"[{ttl[0]} → {ttl[-1]}] ({len(ttl)} edges)"
    else:
        raise ValueError(f"Unknown mode: {mode}")

def probe_label(name: str) -> str:
    """Shorten probe name for log display."""
    return 'ADC' if name == 'OneBox-ADC' else name

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log = logger
        if args and hasattr(args[0], 'name'):
            log = log.bind(probe=probe_label(args[0].name))
        log.success(f"{fn.__name__} completed in {format_duration(elapsed)}")
        return result
    return wrapper

def log_rec(rec: si.BaseRecording):
    """Log recording summary: samples, size, channels, time range, duration."""
    n_samples = rec.get_total_samples()
    logger.info(
        f"  binary: {rec.get_num_segments()} recording(s) · "
        f"{rec.get_num_channels()} ch · {format_unit(rec.get_sampling_frequency(), 'Hz')} · "
        f"{n_samples} samples ({format_size(rec.get_total_memory_size())}) · ({format_duration(rec.get_total_duration())})"
    )

def log_events(event_df):
    """Log sync event summary from a DataFrame with 'timestamp' and 'state' columns."""
    n = len(event_df)
    ts = event_df['timestamp'].to_numpy()
    s, e = ts[0], ts[-1]
    logger.info(
        f"  event: {n} edges ({format_duration(e - s)})"
    )

def log_residuals(residuals: np.ndarray):
    """Log residual error statistics for a sync pair."""
    abs_res = np.abs(residuals - np.median(residuals))
    logger.info(
        f"  residuals: median={format_duration(np.median(residuals))} · "
        f"MAD={format_duration(np.median(abs_res))} · "
        f"max={format_duration(np.max(abs_res))}"
    )
    if np.max(abs_res) > 0.01:
        logger.warning(f"large residual: {format_duration(np.max(abs_res))}")

def log_header():
    import platform
    from spikeinterface import __version__ as si_version
    try:
        from torch import __version__ as torch_version
        from torch import cuda
    except ImportError:
        torch_version = "not installed"
        cuda = None
    try:
        from kilosort import __version__ as kilosort_version
    except ImportError:
        kilosort_version = "not installed"

    with logger.contextualize(stage="env"):
        logger.debug(f"Python:          {platform.python_version()}")
        logger.debug(f"SpikeInterface:  {si_version}")
        logger.debug(f"Kilosort4:       {kilosort_version}")
        logger.debug(f"PyTorch:         {torch_version}")
        if cuda and cuda.is_available():
            logger.debug(f"GPU:             {cuda.get_device_name(0)}")
        else:
            logger.warning("GPU: Not available")


def setup_logger(debug: bool = True, log_path=None):
    logger.configure(extra={"stage": "", "probe": ""})
    logger.remove()

    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{extra[stage]:<8}</cyan> | "
            "<yellow>{extra[probe]:<8}</yellow> | "
            "<level>{message}</level>"
        ),
        level="DEBUG" if debug else "INFO",
        colorize=True,
    )

    if log_path:
        logger.add(
            log_path,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {extra[stage]:<8} | {extra[probe]:<10} | {message}",
        )

def setup_pipeline(config_path: str, debug: bool = False) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate session paths
    session_paths = config.get('session_paths', [])
    missing = [p for p in session_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Sessions not found:\n" + "\n".join(f"  - {p}" for p in missing))
    
    # Resolve output paths
    run_name = config['run_name'].strip().lower().replace(' ', '_')
    local = Path(config['local_output']).resolve() / run_name
    local.mkdir(parents=True, exist_ok=True)
    
    config['run_name'] = run_name
    config['local_output'] = local
    if config.get('remote_output') is not None:
        config['remote_output'] = Path(config['remote_output']).resolve() / run_name
    
    # Logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_dir = local / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{run_name}_{timestamp}.log'
    setup_logger(debug=debug, log_path=log_path)

    logger.info('=' * 80)
    with logger.contextualize(stage="setup"):
        
        logger.debug(f"Config loaded from: {config_path}")
        for k, v in config.items():
            logger.debug(f"{k}: {v}")
        logger.info(f"Run:             {config['run_name']}")
        logger.info(f"Date:            {datetime.now():%Y-%m-%d %H:%M:%S}")
        logger.info(f"Log directory:   {log_dir}")
    logger.info('=' * 80)
    log_header()
    return config


def _copy_with_retry(src: Path, dst: Path, retries: int = 3, backoff: float = 2.0) -> None:
    """Copy a single file with retry, temp-file safety, and size verification.
    
    Writes to a .tmp sibling, then renames on success to avoid leaving
    corrupted partial files on the remote when the network drops.
    """
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    expected_size = src.stat().st_size

    for attempt in range(1, retries + 1):
        try:
            shutil.copy2(src, tmp)
            actual_size = tmp.stat().st_size
            if actual_size != expected_size:
                tmp.unlink(missing_ok=True)
                raise IOError(
                    f"Size mismatch: expected {expected_size}, got {actual_size}"
                )
            tmp.replace(dst)
            return
        except (OSError, IOError) as exc:
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                wait = backoff ** attempt
                logger.warning(
                    f"Retry {attempt}/{retries} for {src.name}: {exc} "
                    f"(waiting {wait:.0f}s)"
                )
                time.sleep(wait)
            else:
                raise


def copy_to_remote(
    local_path: Path,
    remote_path: Path,
    overwrite_mode: str = 'prompt',  # 'prompt', 'all', 'skip-all', 'newer'
    retries: int = 3,
) -> None:
    """Copy files to remote storage with progress, retries, and integrity checks.
    
    Args:
        local_path: Source directory path.
        remote_path: Destination directory path.
        overwrite_mode: How to handle existing files:
            - 'prompt': Ask user for each file
            - 'all': Overwrite all existing files
            - 'skip-all': Skip all existing files
            - 'newer': Only overwrite if local file is newer than remote
        retries: Number of retry attempts per file on network errors.
    """
    log = logger.bind(stage="copy")
    log.info(f"{local_path} → {remote_path}")
    
    remote_path.mkdir(parents=True, exist_ok=True)

    # Collect files and total size for progress reporting
    files = [f for f in local_path.rglob('*') if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)
    log.info(f"Found {len(files)} files ({format_size(total_size)})")

    copied_size = 0
    copied_count = 0
    skipped = 0
    failed = []
    
    for i, local_file in enumerate(files, 1):
        relative = local_file.relative_to(local_path)
        remote_file = remote_path / relative
        file_size = local_file.stat().st_size
        
        if remote_file.exists():
            if overwrite_mode == 'skip-all':
                skipped += 1
                continue
            elif overwrite_mode == 'newer':
                if local_file.stat().st_mtime <= remote_file.stat().st_mtime:
                    skipped += 1
                    continue
            elif overwrite_mode == 'prompt':
                response = input(f"Overwrite '{relative}'? (y/n/all/skip-all): ").strip().lower()
                if response == 'all':
                    overwrite_mode = 'all'
                elif response == 'skip-all':
                    overwrite_mode = 'skip-all'
                    skipped += 1
                    continue
                elif response != 'y':
                    skipped += 1
                    continue
        
        remote_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            _copy_with_retry(local_file, remote_file, retries=retries)
            copied_size += file_size
            copied_count += 1
            log.debug(
                f"[{i}/{len(files)}] {relative} ({format_size(file_size)}) — "
                f"{format_size(copied_size)}/{format_size(total_size)}"
            )
        except (OSError, IOError) as exc:
            failed.append((relative, exc))
            log.error(f"[{i}/{len(files)}] FAILED {relative}: {exc}")
    
    log.info(f"Copied {copied_count} files ({format_size(copied_size)})")
    if skipped:
        log.info(f"Skipped {skipped} files")
    if failed:
        log.error(f"Failed {len(failed)} files:")
        for rel, exc in failed:
            log.error(f"  {rel}: {exc}")
        raise IOError(f"{len(failed)} file(s) failed to copy")