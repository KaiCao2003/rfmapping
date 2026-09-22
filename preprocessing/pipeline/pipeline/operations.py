from .probe import Probe, plot_sync_drift, probe_to_kilosort
from .utils import timed, probe_label, format_unit, log_rec
from .decimation import DecimatedRecording

import subprocess
import sys
import numpy as np
from pathlib import Path

import spikeinterface as si
import spikeinterface.extractors as se

try:
    from kilosort import run_kilosort
except ImportError:
    run_kilosort = None
from loguru import logger
from scipy.interpolate import make_interp_spline

def interpolate(spike_times, sync_map, output_file):
    log = logger.bind(stage="interp")
    spl = make_interp_spline(sync_map[:, 0], sync_map[:, 1], k=1)
    adc_spike_times = spl(spike_times)
    np.save(output_file, adc_spike_times)
    log.info(f"{spike_times.shape[0]} spikes → {Path(output_file).name}")

def load_probes(session_paths: str | list[str], probe_filter = None) -> dict:
    if not isinstance(session_paths, list):
        session_paths = [session_paths]
    probes = {}
    log = logger.bind(stage="load")
    for session_path in session_paths:
        stream_names, stream_ids = se.get_neo_streams('openephysbinary', session_path)
        for stream_name, stream_id in zip(stream_names, stream_ids):
            log.debug(f"Stream: {stream_name} (ID: {stream_id})")
            if "SYNC" not in stream_name:
                name = stream_name.split('.')[-1]
                if probe_filter and name not in probe_filter:
                    continue
                log.info(f"Found {probe_label(name)} (ID: {stream_id})")
                probes[name] = Probe(name=name, stream_id=stream_id)
                probes[name].load_sessions(session_paths)
    return probes

@timed
def downsample(
    rec: si.BaseRecording,
    output_file: str | Path,
    config: dict,
    ):
    log = logger.bind(stage="eeg")

    eeg_file = Path(output_file)
    if eeg_file.exists() and not config['overwrite']:
        log.info(f"EEG file already exists: {eeg_file}")
        return
    
    fs = rec.get_sampling_frequency()
    decimation_factor = int(fs / config['target_fs'])
    actual_fs = fs / decimation_factor
    
    log.info(f"{format_unit(fs, 'Hz')} → {format_unit(actual_fs, 'Hz')} (factor={decimation_factor})")
    dec_rec = DecimatedRecording(rec, decimation_factor)

    # No parallelization. write_binary_recording seems better than naive downsampling
    si.write_binary_recording(
        recording=dec_rec,
        file_paths=eeg_file,
        add_file_extension=False,
        verbose=config['verbose'],
        n_jobs=1,
        chunk_duration=5.0,
        progress_bar=config['job_kwargs']['progress_bar'],
        )
    
    log.info(f"Saved → {eeg_file}")
    

def concatenate_probes(probes: dict, config: dict,) -> None:
    for name, probe in probes.items():
        with logger.contextualize(stage="concat", probe=probe_label(name)):
            output = Path(config['local_output']) / name
            probe.concat(
                output_dir=output / 'concat',
                verbose=config['verbose'],
                overwrite=config['overwrite'],
                **config['job_kwargs']
                )
            if name != 'OneBox-ADC':
                probe.save_geometry(output / 'probe_geometry.png')

def downsample_probes(probe_paths: dict[str, Path],
                      config: dict) -> None:
    for name, probe_path in probe_paths.items():
        with logger.contextualize(stage="downsample", probe=probe_label(name)):
            rec = si.load(probe_path)
            output_file = probe_path.parent / 'eeg.dat'
            downsample(rec,
                       output_file,
                       config
                       )

def sort_probes(probe_paths: dict[str, Path],
                config: dict
                ) -> None:
    if run_kilosort is None:
        raise ImportError("Kilosort is required for spike sorting. Install with: uv sync")
    for name, probe_path in probe_paths.items():
        with logger.contextualize(stage="kilosort", probe=probe_label(name)):
            output = Path(probe_path).parent
            # Assuming probe_path is something like .../probe_name/concat/
            # Save the results to .../probe_name/kilosort/
            results_dir = output / 'kilosort'
            results_dir.mkdir(parents=True, exist_ok=True)

            if (results_dir/'spike_times.npy').exists() and not config['overwrite']:
                logger.info("Results already exist, skipping")
                continue
            
            rec = si.load(probe_path)
            logger.info(f"Loaded concatenated recording: {probe_path}")
            log_rec(rec)

            settings = {'n_chan_bin': rec.get_num_channels(), 'fs': rec.get_sampling_frequency()}
            ks_probe = probe_to_kilosort(rec.get_probe())

            logger.info(f"Saving Kilosort results to: {results_dir}")

            if config['per_shank']:
                for shank_id in np.unique(ks_probe['kcoords']):
                    run_kilosort(
                        settings=settings,
                        probe=ks_probe,
                        filename=probe_path / 'traces_cached_seg0.raw',
                        results_dir=results_dir,
                        device='cuda',
                        verbose_console=config['verbose'],
                        shank_idx=int(shank_id),
                    )
            else:
                run_kilosort(
                    settings=settings,
                    probe=ks_probe,
                    filename=probe_path / 'traces_cached_seg0.raw',
                    results_dir=results_dir,
                    device='cuda',
                    verbose_console=config['verbose'],
                )

def sort_probes_subprocess(probe_paths: dict[str, Path],
                           config: dict
                           ) -> None:
    """Run Kilosort for each probe in subprocess."""
    for name, probe_path in probe_paths.items():
        with logger.contextualize(stage="kilosort", probe=probe_label(name)):
            cmd = [
                sys.executable, '-m', 'pipeline.sort',
                str(probe_path),
            ]
            if config.get('per_shank'):
                cmd.append('--per-shank')
            if config.get('verbose'):
                cmd.append('--verbose')
            if config.get('overwrite'):
                cmd.append('--overwrite')

            openblas = config.get('openblas_threads', 1)
            cmd.extend(['--openblas-threads', str(openblas)])

            log_dir = Path(config['local_output']) / 'logs'
            log_file = log_dir / f'kilosort_{name}.log'
            if log_dir.exists():
                cmd.extend(['--log-file', str(log_file)])

            logger.info(f"Launching sorting subprocess for {probe_label(name)}")
            logger.debug(f"  cmd: {' '.join(cmd)}")

            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.error(f"Kilosort subprocess failed (exit code {result.returncode})")
                raise RuntimeError(
                    f"Kilosort subprocess for {probe_label(name)} exited with code {result.returncode}"
                )
            logger.info(f"Sorting finished for {probe_label(name)}")


def synchronize_probes(
        probes: dict,
        config: dict,
        target: str = 'OneBox-ADC',
        mode: str = 'timestamp'
        ):
    
    target_probe = probes[target]
    for name, probe in probes.items():
        with logger.contextualize(stage="sync", probe=probe_label(name)):
            output_dir = Path(config['local_output']) / name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if name == 'OneBox-ADC':
                logger.info('-' * 80)
                logger.info('ADC samples → timestamps sync map')
                sync_map = (probe.build_global_references(mode='sample_number'),
                            probe.build_global_references(mode='timestamp'),)
                np.save(output_dir / "sync_map.npy", np.column_stack(sync_map))
                logger.info(f"Saved: {output_dir / 'sync_map.npy'}")
                continue

            sync_map = probe.sync_to(target_probe, mode=mode)
            np.save(output_dir / "sync_map.npy", sync_map)
            logger.info(f"Saved sync map: {output_dir / 'sync_map.npy'}")
            _, _ = plot_sync_drift(prb=probe, target=target_probe, save_path=output_dir / "sync_drift.png")
            logger.info(f"Saved sync drift plot: {output_dir / 'sync_drift.png'}")

            if config['per_shank']:
                prb = probe.get_probe(convert_to_kilosort=True)
                for shank_id in np.unique(prb['kcoords']):
                    shank_id = int(shank_id)
                    spike_times_file = output_dir / 'kilosort' / f'shank_{shank_id}' / 'spike_times.npy'
                    if spike_times_file.exists():
                        spike_samples = np.load(spike_times_file).squeeze()
                        interpolate(spike_samples/probe.get_sampling_frequency(),
                                    output_file=output_dir / f'adc_spike_times_{shank_id}.npy',
                                    sync_map=sync_map)
                    else:
                        logger.bind(stage="interp").warning(f"Couldn't find kilosort results: {spike_times_file}")
            else:
                spike_times_file = output_dir / 'kilosort' / 'spike_times.npy'
                if spike_times_file.exists():
                    spike_samples = np.load(spike_times_file).squeeze()
                    interpolate(spike_samples/probe.get_sampling_frequency(),
                                output_file=output_dir / 'adc_spike_times.npy',
                                sync_map=sync_map)
                else:
                    logger.bind(stage="interp").warning(f"Couldn't find kilosort results: {spike_times_file}")