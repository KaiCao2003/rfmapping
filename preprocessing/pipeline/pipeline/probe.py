from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.extractors as se

from loguru import logger

from probeinterface.plotting import plot_probe
from open_ephys.analysis.session import Session
from scipy.signal import find_peaks

from .utils import timed, probe_label, format_duration, log_rec, log_events, log_residuals, format_ttl

class Probe:
    """Probe with multi-session recordings and synchronization events."""
    
    def __init__(self, name: str, stream_id: str):
        self.name = name
        self.stream_id = stream_id

        self.fs                 = None
        self.recordings         = []
        self.intervals          = {'timestamp': [], 'sample_number': []}
        self.events             = []
    
    @timed
    def load_sessions(self, session_paths: str | list[str]):
        """Load recordings and time references from OpenEphys sessions."""
        if not isinstance(session_paths, list):
            session_paths = [session_paths]
        with logger.contextualize(stage="load", probe=probe_label(self.name)):
            logger.info(f"Loading {len(session_paths)} session(s)")
            for session_path in session_paths:
                logger.info('-' * 80)
                logger.info(f"{Path(session_path).name}")
                
                rec = se.read_openephys(session_path, stream_id=self.stream_id)
                self.recordings.append(rec)
                
                log_rec(rec)
                logger.debug(str(rec))
                
                # Extract sync events per recording segment
                for i, segment in enumerate(Session(session_path).recordnodes[0].recordings):

                    cont = segment.continuous[self.name]
                    self.intervals['timestamp'].append((cont.timestamps[0], cont.timestamps[-1]))
                    self.intervals['sample_number'].append((cont.sample_numbers[0], cont.sample_numbers[-1]))
                    
                    logger.debug(f'    segment {i}:')
                    logger.debug(f'      cont timestamp interval: {cont.timestamps[0]:.3f}, {cont.timestamps[-1]:.3f}')
                    logger.debug(f'      cont sample_number interval: {cont.sample_numbers[0]}, {cont.sample_numbers[-1]}')
                    
                    event_df = segment.events
                    event_df = event_df[event_df['stream_name'] == self.name][['sample_number', 'timestamp', 'state']]
                    ttl = event_df['timestamp'].to_numpy()

                    logger.debug(f"      event: {format_ttl(ttl)} ({format_duration(ttl[-1] - ttl[0])})")
                    self.events.append(event_df)



            self.fs = self.recordings[0].get_sampling_frequency()
            logger.success(f"Loaded {len(self.recordings)} recording(s)")

    @timed
    def build_global_references(self, mode: str = 'timestamp') -> np.ndarray:
        """Concatenate sync events"""
        if mode not in ['timestamp', 'sample_number']:
            raise ValueError(f"mode should be either 'timestamp' or 'sample_number. Received {mode}")
        offset = 0.0
        total_globals = []
        logger.info(f"Building global references ({mode})")
        logger.info('-' * 80)
        for i, (interval, event) in enumerate(zip(self.intervals[mode], self.events)):
            locals = event[mode].to_numpy()
            logger.info(f"Recording {i}")
            if mode == 'timestamp':
                logger.info(f"  local TTL : {format_ttl(locals, mode=mode)} ({format_duration(locals[-1] - locals[0])})")
            else:
                logger.info(f"  local TTL : {format_ttl(locals, mode=mode)} ({locals[-1] - locals[0]} samples)")

            logger.debug(f"  source local interval: {format_ttl(interval, mode=mode)}")
            globals = locals - interval[0] + offset
            total_globals.append(globals)
            offset += interval[1] - interval[0] +  1 / self.fs
        logger.info('-' * 80)
        return np.concatenate(total_globals)

    @timed
    def sync_to(self,
                target_probe,
                mode : str = 'timestamp' # 'timestamp' or 'sample_number'
                ) -> np.ndarray:
        """Align and synchronize events to a target probe."""
        assert len(self.intervals[mode]) == len(target_probe.intervals[mode]), "Mismatch in number of recordings/events"
        with logger.contextualize(stage="sync", probe=probe_label(self.name)):
            logger.info(f"{probe_label(self.name)} → {probe_label(target_probe.name)} ({len(self.intervals[mode])} recordings)")

            self_globals, target_globals = [], []
            self_offset, target_offset = 0.0, 0.0

            for i, (self_iv, target_iv) in enumerate(zip(self.intervals[mode], target_probe.intervals[mode])):
                self_ts = self.events[i][mode].to_numpy()
                target_ts = target_probe.events[i][mode].to_numpy()
                self_states = self.events[i]['state']
                target_states = target_probe.events[i]['state']

                logger.info(f"Recording {i}")
                logger.info('-' * 80)
                logger.info(f"  source local TTL : {format_ttl(self_ts)} ({format_duration(self_ts[-1] - self_ts[0])})")
                logger.info(f"  target local TTL : {format_ttl(target_ts)} ({format_duration(target_ts[-1] - target_ts[0])})")
                logger.debug(f"  source local interval: {format_ttl(self_iv)} (init state: {self_states.iloc[0]})")
                logger.debug(f"  target local interval: {format_ttl(target_iv)} (init state: {target_states.iloc[0]})")

                if self_states.iloc[0] != target_states.iloc[0]:
                    logger.warning(f" Initial states differ, aligning edges")
                    logger.debug(f"  source TTL before alignment: {format_ttl(self_ts)}")
                    logger.debug(f"  target TTL before alignment: {format_ttl(target_ts)}")
                    self_ts, target_ts = align_edges(self_ts, target_ts)
                    logger.debug(f"  source TTL after alignment: {format_ttl(self_ts)}")
                    logger.debug(f"  target TTL after alignment: {format_ttl(target_ts)}")

                if len(self_ts) != len(target_ts):
                    n = min(len(self_ts), len(target_ts))
                    logger.warning(f"  Truncating {len(self_ts)} vs {len(target_ts)} → {n} events")
                    self_ts, target_ts = self_ts[:n], target_ts[:n]

                self_global = self_ts - self_iv[0] + self_offset
                target_global = target_ts - target_iv[0] + target_offset

                logger.info(f"  source global TTL : {format_ttl(self_global)} ({format_duration(self_global[-1] - self_global[0])})")
                logger.info(f"  target global TTL : {format_ttl(target_global)} ({format_duration(target_global[-1] - target_global[0])})")

                residuals = self_global - target_global
                log_residuals(residuals)
                logger.info('-' * 80)

                self_globals.append(self_global)
                target_globals.append(target_global)

                self_offset += self_iv[1] - self_iv[0] +  1 / self.fs
                target_offset += target_iv[1] - target_iv[0] + 1 / target_probe.fs

            sync_map = np.column_stack((np.concatenate(self_globals), np.concatenate(target_globals)))
        return sync_map
    
    @timed
    def concat(
        self,
        output_dir: Path | str,
        verbose: bool = True,
        overwrite: bool = False,
        **job_kwargs
    ) -> si.BaseRecording:
        """Concatenate all recordings."""
        logger.info(f"Concatenating {len(self.recordings)} recordings → {output_dir}")
        if (Path(output_dir) / 'traces_cached_seg0.raw').exists() and not overwrite:
            logger.info(f"Already exists at {output_dir}, loading")
            rec = si.load(output_dir)
            logger.debug(str(rec))
            log_rec(rec)
            return rec

        if len(self.recordings) == 0:
            raise ValueError("No sessions to concatenate.")
        
        rec = si.concatenate_recordings(self.recordings)
        log_rec(rec)
        logger.debug(str(rec))
        concat = rec.save(folder=output_dir,
                    verbose=verbose,
                    overwrite=overwrite,
                    **job_kwargs)
        return concat

    def save_geometry(self, output_file: str) -> None:
        """Save probe geometry as png."""
        if Path(output_file).exists():
            logger.info(f"Geometry already exists at {output_file}")
            return
        
        fig, ax = plt.subplots(figsize=(5, 6))
        positions = self.get_probe().to_dict()['contact_positions']
        padding = 100   
        plot_probe(self.get_probe(), ax=ax,
            xlims=(positions[:, 0].min() - padding, positions[:, 0].max() + padding),
            ylims=(positions[:, 1].min() - padding, positions[:, 1].max() + padding)
            )
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        logger.info(f"Saved probe geometry \u2192 {output_file}")
    
    def get_probe(self, convert_to_kilosort: bool = False):
        """Return either ProbeInterface Probe or Kilosort-compatible dict."""
        if self.recordings:
            if convert_to_kilosort:
                return probe_to_kilosort(self.recordings[0].get_probe())
            else:
                return self.recordings[0].get_probe()
        else:
            raise ValueError(f"{self.name}: No recordings loaded, cannot get probe info")
    
    def get_num_channels(self):
        if self.recordings:
            return self.recordings[0].get_num_channels()
        else:
            raise ValueError(f"{self.name}: No recordings loaded.")

    def get_sampling_frequency(self):
        if self.recordings:
            return self.recordings[0].get_sampling_frequency()
        else:
            raise ValueError(f"{self.name}: No recordings loaded.")

    def __repr__(self):
        return f"Probe('{self.name}', {len(self.recordings)} recordings)"


def detect_reset(events: np.ndarray, limit: int) -> int:
    """Detect chirp reset point from interval minima."""
    diffs = np.diff(events[:limit])
    minima = find_peaks(-diffs)[0]
    if len(minima) == 0:
        raise ValueError(f"No chirp reset detected in first {limit} events")
    return minima[0] + 1

def align_edges(source: np.ndarray, target: np.ndarray, limit: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Align two event arrays by trimming to matching chirp phase."""
    source_reset = detect_reset(source, limit)
    target_reset = detect_reset(target, limit)
    logger.debug(f"  source reset index: {source_reset}. {source[source_reset]}")
    logger.debug(f"  target reset index: {target_reset}. {target[target_reset]}")

    if source_reset > target_reset:
        r = source_reset - target_reset
        logger.info(f"Source leads target by {r} events, aligning by trimming source")
        source_aligned, target_aligned = source[r:], target
    elif target_reset > source_reset:
        r = target_reset - source_reset
        logger.info(f"Target leads source by {r} events, aligning by trimming target")
        source_aligned, target_aligned = source, target[r:]
    else:
        source_aligned, target_aligned = source, target

    return source_aligned, target_aligned

def probe_to_kilosort(probe) -> dict:
    return {
        'chanMap': np.arange(probe.get_contact_count(), dtype=int),
        'xc': probe.contact_positions[:, 0].astype('float32'),
        'yc': probe.contact_positions[:, 1].astype('float32'),
        'kcoords': probe.shank_ids.astype('float32'),
        'n_chan': probe.get_contact_count(),
    }

def plot_sync_drift(
    prb: Probe,
    target: Probe,
    mode: str = 'timestamp',
    ncols: int = 4,
    cell_width: float = 4.0,
    cell_height: float = 2.5,
    dpi: int = 320,
    save_path: str | None = None,
):
    """
    Plot detrended clock drift between two probes for each recording session.
    """
    n = len(prb.intervals[mode])
    assert n == len(target.intervals[mode]), "Mismatch in number of recordings"

    # ── Style ─────────────────────────────────────────────────
    with plt.style.context('dark_background'):
        cmap = plt.get_cmap('Set2')

        ncols_actual = min(n, ncols)
        nrows = int(np.ceil(n / ncols_actual))
        fig, axes = plt.subplots(
            nrows, ncols_actual,
            figsize=(cell_width * ncols_actual, cell_height * nrows),
            squeeze=False,
            dpi=dpi,
        )
        axes_flat = axes.ravel()

        src_label = probe_label(prb.name)
        tgt_label = probe_label(target.name)
        fig.suptitle(
            f'{src_label}  →  {tgt_label}   ({n} recordings)',
            fontsize=14, fontweight='bold', y=1.02,
        )

        session_maps = []
        src_off, tgt_off = 0.0, 0.0

        for i, (s_iv, t_iv) in enumerate(zip(prb.intervals[mode], target.intervals[mode])):
            ax = axes_flat[i]

            s_ts = prb.events[i][mode].to_numpy()
            t_ts = target.events[i][mode].to_numpy()

            # align edges if initial states differ
            if prb.events[i]['state'].iloc[0] != target.events[i]['state'].iloc[0]:
                s_ts, t_ts = align_edges(s_ts, t_ts)

            k = min(len(s_ts), len(t_ts))
            s_ts, t_ts = s_ts[:k], t_ts[:k]

            # local drift: elapsed-in-source minus elapsed-in-target
            drift = (s_ts - s_iv[0]) - (t_ts - t_iv[0])
            detrended = (drift - np.median(drift)) * 1e3  # → ms

            # x-axis: elapsed time in source (minutes)
            elapsed_min = (s_ts - s_iv[0]) / 60.0

            color = cmap(i % cmap.N)
            ax.plot(elapsed_min, detrended, color=color, lw=0.6)
            ax.axhline(0, color='grey', lw=0.4, alpha=0.5)

            duration = format_duration(s_iv[1] - s_iv[0])
            ax.set_title(f'Rec {i}  ({duration})', fontsize=10)

            # stats annotation
            mad = np.median(np.abs(detrended - np.median(detrended)))
            pk = np.max(np.abs(detrended))
            ax.text(
                0.97, 0.95,
                f'MAD {mad:.3f} ms\npk   {pk:.2f} ms',
                transform=ax.transAxes, fontsize=7,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5),
            )

            ax.set_xlabel('Time (min)', fontsize=8)
            ax.set_ylabel('Drift (ms)', fontsize=8)
            ax.tick_params(labelsize=7)

            # accumulate global maps
            s_global = s_ts - s_iv[0] + src_off
            t_global = t_ts - t_iv[0] + tgt_off
            session_maps.append(np.column_stack((s_global, t_global)))

            src_off += s_iv[1] - s_iv[0]
            tgt_off += t_iv[1] - t_iv[0]

        # hide unused axes
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return fig, session_maps