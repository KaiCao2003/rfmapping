"""Discover saved RF/EBB pairs and export the session review requested by the user."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from HD.profiles import load_hd_tc
from RF.paths import rf_source_path
from RF.rflocate import load_rf, rf_result_path
from Utils.plotting import LIGHT_PLOT_STYLE
from scripts.prepare_poster_candidates import sorting_source
from scripts.poster_design_figures import plot_profile, rank_ticks


def discover_session_pairs(root, hd_sessions, *, minimum_coverage=.9):
    """Read saved bundles and pair only sessions sharing one Kilosort source."""
    inventory, pairs = [], []
    for mouse in sorted(Path(root).glob('m[0-9][0-9]')):
        for day in sorted(mouse.glob('[0-9]' * 6)):
            for session in sorted(day.glob(day.name + '_*')):
                for folder in sorted((session / 'data/spatial_cells').glob('Probe*/baseline')):
                    required = ('audit.json', 'restored_maps.npz', 'position_correction.npz', 'frame_spike_counts.npz')
                    if not all((folder / name).is_file() for name in required):
                        continue
                    probe = folder.parent.name.removeprefix('Probe')
                    ebb_session = int(session.name.split('_')[-1])
                    hd_session = hd_sessions.get((mouse.name, day.name), ebb_session)
                    hd_folder = day / f'{day.name}_{hd_session}'
                    with np.load(folder / 'restored_maps.npz', allow_pickle=False) as saved:
                        coverage = float((saved['egocentric_occupancy'] > 0).mean())
                        ebb_ids = saved['unit_ids']
                    record = dict(mouse=mouse.name, date=day.name, ebb=ebb_session, hd=hd_session,
                                  probe=probe, coverage=coverage, selected_coverage=coverage > minimum_coverage)
                    record['rf_pairs'] = []
                    inventory.append(record)
                    if not record['selected_coverage']:
                        record['status'] = 'Coverage does not exceed threshold'
                        continue
                    hd = load_hd_tc(hd_folder / f'data/tuning_curves/Probe{probe}/tuning_curves.tc')
                    hd_ids = np.asarray(hd['unit_id'])[np.asarray(hd['unit_data']['hd_class']) == 3]
                    record['hd_class3'] = len(hd_ids)
                    source = sorting_source(session, probe)
                    if sorting_source(hd_folder, probe) != source:
                        raise ValueError(f'HD reference has a different sorting source: {hd_folder}')
                    for rf_folder in sorted(day.glob(day.name + '_*')):
                        raw = rf_source_path(rf_folder, probe)
                        detection_path = rf_result_path(raw, rf_type='excitatory')
                        if not raw.is_file() or not detection_path.is_file():
                            continue
                        if sorting_source(rf_folder, probe) != source:
                            continue
                        detection = load_rf(detection_path)
                        rf_ids = detection.unit_ids[detection.mask_2d.any(axis=(1, 2))]
                        units = np.intersect1d(np.intersect1d(hd_ids, rf_ids), ebb_ids)
                        rf_session = int(rf_folder.name.split('_')[-1])
                        pair = dict(rf=rf_session, n_cells=len(units),
                                    status='ready' if len(units) >= 3 else 'Fewer than three paired cells')
                        record['rf_pairs'].append(pair)
                        if len(units) >= 3:
                            pairs.append(dict(mouse=mouse.name, date=day.name, probe=probe,
                                              rf=rf_session, hd=hd_session, ebb=ebb_session))
                    record['status'] = 'ready' if any(p['status'] == 'ready' for p in record['rf_pairs']) else 'No RF pair with at least three cells'
    return inventory, pairs


@plt.rc_context(LIGHT_PLOT_STYLE)
def sorting_review_figure(profiles, order, *, label, coverage):
    """Use one RF-peak order in every panel and the user's zero-to-maximum scale."""
    panels = [('RF', 'Visual RF'), ('all', 'EBB · all distances'),
              ('0-8', 'EBB · 0–8 cm'), ('8-16', 'EBB · 8–16 cm'), ('16-inf', 'EBB · >16 cm')]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 6), sharey=True,
                             layout='constrained', facecolor='white')
    for axis, (key, title) in zip(axes, panels):
        image = plot_profile(axis, profiles[key], order, title)
        axis.set_xlabel('RF azimuth (°)' if key == 'RF' else 'EBB bearing (°)')
        axis.tick_params(labelsize=11)
        axis.set_facecolor('white')
    rank_ticks(axes[0], len(order))
    colorbar = fig.colorbar(image, ax=axes, fraction=.02, pad=.02, ticks=[0, 1], label='Normalized response')
    colorbar.ax.set_yticklabels(['0', 'max'])
    colorbar.solids.set_rasterized(False)
    fig.suptitle(f'{label}\nCoverage {coverage:.2%} · {len(order)} HD class-3 / ON-RF cells · sorted by RF peak', fontsize=15)
    return fig


def save_review_figure(fig, folder, stem):
    for extension in ('png', 'svg'):
        fig.savefig(Path(folder) / f'{stem}.{extension}', dpi=150,
                    bbox_inches='tight', facecolor='white', transparent=False)
    plt.close(fig)
