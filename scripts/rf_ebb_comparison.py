"""Read matched RF/EBB recordings and prepare the existing TC comparison panels."""

from hashlib import file_digest
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd

from EBB.profiles import boundary_band_profiles
from HD.profiles import load_hd_tc
from RF.profiles import load_rf_comparison, prep_rf_tc
from Utils.direction_comparison import normalize_tc
from Utils.plotting import LIGHT_PLOT_STYLE
from Utils.statistic_utils import circular_distance_matrix_deg
from scripts.prepare_poster_candidates import sorting_source


BAND_LABELS = {'all': 'All saved distances', '0-8': '0–8 cm',
               '8-16': '8–16 cm', '16-inf': '>16 cm'}


def load_pair_inputs(root, spec, *, rf_only=True):
    """Read saved sources; never generate missing maps or comparison CSVs."""
    mouse, date, probe = spec['mouse'], spec['date'], spec['probe']
    base = Path(root) / mouse / date
    sessions = {kind: base / f'{date}_{spec[kind]}' for kind in ('hd', 'rf', 'ebb')}
    folder = sessions['ebb'] / f'data/spatial_cells/Probe{probe}/baseline'
    paths = dict(hd=sessions['hd'] / f'data/tuning_curves/Probe{probe}/tuning_curves.tc',
                 maps=folder / 'restored_maps.npz', audit=folder / 'audit.json',
                 rf=sessions['rf'] / f'data/tc_comparison/rf_excitatory_x_2d{"_rfonly" if rf_only else ""}_Probe{probe}.csv')
    sortings = [sorting_source(session, probe) for session in sessions.values()]
    if not all(value == sortings[0] for value in sortings):
        raise ValueError(f'{mouse} {date} RF/HD/EBB use different sorting sources')
    for kind, session in sessions.items():
        paths[f'sorting_{kind}'] = session / f'kilosort/Probe{probe}/kilosort_{spec[kind]}/params.py'
    with np.load(paths['maps'], allow_pickle=False) as archive:
        maps = {key: archive[key] for key in ('unit_ids', 'egocentric_rate_maps',
                'egocentric_occupancy', 'egocentric_spike_maps', 'theta_edges', 'distance_edges')}
    audit = json.loads(paths['audit'].read_text())
    if audit['status'] != 'complete' or audit['angle_sampling'] != 'bin_centers':
        raise ValueError('Pair comparison requires complete native bin-center EBB maps')
    if audit['heading_convention'] != 'physical north-zero clockwise heading; clockwise egocentric bearings':
        raise ValueError('Select an explicit angle conversion before mixing EBB conventions')
    return dict(spec=spec, maps=maps, audit=audit, hd=load_hd_tc(paths['hd']),
                rf=load_rf_comparison(mouse, root, probe, date, spec['rf'], rf_only=rf_only),
                source_paths=paths, sorting_source=sortings[0])


def prepare_pair_inputs(loaded, *, bins=30, max_zero_bins=2):
    """Apply HD class 3 and the TC notebook's RF filter, then match full unit keys."""
    spec, maps, hd = loaded['spec'], loaded['maps'], loaded['hd']
    prefix = f"{spec['mouse']}:{spec['date']}:{spec['probe']}:"
    hd_ids = np.asarray(hd['unit_id'])[np.asarray(hd['unit_data']['hd_class']) == 3]
    hd_keys = pd.Index([prefix + str(int(unit)) for unit in hd_ids])
    ebb_keys = pd.Index([prefix + str(int(unit)) for unit in maps['unit_ids']], name='unit_id')
    rf = prep_rf_tc(loaded['rf'], bins=bins, max_zero_bins=max_zero_bins)
    rf_after_zero_filter = len(rf)
    shared = rf.index.intersection(hd_keys).intersection(ebb_keys)
    rows = ebb_keys.get_indexer(shared)
    ebb = boundary_band_profiles(maps['egocentric_rate_maps'][rows], maps['egocentric_occupancy'],
                                 maps['theta_edges'], maps['distance_edges'], shared, bins=bins)
    label = f"{spec['mouse'].upper()} {spec['date']} Probe {spec['probe']}"
    rf = rf.loc[shared].copy()
    rf.attrs['label'] = f"{label} RF S{spec['rf']}"
    pairs = {}
    audit_rows = []
    for band, table in ebb.items():
        # An absent or constant curve has no identifiable preferred bearing.
        finite = np.isfinite(table).all(axis=1) & np.isfinite(rf).all(axis=1)
        modulated = (table.max(axis=1) > table.min(axis=1)) & (rf.max(axis=1) > rf.min(axis=1))
        selected = shared[finite & modulated]
        normalized = normalize_tc(table.loc[selected])
        normalized.attrs['label'] = f"{label} EBB S{spec['ebb']} · {BAND_LABELS[band]}"
        pairs[band] = (rf.loc[selected].copy(), normalized)
        for key in shared:
            audit_rows.append(dict(unit_key=key, band=band, selected=bool(key in selected),
                                   complete_angular_support=bool(finite.loc[key]),
                                   nonconstant_curves=bool(modulated.loc[key])))
    return dict(pairs=pairs, ebb_hz=ebb, selection=pd.DataFrame(audit_rows),
                counts=dict(hd_class3=len(hd_ids), rf_saved=len(loaded['rf']),
                            rf_after_zero_filter=rf_after_zero_filter, shared=len(shared)))


def save_pair_result(output, name, band, rf, ebb, stats, figures):
    folder = Path(output) / name / band
    folder.mkdir(parents=True, exist_ok=True)
    rf.to_csv(folder / 'rf_profiles.csv')
    ebb.to_csv(folder / 'ebb_profiles.csv')
    stats['peaks'].to_csv(folder / 'paired_peak_angles.csv')
    peaks = stats['peaks']
    upper = np.triu_indices(len(peaks), k=1)
    keys = peaks.index.to_numpy()
    pairs = pd.DataFrame(dict(unit_i=keys[upper[0]], unit_j=keys[upper[1]],
        rf_distance_deg=circular_distance_matrix_deg(peaks.reference_peak_deg)[upper],
        ebb_distance_deg=circular_distance_matrix_deg(peaks.matched_peak_deg)[upper]))
    pairs.to_csv(folder / 'unique_cell_pairs.csv', index=False)
    record = dict(direction=stats['direction'], cells=len(peaks), unique_pairs=len(pairs),
                  peak_sum_rayleigh_score=stats['rayleigh_score'], peak_sum_rayleigh_p=stats['rayleigh_p'])
    (folder / 'statistics.json').write_text(json.dumps(record, indent=2) + '\n')
    for index, (fig, _) in enumerate(figures):
        for extension in ('svg', 'png'):
            fig.savefig(folder / f'{index+1:02d}_comparison.{extension}', dpi=150,
                        bbox_inches='tight', facecolor='white', transparent=False)
    return record


def source_manifest(loaded):
    records = {}
    for key, path in loaded['source_paths'].items():
        with path.open('rb') as stream:
            sha = file_digest(stream, 'sha256').hexdigest()
        records[key] = dict(path=str(path), sha256=sha)
    return dict(spec=loaded['spec'], sources=records, sorting_source=loaded['sorting_source'])


@plt.rc_context(LIGHT_PLOT_STYLE)
def pairwise_overview(statistics, *, title, single_row=False):
    """Show unique off-diagonal cell pairs; the test permutes whole unit labels."""
    if len(statistics) == 1:
        shape, size = (1, 1), (5.7, 5.4)
    elif single_row:
        shape, size = (1, len(statistics)), (4.4 * len(statistics), 4.8)
    else:
        shape, size = (2, 2), (10, 9)
    fig, axes = plt.subplots(*shape, figsize=size, layout='constrained', facecolor='white')
    axes = np.atleast_1d(axes)
    bin_width = 180 / 14
    edges = np.linspace(-bin_width / 2, 180 + bin_width / 2, 16)
    densities = []
    for stats in statistics.values():
        peaks = stats['peaks']
        upper = np.triu_indices(len(peaks), k=1)
        x = circular_distance_matrix_deg(peaks.reference_peak_deg)[upper]
        y = circular_distance_matrix_deg(peaks.matched_peak_deg)[upper]
        densities.append(np.histogram2d(x, y, bins=(edges, edges))[0])
    norm = PowerNorm(gamma=.5, vmin=0, vmax=max(values.max() for values in densities))
    for axis, (band, stats), density in zip(axes.flat, statistics.items(), densities):
        peaks = stats['peaks']
        upper = np.triu_indices(len(peaks), k=1)
        x = circular_distance_matrix_deg(peaks.reference_peak_deg)[upper]
        y = circular_distance_matrix_deg(peaks.matched_peak_deg)[upper]
        image = axis.pcolormesh(edges, edges, density.T, norm=norm, cmap='viridis', rasterized=False)
        axis.plot([0, 180], [0, 180], color='white', linestyle='--', linewidth=1)
        pair = stats['direction']['pairwise']
        inference = (f"Mantel ρ = {pair['spearman_mantel_rho']:.2f}, p = {pair['permutation_p']:.3g}"
                     if np.isfinite(pair['spearman_mantel_rho']) else 'Mantel undefined: constant peak distances')
        axis.set(xlim=(-3, 183), ylim=(-3, 183), aspect='equal',
                 xticks=[0, 90, 180], yticks=[0, 90, 180],
                 xlabel='Between-cell RF peak distance (°)', ylabel='Between-cell EBB peak distance (°)',
                 title=f"{BAND_LABELS[band]} · {len(peaks)} cells / {len(x)} pairs\n" + inference)
    fig.suptitle(title, fontsize=16)
    fig.colorbar(image, ax=axes, label='Unique cell-pair count', shrink=.7)
    return fig
