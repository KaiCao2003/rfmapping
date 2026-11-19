"""Load and prepare a same-sorting RF/EBB cohort for the poster heatmaps."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from EBB.profiles import boundary_band_profiles
from EBB.quality import boundary_map_quality
from EBB.spatial_cell_analysis import prepare_2d_projection
from HD.profiles import load_hd_tc
from RF.paths import rf_source_path
from RF.profiles import prep_rf_tc
from RF.rflocate import load_rf, load_rfmap, rf_result_path
from Utils.direction_comparison import concat
from scripts.prepare_poster_candidates import sorting_source


# One paired recording per mouse; M19 retains the user's chosen example date.
# M20 uses its earliest complete baseline pairing. Other mice retain the poster dates.
RF_EBB_RECORDINGS = [
    dict(mouse='m19', date='260831', probe='A', rf=1, hd=2, ebb=2),
    dict(mouse='m20', date='260918', probe='A', rf=2, hd=9, ebb=11),
    dict(mouse='m21', date='261006', probe='A', rf=3, hd=9, ebb=11),
    dict(mouse='m22', date='261008', probe='A', rf=2, hd=9, ebb=11),
    dict(mouse='m23', date='261015', probe='A', rf=2, hd=9, ebb=11),
    dict(mouse='m24', date='261104', probe='A', rf=2, hd=10, ebb=11),
]


def load_sorting_inputs(root, spec):
    """Read recording inputs; no exported poster tables are needed."""
    base = Path(root) / spec['mouse'] / spec['date']
    sessions = {kind: base / f"{spec['date']}_{spec[kind]}" for kind in ('rf', 'hd', 'ebb')}
    probe = spec['probe']
    folder = sessions['ebb'] / f'data/spatial_cells/Probe{probe}/baseline'
    raw_rf = rf_source_path(sessions['rf'], probe)
    paths = dict(rf=raw_rf, detection=rf_result_path(raw_rf, rf_type='excitatory'),
        hd=sessions['hd'] / f'data/tuning_curves/Probe{probe}/tuning_curves.tc',
        maps=folder / 'restored_maps.npz', pose=folder / 'position_correction.npz',
        counts=folder / 'frame_spike_counts.npz', audit=folder / 'audit.json')
    sortings = [sorting_source(session, probe) for session in sessions.values()]
    if not all(value == sortings[0] for value in sortings):
        raise ValueError('RF, HD and EBB sessions must share the same Kilosort source')
    for kind, session in sessions.items():
        paths['sorting_' + kind] = session / f'kilosort/Probe{probe}/kilosort_{spec[kind]}/params.py'
    audit = json.loads(paths['audit'].read_text())
    if audit['status'] != 'complete' or audit['angle_sampling'] != 'bin_centers':
        raise ValueError('RF–EBB comparison requires complete bin-center EBB maps')
    if audit['heading_convention'] != 'physical north-zero clockwise heading; clockwise egocentric bearings':
        raise ValueError('Select an explicit angle conversion before pooling EBB conventions')
    with np.load(paths['maps'], allow_pickle=False) as saved:
        maps = {key: saved[key] for key in ('unit_ids', 'egocentric_rate_maps', 'egocentric_spike_maps',
            'egocentric_occupancy', 'theta_edges', 'distance_edges', 'frame_ids', 'frame_times')}
    with np.load(paths['pose'], allow_pickle=False) as saved:
        pose = {key: saved[key] for key in ('frame_ids', 'frame_times', 'theta_deg', 'theta_d_cm', 'position_held')}
    return dict(spec=spec, maps=maps, pose=pose, frame_counts=load_npz(paths['counts']),
        rf=load_rfmap(raw_rf), detection=load_rf(paths['detection']), hd=load_hd_tc(paths['hd']),
        audit=audit, sorting_source=sortings[0], source_paths=paths)


def prepare_sorting_profiles(loaded, *, bins=30, legacy_angular_sigma_deg=30., legacy_distance_sigma_bins=5.):
    """Select all HD-class-3 / ON-RF cells and verify their saved EBB maps."""
    spec, maps, pose = loaded['spec'], loaded['maps'], loaded['pose']
    hd, detection, rf = loaded['hd'], loaded['detection'], loaded['rf']
    hd_ids = np.asarray(hd['unit_id'])[np.asarray(hd['unit_data']['hd_class']) == 3]
    rf_ids = detection.unit_ids[detection.mask_2d.any(axis=(1, 2))]
    units = np.intersect1d(np.intersect1d(hd_ids, rf_ids), maps['unit_ids'])
    rows = pd.Index(maps['unit_ids']).get_indexer(units)
    keys = pd.Index([f"{spec['mouse']}:{spec['date']}:{spec['probe']}:{unit}" for unit in units], name='unit_key')
    np.testing.assert_allclose(detection.manifest['time_range_s'], [0, .2])
    native_rf = pd.DataFrame([rf.by_unit_id(int(unit)).sum(0, .2).to_2d_array().sum(axis=0)
        for unit in units], index=keys, columns=rf[0].x_positions)
    native_rf.attrs.update(label='All-height ON RF', range=[-180, 180], response_units='spike_count')
    rf_profiles = prep_rf_tc(native_rf, bins=bins, max_zero_bins=None)

    np.testing.assert_array_equal(maps['frame_ids'], pose['frame_ids'])
    np.testing.assert_array_equal(maps['frame_times'], pose['frame_times'])
    theta_edges, distance_edges = maps['theta_edges'], maps['distance_edges']
    np.testing.assert_allclose((theta_edges[:-1] + theta_edges[1:]) / 2, pose['theta_deg'])
    sigma = loaded['audit'].get('analysis_config', {}).get('smoothing_sigma_bins',
        [legacy_angular_sigma_deg / np.diff(theta_edges)[0], legacy_distance_sigma_bins])
    projection = prepare_2d_projection(pose['theta_deg'][None], pose['theta_d_cm'], theta_edges, distance_edges)
    quality = boundary_map_quality(projection, loaded['frame_counts'][rows], maps['frame_times'],
        pose['position_held'], maps['egocentric_occupancy'].shape, sigma=sigma)
    np.testing.assert_array_equal(quality['spike_maps'], maps['egocentric_spike_maps'][rows])
    np.testing.assert_allclose(quality['occupancy'], maps['egocentric_occupancy'], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(quality['display_rates'], maps['egocentric_rate_maps'][rows], rtol=1e-12, atol=1e-12)
    ebb = boundary_band_profiles(quality['rates'], quality['occupancy'], theta_edges, distance_edges, keys, bins=bins)
    profiles = {'RF': rf_profiles, **ebb}
    for table in profiles.values():
        assert table.index.equals(keys)
        assert np.isfinite(table.to_numpy()).all()
        assert (table.max(axis=1) > table.min(axis=1)).all()
    metrics = pd.DataFrame(quality['metrics'], index=keys)
    return dict(profiles=profiles, native_rf=native_rf, quality=quality, metrics=metrics, unit_ids=units,
        metadata=dict(**spec, n_cells=len(units), hd_class=3, anatomy='all_hd',
            selection='All same-sorting HD class-3 cells with a saved ON RF and an EBB map; no per-cell stability filter',
            rf_projection='All-height 0–200 ms raw spike-count sum; no RF-only mask or zero-bin exclusion',
            ebb_projection='Equal mean of finite saved rate-map bins in each distance band',
            normalization='Display only: response / row maximum, separately per panel; color scale 0 to max',
            sorting_source=loaded['sorting_source'], raw_reconstruction_passed=True,
            smoothing_sigma_bins=sigma, median_half_r=float(metrics.half_r.median()),
            distance_support_cm={name: table.attrs['distance_support_cm'] for name, table in ebb.items()},
            **quality['summary']))


def pool_sorting_profiles(cohorts):
    """Stack one recording per mouse on matching grids, retaining full cell keys."""
    mice = [cohort['metadata']['mouse'] for cohort in cohorts]
    if len(set(mice)) != len(mice):
        raise ValueError('The pooled poster cohort uses one paired recording per mouse')
    return {name: concat(*(cohort['profiles'][name] for cohort in cohorts),
                         label=f'{len(mice)} mice · {name}')
            for name in cohorts[0]['profiles']}
