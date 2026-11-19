"""Load and calculate a paired same-session HD/EBB example from recording inputs."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from EBB.joint import hd_boundary_joint_maps
from EBB.profiles import boundary_band_profiles
from EBB.quality import boundary_map_quality
from EBB.spatial_cell_analysis import prepare_2d_projection
from HD.profiles import hd_profiles, load_hd_tc


def load_boundary_example(root, spec):
    session = Path(root) / spec['mouse'] / spec['date'] / f"{spec['date']}_{spec['session']}"
    folder = session / f"data/spatial_cells/Probe{spec['probe']}/baseline"
    paths = dict(maps=folder / 'restored_maps.npz', pose=folder / 'position_correction.npz',
                 counts=folder / 'frame_spike_counts.npz', audit=folder / 'audit.json',
                 hd=session / f"data/tuning_curves/Probe{spec['probe']}/tuning_curves.tc")
    with np.load(paths['maps'], allow_pickle=False) as saved:
        row = int(np.flatnonzero(saved['unit_ids'] == spec['unit'])[0])
        maps = {key: saved[key] for key in ('theta_edges', 'distance_edges', 'frame_times',
                                          'frame_ids', 'egocentric_occupancy')}
        maps.update(rate=saved['egocentric_rate_maps'][row], counts=saved['egocentric_spike_maps'][row])
    with np.load(paths['pose'], allow_pickle=False) as saved:
        pose = {key: saved[key] for key in ('frame_ids', 'frame_times', 'theta_deg', 'theta_d_cm',
                                          'position_held', 'hd_deg')}
    return dict(spec=spec, maps=maps, pose=pose, frame_counts=load_npz(paths['counts'])[row],
                audit=json.loads(paths['audit'].read_text()), hd=load_hd_tc(paths['hd']), source_paths=paths)


def calculate_boundary_example(loaded, *, hd_bins=30, joint_smoothing_deg=12.,
                               legacy_angular_sigma_deg=30., legacy_distance_sigma_bins=5.):
    """Reconstruct counts/exposure and verify saved rates before plotting them."""
    maps, pose, spec = loaded['maps'], loaded['pose'], loaded['spec']
    np.testing.assert_array_equal(maps['frame_ids'], pose['frame_ids'])
    np.testing.assert_array_equal(maps['frame_times'], pose['frame_times'])
    theta_edges, distance_edges = maps['theta_edges'], maps['distance_edges']
    np.testing.assert_allclose((theta_edges[:-1] + theta_edges[1:]) / 2, pose['theta_deg'])
    sigma = loaded['audit'].get('analysis_config', {}).get('smoothing_sigma_bins',
        [legacy_angular_sigma_deg / np.diff(theta_edges)[0], legacy_distance_sigma_bins])
    projection = prepare_2d_projection(pose['theta_deg'][None], pose['theta_d_cm'], theta_edges, distance_edges)
    quality = boundary_map_quality(projection, loaded['frame_counts'], maps['frame_times'],
        pose['position_held'], maps['egocentric_occupancy'].shape, sigma=sigma)
    np.testing.assert_array_equal(quality['spike_maps'][0], maps['counts'])
    np.testing.assert_allclose(quality['occupancy'], maps['egocentric_occupancy'], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(quality['display_rates'][0], maps['rate'], rtol=1e-12, atol=1e-12)
    profiles = boundary_band_profiles(quality['rates'], quality['occupancy'], theta_edges, distance_edges,
                                     [spec['unit']], bins=len(theta_edges)-1)
    bands = {name: table.attrs['distance_bin_indices'] for name, table in profiles.items() if name != 'all'}
    joint = hd_boundary_joint_maps(projection, loaded['frame_counts'].toarray().ravel(), maps['frame_times'],
        pose['hd_deg'], theta_edges, distance_edges, bands, hd_bins=hd_bins, smoothing_deg=joint_smoothing_deg)
    np.testing.assert_array_equal(joint['spike_maps'].sum(axis=0), maps['counts'])
    np.testing.assert_allclose(joint['occupancy_maps'].sum(axis=0), quality['occupancy'], rtol=1e-12, atol=1e-9)
    hd = hd_profiles(loaded['hd'], probe=spec['probe'], bins=hd_bins, smoothing_deg=0, hd_class=3)
    curve = hd.loc[(spec['probe'], spec['unit'])]
    np.testing.assert_allclose(curve.index.to_numpy(float) % 360,
                               (joint['hd_edges'][:-1] + joint['hd_edges'][1:]) / 2)
    products = {name: np.outer(curve.to_numpy() / curve.max(),
                table.iloc[0].to_numpy() / table.iloc[0].max()) for name, table in profiles.items() if name != 'all'}
    return dict(spec=spec, rate=quality['display_rates'][0], quality=quality,
                theta_edges=theta_edges, distance_edges=distance_edges, smoothing_sigma_bins=sigma,
                joint=joint, hd=curve, profiles=profiles, products=products)
