"""Save paired candidate inputs and inspect EBB temporal stability for the poster."""

from hashlib import file_digest
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from EBB.ebc_analysis import rates_from_counts
from EBB.spatial_cell_analysis import prepare_2d_projection
from HD.profiles import load_hd_tc, hd_profiles
from RF.rflocate import load_rf, load_rfmap, rf_result_path
from Utils.plotting import LIGHT_PLOT_STYLE
from preprocessing.spikeinterface.split_kilosort_by_recording import joint_sorting_source


def sorting_source(session, probe):
    path = session / f'kilosort/Probe{probe}/kilosort_{session.name.split("_")[-1]}/params.py'
    return joint_sorting_source(path)


def prepare(candidate):
    units = np.asarray(candidate['paired_units'])
    folder = Path(candidate['ebb_path']).parent
    session = folder.parents[3]
    rf_path = Path(candidate['rf_path'])
    rf_session = next(p for p in rf_path.parents if p.name == candidate['rf_session'])
    assert sorting_source(session, 'A') == sorting_source(rf_session, 'A')
    with np.load(folder / 'restored_maps.npz') as saved:
        rows = pd.Index(saved['unit_ids']).get_indexer(units)
        assert (rows >= 0).all()
        rates = saved['egocentric_rate_maps'][rows]
        spikes = saved['egocentric_spike_maps'][rows]
        occupancy = saved['egocentric_occupancy']
        theta_edges, distance_edges = saved['theta_edges'], saved['distance_edges']
        times = saved['frame_times']
    dt = np.median(np.diff(times))
    with np.load(folder / 'position_correction.npz') as saved:
        distances = saved['theta_d_cm']
        theta = np.broadcast_to(saved['theta_deg'][None, :], distances.shape)
        held = saved['position_held']
    projection = prepare_2d_projection(theta, distances, theta_edges, distance_edges)
    counts = load_npz(folder / 'frame_spike_counts.npz')[rows]
    np.testing.assert_allclose((projection @ np.full(len(times), dt)).reshape(occupancy.shape), occupancy)
    np.testing.assert_array_equal((projection @ counts.T).toarray().T.reshape(spikes.shape), spikes)
    np.testing.assert_allclose(rates_from_counts(spikes, occupancy, mask_unvisited=False), rates, rtol=1e-12, atol=1e-12)
    rates[:, occupancy == 0] = np.nan
    real_occupancy = (projection @ ((~held) * dt)).reshape(occupancy.shape)
    halves = []
    for selected in (times <= np.median(times), times > np.median(times)):
        subset = projection[:, selected]
        occ = (subset @ np.full(selected.sum(), dt)).reshape(occupancy.shape)
        raw = (subset @ counts[:, selected].T).toarray().T.reshape(spikes.shape)
        halves.append(rates_from_counts(raw, occ))
    half_rates = np.stack(halves)
    stability = []
    for index, unit in enumerate(units):
        left, right = half_rates[:, index]
        valid = np.isfinite(left) & np.isfinite(right)
        stability.append(float(np.corrcoef(left[valid], right[valid])[0, 1]))

    hd = hd_profiles(load_hd_tc(candidate['hd_path']), probe='A', bins=30, hd_class=3)
    hd.index = hd.index.get_level_values('unit_id')
    hd_values = hd.loc[units].to_numpy()
    detections = load_rf(rf_result_path(rf_path, rf_type='excitatory'))
    rf = load_rfmap(rf_path)
    presentations = rf[0].to_firing_rate(reconstruct_presentations=True).presentation_counts
    rf_rates, raw_rf = [], []
    for unit in units:
        response = rf.by_unit_id(int(unit))
        raw_rf.append(response.sum(0, .2).sum_to_1d(axis='x', rf_only=False).to_1d_array())
        rf_rates.append(response.to_firing_rate(presentation_counts=presentations).mean_rate(0, .2).to_2d_array())
    rf_rates, raw_rf = np.stack(rf_rates), np.stack(raw_rf)
    masks = np.stack([detections.for_unit(int(unit)) for unit in units]).astype(bool)
    modulation = (np.nanmax(rates, axis=(1,2)) - np.nanmin(rates, axis=(1,2))) / np.nanmax(rates,axis=(1,2))
    metrics = pd.DataFrame(dict(unit_id=units, half_map_r=stability, ebb_modulation=modulation,
                                hd_peak_hz=hd_values.max(axis=1), rf_peak_hz=rf_rates.max(axis=(1,2))))
    metrics['example_score'] = np.maximum(metrics.half_map_r, 0) * metrics.ebb_modulation
    metrics = metrics.sort_values('example_score', ascending=False)
    stem = f"{candidate['mouse']}_{candidate['date']}_paired"
    target = Path('poster_assets/data') / stem
    np.savez_compressed(target.with_suffix('.npz'), unit_ids=units, rf_x=rf[0].x_positions,
        rf_y=rf[0].y_positions, rf_raw_profiles=raw_rf, rf_on_hz=rf_rates, rf_masks=masks,
        hd_angles=hd.columns.to_numpy(float), hd_hz=hd_values, ebb_rates=rates,
        ebb_spike_maps=spikes, occupancy_s=occupancy, unheld_occupancy_s=real_occupancy,
        theta_edges=theta_edges, distance_edges=distance_edges, half_rates=half_rates)
    metrics.to_csv(target.with_name(stem+'_units.csv'), index=False)
    inputs = [rf_path, rf_result_path(rf_path, rf_type='excitatory'), Path(candidate['hd_path']),
              Path(candidate['ebb_path']), folder/'restored_maps.npz', folder/'position_correction.npz', folder/'frame_spike_counts.npz',
              Path(__file__), Path('EBB/ebc_analysis.py'), Path('EBB/spatial_cell_analysis.py')]
    provenance = dict(candidate=candidate, same_sorting_source=sorting_source(session,'A'),
        held_fraction=float(held.mean()), unheld_visited_fraction=float((real_occupancy>0).mean()),
        unheld_occupancy_p05_s=float(np.quantile(real_occupancy,.05)),
        rate_reconstruction_matches=True, raw_spikes_and_occupancy_match=True,
        half_map_r_definition='Pearson correlation of finite EBB rate-map bins between first and second temporal halves',
        example_ranking='positive half-map correlation times EBB (max-min)/max; descriptive quality, not a significance test',
        sources=[dict(path=str(p),sha256=file_digest(p.open('rb'),'sha256').hexdigest()) for p in inputs])
    target.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n')
    with plt.rc_context({**LIGHT_PLOT_STYLE,'savefig.transparent':False,'font.size':11}):
        fig, axes = plt.subplots(4,3,figsize=(13,13),layout='constrained',facecolor='white')
        for row, unit in enumerate(metrics.unit_id.head(4)):
            index = int(np.flatnonzero(units==unit)[0])
            order = np.argsort(hd.columns.to_numpy(float))
            axis=axes[row,0]; axis.plot(hd.columns.to_numpy()[order],hd_values[index,order],color='black')
            axis.set(title=f'{unit}: HD',xlabel='Direction (deg)',ylabel='Hz')
            axis=axes[row,1]; axis.imshow(rf_rates[index],cmap='gray',aspect='auto')
            axis.set_title(f'ON RF: {metrics.set_index("unit_id").loc[unit,"rf_peak_hz"]:.1f} Hz max')
            axis=axes[row,2]; im=axis.pcolormesh(distance_edges,theta_edges,rates[index],cmap='viridis')
            axis.set_title(f'EBB: half r={stability[index]:.2f}');fig.colorbar(im,ax=axis,label='Hz')
        fig.suptitle(f"{candidate['mouse'].upper()} {candidate['date']} | {len(units)} RF+HD cells | EBB examples")
        fig.savefig(Path('poster_output')/(stem+'_review.png'),dpi=140,facecolor='white',transparent=False)
        plt.close(fig)
    print(stem,metrics.head(5).to_string(index=False),flush=True)
    print('unheld coverage',provenance['unheld_visited_fraction'],'p05',provenance['unheld_occupancy_p05_s'],flush=True)


def main():
    candidates=json.loads(Path('poster_output/poster_cohort_candidates.json').read_text())
    for mouse,date,ebb in [('m19','260831','260831_2'),('m23','261019','261019_3')]:
        prepare(next(c for c in candidates if c['mouse']==mouse and c['date']==date and c['ebb_session']==ebb))


if __name__ == '__main__':
    main()
