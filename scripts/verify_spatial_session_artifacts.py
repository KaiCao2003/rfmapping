"""Check native ring partition, source identity, current maps, and completed figure counts."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from scripts.audit_restore_spatial_sessions import ROOT,OUT,SESSIONS,sha
from EBB.spatial_cell_analysis import read_good_unit_ids

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-root',type=Path,default=OUT/'current')
args=parser.parse_args()
out=args.output_root;reports=[]
for m,d,e,r in SESSIONS:
    target=out/f'{m}_{d}_{e}'
    if not (target/'audit.json').exists():continue
    audit=json.loads((target/'audit.json').read_text())
    with np.load(target/'restored_maps.npz') as z:
        ids=z['unit_ids'];rates=z['egocentric_rate_maps']
        assert np.isfinite(rates).all() and (rates>=0).all()
        assert set(ids)==set(read_good_unit_ids(Path(audit['source']['kilosort_dir'])))
        all_projection=rates.sum(axis=2)
        summed=np.zeros_like(all_projection)
        for band in ['0-8','8-16','16-']:
            with np.load(target/f'egocentric_rate_map_{band}.rfmap') as ring:
                np.testing.assert_array_equal(ring['unitPool'],ids)
                summed+=np.stack([ring[f'unit_{u}'][:,0,0] for u in ids])
        np.testing.assert_allclose(summed,all_projection,rtol=1e-13,atol=1e-13)
        counts=load_npz(target/'frame_spike_counts.npz')
        assert counts.shape==(len(ids),len(z['frame_times']))
        assert z['egocentric_spike_maps'].sum()<=counts.sum()*60
        assert z['allocentric_spike_maps'].sum()<=counts.sum()
    assert sha(audit['source']['pose_path'])==audit['source_pose_sha256']
    pngs=list((target/'all_units').glob('unit_*.png'))
    reports.append(dict(mouse=m,date=d,ebc_session=e,units=len(ids),data_pass=True,plots_found=len(pngs),plots_complete=len(pngs)==len(ids),outside_bounds_frames=audit['outside_bounds_frames'],valid_frames=audit['valid_frames'],old_cache_max_abs_difference_hz=audit['old_cache_max_abs_difference_hz']))
manifest_path=out/'comparison_manifest.json'
comparisons=json.loads(manifest_path.read_text())
for entry in comparisons:
    if entry['comparison']=='HD_all_descriptive':
        entry['hd_selection']='all tracked nonzero units; no HD classification'
        entry['method']=entry['method'].replace('Class3 HD','All tracked nonzero HD curves (descriptive, no HD classification)')
        (Path(entry['figures'][0]['path']).parent/'manifest.json').write_text(json.dumps(entry,indent=2)+'\n')
    for figure in entry['figures']:
        assert Path(figure['path']).is_file()
manifest_path.write_text(json.dumps(comparisons,indent=2)+'\n')
summary=dict(sessions=reports,all_data_pass=True,all_plots_complete=all(x['plots_complete'] for x in reports),comparison_count=len(comparisons),nonempty_comparisons=sum(x['n_shared']>0 for x in comparisons),comparison_pngs=sum(len(x['figures']) for x in comparisons))
(out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
