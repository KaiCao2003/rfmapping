"""Restore historical spatial maps, verifying cached EBC geometry before export."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix, save_npz
from EBB.ebc_pose import POSITION_CORRECTION_POLICY
from EBB.spatial_cell_results import save_position_correction

ROOT = Path('/mnt/senzailab/Kai/#Recording')
OUT = Path('output/session_comparisons_20261012/spatial_ebc')
SESSIONS = [('m22','261009',3,1),('m22','261008',11,2),('m21','261006',11,2),('m20','260921',11,2),('m20','260918',11,2),('m19','260831',2,1)]

def sha(path):
    return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current",action="store_true")
    parser.add_argument("--exclude",nargs="*",default=[])
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--environment-config",type=Path)
    args=parser.parse_args()
    if args.current:
        environments=json.loads((args.environment_config or OUT/"environment_selection.json").read_text())
        OUT=args.output_root or OUT/"current"
    spec=importlib.util.spec_from_file_location('legacy_spatial',Path('spatial_cell_analysis.py'))
    spatial=importlib.util.module_from_spec(spec);spec.loader.exec_module(spatial)
    audit=[]
    for mouse,date,ebc,rf in SESSIONS:
        if f'{mouse}_{date}_{ebc}' in args.exclude: continue
        session=ROOT/mouse/date/f'{date}_{ebc}'
        target=OUT/f'{mouse}_{date}_{ebc}'
        target.mkdir(parents=True,exist_ok=True)
        if (target/'restored_maps.npz').exists():
            audit.append(json.loads((target/'audit.json').read_text()))
            continue
        cache=session/'data/spatial_cells/ProbeA/baseline/egocentric_rate_map.rfmap'
        with np.load(cache) as z:
            ids=z['unitPool'].astype(int)
            saved=np.stack([z[f'unit_{u}'][...,0] for u in ids])
            meta=json.loads(z['metadata'].tobytes())
        candidates=[('old',(370,920,210,760)),('new',(225,720,280,785))]
        if args.current:
            environment=environments[f'{mouse}_{date}_{ebc}']['selected']
            candidates=[entry for entry in candidates if entry[0]==environment]
            if 'bounds_px' in environments[f'{mouse}_{date}_{ebc}']:
                candidates=[('visible_floor_estimate',tuple(environments[f'{mouse}_{date}_{ebc}']['bounds_px']))]
        bounds=candidates[0][1]
        spatial.x_min,spatial.x_max,spatial.y_min,spatial.y_max=bounds
        pose,times,spikes,clusters,good,source=spatial.load_data(session_dir=session,probe='A',bounds_px=bounds,camera_input_channel=1 if (mouse,date) in [('m20','260918'),('m19','260831')] else 6)
        assert set(ids)==set(good), 'Saved map and current good-unit sets differ'
        keep=np.isin(clusters,ids); spikes=spikes[keep];clusters=clusters[keep]
        # Sparse unit-by-frame counts retain every spike and avoid a units x frames dense table.
        row=pd.Index(ids).get_indexer(clusters)
        frame=spatial.nearest_frame_indices(spikes,times)
        counts=csr_matrix((np.ones(len(frame),dtype=np.int32),(row,frame)),shape=(len(ids),len(times)))
        chosen=int(np.asarray(counts.sum(axis=1)).argmax())
        report=dict(mouse=mouse,date=date,ebc_session=ebc,rf_session=rf,unit_count=len(ids),source=source,cache_path=str(cache),cache_sha256=sha(cache),legacy_module_sha256=sha('spatial_cell_analysis.py'),candidate_checks=[])
        matched=None
        if args.current:
            report['environment_selection']=environments[f'{mouse}_{date}_{ebc}']
        for name,bounds in candidates:
            spatial.x_min,spatial.x_max,spatial.y_min,spatial.y_max=bounds
            spatial.cm_per_px=41/(bounds[1]-bounds[0])
            for sign in ([1] if args.current else [-1,1]):
                # Current geometry consumes CW HD directly; historical probes retain both input signs.
                candidate_pose=pose.copy()
                if not args.current: candidate_pose.hd_deg*=(-sign)
                maps=spatial.prepare_session_maps(candidate_pose,times)
                raw=(maps['spike_projection']@counts[chosen].toarray().ravel()).reshape(60,20)
                rate=spatial.compute_rate_map(raw,maps['smoothed_egocentric_occupancy'],5,('wrap','nearest'))
                delta=float(np.nanmax(np.abs(rate-saved[chosen])))
                equal=bool(np.allclose(rate,saved[chosen],rtol=1e-10,atol=1e-10,equal_nan=True))
                report['candidate_checks'].append(dict(bounds_name=name,bounds_px=bounds,heading_sign=sign,unit=int(ids[chosen]),max_abs_error=delta,match=equal))
                if equal or args.current:
                    matched=(bounds,sign,maps)
                    break
            if matched is not None: break
        if matched is None:
            report['status']='no_geometry_match';audit.append(report)
            (target/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
            (OUT/'geometry_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
            print('NO_MATCH',mouse,date,report['candidate_checks'],flush=True)
            continue
        bounds,sign,maps=matched
        projection=maps['spike_projection']
        ego_spikes=(projection@counts.T).toarray().T.reshape(len(ids),60,20)
        calculated=gaussian_filter(ego_spikes.astype(float),sigma=(0,5,5),mode=('nearest','wrap','nearest'))/maps['smoothed_egocentric_occupancy'][None]
        old_delta=float(np.nanmax(np.abs(calculated-saved)))
        if args.current:
            report['old_cache_max_abs_difference_hz']=old_delta
            saved=calculated
            for suffix,lo,hi in [('',None,None),('_0-8',None,8),('_8-16',8,16),('_16-',16,None)]:
                spatial.save_egocentric_rfmap(target/f'egocentric_rate_map{suffix}.rfmap',calculated,maps,ids,source['selected_interval_s'],distance_band_cm=None if not suffix else (lo,hi),metadata={'sourcePose':source['pose_path'],'sourcePoseSha256':sha(source['pose_path']),'boundaryPx':list(bounds),'headingConvention':'north-zero clockwise HD input; forward-zero clockwise egocentric bearings','angleSampling':'bin_centers','arenaSizeCm':[41,41],'cmPerPixel':[41/(bounds[1]-bounds[0]),41/(bounds[3]-bounds[2])],'outsideArenaPolicy':POSITION_CORRECTION_POLICY,'analysisRun':'session_comparisons_20261012_boundary_hold'})
        else:
            np.testing.assert_allclose(calculated,saved,rtol=1e-10,atol=1e-10,equal_nan=True)
        dt=float(np.median(np.diff(times)))
        occupancy=(projection@np.full(len(times),dt)).reshape(60,20)
        x=(maps['corrected_positions_px'][:,0]-bounds[0])*41/(bounds[1]-bounds[0])
        y=(maps['corrected_positions_px'][:,1]-bounds[2])*41/(bounds[3]-bounds[2])
        xedges=np.linspace(0,41,41);yedges=np.linspace(0,41,41)
        xyproj=spatial.prepare_2d_projection(x,y,xedges,yedges)
        alloc_occ=(xyproj@np.full(len(times),dt)).reshape(40,40)
        alloc_spikes=(xyproj@counts.T).toarray().T.reshape(len(ids),40,40)
        alloc_den=gaussian_filter(alloc_occ,1.5,mode='nearest')
        alloc_num=gaussian_filter(alloc_spikes.astype(float),(0,1.5,1.5),mode='nearest')
        alloc_rate=np.divide(alloc_num,alloc_den[None],out=np.full_like(alloc_num,np.nan),where=alloc_den[None]>0)
        hd=pose.hd_deg.to_numpy()%360
        hd_bins=np.minimum(np.searchsorted(maps['theta_edges'],hd,side='right')-1,59)
        hdproj=csr_matrix((np.ones(len(hd)),(hd_bins,np.arange(len(hd)))),shape=(60,len(hd)))
        hd_occ=np.asarray(hdproj.sum(axis=1)).ravel()*dt
        hd_spikes=(hdproj@counts.T).toarray().T
        hd_den=gaussian_filter(hd_occ,1.5,mode='wrap')
        hd_num=gaussian_filter(hd_spikes,(0,1.5),mode=('nearest','wrap'))
        hd_rate=np.divide(hd_num,hd_den[None],out=np.full_like(hd_num,np.nan),where=hd_den[None]>0)
        np.savez_compressed(target/'restored_maps.npz',unit_ids=ids,theta_edges=maps['theta_edges'],distance_edges=maps['distance_edges'],x_edges=xedges,y_edges=yedges,trajectory_x_cm=x,trajectory_y_cm=y,frame_times=times,frame_ids=pose.frame.to_numpy(),egocentric_occupancy=occupancy,egocentric_spike_maps=ego_spikes,egocentric_rate_maps=saved,allocentric_occupancy=alloc_occ,allocentric_spike_maps=alloc_spikes,allocentric_rate_maps=alloc_rate,allocentric_tuning_curves=hd_rate)
        save_npz(target/'frame_spike_counts.npz',counts)
        save_position_correction(target,pose,times,maps)
        raw_xy=maps['raw_positions_px']
        outside=(raw_xy[:,0]<bounds[0])|(raw_xy[:,0]>bounds[1])|(raw_xy[:,1]<bounds[2])|(raw_xy[:,1]>bounds[3])
        report.update(status='verified',bounds_px=list(bounds),heading_sign=sign,heading_convention='north-zero clockwise HD; forward-zero clockwise EBC',angle_sampling='bin_centers',arena_size_cm=[41,41],cm_per_pixel=[41/(bounds[1]-bounds[0]),41/(bounds[3]-bounds[2])],outside_arena_policy=POSITION_CORRECTION_POLICY,position_correction_policy=POSITION_CORRECTION_POLICY,held_position_frames=int(maps['position_held'].sum()),initial_outside_frames=int(maps['initial_outside'].sum()),max_abs_error_all_units=float(np.nanmax(np.abs(calculated-saved))),analysis_mode='current_pipeline' if args.current else 'historical_cache_reproduction',source_pose_sha256=sha(source['pose_path']),outside_bounds_frames=int(outside.sum()),valid_frames=len(times),nonzero_units=int((np.asarray(counts.sum(axis=1)).ravel()>0).sum()),map_path=str(target/'restored_maps.npz'))
        (target/'audit.json').write_text(json.dumps(report,indent=2)+'\n');audit.append(report)
        (OUT/'geometry_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print('VERIFIED',mouse,date,len(ids),'bounds',bounds,'sign',sign,'outside',int(outside.sum()),flush=True)

    (OUT/'geometry_audit.json').write_text(json.dumps(audit,indent=2)+'\n')

if __name__=='__main__':main()
