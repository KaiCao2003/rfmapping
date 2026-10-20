"""Export three saved EBC rings against paired RF and same-session HD curves."""
import hashlib
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from EBB.profiles import ebc_profiles
from RF.rflocate import load_rfmap
from HD.profiles import load_hd_tc, hd_profiles
from Utils.direction_comparison import normalize_tc, rebin_profiles, concat, save_tc, tcRange, tc_loader, convert_profile_coordinates
from HD.profiles import hd_pick
from RF.profiles import rf_pick
from RF.tc_generation import generate_rf_comparison, generate_rf_tc
from Utils.tc_comparison import calculate_tc_statistics, plot_tc_comparison
from Utils.plotting import LIGHT_PLOT_STYLE
from scripts.audit_restore_spatial_sessions import ROOT, OUT, SESSIONS, sha
from EBB.ebc_pose import POSITION_CORRECTION_POLICY

BANDS=['0-8','8-16','16-']

def current_ebc_metadata(path, *, require_boundary_hold=False):
    with np.load(path) as saved:
        metadata=json.loads(saved['metadata'].tobytes())
        centers=(np.asarray(metadata['yBinEdges'])[:-1]+np.asarray(metadata['yBinEdges'])[1:])/2
        if (metadata.get('angleConvention')!='egocentric clockwise: 0 forward, 90 right'
                or metadata.get('angleSampling')!='bin_centers'
                or not np.array_equal(saved['yPositions'],centers)):
            raise ValueError(f'Current comparisons require regenerated clockwise, bin-center EBC maps: {path}')
        if (metadata.get('arenaSizeCm') != [40, 40]
                or metadata.get('headingCalibration') != 'image_unit_vector_scaled_by_cm_per_pixel_xy'):
            raise ValueError(f'Current comparisons require a 40-cm arena and physically scaled heading: {path}')
        if require_boundary_hold and metadata.get('positionCorrectionPolicy') != POSITION_CORRECTION_POLICY:
            raise ValueError(f'Boundary-hold comparisons require corrected positions: {path}')
    return metadata

def thin_unit_labels(ax,max_labels=25):
    ticks=ax.get_yticks()
    if len(ticks)>max_labels:
        labels=[label.get_text() for label in ax.get_yticklabels()]
        selected=np.linspace(0,len(ticks)-1,max_labels,dtype=int)
        ax.set_yticks(ticks[selected],labels=[labels[i] for i in selected])

def keyed(table,mouse,date,label,angle_range):
    table=table.copy()
    table.index=[f'{mouse}:{date}:A:{u}' for u in table.index.get_level_values('unit_id')]
    table.attrs.update(label=label,range=angle_range)
    table.attrs.pop('unit_info',None)
    return table

def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,np.generic):return clean_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def compare(first,second,target,meta):
    target.mkdir(parents=True,exist_ok=True)
    units=first.index.intersection(second.index)
    first=first.loc[units];second=second.loc[units]
    save_tc(first,target/'reference.csv');save_tc(second,target/'matched.csv')
    result=dict(meta,n_shared=len(units),unit_ids=list(units),figures=[],statistics_bins=15,n_permutations=10000,random_seed=1)
    if not len(units):
        result['status']='no_shared_units'
    else:
        stats=calculate_tc_statistics(first,second,bins=15,n_permutations=10000,random_seed=1)
        stats['peaks'].to_csv(target/'peaks.csv')
        result.update(status='complete',direction=clean_json(stats['direction']),rayleigh_score=float(stats['rayleigh_score']),rayleigh_p=float(stats['rayleigh_p']))
        figs=plot_tc_comparison(first,second,stats,figsize=(7,6.8),show=False)
        for i,(fig,ax) in enumerate(figs):
            if i<12: thin_unit_labels(ax)
            if i==12: ax.set_thetagrids((0,90,180,270),labels=("0°","90°","180°","−90°"))
            path=target/f'comparison_{i:02d}.png'
            fig.savefig(path,dpi=150,facecolor='white',transparent=False,bbox_inches='tight')
            result['figures'].append(dict(path=str(path),title=ax.get_title(),index=i))
            plt.close(fig)
    (target/'manifest.json').write_text(json.dumps(clean_json(result),indent=2)+'\n')
    print(meta['name'],len(units),'units',flush=True)
    return result

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current",action="store_true")
    parser.add_argument("--exclude",nargs="*",default=[])
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--input-root",type=Path)
    parser.add_argument("--changed-session")
    parser.add_argument("--require-boundary-hold",action="store_true")
    parser.add_argument("--generate-inputs",action="store_true",
                        help="Explicitly generate missing RF projection and comparison CSV inputs")
    parser.add_argument("--overwrite-generated-inputs",action="store_true",
                        help="Replace RF CSV inputs when used with --generate-inputs")
    args=parser.parse_args()
    source_root=args.input_root
    OUT=args.output_root or (OUT/"current" if args.current else OUT)
    plt.rcParams.update(LIGHT_PLOT_STYLE)
    pooled={band:{'EBC':[],'RF':[],'HD':[]} for band in BANDS}
    pooled_sources={band:[] for band in BANDS}
    manifest=[]
    for mouse,date,e,r in SESSIONS:
        if f'{mouse}_{date}_{e}' in args.exclude: continue
        cohort=f'{mouse}_{date}_{e}'
        label=f'{mouse} {date}';target=OUT/cohort
        target.mkdir(parents=True,exist_ok=True)
        ebc_session=ROOT/mouse/date/f'{date}_{e}';rf_session=ROOT/mouse/date/f'{date}_{r}'
        hdfile=ebc_session/'data/tuning_curves/ProbeA/tuning_curves.tc'
        all_hd_native=hd_profiles(load_hd_tc(hdfile),probe='A',bins=30,smoothing_deg=0)
        hd_native=hd_pick(all_hd_native,3)
        hd=normalize_tc(keyed(hd_native,mouse,date,f'{label} HD{e}',tcRange(False)))
        raw=rf_session/f'data/rfmapping/good/-100_400_1ms/ProbeA/regular_unitsSpikeCounts_{date}_{r}.rfmap'
        detected=raw.with_suffix('.npz')
        if not detected.exists():
            detected=Path('output/session_comparisons_20261012/overlap/ebc_pair_detections')/f'{mouse}_{date}_{r}_ProbeA/excitatory_2d.npz'
        native=rf_session/f'data/regular_unitsSpikeCounts_{date}_{r}_ProbeA_1d_rfonly.csv'
        rfpath=target/'rf_excitatory_comparison.csv'
        if not native.exists() or not rfpath.exists():
            if not args.generate_inputs:
                raise FileNotFoundError(
                    f"Missing saved RF comparison input(s) for {cohort}; "
                    "rerun with --generate-inputs to create them explicitly."
                )
            native=(source_root/cohort if source_root else target)/'rf_native_rfonly.csv'
        if args.generate_inputs:
            if args.overwrite_generated_inputs or not native.exists():
                generate_rf_comparison(
                    raw, rfpath, projection_path=native, detected_rf_path=detected,
                    time_range_s=(0,.2), rf_only=True, unit_prefix=f'{mouse}:{date}:A',
                    overwrite=args.overwrite_generated_inputs,
                )
            elif not rfpath.exists():
                generate_rf_tc(native, rfpath, select_results=(detected,),
                               unit_prefix=f'{mouse}:{date}:A')
        rf=normalize_tc(rebin_profiles(rf_pick(tc_loader(rfpath,label=f'{label} RF{r}',range=tcRange(True),response_units='spike_count'),2),bins=30))
        for band in BANDS:
            source=ebc_session/f'data/spatial_cells/ProbeA/baseline/egocentric_rate_map_{band}.rfmap'
            if source_root is not None:
                source=source_root/cohort/f'egocentric_rate_map_{band}.rfmap'
            source_metadata=current_ebc_metadata(source,require_boundary_hold=args.require_boundary_hold) if args.current else None
            pooled_sources[band].append(dict(path=str(source),sha256=sha(source),metadata=source_metadata,
                hd_source=str(hdfile),hd_source_sha256=sha(hdfile),rf_source=str(native),
                rf_source_sha256=sha(native),rf_detection=str(detected),rf_detection_sha256=sha(detected)))
            native_ebc=ebc_profiles(load_rfmap(source),probe='A')
            nonzero=np.any(np.isfinite(native_ebc)&(native_ebc!=0),axis=1)
            ebc_tc=normalize_tc(rebin_profiles(keyed(native_ebc.loc[nonzero],mouse,date,f'{label} EBC{e} {band}cm',tcRange(True)),bins=30))
            pooled[band]['EBC'].append(ebc_tc);pooled[band]['RF'].append(rf);pooled[band]['HD'].append(hd)
            if args.changed_session and args.changed_session!=cohort: continue
            comparisons=[('RF',rf),('HD',hd)]
            if not len(hd):
                selected=all_hd_native.loc[np.any(np.isfinite(all_hd_native)&(all_hd_native!=0),axis=1)]
                all_hd=normalize_tc(keyed(selected,mouse,date,f'{label} HD{e} all tracked units (no HD classification)',tcRange(False)))
                comparisons.append(('HD_all_descriptive',all_hd))
            for kind,other in comparisons:
                info=dict(name=f'{mouse}_{date}_EBC{e}_{band}_{kind}',mouse=mouse,date=date,ebc_session=e,rf_session=r,ring=band,comparison=kind,source=str(source),source_sha256=sha(source),hd_source=str(hdfile),rf_source=str(native),rf_detection=str(detected),n_ebc_nonzero=len(ebc_tc),n_hd_class3=len(hd),n_rf_selected=len(rf),method='Saved EBC rates summed over distance-bin centers; 60-to-30 adjacent angular bins; independent maximum normalization; Class3 HD; excitatory 2D RF rows; <=2 native zero RF bins; 10000 permutations seed1.')
                info['hd_selection']='all tracked nonzero units; no HD classification' if kind=='HD_all_descriptive' else 'Class3'
                info['hd_source_sha256']=sha(hdfile)
                info['rf_source_sha256']=sha(native)
                info['rf_detection_sha256']=sha(detected)
                info['ebc_source_metadata']=source_metadata
                if kind=='HD_all_descriptive': info['method']=info['method'].replace('Class3 HD','All tracked nonzero HD curves (descriptive, no HD classification)')
                manifest.append(compare(ebc_tc,other,target/f'ring_{band}_{kind}',info))
                (OUT/'comparison_manifest.json').write_text(json.dumps(clean_json(manifest),indent=2)+'\n')
    for band,tables in pooled.items():
        ebc=concat(*tables['EBC'],label=f'Pooled EBC {band}cm')
        for kind in ['RF','HD']:
            other=concat(*tables[kind],label=f'Pooled {kind}')
            info=dict(name=f'pooled_EBC_{band}_{kind}',mouse='pooled',ring=band,comparison=kind,ebc_sources=pooled_sources[band])
            manifest.append(compare(ebc,other,OUT/f'pooled_{band}_{kind}',info))
            (OUT/'comparison_manifest.json').write_text(json.dumps(clean_json(manifest),indent=2)+'\n')

if __name__=='__main__':main()
