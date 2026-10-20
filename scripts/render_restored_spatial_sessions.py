"""Render all ten historical views per good unit from verified saved arrays."""
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from EBB.spatial_cell_plotting import load_restored_results, prepare_spatial_profiles, restored_unit, plot_maps, MAP_PLOTS, PLOT_LAYOUTS, draw_map
from HD.profiles import load_hd_tc
from Utils.plotting import LIGHT_PLOT_STYLE
from scripts.audit_restore_spatial_sessions import ROOT,OUT,SESSIONS

DATA=None

def render_unit(unit_id):
    metadata,session,counts,target,examples=DATA
    path=target/'all_units'/f'unit_{unit_id}.png'
    if path.exists():
        detail=[dict(name=p.stem.split(f'unit_{unit_id}_',1)[1],path=str(p)) for p in sorted((target/'examples').glob(f'unit_{unit_id}_*.png'))]
        return dict(unit=int(unit_id),path=str(path),details=detail)
    maps=restored_unit(session,counts,unit_id)
    fig=plot_maps(maps,unit_id,metadata)
    fig.savefig(path,dpi=90,facecolor='white',transparent=False)
    plt.close(fig)
    detail=[]
    if unit_id in examples:
        for plot in MAP_PLOTS:
            with plt.rc_context(LIGHT_PLOT_STYLE):
                size,projection=PLOT_LAYOUTS[plot.kind]
                fig,ax=plt.subplots(figsize=size,layout='constrained',subplot_kw={'projection':projection})
                draw_map(ax,maps,plot)
                fig.suptitle(f"{metadata['mouse']} {metadata['date']}_{metadata['ebc_session']} | Unit {unit_id}",fontsize=11)
                p=target/'examples'/f'unit_{unit_id}_{plot.filename}.png'
                fig.savefig(p,dpi=160,facecolor='white',transparent=False)
                plt.close(fig)
                detail.append(dict(name=plot.filename,path=str(p)))
    return dict(unit=int(unit_id),path=str(path),details=detail)

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current",action="store_true")
    parser.add_argument("--exclude",nargs="*",default=[])
    parser.add_argument("--output-root",type=Path)
    args=parser.parse_args()
    if args.current: OUT=args.output_root or OUT/"current"
    global DATA
    manifests=[]
    for mouse,date,e,r in SESSIONS:
        if f'{mouse}_{date}_{e}' in args.exclude: continue
        target=OUT/f'{mouse}_{date}_{e}'
        if not (target/'restored_maps.npz').exists():continue
        metadata,session,counts=load_restored_results(target)
        hd_tc_path=ROOT/mouse/date/f'{date}_{e}'/'data/tuning_curves'/f"Probe{metadata['probe_name']}"/'tuning_curves.tc'
        session=prepare_spatial_profiles(session,load_hd_tc(hd_tc_path))
        (target/'all_units').mkdir(exist_ok=True);(target/'examples').mkdir(exist_ok=True)
        rates=session['egocentric_rate_maps']; totals=np.asarray(counts.sum(axis=1)).ravel()
        modulation=np.nanstd(rates,axis=(1,2))/np.maximum(np.nanmean(rates,axis=(1,2)),1e-12)
        candidates=np.flatnonzero(totals>=100)
        order=candidates[np.argsort(modulation[candidates])[::-1]]
        examples=set(session['unit_ids'][order[:2]].astype(int))
        DATA=metadata,session,counts,target,examples
        outputs=[]
        with ProcessPoolExecutor(max_workers=8,mp_context=get_context('fork')) as ex:
            for output in ex.map(render_unit,session['unit_ids'],chunksize=3):
                outputs.append(output)
                if len(outputs)%50==0:print(mouse,date,len(outputs),'/',len(session['unit_ids']),flush=True)
        report=dict(mouse=mouse,date=date,ebc_session=e,unit_count=len(outputs),example_units=[int(u) for u in sorted(examples)],example_selection='Two largest EBC-map coefficient of variation among >=100-spike units; descriptive only.',plots=outputs,metadata=metadata)
        (target/'spatial_plot_manifest.json').write_text(json.dumps(report,indent=2)+'\n');manifests.append(report)
        (OUT/'spatial_manifest.json').write_text(json.dumps(manifests,indent=2)+'\n')
        print('DONE',mouse,date,len(outputs),flush=True)

if __name__=='__main__':main()
