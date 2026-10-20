"""Show selected pipeline presets and observed pose coverage without changing calibration."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO
from fractions import Fraction
from subprocess import run
import numpy as np
import pandas as pd
from Utils.plotting import LIGHT_PLOT_STYLE
from scripts.audit_restore_spatial_sessions import ROOT,OUT,SESSIONS

plt.rcParams.update(LIGHT_PLOT_STYLE)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-root', type=Path, default=OUT/'current')
args = parser.parse_args()
for m,d,e,r in SESSIONS:
    target=args.output_root/f'{m}_{d}_{e}'
    if not (target/'audit.json').exists():
        continue
    audit=json.loads((target/'audit.json').read_text())
    inferred = 'bounds_source' in audit['environment_selection']
    boundary_label = 'visible floor estimate' if inferred else f'pipeline {audit["environment_selection"]["selected"]} preset'
    s=ROOT/m/d/f'{d}_{e}'
    pose=pd.read_csv(audit['source']['pose_path'])
    bounds=audit['bounds_px'];left,right,top,bottom=bounds
    out=(pose.center_x<left)|(pose.center_x>right)|(pose.center_y<top)|(pose.center_y>bottom)
    video=s/f'{d}.avi'
    probe=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,avg_frame_rate','-of','json',str(video)],check=True,capture_output=True).stdout)['streams'][0]
    frame=int(probe['nb_frames'])//2;seconds=frame/float(Fraction(probe['avg_frame_rate']))
    png=run(['ffmpeg','-v','error','-ss',str(seconds),'-i',str(video),'-frames:v','1','-f','image2pipe','-vcodec','png','-'],check=True,capture_output=True).stdout
    img=plt.imread(BytesIO(png),format='png')
    fig,axes=plt.subplots(1,2,figsize=(14,6.4),layout='constrained')
    axes[0].imshow(img)
    axes[0].set_title(f'Video t={seconds:.1f}s | {boundary_label}')
    for ax in axes:ax.add_patch(Rectangle((left,top),right-left,bottom-top,fill=False,color='#df4b33',lw=2))
    correction_path=target/'position_correction.npz'
    if correction_path.exists():
        with np.load(correction_path) as correction:
            corrected=correction['corrected_xy_px'];raw=correction['raw_xy_px']
            held=correction['position_held'];initial=correction['initial_outside']
        axes[1].scatter(corrected[:,0],corrected[:,1],s=.4,color='0.5',alpha=.3,label='Corrected positions')
        axes[1].scatter(raw[held|initial,0],raw[held|initial,1],s=.8,color='#df4b33',alpha=.4,label='Raw outside positions (audit only)')
        axes[1].legend(fontsize=9)
        axes[1].set_title(f'Boundary hold: {held.sum():,} frames | leading outside missing: {initial.sum():,}')
    else:
        axes[1].scatter(pose.center_x,pose.center_y,s=.4,color='0.65',alpha=.3)
        axes[1].scatter(pose.center_x[out],pose.center_y[out],s=.6,color='#df4b33',alpha=.3)
        axes[1].set_title(f'All tracked centers | outside bounds: {out.sum():,}/{len(pose):,} ({out.mean():.1%})')
    axes[1].set_ylim(img.shape[0],0);axes[1].set_xlim(0,img.shape[1]);axes[1].set_aspect('equal')
    for ax in axes:ax.set_xlabel('Video x (px)');ax.set_ylabel('Video y (px, down)')
    note = ' | visual estimate, ~5-10 px uncertainty' if inferred else ''
    fig.suptitle(f'{m} {d}_{e} | bounds {bounds} px{note}',fontsize=15)
    path=target/'geometry_qc.png';fig.savefig(path,dpi=160,facecolor='white',transparent=False);plt.close(fig)
    print(path,flush=True)
