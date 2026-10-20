"""Inspect visible floor edges against the configured rectangular EBC boundary."""
import json
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from subprocess import run

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from Utils.plotting import LIGHT_PLOT_STYLE
OUT=Path('output/session_comparisons_20261012/spatial_ebc/boundary_review_1009')
SESSION=Path('/mnt/senzailab/Kai/#Recording/m22/261009/261009_3')
PRESET=(225,720,280,785)
INFERRED=(230,740,335,835)

OUT.mkdir(parents=True,exist_ok=True)
video=SESSION/'261009.avi'
info=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,avg_frame_rate','-of','json',str(video)],check=True,capture_output=True).stdout)['streams'][0]
fps=float(Fraction(info['avg_frame_rate']));n=int(info['nb_frames'])
plt.rcParams.update(LIGHT_PLOT_STYLE)
fig,axes=plt.subplots(2,3,figsize=(15,10.5),layout='constrained')
frames=[]
for ax,fraction in zip(axes.ravel(),[0,.2,.4,.6,.8,.99]):
    frame=int(fraction*(n-1));seconds=frame/fps
    png=run(['ffmpeg','-v','error','-ss',str(seconds),'-i',str(video),'-frames:v','1','-f','image2pipe','-vcodec','png','-'],check=True,capture_output=True).stdout
    (OUT/f'frame_{frame:06d}.png').write_bytes(png)
    image=plt.imread(BytesIO(png),format='png');ax.imshow(image)
    for bounds,color,style,label in [(PRESET,'#e34a33','--','Pipeline preset'),(INFERRED,'#00dadf','-','Visible floor estimate')]:
        left,right,top,bottom=bounds
        ax.add_patch(Rectangle((left,top),right-left,bottom-top,fill=False,color=color,lw=1.8,ls=style,label=label))
    ax.set_xlim(170,800);ax.set_ylim(905,240);ax.set_title(f'frame {frame:,} | {seconds:.1f}s');ax.set_xlabel('x (px)');ax.set_ylabel('y (px)')
    frames.append(dict(frame=frame,time_s=seconds,path=str(OUT/f'frame_{frame:06d}.png')))
axes[0,0].legend(loc='upper right',fontsize=8,facecolor='white',framealpha=1)
fig.suptitle('m22 261009_3 | visible floor boundary review | estimated rectangle',fontsize=16)
fig.savefig(OUT/'boundary_overlay_review.png',dpi=160,facecolor='white',transparent=False);plt.close(fig)
pose=pd.read_csv(SESSION/'261009.csv')
report=dict(session=str(SESSION),video=str(video),method='Rectangle estimated from stable visible floor-wall intersections across six raw video frames; no pose points used to choose bounds. Perspective/edge uncertainty approximately 5-10 pixels.',preset_bounds_px=PRESET,inferred_bounds_px=INFERRED,frame_samples=frames)
for key,bounds in [('preset',PRESET),('inferred',INFERRED)]:
    l,r,t,b=bounds;valid=pose[['center_x','center_y','hd_deg']].notna().all(axis=1)
    inside=valid&(pose.center_x>=l)&(pose.center_x<=r)&(pose.center_y>=t)&(pose.center_y<=b)
    report[key]=dict(inside_frames=int(inside.sum()),valid_frames=int(valid.sum()),inside_fraction=float(inside.sum()/valid.sum()),outside_frames=int((valid&~inside).sum()))
(OUT/'boundary_review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
