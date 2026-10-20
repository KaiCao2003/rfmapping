"""Export each recording's trajectory with discontinuous tracking edges marked."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from Utils.plotting import LIGHT_PLOT_STYLE, plot_trajectory_spikes, trajectory_breaks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_root',type=Path)
    args=parser.parse_args()
    manifest=[]
    for audit_path in sorted(args.output_root.glob('*/audit.json')):
        directory=audit_path.parent
        audit=json.loads(audit_path.read_text())
        with np.load(directory/'restored_maps.npz') as saved:
            x,y,times=(saved[key] for key in ['trajectory_x_cm','trajectory_y_cm','frame_times'])
            xlimits=saved['x_edges'][[0,-1]];ylimits=saved['y_edges'][[0,-1]]
            frame_ids=saved['frame_ids'] if 'frame_ids' in saved else None
        if (directory/'trajectory_display_inputs.npz').exists():
            with np.load(directory/'trajectory_display_inputs.npz') as display:
                np.testing.assert_array_equal(times,display['frame_times'])
                np.testing.assert_array_equal(x,display['trajectory_x_cm'])
                np.testing.assert_array_equal(y,display['trajectory_y_cm'])
                frame_ids=display['frame_ids']
        breaks,gaps,jumps=trajectory_breaks(x,y,frame_times=times,frame_ids=frame_ids)
        with plt.rc_context({**LIGHT_PLOT_STYLE,'font.size':12}):
            fig,ax=plt.subplots(figsize=(7,6.5),layout='constrained')
            plot_trajectory_spikes(ax,x,y,[],[],x_limits=xlimits,y_limits=ylimits,
                                   frame_times=times,frame_ids=frame_ids)
            ax.set_title('');ax.set_xlabel('x (cm)',fontsize=14);ax.set_ylabel('y (cm)',fontsize=14)
            ax.legend(loc='upper right',fontsize=9)
            ax.texts[0].set_fontsize(8)
            if 'held_position_frames' in audit:
                ax.texts[0].set_text(ax.texts[0].get_text()
                                    + f"\nBoundary hold: {audit['held_position_frames']:,} frames")
            path=directory/'trajectory_overview.png'
            fig.savefig(path,dpi=180,facecolor='white',transparent=False)
            plt.close(fig)
        entry=dict(cohort=directory.name,path=str(path),frame_count=len(times),
                   breaks=int(breaks.sum()),speed_jumps=int(jumps.sum()),frame_or_time_gaps=int(gaps.sum()),
                   max_speed_cm_s=250,all_positions_retained=True,
                   position_correction_policy=audit.get('position_correction_policy','none'),
                   held_position_frames=audit.get('held_position_frames',0),
                   initial_outside_frames=audit.get('initial_outside_frames',0),
                   scientific_arrays_changed='held_position_frames' in audit,raw_tracking_repaired=False)
        manifest.append(entry)
        (directory/'trajectory_display.json').write_text(json.dumps(entry,indent=2)+'\n')
        print(directory.name,entry['breaks'],flush=True)
    (args.output_root/'trajectory_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
