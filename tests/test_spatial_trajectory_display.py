import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from Utils.plotting import plot_trajectory_spikes, trajectory_breaks


def test_breaks_use_actual_time_and_missing_frame_ids():
    x = np.array([0., 1., 10., 11., 12., 13.])
    y = np.zeros(6)
    times = np.array([0., .02, .04, .06, .16, .18])
    breaks, gaps, jumps = trajectory_breaks(x,y,frame_times=times,
                                           frame_ids=[0,1,2,4,9,10])
    np.testing.assert_array_equal(breaks,[False,True,True,True,False])
    np.testing.assert_array_equal(jumps,[False,True,False,False,False])
    np.testing.assert_array_equal(gaps,[False,False,True,True,False])
    # A 9 cm displacement over a genuinely slow sample interval is not a speed jump.
    assert not trajectory_breaks([0.,9.],[0.,0.],frame_times=[0.,.1])[2][0]


def test_plot_retains_jump_endpoints_and_source_positions():
    x = np.array([1.,2.,48.,3.,4.]); y = np.array([1.,2.,39.,3.,4.])
    original = x.copy(),y.copy()
    fig,ax = plt.subplots()
    plot_trajectory_spikes(ax,x,y,[],[],x_limits=[0,41],y_limits=[0,41],
                           frame_times=np.arange(5)*.02)
    line_x,line_y = ax.lines[0].get_data()
    np.testing.assert_array_equal(line_x[np.isfinite(line_x)],x)
    np.testing.assert_array_equal(line_y[np.isfinite(line_y)],y)
    assert np.isnan(line_x).sum() == 2
    np.testing.assert_array_equal(ax.collections[0].get_offsets(),np.c_[x[[1,2,3]],y[[1,2,3]]])
    assert ax.get_xlim()[1] > 48
    np.testing.assert_array_equal(x,original[0]); np.testing.assert_array_equal(y,original[1])
    assert "2 > 250 cm/s" in ax.texts[0].get_text()
    plt.close(fig)
