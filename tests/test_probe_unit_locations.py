from types import SimpleNamespace
from unittest.mock import Mock, call

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.figure import Figure

from Utils import probe_plotting


def test_unit_artifact_generation_with_no_outputs_does_not_open_session(tmp_path, monkeypatch):
    from Utils import si_utils

    def forbidden(*args, **kwargs):
        pytest.fail("No requested outputs must not read or compute recording data")

    monkeypatch.setattr(si_utils, "validate_data", forbidden)
    monkeypatch.setattr(si_utils.si, "create_sorting_analyzer", forbidden)
    missing_session = tmp_path / "missing-session"
    result = si_utils.generate_unit_artifacts(
        missing_session, "260922", "3", "A", unit_waveform=False, unit_position=False,
    )
    assert result == si_utils.UnitArtifactPaths()
    assert not missing_session.exists()


def test_template_locations_use_only_contacts_near_each_peak():
    from probeinterface import Probe
    from Utils.si_utils import compute_template_ptp_summary, compute_template_unit_locations

    channel_locations = np.array([[0., 0.], [20., 0.], [200., 0.]])
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(channel_locations, shapes="circle", shape_params={"radius": 6.})
    probe.set_device_channel_indices(np.arange(3))
    analyzer = SimpleNamespace(
        unit_ids=np.array([7, 31]), channel_ids=np.arange(3), sampling_frequency=30_000.,
        get_channel_locations=lambda: channel_locations,
        get_num_channels=lambda: 3,
        get_probe=lambda: probe,
    )
    templates = np.zeros((2, 3, 3))
    templates[:, 1] = [[-10., -5., -8.], [-1., -2., -10.]]
    locations = compute_template_unit_locations(
        analyzer, templates, compute_template_ptp_summary(templates), nbefore=1,
    )
    np.testing.assert_allclose(locations, [[20. / 3., 0.], [200., 0.]])


@pytest.fixture
def summary_store(monkeypatch):
    summaries = {
        7: SimpleNamespace(unit_id=7, unit_x_um=4.5, unit_y_um=11.5,
                           best_channel_index=0, best_channel_x_um=0., best_channel_y_um=0.),
        31: SimpleNamespace(unit_id=31, unit_x_um=25.5, unit_y_um=33.5,
                            best_channel_index=1, best_channel_x_um=20., best_channel_y_um=40.),
    }
    store = SimpleNamespace(
        channel_ids=np.array([10, 11]),
        channel_locations=np.array([[0., 0.], [20., 40.]]),
        time_ms=np.array([-1., 0., 1.]),
        unit_summaries=summaries,
    )
    store.load_unit = Mock(side_effect=lambda unit_id: SimpleNamespace(
        summary=summaries[unit_id],
        template_uv=np.array([[0., 1.], [-2., 3.], [1., 2.]]),
    ))
    monkeypatch.setattr(probe_plotting, 'WaveformArtifactStore', lambda _: store)
    yield store
    plt.close('all')


def test_summary_only_external_axis_keeps_positions_missing_points_and_color_mapping(
        summary_store, monkeypatch, tmp_path):
    plots = probe_plotting.WaveformUnitPlotCollection(
        tmp_path, [], save_figures=True, output_dir=tmp_path / 'should_not_be_created',
    )
    summary_store.load_unit.assert_not_called()
    figure, ax = plt.subplots()
    initial_figures = plt.get_fignums()
    norm = Normalize(0., 360.)
    cmap = plt.get_cmap('twilight')
    values = {31: np.nan, 7: 180.}  # Different order from the saved summaries.

    def forbidden(*args, **kwargs):
        pytest.fail('A supplied axis must not create, show, or save a figure')

    monkeypatch.setattr(plt, 'subplots', forbidden)
    monkeypatch.setattr(plt, 'figure', forbidden)
    monkeypatch.setattr(plt, 'show', forbidden)
    monkeypatch.setattr(Figure, 'savefig', forbidden)
    result = plots.plot_unit_locations(ax=ax, unit_values=values, cmap=cmap, norm=norm)
    figure.canvas.draw()

    assert result.figure is figure
    assert result.plotted_unit_ids == [31, 7]
    assert result.message == 'Plotted 2 unit(s).'
    assert plt.get_fignums() == initial_figures
    assert not plots.output_dir.exists()
    summary_store.load_unit.assert_not_called()
    points, = [item for item in ax.collections if isinstance(item, PathCollection)]
    assert points.norm is norm
    np.testing.assert_allclose(points.get_offsets(), [[25.5, 33.5], [4.5, 11.5]])
    assert not np.ma.getmaskarray(points.get_offsets()).any()
    np.testing.assert_allclose(points.get_facecolors(), [to_rgba('#9ca3af'), cmap(norm(180.))])
    assert points.get_facecolors()[0, 3] == 1.
    assert list(values) == [31, 7]
    assert np.isnan(values[31]) and values[7] == 180.


def test_other_units_are_background_only_and_use_unit_positions(summary_store, tmp_path):
    plots = probe_plotting.WaveformUnitPlotCollection(tmp_path, [])
    figure, ax = plt.subplots()
    result = plots.plot_unit_locations(
        ax=ax, unit_values={7: np.nan}, show_other_units=True,
        norm=Normalize(-180., 180.), missing_color='#808080',
    )
    figure.canvas.draw()

    assert result.plotted_unit_ids == [7]
    background, selected = [item for item in ax.collections if isinstance(item, PathCollection)]
    np.testing.assert_allclose(background.get_offsets(), [[25.5, 33.5]])
    np.testing.assert_allclose(selected.get_offsets(), [[4.5, 11.5]])
    np.testing.assert_allclose(selected.get_facecolors(), [to_rgba('#808080')])
    summary_store.load_unit.assert_not_called()


@pytest.mark.parametrize('scalar_map', [False, True], ids=['legacy-waveforms', 'summary-scalars'])
def test_owned_figure_returns_requested_ids_and_preserves_legacy_finish(
        summary_store, monkeypatch, tmp_path, scalar_map):
    plots = probe_plotting.WaveformUnitPlotCollection(
        tmp_path, [] if scalar_map else [31, 7], save_figures=True,
    )
    show, save = Mock(), Mock()
    monkeypatch.setattr(plt, 'show', show)
    monkeypatch.setattr(Figure, 'savefig', save)
    kwargs = {'unit_values': {31: 90., 7: 270.}, 'norm': Normalize(0., 360.)} if scalar_map else {}
    result = plots.plot_unit_locations(**kwargs)

    assert result.plotted_unit_ids == [31, 7]
    assert result.message == 'Plotted 2 unit(s).'
    points = [item for item in result.figure.axes[0].collections if isinstance(item, PathCollection)]
    np.testing.assert_allclose(
        np.concatenate([point.get_offsets() for point in points]), [[25.5, 33.5], [4.5, 11.5]],
    )
    assert summary_store.load_unit.call_args_list == ([] if scalar_map else [call(31), call(7)])
    show.assert_called_once_with()
    save.assert_called_once_with(
        tmp_path / 'figures' / 'spikeinterface_unit_locations.png', dpi=200, bbox_inches='tight',
    )
