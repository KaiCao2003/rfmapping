import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import matplotlib.pyplot as plt
import numpy as np

from Utils.probe_plotting import WaveformUnitPlot, WaveformUnitPlotCollection


def _artifact(unit_id: int, offset: float) -> SimpleNamespace:
    return SimpleNamespace(
        summary=SimpleNamespace(
            unit_id=unit_id,
            best_channel_index=0,
            best_channel_id=10,
            best_channel_x_um=0.0,
            best_channel_y_um=0.0,
            max_ptp_uv=5.0 + offset,
        ),
        template_uv=np.asarray([
            [offset, 1.0 + offset],
            [-2.0 + offset, 3.0 + offset],
            [1.0 + offset, 2.0 + offset],
        ]),
    )


class WaveformUnitPlotCollectionTests(unittest.TestCase):
    @patch('Utils.probe_plotting.WaveformArtifactStore')
    def test_collection_loads_and_plots_units_in_requested_order(
            self,
            store_type,
    ) -> None:
        store = store_type.return_value
        store.channel_ids = np.asarray([10, 11])
        store.channel_locations = np.asarray([[0.0, 0.0], [20.0, 40.0]])
        store.time_ms = np.asarray([-1.0, 0.0, 1.0])
        store.load_unit.side_effect = [_artifact(520, 0.0), _artifact(115, 1.0)]

        plots = WaveformUnitPlotCollection('/analysis', [520, 115])
        with patch('Utils.probe_plotting.plt.show'):
            result = plots.plot_best_channel_averages()

        self.assertEqual(store.load_unit.call_args_list, [call(520), call(115)])
        self.assertEqual(result.plotted_unit_ids, [520, 115])
        self.assertEqual(result.message, 'Plotted 2 unit(s).')
        self.assertEqual(len(result.figure.axes), 2)
        plt.close(result.figure)

    @patch('Utils.probe_plotting.WaveformArtifactStore')
    def test_collection_fails_fast_when_a_unit_cannot_be_loaded(
            self,
            store_type,
    ) -> None:
        store = store_type.return_value
        store.channel_ids = np.asarray([10])
        store.channel_locations = np.asarray([[0.0, 0.0]])
        store.load_unit.side_effect = KeyError('Unit 999 is unavailable.')

        with self.assertRaisesRegex(KeyError, 'Unit 999 is unavailable'):
            WaveformUnitPlotCollection('/analysis', [999])

    def test_local_channels_are_returned_from_probe_top_to_bottom(self) -> None:
        unit = object.__new__(WaveformUnitPlot)
        unit.summary = SimpleNamespace(best_channel_index=2)
        unit.store = SimpleNamespace(
            channel_locations=np.asarray([
                [0.0, 0.0],
                [0.0, 20.0],
                [0.0, 40.0],
                [0.0, 60.0],
                [0.0, 80.0],
                [20.0, 100.0],
            ]),
            channel_shank_ids=np.zeros(6, dtype=int),
        )

        indices = unit.select_local_channel_indices('same_x_column', 5)

        self.assertEqual(indices.tolist(), [4, 3, 2, 1, 0])
        selected_y = unit.store.channel_locations[indices, 1]
        self.assertTrue(np.all(np.diff(selected_y) <= 0))


if __name__ == '__main__':
    unittest.main()
