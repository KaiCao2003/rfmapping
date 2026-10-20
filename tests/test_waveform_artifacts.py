import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Utils.si_utils import compute_template_ptp_summary
from Utils.waveform_artifacts import (
    WaveformArtifactStore,
    export_spike_position_data,
    export_waveform_data,
    spike_position_analysis_dir,
    spike_position_root_dir,
    waveform_analysis_dir,
    waveform_root_dir,
)


class _FakeSorting:
    def __init__(self) -> None:
        self._spike_samples = {
            115: np.asarray([100, 200, 300], dtype=np.int64),
            520: np.asarray([150, 350], dtype=np.int64),
        }

    def get_property(self, name: str) -> np.ndarray:
        assert name == 'KSLabel'
        return np.asarray(['good', 'good'])

    def get_unit_spike_train(self, *, unit_id: int) -> np.ndarray:
        return self._spike_samples[unit_id]


def _export_small_waveform(
        output_dir: Path,
        *,
        source_probe: str = 'A',
) -> Path:
    unit_ids = np.asarray([115, 520], dtype=np.int64)
    selected_spikes = np.asarray(
        [(0, 100), (0, 300), (1, 150), (1, 350)],
        dtype=[('unit_index', np.int64), ('sample_index', np.int64)],
    )
    template_array = np.asarray(
        [
            [[0.0, 1.0], [-2.0, 3.0], [1.0, 2.0]],
            [[1.0, 0.0], [4.0, -1.0], [2.0, 1.0]],
        ],
        dtype=np.float32,
    )
    return export_waveform_data(
        output_dir,
        source_session='260615_3',
        source_probe=source_probe,
        source_kilosort_dir='/recording/kilosort/ProbeA/kilosort_3',
        source_raw_file='/recording/continuous.dat',
        only_good_units=True,
        sorting=_FakeSorting(),
        unit_ids=unit_ids,
        selected_spikes=selected_spikes,
        template_array=template_array,
        template_ptp_summary=compute_template_ptp_summary(template_array),
        channel_ids=np.asarray([10, 11], dtype=np.int64),
        channel_locations=np.asarray([[0.0, 0.0], [20.0, 40.0]]),
        channel_shank_ids=np.asarray([0, 1], dtype=np.int64),
        sampling_frequency=1_000.0,
        recording_num_frames=1_000,
        recording_duration_minutes=1.0 / 60.0,
        time_ms=np.asarray([-1.0, 0.0, 1.0]),
        nbefore=1,
        pre_spike_ms=1.0,
        post_spike_ms=1.0,
        max_spikes_per_unit=2,
        waveform_seed=7,
        run_config={
            'n_jobs': 2,
        },
    )


def _export_small_positions(
        output_dir: Path,
        *,
        source_probe: str = 'A',
) -> Path:
    return export_spike_position_data(
        output_dir,
        source_session='260615_3',
        source_probe=source_probe,
        source_kilosort_dir='/recording/kilosort/ProbeA/kilosort_3',
        source_raw_file='/recording/continuous.dat',
        only_good_units=True,
        unit_ids=np.asarray([115, 520], dtype=np.int64),
        unit_locations=np.asarray([[10.0, 20.0], [30.0, 40.0]]),
        unit_location_feature='ptp',
        unit_location_radius_um=75.0,
        run_config={
            'n_jobs': 2,
        },
    )


class WaveformArtifactLayoutTests(unittest.TestCase):
    def test_waveform_analysis_dir_uses_short_probe_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'session' / 'data'

            self.assertEqual(
                waveform_analysis_dir(output_dir, 'A'),
                output_dir / 'waveform' / 'ProbeA',
            )
            self.assertEqual(
                waveform_analysis_dir(output_dir, 'ProbeB'),
                output_dir / 'waveform' / 'ProbeB',
            )
            self.assertEqual(
                spike_position_analysis_dir(output_dir, 'A'),
                output_dir / 'spike_position' / 'ProbeA',
            )

    def test_waveform_export_does_not_create_spike_position_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'data'
            waveform_dir = waveform_analysis_dir(output_dir, 'A')

            exported_dir = _export_small_waveform(output_dir)

            self.assertEqual(exported_dir, waveform_dir)
            self.assertFalse(spike_position_root_dir(output_dir).exists())
            self.assertTrue((waveform_root_dir(output_dir) / 'config.json').is_file())
            self.assertTrue((waveform_root_dir(output_dir) / 'run.log').is_file())
            self.assertFalse((waveform_dir / 'config.json').exists())
            self.assertFalse((waveform_dir / 'run.log').exists())
            self.assertTrue((waveform_dir / 'manifest.json').is_file())
            self.assertTrue((waveform_dir / 'Unit115').is_dir())
            self.assertTrue((waveform_dir / 'Unit520').is_dir())

            waveform_config = json.loads(
                (waveform_root_dir(output_dir) / 'config.json').read_text()
            )
            probe_config = waveform_config['probes']['ProbeA']
            self.assertEqual(probe_config['manifest'], 'ProbeA/manifest.json')
            self.assertEqual(probe_config['run']['n_jobs'], 2)

            waveform_log = (waveform_root_dir(output_dir) / 'run.log').read_text()
            self.assertIn('status=complete', waveform_log)
            self.assertIn('probe=ProbeA', waveform_log)
            self.assertIn('units=2', waveform_log)

            with (waveform_dir / 'units.csv').open() as units_file:
                units_header = units_file.readline()
            self.assertNotIn('unit_x_um', units_header)
            self.assertNotIn('unit_y_um', units_header)

            store = WaveformArtifactStore(waveform_dir)
            unit = store.load_unit(115)
            self.assertEqual(unit.summary.unit_data_dir, 'Unit115')
            self.assertIsNone(unit.summary.unit_x_um)
            self.assertIsNone(unit.summary.unit_y_um)
            np.testing.assert_array_equal(unit.all_spike_samples, [100, 200, 300])
            np.testing.assert_array_equal(unit.selected_spike_samples, [100, 300])

    def test_spike_position_export_does_not_create_waveform_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'data'
            spike_position_dir = spike_position_analysis_dir(output_dir, 'A')

            exported_dir = _export_small_positions(output_dir)

            self.assertEqual(exported_dir, spike_position_dir)
            self.assertFalse(waveform_root_dir(output_dir).exists())
            self.assertTrue((spike_position_root_dir(output_dir) / 'config.json').is_file())
            self.assertTrue((spike_position_root_dir(output_dir) / 'run.log').is_file())
            self.assertTrue((spike_position_dir / 'manifest.json').is_file())
            self.assertTrue((spike_position_dir / 'positions.csv').is_file())

    def test_independent_exports_round_trip_together(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'data'
            waveform_dir = _export_small_waveform(output_dir)
            _export_small_positions(output_dir)

            unit = WaveformArtifactStore(waveform_dir).load_unit(115)

            self.assertEqual(unit.summary.unit_x_um, 10.0)
            self.assertEqual(unit.summary.unit_y_um, 20.0)

    def test_split_artifacts_remain_linked_after_output_root_moves(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'data'
            moved_output_dir = Path(temp_dir) / 'moved_data'
            _export_small_waveform(output_dir)
            _export_small_positions(output_dir)

            output_dir.rename(moved_output_dir)

            store = WaveformArtifactStore(
                waveform_analysis_dir(moved_output_dir, 'A')
            )
            unit = store.load_unit(520)
            self.assertEqual(unit.summary.unit_x_um, 30.0)
            self.assertEqual(unit.summary.unit_y_um, 40.0)

    def test_combined_schema_v3_inline_positions_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'data'
            _export_small_waveform(output_dir)
            _export_small_positions(output_dir)
            waveform_dir = waveform_analysis_dir(output_dir, 'A')
            position_file = spike_position_analysis_dir(
                output_dir,
                'A',
            ) / 'positions.csv'

            with position_file.open(newline='') as csv_file:
                positions = {
                    int(row['unit_id']): row
                    for row in csv.DictReader(csv_file)
                }
            units_file = waveform_dir / 'units.csv'
            with units_file.open(newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                unit_rows = list(reader)
                unit_columns = list(reader.fieldnames or [])
            unit_columns.extend(['unit_x_um', 'unit_y_um'])
            for row in unit_rows:
                position = positions[int(row['unit_id'])]
                row['unit_x_um'] = position['x_um']
                row['unit_y_um'] = position['y_um']
            with units_file.open('w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=unit_columns)
                writer.writeheader()
                writer.writerows(unit_rows)

            manifest_file = waveform_dir / 'manifest.json'
            manifest = json.loads(manifest_file.read_text())
            manifest['schema_version'] = 3
            manifest['files'].pop('spike_positions')
            manifest_file.write_text(json.dumps(manifest))
            shutil.rmtree(spike_position_root_dir(output_dir))

            store = WaveformArtifactStore(waveform_dir)
            unit = store.load_unit(115)
            self.assertEqual(unit.summary.unit_x_um, 10.0)
            self.assertEqual(unit.summary.unit_y_um, 20.0)

    def test_export_appends_run_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'data'

            _export_small_waveform(output_dir)
            _export_small_waveform(output_dir)
            _export_small_positions(output_dir)
            _export_small_positions(output_dir)

            self.assertEqual(
                len((waveform_root_dir(output_dir) / 'run.log').read_text().splitlines()),
                2,
            )
            self.assertEqual(
                len((spike_position_root_dir(output_dir) / 'run.log').read_text().splitlines()),
                2,
            )

    def test_root_configs_keep_multiple_probes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'data'

            _export_small_waveform(output_dir, source_probe='A')
            _export_small_waveform(output_dir, source_probe='B')
            _export_small_positions(output_dir, source_probe='A')
            _export_small_positions(output_dir, source_probe='B')

            for root_dir in (
                waveform_root_dir(output_dir),
                spike_position_root_dir(output_dir),
            ):
                config = json.loads((root_dir / 'config.json').read_text())
                self.assertEqual(set(config['probes']), {'ProbeA', 'ProbeB'})


if __name__ == '__main__':
    unittest.main()
