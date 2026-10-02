import json
from pathlib import Path

import numpy as np
import pytest
import spikeinterface as si

from Utils import si_utils


@pytest.fixture
def source_session(tmp_path):
    session = tmp_path / '260922_3'
    recording_dir = session / '260922' / 'Record Node 102' / 'experiment2' / 'recording3'
    stream_name = 'OneBox-104.ProbeA'
    stream_dir = recording_dir / 'continuous' / stream_name
    stream_dir.mkdir(parents=True)
    traces = np.arange(120, dtype=np.int16).reshape(40, 3) - 60
    traces.tofile(stream_dir / 'continuous.dat')
    sample_rate = 1_200.
    times = 21. + np.arange(40) / sample_rate
    times[1::2] += 1e-6
    np.save(stream_dir / 'sample_numbers.npy', np.arange(40))
    np.save(stream_dir / 'timestamps.npy', times)
    gains = np.array([.5, 1., 2.])
    structure = {
        'GUI version': '1.0.1',
        'continuous': [{
            'folder_name': stream_name + '/', 'sample_rate': sample_rate,
            'source_processor_name': 'OneBox', 'source_processor_id': 104,
            'stream_name': 'ProbeA', 'recorded_processor': 'Record Node',
            'recorded_processor_id': 102, 'num_channels': 3,
            'channels': [{'channel_name': f'CH{i}', 'bit_volts': gain, 'units': 'uV'}
                         for i, gain in enumerate(gains)],
        }],
        'events': [], 'spikes': [],
    }
    (recording_dir / 'structure.oebin').write_text(json.dumps(structure))
    info = {'base_path': str(session / '260922'), 'record_nodes': 'Record Node 102',
            'experiment_id': 'experiment2', 'recording_name': 'recording3',
            'continuous_probe_A_folder': stream_name}
    (session / 'data').mkdir()
    (session / 'data/session_info.json').write_text(json.dumps({'session_info': info}))
    kilosort = session / 'kilosort/ProbeA/kilosort_3'
    kilosort.mkdir(parents=True)
    (kilosort / 'params.py').write_text(f'sample_rate = {sample_rate}\n')
    np.save(kilosort / 'spike_times.npy', np.array([3, 8, 12]))
    np.save(kilosort / 'spike_clusters.npy', np.array([7, 9, 7]))
    np.save(kilosort / 'channel_map.npy', np.array([2, 0]))
    np.save(kilosort / 'channel_positions.npy', np.array([[250., 20.], [0., 0.]]))
    np.save(kilosort / 'ops.npy', {'probe': {'chanMap': np.array([2, 0]), 'kcoords': np.array([4, 1])}})
    (kilosort / 'cluster_KSLabel.tsv').write_text('cluster_id\tKSLabel\n7\tgood\n9\tmua\n11\tgood\n')
    return session, info, kilosort, stream_dir / 'continuous.dat', traces, times, gains


def test_open_ephys_reader_uses_metadata_stream_gain_and_actual_timestamps(source_session):
    _, info, _, raw_file, traces, times, gains = source_session
    recording = si_utils.read_open_ephys_probe(info, 'A', load_sync_timestamps=True)
    np.testing.assert_array_equal(recording.channel_ids, [0, 1, 2])
    np.testing.assert_array_equal(recording.get_traces(), traces)
    np.testing.assert_array_equal(recording.get_times(), times)
    np.testing.assert_allclose(recording.get_traces(return_in_uV=True), traces * gains)
    assert recording.get_sampling_frequency() == 1_200.
    assert recording.get_annotation('open_ephys_raw_file') == str(raw_file)
    assert recording.get_annotation('open_ephys_stream_name') == 'Record Node 102#OneBox-104.ProbeA'
    assert recording.get_annotation('load_sync_timestamps') is True
    restored = recording.clone()
    np.testing.assert_array_equal(restored.get_times(), times)
    np.testing.assert_array_equal(restored.get_traces(), traces)


def test_open_ephys_reader_can_explicitly_request_sample_clock(source_session):
    _, info, _, _, traces, _, _ = source_session
    recording = si_utils.read_open_ephys_probe(info, 'ProbeA', load_sync_timestamps=False)
    assert not recording.has_time_vector()
    np.testing.assert_array_equal(recording.get_traces(), traces)
    np.testing.assert_allclose(recording.get_times(), np.arange(40) / 1_200.)


def test_missing_requested_timestamp_vector_is_not_silently_accepted(source_session, monkeypatch):
    _, info, _, _, traces, _, _ = source_session
    without_times = si.NumpyRecording(traces, sampling_frequency=1_200.)
    monkeypatch.setattr(si_utils.se, 'read_openephys', lambda *args, **kwargs: without_times)
    with pytest.raises(ValueError, match='no synchronized'):
        si_utils.read_open_ephys_probe(info, 'A', load_sync_timestamps=True)


def test_kilosort_probe_keeps_raw_channel_identity_and_physical_shanks(source_session):
    _, info, kilosort, raw_file, traces, times, _ = source_session
    recording = si_utils.read_open_ephys_probe(info, 'A')
    data = si_utils.validate_data(kilosort_dir=kilosort, recording_file=raw_file, recording=recording)
    mapped = si_utils.attach_kilosort_probe(recording, data)
    np.testing.assert_array_equal(mapped.channel_ids, [2, 0])
    np.testing.assert_array_equal(mapped.get_traces(), traces[:, [2, 0]])
    np.testing.assert_array_equal(mapped.get_times(), times)
    np.testing.assert_array_equal(mapped.get_channel_locations(), [[250., 20.], [0., 0.]])
    np.testing.assert_array_equal(mapped.get_probe().shank_ids, ['4', '1'])


@pytest.mark.parametrize('keep_good,remove_empty,expected', [
    (True, True, [7]), (False, True, [7, 9]), (True, False, [7, 11]),
])
def test_generator_passes_explicit_unit_retention_before_computation(
    source_session, monkeypatch, keep_good, remove_empty, expected,
):
    session, _, _, _, _, _, _ = source_session
    requested = []
    read_kilosort = si_utils.se.read_kilosort

    def read_selected(folder, **kwargs):
        requested.append(kwargs)
        return read_kilosort(folder, **kwargs)

    class BeforeComputation(Exception):
        pass

    def stop_before_computation(sorting, recording, **kwargs):
        np.testing.assert_array_equal(sorting.unit_ids, expected)
        np.testing.assert_array_equal(recording.channel_ids, [2, 0])
        assert recording.has_time_vector()
        raise BeforeComputation

    monkeypatch.setattr(si_utils.se, 'read_kilosort', read_selected)
    monkeypatch.setattr(si_utils.si, 'create_sorting_analyzer', stop_before_computation)
    with pytest.raises(BeforeComputation):
        si_utils.generate_unit_artifacts(
            session.parent, '260922', '3', 'A', session_dir=session,
            only_good_units=keep_good, remove_empty_units=remove_empty,
        )
    assert requested == [{'keep_good_only': keep_good, 'remove_empty_units': remove_empty}]


@pytest.mark.parametrize('change', ['clock_origin', 'gain', 'sampling_frequency', 'missing_clock'])
def test_cached_analyzer_metadata_must_match_current_reader(change):
    traces = np.zeros((8, 2), dtype=np.int16)
    current = si.NumpyRecording(traces, sampling_frequency=1_000.)
    cached = si.NumpyRecording(traces, sampling_frequency=2_000. if change == 'sampling_frequency' else 1_000.)
    for recording in (current, cached):
        recording.set_channel_gains(.5)
        recording.set_channel_offsets(0.)
    current.set_times(20. + np.arange(8) / 1_000., with_warning=False)
    if change != 'missing_clock':
        cached.set_times((0. if change == 'clock_origin' else 20.) + np.arange(8) / 1_000., with_warning=False)
    if change == 'gain':
        cached.set_channel_gains(.195)
    with pytest.raises(ValueError, match='Rebuild explicitly'):
        si_utils._validate_cached_recording(cached, current)
    si_utils._validate_cached_recording(current, current)
