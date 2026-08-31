from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

import Utils.rf_trials as rf_trials
from Utils.rf_trials import load_regular_rf_trials
from Utils.rfmap import RFMapList, load_rf_maps


def test_trial_loader_has_one_plain_public_entry_point() -> None:
    assert rf_trials.__all__ == ["load_regular_rf_trials"]
    assert not hasattr(rf_trials, "RegularRFSourcePaths")
    assert not hasattr(rf_trials, "load_regular_rf_trial_data")


@dataclass(frozen=True)
class _SyntheticRegularRF:
    session: Path
    rf_maps: RFMapList
    onsets: np.ndarray
    position_ids: np.ndarray
    polarities: np.ndarray
    strata: np.ndarray
    expected_responses: np.ndarray


def _write_regular_fixture(
    tmp_path: Path,
    *,
    pooled_delta: int = 0,
    corrupt_last_block: bool = False,
    n_time_bins: int = 1,
    source_suffix: str = ".json",
    trial_polarities: tuple[int, ...] = (0, 1),
    pooled_polarity: int = 1,
) -> _SyntheticRegularRF:
    session = tmp_path / "260101_3"
    spike_dir = session / "data" / "probeA"
    kilosort_dir = session / "kilosort" / "ProbeA" / "kilosort_3"
    spike_dir.mkdir(parents=True)
    kilosort_dir.mkdir(parents=True)

    x_positions = np.asarray([-10.0, 10.0])
    y_positions = np.asarray([-5.0, 5.0])
    n_positions = x_positions.size * y_positions.size
    observed_polarities = np.asarray(trial_polarities, dtype=int)
    assert np.array_equal(observed_polarities, np.unique(observed_polarities))
    assert set(observed_polarities.tolist()).issubset({0, 1})
    assert pooled_polarity in {0, 1}
    block_size = n_positions * observed_polarities.size
    n_blocks = 2

    # Each block is a randomized permutation of the complete
    # position-by-observed-polarity factorial.
    conditions = (
        observed_polarities[:, np.newaxis] * n_positions
        + np.arange(n_positions)
    ).reshape(-1)
    rng = np.random.default_rng(12345)
    block_orders = [rng.permutation(conditions) for _ in range(n_blocks)]
    encoded = np.concatenate(block_orders)
    position_ids = encoded % n_positions
    polarities = encoded // n_positions
    strata = np.repeat(np.arange(n_blocks), block_size)
    if corrupt_last_block:
        position_ids[-1] = position_ids[-block_size]
        polarities[-1] = polarities[-block_size]

    x = x_positions[position_ids % x_positions.size]
    y = y_positions[position_ids // x_positions.size]
    trial_dtype = np.dtype(
        [
            ("Stimulus_Type", object),
            ("Square_PositionX", object),
            ("Square_PositionY", object),
            ("Square_Luminance", object),
            ("Square_Size", object),
            ("Timing", object),
        ]
    )
    mat_trials = np.empty((position_ids.size, 1), dtype=trial_dtype)
    for index in range(position_ids.size):
        mat_trials[index, 0] = (
            "Receptive Field Mapping",
            x[index],
            y[index],
            polarities[index],
            20,
            np.asarray([0.0, 0.095, 0.0]),
        )
    savemat(session / "260101.mat", {"trials": mat_trials})

    onsets = 1.0 + np.arange(position_ids.size, dtype=float) * 0.1
    np.save(
        session / "data" / "on_list_times.npy",
        np.append(onsets, onsets[-1] + 0.1),
    )

    unit_ids = np.asarray([41, 7], dtype=np.int64)
    all_responses = np.empty((unit_ids.size, position_ids.size), dtype=np.int64)
    all_responses[0] = np.where(
        polarities == 1,
        position_ids + 1,
        n_positions - position_ids,
    )
    all_responses[1] = np.where(
        polarities == 1,
        position_ids,
        n_positions - 1 - position_ids,
    )

    spike_events: list[tuple[float, int]] = []
    for trial_index, onset in enumerate(onsets):
        for unit_index, unit_id in enumerate(unit_ids):
            for spike_index in range(all_responses[unit_index, trial_index]):
                spike_events.append(
                    (onset + 0.01 + spike_index * 0.005, int(unit_id))
                )
            # A spike exactly at the stop edge must be excluded by the
            # loader's half-open [start, stop) response window.
            spike_events.append((onset + 0.05, int(unit_id)))
    spike_events.append((0.25, 999))
    spike_events.sort(key=lambda item: item[0])
    np.save(
        spike_dir / "adc_spike_time.npy",
        np.asarray([event[0] for event in spike_events], dtype=float),
    )
    np.save(
        kilosort_dir / "spike_clusters.npy",
        np.asarray([event[1] for event in spike_events], dtype=np.int32),
    )

    pooled_trials = polarities == pooled_polarity
    pooled = np.zeros((unit_ids.size, 2, 2), dtype=np.int64)
    for unit_index in range(unit_ids.size):
        pooled[unit_index].flat[:] = np.bincount(
            position_ids[pooled_trials],
            weights=all_responses[unit_index, pooled_trials],
            minlength=n_positions,
        )
    pooled[0, 0, 0] += pooled_delta

    if n_time_bins == 1:
        json_counts = pooled[..., np.newaxis]
        time_edges = [0.0, 0.05]
    elif n_time_bins == 2:
        json_counts = np.stack([pooled, np.zeros_like(pooled)], axis=-1)
        time_edges = [0.0, 0.025, 0.05]
    else:
        raise AssertionError("test helper supports one or two time bins")

    payload = {
        "unitsSpikeCounts": json_counts.tolist(),
        "unitsSpikeCountsSize": list(json_counts.shape),
        "unitPool": unit_ids.tolist(),
        "xPositions": x_positions.tolist(),
        "yPositions": y_positions.tolist(),
        "timeBinEdges": time_edges,
        "stimulusPresentationCounts": np.full(
            (2, 2),
            n_blocks if pooled_polarity in observed_polarities else 0,
        ).tolist(),
    }
    polarity_suffix = "_off" if pooled_polarity == 0 else ""
    source_path = session / (
        f"regular_unitsSpikeCounts_260101_3{polarity_suffix}{source_suffix}"
    )
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    rf_maps = load_rf_maps(source_path)
    return _SyntheticRegularRF(
        session=session,
        rf_maps=rf_maps,
        onsets=onsets,
        position_ids=position_ids,
        polarities=polarities,
        strata=strata,
        expected_responses=all_responses,
    )


def test_loader_resolves_canonical_regular_rf_paths(tmp_path: Path) -> None:
    fixture = _write_regular_fixture(tmp_path)

    trial_data = load_regular_rf_trials(
        fixture.session,
        "probea",
        fixture.rf_maps,
    )
    provenance = trial_data["provenance"]

    assert provenance["trials_mat"] == str(fixture.session / "260101.mat")
    assert provenance["onsets_npy"] == str(
        fixture.session / "data/on_list_times.npy"
    )
    assert provenance["spike_times_npy"] == str(
        fixture.session / "data/probeA/adc_spike_time.npy"
    )
    assert provenance["spike_clusters_npy"] == str(
        fixture.session
        / "kilosort/ProbeA/kilosort_3/spike_clusters.npy"
    )


def test_loader_accepts_json_text_rfmap_source(tmp_path: Path) -> None:
    fixture = _write_regular_fixture(tmp_path, source_suffix=".rfmap")

    trial_data = load_regular_rf_trials(fixture.session, "A", fixture.rf_maps)

    assert trial_data["provenance"]["pooled_rf_json"] == str(
        fixture.session / "regular_unitsSpikeCounts_260101_3.rfmap"
    )


def test_loader_rejects_regular_source_with_unrelated_suffix(
    tmp_path: Path,
) -> None:
    fixture = _write_regular_fixture(tmp_path, source_suffix=".txt")

    with pytest.raises(ValueError, match=r"\.json or \.rfmap"):
        load_regular_rf_trials(fixture.session, "A", fixture.rf_maps)


@pytest.mark.parametrize("probe", ["", "C", "Probe1", 1])
def test_loader_rejects_unknown_probe(tmp_path: Path, probe: object) -> None:
    fixture = _write_regular_fixture(tmp_path)

    with pytest.raises(ValueError, match="probe"):
        load_regular_rf_trials(
            fixture.session,
            probe,  # type: ignore[arg-type]
            fixture.rf_maps,
        )


def test_load_builds_unit_by_on_trial_data_and_excludes_terminal_edge(
    tmp_path: Path,
) -> None:
    fixture = _write_regular_fixture(tmp_path)
    on_trials = fixture.polarities == 1

    trial_data = load_regular_rf_trials(
        fixture.session,
        "A",
        fixture.rf_maps,
    )

    assert type(trial_data) is dict
    assert trial_data["shape"] == (2, 2)
    assert trial_data["responses"].shape == (2, 8)
    np.testing.assert_array_equal(
        trial_data["responses"],
        fixture.expected_responses[:, on_trials],
    )
    np.testing.assert_array_equal(
        trial_data["position_ids"],
        fixture.position_ids[on_trials],
    )
    np.testing.assert_array_equal(
        trial_data["stratum_ids"],
        fixture.strata[on_trials],
    )
    np.testing.assert_array_equal(trial_data["unit_ids"], [41, 7])
    np.testing.assert_array_equal(trial_data["x_positions"], [-10.0, 10.0])
    np.testing.assert_array_equal(trial_data["y_positions"], [-5.0, 5.0])
    assert trial_data["time_range_s"] == (0.0, 0.05)
    assert trial_data["polarity"] == "on"
    assert trial_data["provenance"]["synthetic_terminal_edge_excluded"] is True
    assert trial_data["provenance"]["n_all_trials"] == 16
    for value in trial_data.values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable


def test_load_supports_on_only_repeat_blocks(tmp_path: Path) -> None:
    fixture = _write_regular_fixture(tmp_path, trial_polarities=(1,))

    trial_data = load_regular_rf_trials(
        fixture.session,
        "A",
        fixture.rf_maps,
        on=True,
        off=False,
    )

    np.testing.assert_array_equal(
        trial_data["responses"],
        fixture.expected_responses,
    )
    np.testing.assert_array_equal(
        trial_data["position_ids"],
        fixture.position_ids,
    )
    np.testing.assert_array_equal(trial_data["stratum_ids"], fixture.strata)
    assert trial_data["polarity"] == "on"
    assert trial_data["provenance"]["n_repeat_blocks"] == 2

    with pytest.raises(ValueError, match="no off trials"):
        load_regular_rf_trials(
            fixture.session,
            "A",
            fixture.rf_maps,
            on=False,
            off=True,
            validate_pooled=False,
        )


def test_load_supports_off_only_repeat_blocks(tmp_path: Path) -> None:
    fixture = _write_regular_fixture(
        tmp_path,
        trial_polarities=(0,),
        pooled_polarity=0,
    )

    with pytest.raises(ValueError, match="no on trials"):
        load_regular_rf_trials(fixture.session, "A", fixture.rf_maps)

    trial_data = load_regular_rf_trials(
        fixture.session,
        "A",
        fixture.rf_maps,
        on=False,
        off=True,
    )

    np.testing.assert_array_equal(
        trial_data["responses"],
        fixture.expected_responses,
    )
    np.testing.assert_array_equal(
        trial_data["position_ids"],
        fixture.position_ids,
    )
    np.testing.assert_array_equal(trial_data["stratum_ids"], fixture.strata)
    assert trial_data["polarity"] == "off"
    assert trial_data["provenance"]["n_repeat_blocks"] == 2


@pytest.mark.parametrize("on, off", [(True, True), (False, False)])
def test_load_requires_exactly_one_polarity_flag(
    tmp_path: Path,
    on: bool,
    off: bool,
) -> None:
    fixture = _write_regular_fixture(tmp_path)

    with pytest.raises(ValueError, match="exactly one"):
        load_regular_rf_trials(
            fixture.session,
            "A",
            fixture.rf_maps,
            on=on,
            off=off,
        )


def test_load_requires_boolean_polarity_flags(tmp_path: Path) -> None:
    fixture = _write_regular_fixture(tmp_path)

    with pytest.raises(TypeError, match="on must be bool"):
        load_regular_rf_trials(
            fixture.session,
            "A",
            fixture.rf_maps,
            on=1,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="off must be bool"):
        load_regular_rf_trials(
            fixture.session,
            "A",
            fixture.rf_maps,
            off=0,  # type: ignore[arg-type]
        )


def test_half_open_counts_match_pooled_json_despite_stop_edge_spikes(
    tmp_path: Path,
) -> None:
    fixture = _write_regular_fixture(tmp_path)

    # Every unit has one synthetic spike exactly at every trial's +50 ms
    # edge. Successful pooled validation proves those spikes were excluded.
    trial_data = load_regular_rf_trials(fixture.session, "A", fixture.rf_maps)

    assert int(trial_data["responses"].sum()) == int(
        fixture.expected_responses[:, fixture.polarities == 1].sum()
    )


def test_off_loading_validates_the_supplied_pooled_polarity(
    tmp_path: Path,
) -> None:
    on_fixture = _write_regular_fixture(tmp_path / "on")

    with pytest.raises(ValueError, match="raw trial counts do not match"):
        load_regular_rf_trials(
            on_fixture.session,
            "A",
            on_fixture.rf_maps,
            on=False,
            off=True,
        )

    fixture = _write_regular_fixture(tmp_path / "off", pooled_polarity=0)
    off_trials = fixture.polarities == 0

    trial_data = load_regular_rf_trials(
        fixture.session,
        "A",
        fixture.rf_maps,
        on=False,
        off=True,
    )

    assert trial_data["polarity"] == "off"
    np.testing.assert_array_equal(
        trial_data["responses"],
        fixture.expected_responses[:, off_trials],
    )
    np.testing.assert_array_equal(
        trial_data["position_ids"],
        fixture.position_ids[off_trials],
    )


def test_load_rejects_unsummed_multi_bin_rf_maps(tmp_path: Path) -> None:
    fixture = _write_regular_fixture(tmp_path, n_time_bins=2)

    with pytest.raises(ValueError, match="summed|one response-window"):
        load_regular_rf_trials(fixture.session, "A", fixture.rf_maps)


@pytest.mark.parametrize(
    "trial_polarities, pooled_polarity",
    [((0,), 0), ((1,), 1), ((0, 1), 1)],
)
def test_load_rejects_incomplete_factorial_repeat_block(
    tmp_path: Path,
    trial_polarities: tuple[int, ...],
    pooled_polarity: int,
) -> None:
    fixture = _write_regular_fixture(
        tmp_path,
        corrupt_last_block=True,
        trial_polarities=trial_polarities,
        pooled_polarity=pooled_polarity,
    )

    with pytest.raises(ValueError, match="factorial"):
        load_regular_rf_trials(
            fixture.session,
            "A",
            fixture.rf_maps,
            on=pooled_polarity == 1,
            off=pooled_polarity == 0,
            validate_pooled=False,
        )


def test_load_requires_one_synthetic_terminal_edge(tmp_path: Path) -> None:
    fixture = _write_regular_fixture(tmp_path)
    np.save(fixture.session / "data/on_list_times.npy", fixture.onsets)

    with pytest.raises(ValueError, match="synthetic terminal edge"):
        load_regular_rf_trials(fixture.session, "A", fixture.rf_maps)


def test_load_validates_synthetic_terminal_edge_interval(tmp_path: Path) -> None:
    fixture = _write_regular_fixture(tmp_path)
    bad_edges = np.append(fixture.onsets, fixture.onsets[-1] + 0.2)
    np.save(fixture.session / "data/on_list_times.npy", bad_edges)

    with pytest.raises(ValueError, match=r"\+0\.1 s"):
        load_regular_rf_trials(fixture.session, "A", fixture.rf_maps)


def test_load_fails_when_raw_counts_disagree_with_pooled_json(
    tmp_path: Path,
) -> None:
    fixture = _write_regular_fixture(tmp_path, pooled_delta=1)

    with pytest.raises(ValueError, match="unit_id 41"):
        load_regular_rf_trials(fixture.session, "A", fixture.rf_maps)


def test_loader_fails_closed_when_a_canonical_source_is_missing(
    tmp_path: Path,
) -> None:
    fixture = _write_regular_fixture(tmp_path)
    (fixture.session / "data/probeA/adc_spike_time.npy").unlink()

    with pytest.raises(FileNotFoundError, match="spike_times_npy"):
        load_regular_rf_trials(fixture.session, "A", fixture.rf_maps)
