from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

import Utils.rf_detection as rf_detection
from Utils.rf_detection import detect_rf


EXPECTED_KEYS = {
    "unit_ids",
    "response_map",
    "null_mean_map",
    "null_sd_map",
    "z_map",
    "valid_mask",
    "candidate_mask",
    "cluster_labels",
    "cluster_masses",
    "cluster_pvalues",
    "significant_mask",
    "filled_mask",
    "final_mask",
    "null_max_masses",
    "is_significance_tested",
    "n_workers",
}


def test_module_exposes_one_plain_function_and_no_result_wrappers() -> None:
    assert rf_detection.__all__ == ["detect_rf"]
    assert inspect.isfunction(detect_rf)
    for removed_name in (
        "RFClusterConfig",
        "RFTrialData",
        "RFClusterResult",
        "RFClusterBatchResult",
        "detect_rf_2d",
        "detect_rf_2d_batch",
        "cluster_permutation_2d",
    ):
        assert not hasattr(rf_detection, removed_name)


def test_analytic_null_moments_and_uniform_batch_shapes() -> None:
    # Across all C(4, 2)=6 assignments, each position mean has mean 4 and
    # variance 2.5. The observed position means are 1.5 and 6.5.
    result = detect_rf(
        [[1.0, 2.0, 5.0, 8.0]],
        [0, 0, 1, 1],
        (1, 2),
        unit_ids=[7],
        is_shuffle=False,
    )

    assert set(result) == EXPECTED_KEYS
    np.testing.assert_array_equal(result["unit_ids"], [7])
    np.testing.assert_allclose(result["response_map"], [[[1.5, 6.5]]])
    np.testing.assert_allclose(result["null_mean_map"], [[[4.0, 4.0]]])
    np.testing.assert_allclose(
        result["null_sd_map"],
        [[[math.sqrt(2.5), math.sqrt(2.5)]]],
    )
    np.testing.assert_allclose(
        result["z_map"],
        [[[-math.sqrt(2.5), math.sqrt(2.5)]]],
    )
    np.testing.assert_array_equal(
        result["candidate_mask"],
        [[[False, True]]],
    )

    assert result["response_map"].shape == (1, 1, 2)
    assert result["cluster_labels"].shape == (1, 1, 2)
    assert isinstance(result["cluster_masses"], tuple)
    assert len(result["cluster_masses"]) == 1
    assert result["null_max_masses"].shape == (1, 0)
    assert not result["is_significance_tested"]
    assert result["n_workers"] == 1

    # No-shuffle mode is explicitly exploratory.
    np.testing.assert_array_equal(
        result["final_mask"],
        result["candidate_mask"],
    )
    assert not result["significant_mask"].any()
    assert not result["filled_mask"].any()
    assert np.isnan(result["cluster_pvalues"][0]).all()


def test_all_output_arrays_are_read_only() -> None:
    result = detect_rf(
        [[1.0, 2.0, 5.0, 8.0], [8.0, 5.0, 2.0, 1.0]],
        [0, 0, 1, 1],
        (1, 2),
        unit_ids=[7, 8],
        is_shuffle=False,
    )

    for key, value in result.items():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable, key
        elif isinstance(value, tuple):
            for item in value:
                assert isinstance(item, np.ndarray)
                assert not item.flags.writeable, key
    with pytest.raises(ValueError):
        result["final_mask"][0, 0, 0] = True


def test_strata_preserve_between_block_drift() -> None:
    # There is a 100-spike block offset but no within-block spatial difference.
    result = detect_rf(
        [[0.0, 0.0, 100.0, 100.0]],
        [0, 1, 0, 1],
        (1, 2),
        stratum_ids=[10, 10, 20, 20],
        is_shuffle=False,
    )

    np.testing.assert_array_equal(result["null_mean_map"], [[[50.0, 50.0]]])
    np.testing.assert_array_equal(result["null_sd_map"], [[[0.0, 0.0]]])
    np.testing.assert_array_equal(result["z_map"], [[[0.0, 0.0]]])
    assert not result["valid_mask"].any()
    assert not result["candidate_mask"].any()


def test_response_map_uses_per_position_means_with_unbalanced_repeats() -> None:
    result = detect_rf(
        [[2.0, 2.0, 2.0, 6.0]],
        [0, 0, 0, 1],
        (1, 2),
        is_shuffle=False,
    )

    # Both positions have the same total (6), but their trial means differ.
    np.testing.assert_allclose(result["response_map"], [[[2.0, 6.0]]])
    np.testing.assert_allclose(
        result["z_map"],
        [[[-math.sqrt(3), math.sqrt(3)]]],
    )
    np.testing.assert_array_equal(
        result["candidate_mask"],
        [[[False, True]]],
    )


def test_less_alternative_detects_the_opposite_tail() -> None:
    result = detect_rf(
        [[1.0, 2.0, 5.0, 8.0]],
        [0, 0, 1, 1],
        (1, 2),
        is_shuffle=False,
        alternative="less",
    )
    np.testing.assert_array_equal(
        result["candidate_mask"],
        [[[True, False]]],
    )


def test_four_connectivity_horizontal_wrap_and_diagonal_separation() -> None:
    score = np.asarray([[2.0, 2.0, 0.0], [2.0, 0.0, 2.0]])
    candidate = score >= 1.5

    labels, masses = rf_detection._label_clusters(
        score,
        candidate,
        wrap_x=False,
    )
    assert labels.max() == 2
    np.testing.assert_allclose(np.sort(masses), [2.0, 6.0])

    wrapped_labels, wrapped_masses = rf_detection._label_clusters(
        score,
        candidate,
        wrap_x=True,
    )
    assert wrapped_labels.max() == 1
    np.testing.assert_allclose(wrapped_masses, [8.0])

    diagonal = np.asarray([[2.0, 0.0], [0.0, 2.0]])
    diagonal_labels, diagonal_masses = rf_detection._label_clusters(
        diagonal,
        diagonal >= 1.5,
        wrap_x=False,
    )
    assert diagonal_labels.max() == 2
    np.testing.assert_allclose(diagonal_masses, [2.0, 2.0])


def test_scipy_batch_cluster_mass_keeps_maps_independent_and_wraps_seam() -> None:
    scores = np.asarray(
        [
            [[2.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
            [[2.0, 2.0, 0.0], [0.0, 0.0, 4.0]],
        ]
    )
    valid = np.ones((2, 3), dtype=bool)

    plain = rf_detection._max_cluster_masses_batch(
        scores,
        valid,
        cluster_forming_z=1.5,
        wrap_x=False,
    )
    wrapped = rf_detection._max_cluster_masses_batch(
        scores,
        valid,
        cluster_forming_z=1.5,
        wrap_x=True,
    )

    np.testing.assert_array_equal(plain, [3.0, 4.0])
    np.testing.assert_array_equal(wrapped, [5.0, 4.0])


def test_cluster_threshold_is_inclusive_and_pvalues_use_plus_one_ties() -> None:
    maximum = rf_detection._max_cluster_masses_batch(
        np.asarray([[[1.5]]]),
        np.asarray([[True]]),
        cluster_forming_z=1.5,
        wrap_x=False,
    )
    np.testing.assert_array_equal(maximum, [1.5])

    pvalues = rf_detection._cluster_pvalues(
        np.asarray([5.0, 5.1]),
        np.asarray([0.0, 2.0, 5.0, 5.0]),
    )
    np.testing.assert_array_equal(pvalues, [0.6, 0.2])

    null = np.arange(19, dtype=float)
    assert rf_detection._cluster_pvalues(
        np.asarray([18.0, 18.1]),
        null,
    ).tolist() == [0.1, 0.05]


def test_single_hole_fill_requires_same_significant_component() -> None:
    valid = np.ones((2, 3), dtype=bool)
    same_component = np.asarray([[1, 0, 1], [1, 1, 1]], dtype=np.int32)
    mixed_components = np.asarray([[1, 0, 2], [1, 2, 2]], dtype=np.int32)

    np.testing.assert_array_equal(
        rf_detection._fill_single_holes(
            same_component,
            valid,
            wrap_x=False,
            min_hole_neighbors=3,
        ),
        [[False, True, False], [False, False, False]],
    )
    assert not rf_detection._fill_single_holes(
        mixed_components,
        valid,
        wrap_x=False,
        min_hole_neighbors=3,
    ).any()

    invalid_hole = valid.copy()
    invalid_hole[0, 1] = False
    assert not rf_detection._fill_single_holes(
        same_component,
        invalid_hole,
        wrap_x=False,
        min_hole_neighbors=3,
    ).any()


def _permutation_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.tile(np.arange(4), 4)
    strata = np.repeat(np.arange(4), 4)
    first = np.tile([9.0, 7.0, 1.0, 0.0], 4)
    second = np.tile([0.0, 1.0, 7.0, 9.0], 4)
    return np.stack([first, second]), positions, strata


def test_fixed_seed_is_exact_across_batch_size_and_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, positions, strata = _permutation_fixture()
    monkeypatch.setattr(rf_detection, "_available_cpu_count", lambda: 16)
    serial = detect_rf(
        responses,
        positions,
        (2, 2),
        stratum_ids=strata,
        unit_ids=[11, 22],
        cluster_forming_z=0.5,
        n_permutations=37,
        random_seed=1234,
        batch_size=1,
        n_jobs=1,
    )
    threaded = detect_rf(
        responses,
        positions,
        (2, 2),
        stratum_ids=strata,
        unit_ids=[11, 22],
        cluster_forming_z=0.5,
        n_permutations=37,
        random_seed=1234,
        batch_size=17,
        n_jobs=None,
    )

    assert serial["n_workers"] == 1
    assert threaded["n_workers"] == 2
    for key in (
        "response_map",
        "null_mean_map",
        "null_sd_map",
        "z_map",
        "candidate_mask",
        "cluster_labels",
        "significant_mask",
        "final_mask",
        "null_max_masses",
    ):
        np.testing.assert_array_equal(serial[key], threaded[key])
    for key in ("cluster_masses", "cluster_pvalues"):
        for serial_unit, threaded_unit in zip(
            serial[key],
            threaded[key],
            strict=True,
        ):
            np.testing.assert_array_equal(serial_unit, threaded_unit)


def test_single_unit_auto_serial_and_explicit_threads_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, positions, strata = _permutation_fixture()
    monkeypatch.setattr(rf_detection, "_available_cpu_count", lambda: 16)
    common = {
        "position_ids": positions,
        "shape": (2, 2),
        "stratum_ids": strata,
        "unit_ids": [11],
        "cluster_forming_z": 0.5,
        "n_permutations": 37,
        "random_seed": 1234,
    }
    serial = detect_rf(
        responses[:1],
        batch_size=1,
        n_jobs=1,
        **common,
    )
    automatic = detect_rf(
        responses[:1],
        batch_size=17,
        n_jobs=None,
        **common,
    )
    explicit = detect_rf(
        responses[:1],
        batch_size=8,
        n_jobs=2,
        **common,
    )

    assert serial["n_workers"] == 1
    assert automatic["n_workers"] == 1
    assert explicit["n_workers"] == 2
    for threaded in (automatic, explicit):
        for key in EXPECTED_KEYS - {"n_workers"}:
            serial_value = serial[key]
            threaded_value = threaded[key]
            if isinstance(serial_value, tuple):
                for serial_unit, threaded_unit in zip(
                    serial_value,
                    threaded_value,
                    strict=True,
                ):
                    np.testing.assert_array_equal(serial_unit, threaded_unit)
            elif isinstance(serial_value, np.ndarray):
                np.testing.assert_array_equal(serial_value, threaded_value)
            else:
                assert serial_value == threaded_value


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_rf_progress_tracks_completed_permutations_for_all_worker_modes(
    monkeypatch: pytest.MonkeyPatch,
    n_jobs: int,
) -> None:
    responses, positions, strata = _permutation_fixture()
    monkeypatch.setattr(rf_detection, "_available_cpu_count", lambda: 16)
    created: list[dict[str, object]] = []

    class RecordingProgress:
        def __init__(self, *, total: int, n_workers: int) -> None:
            self.total = total
            self.n_workers = n_workers
            self.updates: list[int] = []
            self.closed = False

        def update(self, amount: int) -> None:
            self.updates.append(amount)

        def close(self) -> None:
            self.closed = True

    def make_progress_bar(*, total: int, n_workers: int) -> RecordingProgress:
        progress = RecordingProgress(total=total, n_workers=n_workers)
        created.append({"progress": progress})
        return progress

    monkeypatch.setattr(rf_detection, "_make_progress_bar", make_progress_bar)
    result = detect_rf(
        responses,
        positions,
        (2, 2),
        stratum_ids=strata,
        cluster_forming_z=0.5,
        n_permutations=7,
        batch_size=3,
        n_jobs=n_jobs,
        show_progress=True,
    )

    assert result["n_workers"] == n_jobs
    assert len(created) == 1
    progress = created[0]["progress"]
    assert isinstance(progress, RecordingProgress)
    assert progress.total == 7
    assert progress.n_workers == n_jobs
    assert progress.updates == [3, 3, 1]
    assert sum(progress.updates) == progress.total
    assert progress.closed


def test_rf_progress_can_be_disabled_and_is_skipped_without_shuffle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_progress(**_kwargs: object) -> None:
        pytest.fail("progress bar should not be created")

    monkeypatch.setattr(rf_detection, "_make_progress_bar", unexpected_progress)
    detect_rf(
        [[1.0, 2.0, 5.0, 8.0]],
        [0, 0, 1, 1],
        (1, 2),
        n_permutations=3,
        show_progress=False,
    )
    detect_rf(
        [[1.0, 2.0, 5.0, 8.0]],
        [0, 0, 1, 1],
        (1, 2),
        is_shuffle=False,
        show_progress=True,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_automatic_worker_count_uses_the_measured_safe_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rf_detection, "_available_cpu_count", lambda: 32)

    assert rf_detection._resolve_n_workers(None, 146, is_shuffle=True) == 2
    assert rf_detection._resolve_n_workers(None, 1, is_shuffle=True) == 1
    assert rf_detection._resolve_n_workers(None, 2, is_shuffle=True) == 2
    assert rf_detection._resolve_n_workers(None, 146, is_shuffle=False) == 1


def test_shuffle_pvalues_use_each_units_maximum_null() -> None:
    responses, positions, strata = _permutation_fixture()
    result = detect_rf(
        responses,
        positions,
        (2, 2),
        stratum_ids=strata,
        unit_ids=[11, 22],
        cluster_forming_z=0.5,
        n_permutations=29,
        random_seed=9,
        batch_size=8,
        n_jobs=2,
    )

    assert result["is_significance_tested"]
    assert result["null_max_masses"].shape == (2, 29)
    for unit_index in range(2):
        for mass, pvalue in zip(
            result["cluster_masses"][unit_index],
            result["cluster_pvalues"][unit_index],
            strict=True,
        ):
            expected = (
                1
                + np.count_nonzero(
                    result["null_max_masses"][unit_index] >= mass
                )
            ) / 30
            assert pvalue == expected


def test_unpresented_positions_remain_invalid() -> None:
    result = detect_rf(
        [[1.0, 2.0, 4.0, 8.0]],
        [0, 0, 1, 1],
        (1, 3),
        unit_ids=[4],
        is_shuffle=False,
    )

    np.testing.assert_array_equal(
        result["valid_mask"],
        [[[True, True, False]]],
    )
    assert np.isnan(result["response_map"][0, 0, 2])
    assert np.isnan(result["null_mean_map"][0, 0, 2])
    assert np.isnan(result["null_sd_map"][0, 0, 2])
    assert result["z_map"][0, 0, 2] == 0
    assert not result["final_mask"][0, 0, 2]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cluster_forming_z": 0},
        {"cluster_forming_z": np.nan},
        {"alpha": 0},
        {"alpha": 1},
        {"n_permutations": -1},
        {"alternative": "two-sided"},
        {"batch_size": 0},
        {"min_hole_neighbors": 5},
        {"random_seed": -1},
        {"n_jobs": 0},
        {"wrap_x": 1},
        {"fill_single_holes": 1},
        {"show_progress": 1},
    ],
)
def test_rejects_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        detect_rf(
            [[1.0, 2.0]],
            [0, 1],
            (1, 2),
            is_shuffle=False,
            **kwargs,
        )


def test_shuffle_mode_and_response_rank_are_explicit() -> None:
    exploratory = detect_rf(
        [[1.0, 2.0, 5.0, 8.0]],
        [0, 0, 1, 1],
        (1, 2),
        is_shuffle=False,
        n_permutations=0,
    )
    assert not exploratory["is_significance_tested"]
    assert exploratory["null_max_masses"].shape == (1, 0)

    with pytest.raises(ValueError, match="n_permutations"):
        detect_rf(
            [[1.0, 2.0]],
            [0, 1],
            (1, 2),
            n_permutations=0,
        )
    with pytest.raises(ValueError, match="is_shuffle"):
        detect_rf(
            [[1.0, 2.0]],
            [0, 1],
            (1, 2),
            is_shuffle=1,
        )
    with pytest.raises(ValueError, match=r"\(unit, trial\)"):
        detect_rf(
            [1.0, 2.0],
            [0, 1],
            (1, 2),
            is_shuffle=False,
        )


def test_rejects_misaligned_labels_and_unit_ids() -> None:
    with pytest.raises(ValueError, match="position_ids"):
        detect_rf([[1.0, 2.0]], [0], (1, 2), is_shuffle=False)
    with pytest.raises(ValueError, match="position_ids"):
        detect_rf([[1.0, 2.0]], [0.0, 1.0], (1, 2), is_shuffle=False)
    with pytest.raises(ValueError, match="stratum_ids"):
        detect_rf(
            [[1.0, 2.0]],
            [0, 1],
            (1, 2),
            stratum_ids=[0],
            is_shuffle=False,
        )
    with pytest.raises(ValueError, match="unit_ids"):
        detect_rf(
            [[1.0, 2.0], [3.0, 4.0]],
            [0, 1],
            (1, 2),
            unit_ids=[7, 7],
            is_shuffle=False,
        )
