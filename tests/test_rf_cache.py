from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import Utils.rf_cache as rf_cache
from Utils.rf_cache import (
    RF_RESULT_CACHE_SCHEMA_VERSION,
    RFResultCacheError,
    build_rf_result_cache_key,
    load_rf_result,
    rf_result_path,
    save_rf_result,
)


def _key_arrays() -> dict[str, object]:
    return {
        "responses": np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        "position_ids": np.asarray([0, 1, 0], dtype=np.int64),
        "stratum_ids": np.asarray([4, 4, 5], dtype=np.int64),
        "unit_ids": np.asarray([10, 11], dtype=np.int64),
        "x_positions": np.asarray([-10.0, 10.0]),
        "y_positions": np.asarray([-5.0]),
    }


def _cache_key() -> str:
    return build_rf_result_cache_key(
        arrays=_key_arrays(),
        manifest={"is_shuffle": True, "cluster_forming_z": 1.5},
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("foo.json", "foo.npz"),
        ("foo.rfmap", "foo.npz"),
        ("foo", "foo.npz"),
        ("recording.v2.json", "recording.v2.npz"),
    ],
)
def test_rf_result_path_replaces_only_the_source_suffix(
    source: str,
    expected: str,
) -> None:
    assert rf_result_path(source) == Path(expected)


def test_rf_result_path_rejects_npz_source() -> None:
    with pytest.raises(ValueError, match=r"already end with '\.npz'"):
        rf_result_path("source.npz")


@pytest.mark.parametrize(
    "filename",
    ["source.rfmap", "source.json", "result"],
)
def test_result_io_rejects_unsafe_suffixes(
    tmp_path: Path,
    filename: str,
) -> None:
    target = tmp_path / filename
    with pytest.raises(ValueError, match=r"\.npz"):
        save_rf_result(
            target,
            mask_2d=[[1]],
            center_2d=[[1]],
            unit_ids=[1],
            manifest={},
            cache_key="key",
        )
    with pytest.raises(ValueError, match=r"\.npz"):
        load_rf_result(target, expected_cache_key="key")
    assert not target.exists()


def test_cache_key_is_canonical_and_hashes_array_metadata_and_values() -> None:
    arrays = _key_arrays()
    manifest = {"z": [1, 2], "a": {"enabled": True}}
    key = build_rf_result_cache_key(arrays=arrays, manifest=manifest)

    reordered_arrays = dict(reversed(list(arrays.items())))
    reordered_manifest = {"a": {"enabled": True}, "z": [1, 2]}
    assert build_rf_result_cache_key(
        arrays=reordered_arrays,
        manifest=reordered_manifest,
    ) == key

    fortran_arrays = dict(arrays)
    fortran_arrays["responses"] = np.asfortranarray(arrays["responses"])
    assert build_rf_result_cache_key(
        arrays=fortran_arrays,
        manifest=manifest,
    ) == key

    changed_values = dict(arrays)
    changed_values["responses"] = np.asarray(arrays["responses"]).copy()
    changed_values["responses"][0, 0] = 99.0
    assert build_rf_result_cache_key(
        arrays=changed_values,
        manifest=manifest,
    ) != key

    changed_dtype = dict(arrays)
    changed_dtype["responses"] = np.asarray(
        arrays["responses"],
        dtype=np.float32,
    )
    assert build_rf_result_cache_key(
        arrays=changed_dtype,
        manifest=manifest,
    ) != key
    assert build_rf_result_cache_key(
        arrays=arrays,
        manifest={"z": [1, 3], "a": {"enabled": True}},
    ) != key

    scalar_extra = dict(arrays, extra=np.asarray(1, dtype=np.int64))
    vector_extra = dict(arrays, extra=np.asarray([1], dtype=np.int64))
    assert build_rf_result_cache_key(
        arrays=scalar_extra,
        manifest=manifest,
    ) != build_rf_result_cache_key(
        arrays=vector_extra,
        manifest=manifest,
    )


def test_cache_key_supports_an_explicit_absent_stratum() -> None:
    arrays = _key_arrays()
    arrays["stratum_ids"] = None

    key = build_rf_result_cache_key(arrays=arrays, manifest={})

    assert len(key) == 64
    assert set(key) <= set("0123456789abcdef")


def test_cache_key_rejects_missing_misaligned_and_object_inputs() -> None:
    missing = _key_arrays()
    del missing["position_ids"]
    with pytest.raises(ValueError, match="missing cache-key inputs"):
        build_rf_result_cache_key(arrays=missing, manifest={})

    misaligned = _key_arrays()
    misaligned["stratum_ids"] = np.asarray([1, 2])
    with pytest.raises(ValueError, match="stratum_ids must align"):
        build_rf_result_cache_key(arrays=misaligned, manifest={})

    object_array = _key_arrays()
    object_array["responses"] = np.asarray([[object()]], dtype=object)
    with pytest.raises(ValueError, match="object dtype"):
        build_rf_result_cache_key(arrays=object_array, manifest={})

    with pytest.raises(ValueError, match="JSON-compatible"):
        build_rf_result_cache_key(
            arrays=_key_arrays(),
            manifest={"threshold": np.nan},
        )


def test_batch_cache_round_trip_is_pickle_free_and_read_only(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "session.npz"
    mask = np.asarray(
        [
            [[1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    center = np.asarray(
        [
            [[0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    unit_ids = np.asarray([41, 7], dtype=np.int32)
    manifest = {"alternative": "greater", "options": {"drop_bins": 1}}
    key = _cache_key()

    assert save_rf_result(
        target,
        mask_2d=mask,
        center_2d=center,
        unit_ids=unit_ids,
        manifest=manifest,
        cache_key=key,
    ) == target
    assert target.is_file()

    with np.load(target, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema_version",
            "mask_2d",
            "center_2d",
            "unit_ids",
            "manifest_json",
            "cache_key",
        }
        assert archive["manifest_json"].item() == json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        assert all(archive[name].dtype != object for name in archive.files)

    loaded = load_rf_result(target, expected_cache_key=key)
    assert loaded is not None
    assert loaded["schema_version"] == RF_RESULT_CACHE_SCHEMA_VERSION
    assert loaded["manifest"] == manifest
    assert loaded["cache_key"] == key
    np.testing.assert_array_equal(loaded["mask_2d"], mask)
    np.testing.assert_array_equal(loaded["center_2d"], center)
    np.testing.assert_array_equal(loaded["unit_ids"], unit_ids)
    assert loaded["mask_2d"].dtype == np.dtype(np.uint8)
    assert loaded["center_2d"].dtype == np.dtype(np.uint8)
    assert loaded["unit_ids"].dtype == np.dtype(np.int64)
    for name in ("mask_2d", "center_2d", "unit_ids"):
        assert not loaded[name].flags.writeable
        with pytest.raises(ValueError):
            loaded[name].flat[0] = 99


def test_single_map_result_normalizes_to_canonical_unit_axis(tmp_path: Path) -> None:
    target = tmp_path / "single.npz"
    key = _cache_key()

    save_rf_result(
        target,
        mask_2d=np.zeros((2, 3), dtype=bool),
        center_2d=np.zeros((2, 3), dtype=bool),
        unit_ids=[91],
        manifest={},
        cache_key=key,
    )
    loaded = load_rf_result(target, expected_cache_key=key)

    assert loaded is not None
    assert loaded["mask_2d"].shape == (1, 2, 3)
    assert loaded["center_2d"].shape == (1, 2, 3)
    np.testing.assert_array_equal(loaded["unit_ids"], [91])


def test_missing_file_and_key_mismatch_are_ordinary_cache_misses(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.npz"
    key = _cache_key()
    assert load_rf_result(target, expected_cache_key=key) is None

    save_rf_result(
        target,
        mask_2d=[[1, 0]],
        center_2d=[[1, 0]],
        unit_ids=[3],
        manifest={},
        cache_key=key,
    )
    assert load_rf_result(target, expected_cache_key="different") is None


@pytest.mark.parametrize(
    ("mask", "center", "unit_ids", "message"),
    [
        ([[1, 0]], [[0, 1]], [1], "must also belong"),
        ([[1, 1]], [[1, 1]], [1], "exactly one bin"),
        ([[1, 1]], [[0, 0]], [1], "exactly one bin"),
        ([[1, 0]], [[1], [0]], [1], "identical shapes"),
        ([[1, 0]], [[1, 0]], [1, 2], "aligned to units"),
        (
            [[[1, 0]], [[0, 1]]],
            [[[1, 0]], [[0, 1]]],
            [1, 1],
            "must be unique",
        ),
        ([[2, 0]], [[1, 0]], [1], "only zero and one"),
    ],
)
def test_save_rejects_inconsistent_final_results(
    tmp_path: Path,
    mask: object,
    center: object,
    unit_ids: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        save_rf_result(
            tmp_path / "bad.npz",
            mask_2d=mask,
            center_2d=center,
            unit_ids=unit_ids,
            manifest={},
            cache_key=_cache_key(),
        )


def test_atomic_replace_failure_preserves_old_cache_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "result.npz"
    key = _cache_key()
    save_rf_result(
        target,
        mask_2d=[[1, 0]],
        center_2d=[[1, 0]],
        unit_ids=[3],
        manifest={"generation": 1},
        cache_key=key,
    )
    original_bytes = target.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(rf_cache.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        save_rf_result(
            target,
            mask_2d=[[0, 1]],
            center_2d=[[0, 1]],
            unit_ids=[3],
            manifest={"generation": 2},
            cache_key=key,
        )

    assert target.read_bytes() == original_bytes
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_existing_corrupt_or_incompatible_cache_raises_explicit_error(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not a zip archive")
    with pytest.raises(RFResultCacheError, match="unable to load"):
        load_rf_result(corrupt, expected_cache_key=_cache_key())

    incompatible = tmp_path / "incompatible.npz"
    np.savez_compressed(
        incompatible,
        schema_version=np.asarray(
            RF_RESULT_CACHE_SCHEMA_VERSION + 1,
            dtype=np.int64,
        ),
        mask_2d=np.asarray([[1]], dtype=np.uint8),
        center_2d=np.asarray([[1]], dtype=np.uint8),
        unit_ids=np.asarray([1], dtype=np.int64),
        manifest_json=np.asarray("{}"),
        cache_key=np.asarray(_cache_key()),
    )
    with pytest.raises(RFResultCacheError, match="unsupported.*schema"):
        load_rf_result(incompatible, expected_cache_key=_cache_key())
