from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from Utils.rf_cache import RF_RESULT_CACHE_SCHEMA_VERSION, build_rf_result_cache_key


def _legacy_cache_key(arrays: dict[str, object], manifest: dict[str, object]) -> str:
    """Reproduce the persisted byte format used before buffer-based hashing."""
    digest = hashlib.sha256()

    def chunk(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)

    chunk(f"rf-result-cache-v{RF_RESULT_CACHE_SCHEMA_VERSION}".encode("ascii"))
    chunk(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8"))
    for name in sorted(arrays):
        chunk(name.encode("utf-8"))
        value = arrays[name]
        if value is None:
            chunk(b"none")
            continue
        array = np.asarray(value)
        contiguous = np.ascontiguousarray(array)
        chunk(contiguous.dtype.str.encode("ascii"))
        chunk(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        chunk(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "reversed"])
@pytest.mark.parametrize("dtype", ["<f8", ">f8", "<i8"])
def test_buffer_hash_preserves_persisted_keys_and_detects_mutation(
    layout: str, dtype: str,
) -> None:
    responses = np.arange(24, dtype=dtype).reshape(3, 8)
    if layout == "fortran":
        responses = np.asfortranarray(responses)
    elif layout == "strided":
        responses = responses[:, ::2]
    elif layout == "reversed":
        responses = responses[:, ::-1]
    arrays = {
        "responses": responses,
        "position_ids": np.arange(responses.shape[1], dtype=np.int64),
        "stratum_ids": None,
        "unit_ids": np.arange(3, dtype=np.int64),
        "x_positions": np.arange(2, dtype=float),
        "y_positions": np.arange(2, dtype=float),
        "extra_text": np.asarray(["响应", "rate"]),
        "extra_empty": np.empty((0, 2)),
        "extra_time": np.asarray(["2026-09-21"], dtype="datetime64[D]"),
    }
    manifest = {"parameters": {"seed": 0}, "name": "RF 响应"}

    initial = build_rf_result_cache_key(arrays=arrays, manifest=manifest)
    assert initial == _legacy_cache_key(arrays, manifest)
    responses[0, 0] += 1
    changed = build_rf_result_cache_key(arrays=arrays, manifest=manifest)
    assert changed != initial
    assert changed == _legacy_cache_key(arrays, manifest)
