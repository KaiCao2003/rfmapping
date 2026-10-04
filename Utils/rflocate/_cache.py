"""Internal, pickle-free persistence for versioned RF result sidecars.

An RF result sidecar contains both possible public ``rf_2d`` outputs: the full
binary RF mask and its single-bin center.  Consequently, choosing one of those
outputs is not part of the result identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .results import (
    RF_RESULT_CACHE_SCHEMA_VERSION,
    RFResult,
    _binary_array,
    _cache_key,
    _canonical_manifest_json,
    _rf_result_for_map,
    _rf_source_path,
    _unit_id_array,
    _validated_result_arrays,
)

__all__ = ["RFResult", "RFResultCacheError", "read_rf_result", "rf_result_path"]

RFResultCache = RFResult

_ARCHIVE_KEYS = frozenset(
    {
        "schema_version",
        "mask_2d",
        "center_2d",
        "unit_ids",
        "manifest_json",
        "cache_key",
    }
)
_KEY_ARRAY_NAMES = frozenset(
    {
        "responses",
        "position_ids",
        "stratum_ids",
        "unit_ids",
        "x_positions",
        "y_positions",
    }
)


class RFResultCacheError(RuntimeError):
    """Raised when an existing RF result cache is malformed or incompatible."""


def rf_result_path(
    source_path: str | os.PathLike[str],
    *,
    dimension: str = "2d",
    rf_type: str = "excitatory",
) -> Path:
    """Return a saved detection path, preserving dimension and polarity."""

    source = Path(source_path)
    if source.suffix.lower() == ".npz":
        raise ValueError("source_path must not already end with '.npz'")
    if dimension not in ("2d", "1d"):
        raise ValueError("dimension must be '2d' or '1d'")
    if rf_type not in ("excitatory", "inhibitory"):
        raise ValueError("rf_type must be 'excitatory' or 'inhibitory'")
    stem = source.stem if source.suffix else source.name
    if rf_type == "inhibitory":
        stem += "_inhibitory"
    if dimension == "1d":
        stem += "_1d"
    return source.with_name(stem + ".npz")


def _result_archive_path(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError("RF result paths must end with '.npz'")
    return target


def _key_array(value: Any, name: str) -> NDArray[Any]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"cache-key array {name!r} is invalid: {exc}") from exc
    if array.dtype.hasobject:
        raise ValueError(f"cache-key array {name!r} must not have object dtype")
    return array


def _validated_key_arrays(
    arrays: Mapping[str, Any],
) -> dict[str, NDArray[Any] | None]:
    if not isinstance(arrays, Mapping):
        raise ValueError("arrays must be a mapping")
    if not all(isinstance(name, str) and name for name in arrays):
        raise ValueError("cache-key array names must be non-empty strings")
    missing = _KEY_ARRAY_NAMES.difference(arrays)
    if missing:
        raise ValueError(
            f"arrays is missing cache-key inputs: {sorted(missing)}"
        )

    validated: dict[str, NDArray[Any] | None] = {}
    for name, value in arrays.items():
        if name == "stratum_ids" and value is None:
            validated[name] = None
        else:
            validated[name] = _key_array(value, name)

    responses = validated["responses"]
    position_ids = validated["position_ids"]
    stratum_ids = validated["stratum_ids"]
    unit_ids = validated["unit_ids"]
    x_positions = validated["x_positions"]
    y_positions = validated["y_positions"]
    assert responses is not None
    assert position_ids is not None
    assert unit_ids is not None
    assert x_positions is not None
    assert y_positions is not None

    if responses.ndim != 2 or 0 in responses.shape:
        raise ValueError("cache-key responses must have shape (unit, trial)")
    if position_ids.ndim != 1 or position_ids.size != responses.shape[1]:
        raise ValueError("cache-key position_ids must align with response trials")
    if stratum_ids is not None and (
        stratum_ids.ndim != 1 or stratum_ids.size != responses.shape[1]
    ):
        raise ValueError("cache-key stratum_ids must align with response trials")
    if unit_ids.ndim != 1 or unit_ids.size != responses.shape[0]:
        raise ValueError("cache-key unit_ids must align with response units")
    for name, positions in (
        ("x_positions", x_positions),
        ("y_positions", y_positions),
    ):
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError(f"cache-key {name} must be a non-empty 1-D array")
    return validated


def _hash_chunk(hasher: Any, payload: bytes | memoryview) -> None:
    """Hash a length-delimited byte string to avoid concatenation ambiguity."""

    hasher.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    hasher.update(payload)


def build_rf_result_cache_key(
    *,
    arrays: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    """Hash all RF inputs and detection settings into one stable SHA-256 key.

    The required arrays are ``responses``, ``position_ids``, ``stratum_ids``
    (which may be ``None``), ``unit_ids``, ``x_positions``, and
    ``y_positions``.  Extra named arrays are included as well.  Logical array
    order, dtype, shape, and C-order bytes all contribute to the key.
    """

    validated = _validated_key_arrays(arrays)
    manifest_json = _canonical_manifest_json(manifest)
    hasher = hashlib.sha256()
    _hash_chunk(
        hasher,
        f"rf-result-cache-v{RF_RESULT_CACHE_SCHEMA_VERSION}".encode("ascii"),
    )
    _hash_chunk(hasher, manifest_json.encode("utf-8"))

    for name in sorted(validated):
        _hash_chunk(hasher, name.encode("utf-8"))
        array = validated[name]
        if array is None:
            _hash_chunk(hasher, b"none")
            continue
        contiguous = np.ascontiguousarray(array)
        _hash_chunk(hasher, contiguous.dtype.str.encode("ascii"))
        _hash_chunk(
            hasher,
            json.dumps(array.shape, separators=(",", ":")).encode("ascii"),
        )
        byte_view = contiguous.view(np.uint8).reshape(-1)
        _hash_chunk(hasher, memoryview(byte_view))
    return hasher.hexdigest()


def save_rf_result(
    path: str | os.PathLike[str],
    *,
    mask_2d: Any,
    center_2d: Any,
    unit_ids: Any,
    manifest: Mapping[str, Any],
    cache_key: str,
) -> Path:
    """Atomically save one validated, pickle-free RF result archive."""

    target = _result_archive_path(path)
    mask, center, ids = _validated_result_arrays(
        mask_2d,
        center_2d,
        unit_ids,
    )
    manifest_json = _canonical_manifest_json(manifest)
    key = _cache_key(cache_key)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                schema_version=np.asarray(
                    RF_RESULT_CACHE_SCHEMA_VERSION,
                    dtype=np.int64,
                ),
                mask_2d=mask,
                center_2d=center,
                unit_ids=ids,
                manifest_json=np.asarray(manifest_json, dtype=np.str_),
                cache_key=np.asarray(key, dtype=np.str_),
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                # Preserve the original save failure; the uniquely named file
                # is never a valid cache target and can be cleaned later.
                pass
    return target


def _stored_scalar_text(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"stored {name} must be one Unicode scalar")
    parsed = array.item()
    if not isinstance(parsed, str) or not parsed:
        raise ValueError(f"stored {name} must be one non-empty Unicode scalar")
    return parsed


def _read_rf_result(
    path: str | os.PathLike[str],
    *,
    expected_cache_key: str | None = None,
) -> RFResult | None:
    target = _result_archive_path(path)
    try:
        with np.load(target, allow_pickle=False) as archive:
            found_keys = frozenset(archive.files)
            if found_keys != _ARCHIVE_KEYS:
                missing = sorted(_ARCHIVE_KEYS.difference(found_keys))
                unexpected = sorted(found_keys.difference(_ARCHIVE_KEYS))
                raise ValueError(
                    "archive fields do not match the RF result schema; "
                    f"missing={missing}, unexpected={unexpected}"
                )

            schema = np.asarray(archive["schema_version"])
            if (
                schema.shape != ()
                or np.issubdtype(schema.dtype, np.bool_)
                or not np.issubdtype(schema.dtype, np.integer)
            ):
                raise ValueError("stored schema_version must be one integer")
            stored_version = int(schema.item())
            if stored_version != RF_RESULT_CACHE_SCHEMA_VERSION:
                raise ValueError(
                    "unsupported RF result cache schema version "
                    f"{stored_version}; expected "
                    f"{RF_RESULT_CACHE_SCHEMA_VERSION}"
                )

            stored_key = _stored_scalar_text(archive["cache_key"], "cache_key")
            if expected_cache_key is not None and stored_key != expected_cache_key:
                return None

            manifest_json = _stored_scalar_text(
                archive["manifest_json"],
                "manifest_json",
            )
            parsed_manifest = json.loads(manifest_json)
            if not isinstance(parsed_manifest, dict):
                raise ValueError("stored manifest_json must encode one JSON object")
            if _canonical_manifest_json(parsed_manifest) != manifest_json:
                raise ValueError("stored manifest_json is not canonical")

            result = RFResult(
                mask_2d=archive["mask_2d"],
                center_2d=archive["center_2d"],
                unit_ids=archive["unit_ids"],
                manifest=parsed_manifest,
                cache_key=stored_key,
                schema_version=stored_version,
            )
    except FileNotFoundError:
        raise
    except RFResultCacheError:
        raise
    except Exception as exc:
        raise RFResultCacheError(
            f"unable to load RF result cache {target}: {exc}"
        ) from exc

    return result


def read_rf_result(path: str | os.PathLike[str]) -> RFResult:
    """Read a saved RF result without computing or replacing it.

    Missing files raise ``FileNotFoundError``. Invalid schemas, unit alignment,
    masks, and centers raise ``RFResultCacheError``. Arrays are read-only.
    Both 1-D and 2-D results retain the stored ``(unit, y, x)`` axes.
    """
    result = _read_rf_result(path)
    assert result is not None
    return result


def load_rf_result(
    path: str | os.PathLike[str],
    *,
    expected_cache_key: str,
) -> RFResult | None:
    """Load a matching cached RF result; missing files and key mismatches miss.

    Existing incompatible or corrupt results raise ``RFResultCacheError`` so
    detection never silently overwrites an unexplained bad artifact.
    """
    expected_key = _cache_key(expected_cache_key, "expected_cache_key")
    try:
        return _read_rf_result(path, expected_cache_key=expected_key)
    except FileNotFoundError:
        return None
