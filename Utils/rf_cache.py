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
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

__all__: list[str] = []


RF_RESULT_CACHE_SCHEMA_VERSION = 1

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


class RFResultCache(TypedDict):
    """Validated in-memory representation returned by :func:`load_rf_result`."""

    schema_version: int
    mask_2d: NDArray[np.uint8]
    center_2d: NDArray[np.uint8]
    unit_ids: NDArray[np.int64]
    manifest: dict[str, Any]
    cache_key: str


def rf_result_path(source_path: str | os.PathLike[str]) -> Path:
    """Return the ``.npz`` result path for an RF source file."""

    source = Path(source_path)
    if source.suffix.lower() == ".npz":
        raise ValueError("source_path must not already end with '.npz'")
    if source.suffix:
        return source.with_suffix(".npz")
    return source.with_name(source.name + ".npz")


def _result_archive_path(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError("RF result paths must end with '.npz'")
    return target


def _canonical_manifest_json(manifest: Mapping[str, Any]) -> str:
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a JSON-compatible mapping")
    try:
        serialized = json.dumps(
            dict(manifest),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"manifest must be a JSON-compatible mapping: {exc}"
        ) from exc

    # Parsing once catches encoder extensions or unusual mapping keys that do
    # not round-trip as one ordinary JSON object.
    parsed = json.loads(serialized)
    if not isinstance(parsed, dict):
        raise ValueError("manifest must encode one JSON object")
    if json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) != serialized:
        raise ValueError("manifest must use unambiguous JSON object keys")
    return serialized


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


def _binary_array(value: Any, name: str) -> NDArray[np.uint8]:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is invalid: {exc}") from exc
    if (
        raw.dtype.hasobject
        or not (
            np.issubdtype(raw.dtype, np.bool_)
            or np.issubdtype(raw.dtype, np.number)
        )
        or np.issubdtype(raw.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be a binary numeric array")
    if raw.ndim == 2:
        raw = raw[np.newaxis, ...]
    if raw.ndim != 3 or 0 in raw.shape:
        raise ValueError(f"{name} must have shape (unit, y, x)")
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError(f"{name} must contain only zero and one")
    return np.array(raw, dtype=np.uint8, copy=True, order="C")


def _unit_id_array(value: Any, n_units: int) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or raw.size != n_units
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError("unit_ids must be a 1-D integer array aligned to units")
    if np.issubdtype(raw.dtype, np.unsignedinteger) and raw.size:
        if int(raw.max()) > np.iinfo(np.int64).max:
            raise ValueError("unit_ids values must fit in signed 64-bit integers")
    unit_ids = np.array(raw, dtype=np.int64, copy=True, order="C")
    if np.unique(unit_ids).size != n_units:
        raise ValueError("unit_ids must be unique")
    return unit_ids


def _validated_result_arrays(
    mask_2d: Any,
    center_2d: Any,
    unit_ids: Any,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.int64]]:
    mask = _binary_array(mask_2d, "mask_2d")
    center = _binary_array(center_2d, "center_2d")
    if center.shape != mask.shape:
        raise ValueError("mask_2d and center_2d must have identical shapes")

    if np.any((center != 0) & (mask == 0)):
        raise ValueError("every center_2d bin must also belong to mask_2d")

    mask_nonempty = np.any(mask != 0, axis=(1, 2))
    center_counts = np.sum(center != 0, axis=(1, 2))
    if not np.array_equal(center_counts, mask_nonempty.astype(np.int64)):
        raise ValueError(
            "center_2d must contain exactly one bin for each non-empty unit "
            "and zero bins for each empty unit"
        )

    ids = _unit_id_array(unit_ids, mask.shape[0])
    for array in (mask, center, ids):
        array.setflags(write=False)
    return mask, center, ids


def _cache_key(value: Any, name: str = "cache_key") -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


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


def load_rf_result(
    path: str | os.PathLike[str],
    *,
    expected_cache_key: str,
) -> RFResultCache | None:
    """Load a matching RF result or return ``None`` for an ordinary miss.

    A missing file and a cache-key mismatch are ordinary misses.  An existing
    but corrupt, non-canonical, or incompatible file raises
    :class:`RFResultCacheError` so scientific results are never silently
    recomputed over an unexplained bad artifact.
    """

    target = _result_archive_path(path)
    expected_key = _cache_key(expected_cache_key, "expected_cache_key")
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
            if stored_key != expected_key:
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

            mask, center, ids = _validated_result_arrays(
                archive["mask_2d"],
                archive["center_2d"],
                archive["unit_ids"],
            )
    except FileNotFoundError:
        return None
    except RFResultCacheError:
        raise
    except Exception as exc:
        raise RFResultCacheError(
            f"unable to load RF result cache {target}: {exc}"
        ) from exc

    return {
        "schema_version": RF_RESULT_CACHE_SCHEMA_VERSION,
        "mask_2d": mask,
        "center_2d": center,
        "unit_ids": ids,
        "manifest": parsed_manifest,
        "cache_key": stored_key,
    }
