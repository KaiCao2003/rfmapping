"""Stage existing RF CSV derivatives without overwriting recordings or 1-D-only exports."""

import csv
from pathlib import Path
import re

import numpy as np

from RF.rflocate import RFMapList, load_rf, rf_result_path, rf_tc_table, save_rf_tc
from RF.tc_generation import calculate_rf_tc
from Utils.direction_comparison import save_tc


def stage_existing_csvs(item, counts, result, polarity, stage):
    """Refresh only existing CSVs whose values or selection depend on this 2-D RF.

    ``1d_rfonly`` retains the saved independent 1-D unit selection, but its
    response values sum rows chosen by the new 2-D mask. Plain ``1d`` and ``all``
    exports have no 2-D dependency and remain untouched.
    """
    source = Path(item["path"])
    probe = item["probe"].removeprefix("Probe")
    data_dir = Path(item["session_path"]) / "data"
    suffix = "_inhibitory" if polarity == "inhibitory" else ""
    native = re.compile(
        re.escape(f"{source.stem}_Probe{probe}{suffix}_1d") + r"(?:_([xy]))?_rfonly\.csv"
    )
    selection_pattern = rf"([xy])_(2d|1d|either|all)(_rfonly)?_Probe{re.escape(probe)}\.csv"
    qualified = re.compile(re.escape(f"{source.stem}_{polarity}_") + selection_pattern)
    alias = re.compile(re.escape(f"rf_{polarity}_") + selection_pattern)
    canonical_on = (
        item["root_kind"] == "rfmapping"
        and source.stem == f"regular_unitsSpikeCounts_{Path(item['session_path']).name}"
    )
    jobs = []
    existing = {Path(record["path"]) for record in item["session_rf_csvs_shared_or_source_specific"]}
    for target in sorted(existing):
        match = native.fullmatch(target.name)
        if match:
            jobs.append((target, match[1] or "x", "all", True))
            continue
        match = qualified.fullmatch(target.name)
        if match is None and canonical_on:
            match = alias.fullmatch(target.name)
        if match:
            axis, selection, rf_only = match[1], match[2], bool(match[3])
            if rf_only or selection in ("2d", "either"):
                jobs.append((target, axis, selection, rf_only))
    if not jobs:
        return []

    saved_1d = None
    if any(selection in ("1d", "either") for _, _, selection, _ in jobs):
        saved_1d = load_rf(rf_result_path(source, dimension="1d", rf_type=polarity))
    window = counts.sum(0., .2, show_progress=False)
    by_id = {m.unit_id: m for m in window}
    selected_window = RFMapList([by_id[int(uid)] for uid in result.unit_ids], source)
    projections = {}
    prefix = f"{item['mouse']}:{item['date']}:{probe}"
    replacements = []
    for target, axis, selection, rf_only in jobs:
        key = (axis, rf_only)
        if key not in projections:
            maps = selected_window if rf_only else window
            projections[key] = rf_tc_table(
                maps.sum_to_1d(axis, rf_only=rf_only, detected_rf=result if rf_only else None),
                axis=axis,
            )
        table = projections[key]
        selected_results = {
            "all": (), "2d": (result,), "1d": (saved_1d,), "either": (result, saved_1d),
        }[selection]
        if selected_results:
            ids = set()
            for detected in selected_results:
                ids.update(detected.unit_ids[detected.mask_2d.any(axis=(1, 2))].tolist())
            table = table.loc[table.index.isin(ids)]

        with target.open(newline="") as stream:
            rows = csv.reader(stream)
            header, first = next(rows), next(rows, None)
        if not np.array_equal(np.asarray(header[1:], dtype=float), table.columns.to_numpy(dtype=float)):
            raise ValueError(f"Existing RF CSV does not match the native {axis} grid: {target}")
        prefixed = ":" in first[0] if first else target.parent.name == "tc_comparison"
        if prefixed and first and first[0].rsplit(":", 1)[0] != prefix:
            raise ValueError(f"Existing RF CSV has a different recording identity: {target}")
        staged = Path(stage) / "existing_csvs" / polarity / target.relative_to(data_dir)
        staged.parent.mkdir(parents=True, exist_ok=True)
        if prefixed:
            save_tc(calculate_rf_tc(table, unit_prefix=prefix), staged)
        else:
            save_rf_tc(table, staged)
        replacements.append({"staged": str(staged), "target": str(target)})
    return replacements
