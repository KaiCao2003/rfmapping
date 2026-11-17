"""Explicit RF TC generation from saved response maps and saved detections."""

from pathlib import Path

from RF.rflocate import RFMapList, load_rf, load_rfmap, load_rf_tc, rf_tc_table, save_rf_tc
from Utils.direction_comparison import save_tc
from Utils.tc_preparation import check_tc_outputs, prefix_tc


def calculate_rf_projection(maps, detected, *, time_range_s, rf_only):
    """Sum loaded counts over time and space on their native grid; perform no I/O."""
    if rf_only:
        by_id = {item.unit_id: item for item in maps}
        maps = RFMapList([by_id[int(unit_id)] for unit_id in detected.unit_ids], maps.source_path)
    window = maps.sum(*time_range_s, show_progress=False)
    projection = window.sum_to_1d(axis="x", rf_only=rf_only,
                                  detected_rf=detected if rf_only else None)
    return rf_tc_table(projection)


def calculate_rf_tc(table, *, detected_results=(), unit_prefix):
    """Select saved detected units and add recording identity; perform no I/O."""
    if detected_results:
        selected_ids = set()
        for result in detected_results:
            selected_ids.update(result.unit_ids[result.mask_2d.any(axis=(1, 2))])
        table = table.loc[table.index.isin(selected_ids)]
    return prefix_tc(table, unit_prefix)


def generate_rf_tc(source, output, *, select_results=(), unit_prefix, overwrite=False):
    """Read a saved native projection and detections, then explicitly save a TC."""
    output = Path(output)
    check_tc_outputs([output], inputs=[source, *select_results], overwrite=overwrite)
    table = load_rf_tc(source)
    results = [load_rf(path) for path in select_results]
    table = calculate_rf_tc(table, detected_results=results, unit_prefix=unit_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_tc(table, output)
    return output


def generate_rf_projection(source, output, *, detected_rf_path, time_range_s,
                           rf_only, overwrite=False):
    """Explicitly read saved inputs, calculate the native projection, and save it."""
    output = Path(output)
    check_tc_outputs([output], inputs=[source, detected_rf_path], overwrite=overwrite)
    maps = load_rfmap(source)
    detected = load_rf(detected_rf_path) if rf_only else None
    table = calculate_rf_projection(maps, detected, time_range_s=time_range_s, rf_only=rf_only)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_rf_tc(table, output)
    return output


def generate_rf_comparison(source, output, *, projection_path, detected_rf_path,
                           time_range_s, rf_only, unit_prefix, overwrite=False):
    """Generate both CSVs in one pass; read each source once and keep results in memory."""
    output, projection_path = Path(output), Path(projection_path)
    check_tc_outputs([output, projection_path], inputs=[source, detected_rf_path], overwrite=overwrite)
    maps, detected = load_rfmap(source), load_rf(detected_rf_path)
    projection = calculate_rf_projection(maps, detected, time_range_s=time_range_s, rf_only=rf_only)
    table = calculate_rf_tc(projection, detected_results=[detected], unit_prefix=unit_prefix)
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_rf_tc(projection, projection_path)
    save_tc(table, output)
    return output
