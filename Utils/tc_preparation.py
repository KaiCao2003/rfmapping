"""Prepare comparison CSVs once; existing outputs are read without regeneration."""

from pathlib import Path

from Utils.direction_comparison import hd_pick, load_hd_profiles, save_tc
from Utils.rflocate import RFMapList, load_rf, load_rfmap, load_rf_tc, save_rf_tc


def _save_prefixed_tc(table, output, unit_prefix):
    table = table.copy()
    table.index = table.index.get_level_values("unit_id").map(
        lambda unit_id: f"{unit_prefix}:{unit_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_tc(table, output)
    return output


def prepare_hd_tc(source, output, *, bins, hd_class=None, unit_prefix, overwrite=False):
    """Save unsmoothed HD rates with explicit bins, class, and shared unit identity."""
    output = Path(output)
    if output.exists() and not overwrite:
        return output
    if Path(source).resolve() == output.resolve():
        raise ValueError("output must differ from the input path")
    table = load_hd_profiles(source, probe=None, bins=bins, smoothing_deg=0)
    if hd_class is not None:
        table = hd_pick(table, hd_class)
    return _save_prefixed_tc(table, output, unit_prefix)


def prepare_rf_tc(source, output, *, select_results=(), unit_prefix, overwrite=False):
    """Copy a native RF projection, optionally selecting saved detected unit IDs."""
    output = Path(output)
    if output.exists() and not overwrite:
        return output
    if output.resolve() in {Path(path).resolve() for path in (source, *select_results)}:
        raise ValueError("output must differ from every input path")
    table = load_rf_tc(source)
    if select_results:
        selected_ids = set()
        for path in select_results:
            result = load_rf(path)
            selected_ids.update(result.unit_ids[result.mask_2d.any(axis=(1, 2))])
        table = table.loc[table.index.isin(selected_ids)]
    return _save_prefixed_tc(table, output, unit_prefix)


def prepare_rf_projection(source, output, *, detected_rf_path, time_range_s,
                          rf_only, overwrite=False):
    """Save native x-bin raw counts; RF-only includes whole rows touched by the RF."""
    output = Path(output)
    if output.exists() and not overwrite:
        return output
    if output.resolve() in {Path(source).resolve(), Path(detected_rf_path).resolve()}:
        raise ValueError("output must differ from every input path")
    raw = load_rfmap(source)
    detected = load_rf(detected_rf_path) if rf_only else None
    if rf_only:
        raw = RFMapList([raw.by_unit_id(int(unit_id)) for unit_id in detected.unit_ids],
                        raw.source_path)
    window = raw.sum(*time_range_s, show_progress=False)
    projection = window.sum_to_1d(axis="x", rf_only=rf_only, detected_rf=detected)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_rf_tc(projection, output)
    return output


def prepare_rf_comparison(source, output, *, projection_path, detected_rf_path,
                          time_range_s, rf_only, unit_prefix, overwrite=False):
    """Prepare one requested RF comparison CSV without recomputing detection."""
    output = Path(output)
    if output.exists() and not overwrite:
        return output
    if output.resolve() in {Path(path).resolve() for path in (source, projection_path, detected_rf_path)}:
        raise ValueError("output must differ from every input path")
    prepare_rf_projection(source, projection_path, detected_rf_path=detected_rf_path,
                          time_range_s=time_range_s, rf_only=rf_only, overwrite=overwrite)
    return prepare_rf_tc(projection_path, output, select_results=(detected_rf_path,),
                         unit_prefix=unit_prefix, overwrite=overwrite)
