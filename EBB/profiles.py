"""Prepare EBB angular profiles from already loaded maps."""

from Utils.direction_comparison import profile_table, recording_profiles, resample_profiles, tcRange


def ebc_profiles(maps, *, probe="A"):
    """Collapse supplied maps over distance, retaining native angles and NaNs."""
    angles = maps[0].y_positions
    projected = maps.sum_to_1d(axis="y")
    profiles = profile_table(projected.to_1d_array(axis="y"), maps.unit_ids, angles, probe=probe)
    profiles.attrs["response_units"] = maps[0].metadata.get("responseUnits", "spike_count")
    profiles.attrs["unit_info"] = {key: {} for key in profiles.index}
    return profiles


def prepare_ebc(maps, *, mouse, date, probe="A", label="EBC", range: list[int] | None = None):
    """Resample supplied EBB maps to the comparison grid and add recording keys."""
    profiles = ebc_profiles(maps, probe=probe)
    profiles = resample_profiles(profiles, range=tcRange(True) if range is None else range)
    table = recording_profiles(profiles, mouse=mouse, date=date)
    table.attrs["label"] = label
    return table
