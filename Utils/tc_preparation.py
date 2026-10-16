"""Shared unit identity and output checks for explicit TC generation."""

from pathlib import Path


def prefix_tc(table, unit_prefix):
    """Add the supplied recording identity without changing response values."""
    table = table.copy()
    table.index = table.index.get_level_values("unit_id").map(
        lambda unit_id: f"{unit_prefix}:{unit_id}")
    return table


def check_tc_outputs(outputs, *, inputs, overwrite):
    """Protect source files and require an explicit choice to replace saved TCs."""
    source_paths = {Path(path).resolve() for path in inputs}
    output_paths = [Path(path).resolve() for path in outputs]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("output must differ from other output paths")
    for path in map(Path, outputs):
        if path.resolve() in source_paths:
            raise ValueError("output must differ from every input path")
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists; use overwrite=True to regenerate")
