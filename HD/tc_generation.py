"""Explicit comparison TC generation from a saved HD tuning curve."""

from pathlib import Path

from HD.profiles import hd_profiles, load_hd_tc
from Utils.direction_comparison import save_tc
from Utils.tc_preparation import check_tc_outputs, prefix_tc


def calculate_hd_tc(data, *, bins, hd_class=None, unit_prefix):
    """Rebin loaded counts and occupancy, retaining saved classes; perform no I/O."""
    table = hd_profiles(data, probe=None, bins=bins, smoothing_deg=0, hd_class=hd_class)
    return prefix_tc(table, unit_prefix)


def generate_hd_tc(source, output, *, bins, hd_class=None, unit_prefix, overwrite=False):
    """Read a saved HD TC, calculate its comparison table, and explicitly save it."""
    output = Path(output)
    check_tc_outputs([output], inputs=[source], overwrite=overwrite)
    table = calculate_hd_tc(load_hd_tc(source), bins=bins, hd_class=hd_class, unit_prefix=unit_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_tc(table, output)
    return output
