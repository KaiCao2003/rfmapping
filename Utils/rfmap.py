"""Compatibility imports for the RF package; prefer ``Utils.rflocate``."""
from .rflocate.models import RFMap, RFMapList, asrfmap
from .rflocate.io import load_rf_maps

__all__ = ["RFMap", "RFMapList", "asrfmap", "load_rf_maps", "plot_2d_rfmap", "plot_1d_rfmap"]


def __getattr__(name):
    # Keep optional plotting imports out of data-only callers.
    if name in {"plot_2d_rfmap", "plot_1d_rfmap"}:
        from .rflocate import plotting
        return getattr(plotting, name)
    raise AttributeError(name)
