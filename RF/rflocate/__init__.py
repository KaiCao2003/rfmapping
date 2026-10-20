"""RF response data, saved results, and explicit detection.

Load values with ``load_rfmap`` and saved detections with ``load_rf``.
Transform RFMap objects explicitly; only ``detect_rf`` runs detection.
Plotting and complete analysis workflows live in their named submodules.
"""

from .models import RFMap, RFMapList, asrfmap
from .results import RFResult
from .io import load_rfmap, load_rf, save_rf, rf_result_path, rf_tc_table, save_rf_tc, load_rf_tc
from .detection import detect_rf
from .fitting import fit_rf_gaussians

__all__ = [
    "RFMap", "RFMapList", "RFResult", "asrfmap",
    "load_rfmap", "load_rf", "save_rf", "rf_result_path", "detect_rf",
    "rf_tc_table", "save_rf_tc", "load_rf_tc", "fit_rf_gaussians",
]
