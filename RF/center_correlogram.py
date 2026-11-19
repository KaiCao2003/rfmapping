"""Within-neuron RF center pairs and their raw angular histograms."""

import numpy as np
import pandas as pd


def rf_center_pairs(excitatory, inhibitory):
    """Pair loaded center rows within their complete neuronal index.

    Inputs have the named index of ``RF.profiles.rf_centers``. Include session
    levels before pooling recordings. E–E counts both directions without self
    pairs; E→I is inhibitory minus excitatory azimuth, wrapped to [-180, 180).
    Elevation does not enter the difference; distinct centers sharing x remain.
    """
    keys = list(excitatory.index.names)
    if keys != list(inhibitory.index.names) or None in keys or "unit_id" not in keys:
        raise ValueError("Center tables must share named neuronal index levels including unit_id")
    reference = excitatory.assign(center_id=np.arange(len(excitatory))).reset_index()
    inhibitory = inhibitory.assign(center_id=np.arange(len(inhibitory))).reset_index()
    pairs = {}
    for label, target in (("E–E", reference), ("E→I", inhibitory)):
        table = reference.merge(target, on=keys, suffixes=("_ref", "_target"))
        if label == "E–E":
            table = table.loc[table.center_id_ref != table.center_id_target].copy()
        table["delta_deg"] = (table.rf_x_deg_target - table.rf_x_deg_ref + 180) % 360 - 180
        pairs[label] = table.set_index(keys)
    return pairs


def rf_center_histograms(pairs, *, bins=30):
    """Bin ordered angular differences without smoothing or normalization.

    Bin centers run from 0 to <360 degrees, with a bin centered at zero.
    Unit counts use the entire neuronal index (including session if supplied).
    More centers contribute more pairs; empty populations retain zero counts.
    """
    if bins < 1:
        raise ValueError("bins must be positive")
    width = 360 / bins
    edges = np.linspace(0, 360, bins + 1) - width / 2
    counts = {}
    for label, table in pairs.items():
        angles = (table.delta_deg.to_numpy() + width / 2) % 360 - width / 2
        counts[label] = np.histogram(angles, bins=edges)[0]
    histograms = pd.DataFrame(counts, index=pd.Index((edges[:-1] + edges[1:]) / 2, name="angle_deg"))
    histograms.attrs["pair_counts"] = {label: len(table) for label, table in pairs.items()}
    histograms.attrs["unit_counts"] = {label: table.index.nunique() for label, table in pairs.items()}
    return histograms
