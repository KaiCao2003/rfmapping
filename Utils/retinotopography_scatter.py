"""Prepare ADN angle/depth points separately from scatter rendering."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from Utils.plotting import LIGHT_PLOT_STYLE


IDENTITY = ["mouse", "date", "probe", "unit_id"]
SHANK = ["mouse", "date", "probe", "shank"]


def select_adn_depth(units: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """Select annotated units, then shift each shank's uppermost HD cell to 0 µm.

    No distance scaling is applied. RF centers later inherit this same origin,
    including shanks whose highest HD cell has no detected RF.
    """
    selected = units.merge(selection[IDENTITY], on=IDENTITY, how="inner",
                           sort=False, validate="one_to_one")
    selected = selected.loc[
        np.isfinite(selected.y_um) & np.isfinite(selected.hd_preferred_deg)
    ].copy()
    selected["topmost_hd_y_um"] = selected.groupby(SHANK).y_um.transform("max")
    selected["depth_um"] = selected.topmost_hd_y_um - selected.y_um
    return selected


def normalize_shank_depth(units: pd.DataFrame) -> pd.DataFrame:
    """Map each selected HD shank's physical span to [0, 1], before RF selection.

    A shank with zero depth span remains at 0; it has no distinct bottom.
    """
    normalized = units.copy()
    span = normalized.groupby(SHANK).depth_um.transform("max").to_numpy()
    normalized["depth_fraction"] = np.divide(
        normalized.depth_um.to_numpy(), span, out=np.zeros(len(normalized)), where=span > 0,
    )
    return normalized


def prepare_depth_angles(
    units: pd.DataFrame, centers: pd.DataFrame, *, kind: Literal["hd", "rf"],
    depth_column: Literal["depth_um", "depth_fraction"] = "depth_um",
) -> pd.DataFrame:
    """Return one row per HD cell or per RF center, retaining neuronal identity."""
    columns = IDENTITY + ["shank", depth_column]
    if kind == "hd":
        points = units[columns + ["hd_preferred_deg"]].rename(
            columns={"hd_preferred_deg": "angle_deg"})
        points["angle_deg"] = points.angle_deg % 360
    elif kind == "rf":
        points = units[columns].merge(
            centers[IDENTITY + ["rf_x_deg", "rf_y_deg"]],
            on=IDENTITY, how="inner", sort=False, validate="one_to_many",
        ).rename(columns={"rf_x_deg": "angle_deg"})
        # Preserve separate centers even when their horizontal angles coincide.
        points["angle_deg"] = (points.angle_deg + 180) % 360 - 180
    else:
        raise ValueError("kind must be 'hd' or 'rf'")
    return points


def prepare_depth_panels(
    points: pd.DataFrame, mice: Sequence[str], *, missing_mice: Sequence[str] = (),
) -> list[pd.DataFrame]:
    """Select each mouse and pooled panel without binning or changing coordinates."""
    if not set(points.mouse).issubset(mice):
        raise ValueError("points contain mice outside the requested panels")
    if set(points.mouse) & set(missing_mice):
        raise ValueError("a mouse with missing ADN selection cannot contribute points")
    panels = []
    for label in [*mice, "All"]:
        group = points.copy() if label == "All" else points.loc[points.mouse.eq(label)].copy()
        group.attrs.update(
            label=label, n_points=len(group), n_units=len(group[IDENTITY].drop_duplicates()),
            missing_selection=label in missing_mice,
            missing_mice=list(missing_mice) if label == "All" else [],
        )
        panels.append(group)
    return panels


@plt.rc_context(LIGHT_PLOT_STYLE)
def plot_depth_scatter(
    panels: Sequence[pd.DataFrame], *, xlabel: str, title: str,
    angle_range: tuple[float, float], depth_max: float,
    depth_column: Literal["depth_um", "depth_fraction"] = "depth_um",
    figsize: tuple[float, float] = (15, 9),
) -> tuple[Figure, np.ndarray]:
    """Draw supplied angle/depth points without binning, fitting, or saving."""
    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False, sharex=True, sharey=True,
        layout="constrained", facecolor="white",
    )
    for ax, points in zip(axes.flat, panels):
        info = points.attrs
        ax.scatter(points.angle_deg, points[depth_column], s=24, color="#2468a2",
                   alpha=0.75, linewidths=0, clip_on=False)
        ax.set(
            xlabel=xlabel, ylabel=("Normalized depth within shank" if depth_column == "depth_fraction"
                                    else "Depth below shank top (µm)"),
            xlim=angle_range, ylim=(depth_max, 0),
            xticks=np.linspace(*angle_range, 5),
        )
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        ax.grid(alpha=0.35, linewidth=0.6)
        if info["missing_selection"]:
            ax.set_title(info["label"])
            ax.text(0.5, 0.5, "ADN selection needed", transform=ax.transAxes,
                    ha="center", va="center", color="black")
        else:
            label = f"{info['label']} · {info['n_units']} cells · {info['n_points']} points"
            if info["missing_mice"]:
                label += "\nExcludes " + ", ".join(info["missing_mice"])
            ax.set_title(label)
    for ax in axes.flat[len(panels):]:
        ax.remove()
    figure.suptitle(title)
    return figure, axes.ravel()[:len(panels)]
