import matplotlib.pyplot as plt
import numpy as np

is_norm: bool = False


def plot_2d_rfmap(data: np.ndarray, *, cmap: str = "viridis", is_save: bool = False, save_path: str = "rfmap.png"):
    """Plot a 2D array as a heatmap without changing its orientation.

    The first array dimension is shown vertically (rows), and the second
    dimension is shown horizontally (columns).
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be a 2D array.")
    if 0 in data.shape:
        raise ValueError("data must not have an empty dimension.")

    n_rows, n_columns = data.shape
    fig, ax = plt.subplots()
    image = ax.imshow(
        data,
        aspect="equal",
        cmap=cmap,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(n_columns))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    if is_save:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax


def plot_1d_rfmap(unitsSpikeCounts: np.ndarray, label_list, *, isNormalize: bool = False, isLineplot: bool = False,
                  isHeatmap: bool = False, offset: float = 1.0, xinDeg: bool = False):
    # Validation
    if isLineplot == isHeatmap:
        raise ValueError("Exactly one of isLineplot or isHeatmap must be True.")

    n_units, n_x = unitsSpikeCounts.shape
    x_values = np.linspace(0, 360, n_x, endpoint=False) if xinDeg else np.arange(n_x)
    x_label = "Angle (deg)" if xinDeg else "x"

    if isNormalize:
        max_per_unit = unitsSpikeCounts.max(axis=1, keepdims=True)
        unitsSpikeCounts = np.divide(
            unitsSpikeCounts,
            max_per_unit,
            out=np.zeros_like(unitsSpikeCounts, dtype=float),
            where=max_per_unit != 0,
        )

    fig, ax = plt.subplots(figsize=(10, 8))

    if isLineplot:
        yticks_height = []

        for unit_idx, spikeCounts in enumerate(unitsSpikeCounts):
            y = spikeCounts + (n_units - 1 - unit_idx) * offset
            yticks_height.append(np.average(y))
            ax.plot(x_values, y, linewidth=1)

        ax.set_yticks(yticks_height)
        ax.set_yticklabels(label_list)

    if isHeatmap:
        imshow_kwargs = dict(
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )
        if xinDeg:
            imshow_kwargs["extent"] = [0, 360, n_units - 0.5, -0.5]

        im = ax.imshow(unitsSpikeCounts, **imshow_kwargs)

        ax.set_yticks(np.arange(n_units))
        ax.set_yticklabels(label_list)
        fig.colorbar(im, ax=ax, label="Normalized spikes" if isNormalize else "Spikes")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Unit ID")
    if xinDeg:
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 60))

    plt.tight_layout()
    plt.show()
