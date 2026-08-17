from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from plotting import plot_2d_rfmap


def test_plot2dheatmap_keeps_rows_vertical_and_columns_horizontal(monkeypatch) -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]])
    monkeypatch.setattr(plt, "show", lambda: None)

    fig, ax = plot_2d_rfmap(data)

    try:
        image = ax.images[0]
        np.testing.assert_array_equal(image.get_array(), data)
        assert image.get_array().shape == (2, 3)
        np.testing.assert_array_equal(ax.get_xticks(), [0, 1, 2])
        np.testing.assert_array_equal(ax.get_yticks(), [0, 1])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("shape", [(3,), (1, 2, 3), (0, 3), (2, 0)])
def test_plot2dheatmap_rejects_non_2d_or_empty_data(shape) -> None:
    with pytest.raises(ValueError, match="2D|empty"):
        plot_2d_rfmap(np.empty(shape))
