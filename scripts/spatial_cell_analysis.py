"""Run the shared spatial-cell analysis through the historical script path."""

import sys

from EBB import spatial_cell_analysis as analysis


if __name__ == "__main__":
    analysis.main()
else:
    sys.modules[__name__] = analysis
