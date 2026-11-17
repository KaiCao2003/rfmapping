"""Compatibility import for :mod:`RF.rflocate.plotting`."""

import sys

from .rflocate import plotting as _implementation

sys.modules[__name__] = _implementation
