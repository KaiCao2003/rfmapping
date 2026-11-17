"""Compatibility import for :mod:`RF.rflocate.trials`."""

import sys

from .rflocate import trials as _implementation

sys.modules[__name__] = _implementation
