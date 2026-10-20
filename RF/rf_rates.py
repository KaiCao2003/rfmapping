"""Compatibility import for :mod:`RF.rflocate.rates`."""

import sys

from .rflocate import rates as _implementation

sys.modules[__name__] = _implementation
