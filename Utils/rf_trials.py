"""Compatibility import for :mod:`Utils.rflocate.trials`."""

import sys

from .rflocate import trials as _implementation

sys.modules[__name__] = _implementation
