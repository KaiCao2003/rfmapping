"""Compatibility import for :mod:`Utils.rflocate._detector`."""

import sys

from .rflocate import _detector as _implementation

sys.modules[__name__] = _implementation
