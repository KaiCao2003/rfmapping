"""Compatibility import for :mod:`RF.rflocate.workflow`."""

import sys

from .rflocate import workflow as _implementation

sys.modules[__name__] = _implementation
