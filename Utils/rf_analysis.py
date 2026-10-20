"""Compatibility import for :mod:`Utils.rflocate.workflow`."""

import sys

from .rflocate import workflow as _implementation

sys.modules[__name__] = _implementation
