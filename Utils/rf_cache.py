"""Compatibility alias for RF result persistence in ``Utils.rflocate``."""

import sys

from Utils.rflocate import _cache

sys.modules[__name__] = _cache
