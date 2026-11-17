"""Compatibility alias for RF result persistence in ``RF.rflocate``."""

import sys

from RF.rflocate import _cache

sys.modules[__name__] = _cache
