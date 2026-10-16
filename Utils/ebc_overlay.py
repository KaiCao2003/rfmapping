"""Compatibility import; EBC implementation lives in EBB.ebc_overlay."""

import sys

from EBB import ebc_overlay as implementation

sys.modules[__name__] = implementation
