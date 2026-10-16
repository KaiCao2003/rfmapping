"""Compatibility import; EBC implementation lives in EBB.ebc_video."""

import sys

from EBB import ebc_video as implementation

sys.modules[__name__] = implementation
