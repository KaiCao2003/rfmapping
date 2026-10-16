"""Compatibility import; EBC implementation lives in EBB.ebc_geometry."""

import sys

from EBB import ebc_geometry as implementation

sys.modules[__name__] = implementation
