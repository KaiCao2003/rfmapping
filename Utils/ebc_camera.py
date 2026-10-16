"""Compatibility import; EBC implementation lives in EBB.ebc_camera."""

import sys

from EBB import ebc_camera as implementation

sys.modules[__name__] = implementation
