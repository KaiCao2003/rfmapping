"""Compatibility import; EBC implementation lives in EBB.ebc_pose."""

import sys

from EBB import ebc_pose as implementation

sys.modules[__name__] = implementation
