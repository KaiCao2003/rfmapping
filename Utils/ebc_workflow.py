"""Compatibility import; EBC implementation lives in EBB.ebc_workflow."""

import sys

from EBB import ebc_workflow as implementation

sys.modules[__name__] = implementation
