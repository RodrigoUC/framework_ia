"""Compatibility wrapper for prepare_delivery.py."""
from __future__ import annotations

from scripts.prepare_delivery import *

if __name__ == "__main__":
    import sys
    from scripts.prepare_delivery import main
    raise SystemExit(main(sys.argv[1:]))
