"""Compatibility alias for the report generator."""
import sys
from scripts import generate_report as implementation
if __name__ == "__main__":
    raise SystemExit(implementation.main())
sys.modules[__name__] = implementation
