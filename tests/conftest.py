"""Shared test configuration.

The project keeps its modules in ``code/``, ``code/utils/`` and ``data/`` rather
than in an installable package, so make those directories importable for the
test suite.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _sub in ("code", os.path.join("code", "utils"), "data"):
    _path = os.path.join(ROOT, _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)
