"""Compatibility package shim so imports like `from solarModuleDefect.utils import ...` work

This project places packages at `src/<name>` (e.g., `src/utils`, `src/data`).
Some modules expect a top-level package named `solarModuleDefect`. To avoid moving files,
this shim dynamically re-exports the existing top-level packages under the
`solarModuleDefect` package namespace.
"""
from __future__ import annotations

import importlib
import sys

__all__ = ["utils", "data", "models", "scripts"]

for pkg in __all__:
    # import the existing top-level package (e.g., `utils`) and register it
    # as a submodule of `solarModuleDefect` (so `solarModuleDefect.utils` works).
    try:
        mod = importlib.import_module(pkg)
    except Exception:
        # best-effort import; if it fails we'll let the original import error surface
        continue
    sys.modules[f"solarModuleDefect.{pkg}"] = mod
    setattr(sys.modules[__name__], pkg, mod)
