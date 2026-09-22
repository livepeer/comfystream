"""Load comfystream submodules without ComfyUI when the stream stack is absent.

Existing stream tests (test_utils.py) still require ComfyUI. Catalog, job-pool,
and batch HTTP tests only need capabilities / BatchPipeline.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "comfystream"

try:
    import comfy_compatibility  # noqa: F401
except ImportError:
    if "comfystream" not in sys.modules:
        pkg = types.ModuleType("comfystream")
        pkg.__path__ = [str(SRC)]
        pkg.__file__ = str(SRC / "__init__.py")
        pkg.__package__ = "comfystream"
        sys.modules["comfystream"] = pkg
