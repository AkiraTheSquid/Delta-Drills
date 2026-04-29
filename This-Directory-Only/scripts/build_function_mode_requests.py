#!/usr/bin/env python3

from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parents[1] / "Local_Deployed_Shared"

sys.path.insert(0, str(SHARED_DIR / "pipeline"))
sys.path.insert(0, str(SHARED_DIR))

runpy.run_path(str(SHARED_DIR / "pipeline" / "build_function_mode_requests.py"), run_name="__main__")
