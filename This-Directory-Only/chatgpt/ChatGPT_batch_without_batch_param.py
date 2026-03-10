#!/usr/bin/env python3

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parents[1] / "Local_Deployed_Shared"

os.environ.setdefault("DELTA_CHATGPT_CODE_DIR", str(SCRIPT_DIR))
sys.path.insert(0, str(SHARED_DIR))

runpy.run_path(str(SHARED_DIR / "ChatGPT_batch_without_batch_param.py"), run_name="__main__")
