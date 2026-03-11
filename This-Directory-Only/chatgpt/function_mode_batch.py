#!/usr/bin/env python3

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parents[1] / "Local_Deployed_Shared"
SHARED_CHATGPT_DIR = SHARED_DIR / "chatgpt"

os.environ.setdefault("DELTA_CHATGPT_CODE_DIR", str(SHARED_CHATGPT_DIR))
os.environ.setdefault("DELTA_CHATGPT_RUNTIME_DIR", str(SCRIPT_DIR))
os.environ.setdefault("DELTA_CHATGPT_DATA_BACKEND", "supabase")
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(SHARED_CHATGPT_DIR))

runpy.run_path(str(SHARED_CHATGPT_DIR / "function_mode_batch.py"), run_name="__main__")
