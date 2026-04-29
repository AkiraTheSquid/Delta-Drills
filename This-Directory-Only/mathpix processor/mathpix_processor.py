import os
import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parents[1] / "Local_Deployed_Shared"
SHARED_SCRIPT = SHARED_DIR / "tools" / "mathpix_processor.py"


def main() -> None:
    os.environ.setdefault("DELTA_MATHPIX_CODE_DIR", str(SCRIPT_DIR))
    sys.path.insert(0, str(SHARED_DIR / "tools"))
    sys.path.insert(0, str(SHARED_DIR))
    runpy.run_path(str(SHARED_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
