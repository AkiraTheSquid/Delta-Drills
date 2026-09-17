import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_CONFIG = SCRIPT_DIR.parents[1] / "Local_Deployed_Shared" / "mathpix_config.py"


spec = importlib.util.spec_from_file_location("delta_shared_mathpix_config", SHARED_CONFIG)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)

MATHPIX_APP_ID = module.MATHPIX_APP_ID
MATHPIX_APP_KEY = module.MATHPIX_APP_KEY
MATHPIX_URL = module.MATHPIX_URL
DEFAULT_BULK_PDF_DIR = module.DEFAULT_BULK_PDF_DIR
DEFAULT_BULK_START = module.DEFAULT_BULK_START
DEFAULT_BULK_END = module.DEFAULT_BULK_END
DEFAULT_MD_INPUT_DIR = module.DEFAULT_MD_INPUT_DIR
DEFAULT_MD_CSV_OUTPUT_DIR = module.DEFAULT_MD_CSV_OUTPUT_DIR
get_credentials = module.get_credentials
