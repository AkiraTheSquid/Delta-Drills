# Mathpix API Configuration
# Get your credentials from: https://accounts.mathpix.com/
# Credentials are loaded from (in order):
#   1. Environment variables MATHPIX_APP_ID / MATHPIX_APP_KEY
#   2. ~/secrets/delta-drills/keys.json

import json
import os
from pathlib import Path

_SECRETS_FILE = Path.home() / "secrets" / "delta-drills" / "keys.json"


def _load_credential(env_var: str, secrets_key: str) -> str:
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    try:
        with open(_SECRETS_FILE) as f:
            data = json.load(f)
        return (data.get(secrets_key) or "").strip()
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


MATHPIX_APP_ID = _load_credential("MATHPIX_APP_ID", "mathpix_app_id")
MATHPIX_APP_KEY = _load_credential("MATHPIX_APP_KEY", "mathpix_app_key")

# API endpoint (usually doesn't need to change)
MATHPIX_URL = "https://api.mathpix.com"

# Defaults for bulk PDF processing (can be changed in one place)
# These control the folder and numeric filename range for the pdf-bulk command
DEFAULT_BULK_PDF_DIR = r"C:\Users\prime\Documents\NEW EXPORT\New export\Autohotkey\AutoHotkey\AHK Stuffs\Mech Interp Utralearning Project\pdf_2_problem\exercise_sections"
DEFAULT_BULK_START = 34  # inclusive
DEFAULT_BULK_END = 35    # inclusive

# Defaults for Markdown → CSV bulk processing
DEFAULT_MD_INPUT_DIR = r"C:\Users\prime\Documents\NEW EXPORT\New export\Autohotkey\AutoHotkey\AHK Stuffs\Mech Interp Utralearning Project\pdf_2_problem\output"
DEFAULT_MD_CSV_OUTPUT_DIR = r"C:\Users\prime\Documents\NEW EXPORT\New export\Autohotkey\AutoHotkey\AHK Stuffs\Mech Interp Utralearning Project\pdf_2_problem\csv_output"

def get_credentials():
    """
    Returns the Mathpix credentials.
    Raises an exception if credentials are not properly configured.
    """
    if not MATHPIX_APP_ID or not MATHPIX_APP_KEY:
        raise Exception(
            f"Mathpix credentials not found.\n"
            f"Set MATHPIX_APP_ID/MATHPIX_APP_KEY env vars or add them to {_SECRETS_FILE}"
        )
    
    return {
        "app_id": MATHPIX_APP_ID,
        "app_key": MATHPIX_APP_KEY,
        "url": MATHPIX_URL
    } 
