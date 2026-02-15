# Mathpix API Configuration
# Get your credentials from: https://accounts.mathpix.com/

import os


def _env_or_default(name, default_value):
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    return default_value


# You need to set both of these values:
MATHPIX_APP_ID = _env_or_default("MATHPIX_APP_ID", "seth_spersonalsnips_e8fea9_996429")  # Your Mathpix App ID
MATHPIX_APP_KEY = _env_or_default("MATHPIX_APP_KEY", "fa9647d163bac68dad31a136bdd4c23027e5a52325f6eb0221dfea792e501d26")

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
    if MATHPIX_APP_ID == "your-app-id-here":
        raise Exception(
            "Please update MATHPIX_APP_ID in mathpix_config.py with your actual app ID.\n"
            "Get your credentials from: https://accounts.mathpix.com/"
        )
    
    return {
        "app_id": MATHPIX_APP_ID,
        "app_key": MATHPIX_APP_KEY,
        "url": MATHPIX_URL
    } 
