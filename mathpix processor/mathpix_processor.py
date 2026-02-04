r"""
Example (PowerShell, PDF → Markdown):

  py -m pip install mpxpy; py 'pdf_2_problem\mathpix_processor.py' pdf 'pdf_2_problem\exercise_sections\1. Exercises 1A - 𝐑𝑛 and 𝐂𝑛.pdf' --out 'pdf_2_problem\output' --timeout 600

Bulk (explicit folder and inclusive range):

  py 'pdf_2_problem\mathpix_processor.py' pdf-bulk --dir 'pdf_2_problem\exercise_sections' --start 2 --end 5 --out 'pdf_2_problem\output' --timeout 600
"""

import os
import sys
import csv
import argparse
import logging
import re
import time
import threading
from pathlib import Path
import subprocess

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=30):
    """Run a function with a timeout (Windows compatible)"""
    if kwargs is None:
        kwargs = {}
    
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]

 


def extract_pdf_with_mathpix(pdf_path, output_dir, logger, timeout_seconds=300):
    """
    Extract text from a PDF using Mathpix PDF API and save outputs.
    Produces Markdown files in output_dir.
    """
    try:
        from mpxpy.mathpix_client import MathpixClient
        from mathpix_config import get_credentials
    except ImportError:
        logger.error("mpxpy package not found. Please install with: py -m pip install mpxpy")
        print("ERROR: mpxpy package not found. Please install with: py -m pip install mpxpy")
        return False

    try:
        credentials = get_credentials()
        client = MathpixClient(app_id=credentials["app_id"], app_key=credentials["app_key"]) 
        logger.info("Mathpix client initialized for PDF processing")
    except Exception as e:
        logger.error(f"Failed to initialize Mathpix client: {e}")
        print(f"ERROR: Failed to initialize Mathpix client: {e}")
        return False

    pdf_path = str(pdf_path)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Submitting PDF to Mathpix: {pdf_path}")

    try:
        # Submit PDF job (Markdown only)
        pdf_job = client.pdf_new(file_path=pdf_path, convert_to_md=True, convert_to_docx=False)
        logger.info(f"PDF job created. Waiting up to {timeout_seconds}s for completion...")

        # Wait for completion with overall timeout
        def wait_job():
            pdf_job.wait_until_complete(timeout=timeout_seconds)
            return True

        run_with_timeout(wait_job, timeout_seconds=timeout_seconds)

        # Save outputs (Markdown only)
        md_path = os.path.join(output_dir, Path(pdf_path).stem + ".md")

        try:
            saved_md = pdf_job.to_md_file(path=md_path)
            logger.info(f"Saved Markdown to: {saved_md}")
            print(f"Markdown saved: {saved_md}")
        except Exception as e:
            logger.error(f"Failed to save Markdown: {e}")

        return True

    except TimeoutError as e:
        logger.error(f"TIMEOUT waiting for PDF processing: {e}")
        print("TIMEOUT waiting for PDF processing")
        return False
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        print(f"ERROR: PDF processing failed: {e}")
        return False

def setup_logging(log_file_path):
    """Set up logging to help with debugging"""
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_leading_number(filename):
    """Extract leading integer before first non-digit/period/space in filename stem.
    Accepts patterns like '1. Title.pdf', '02 Title.pdf', '10. Something.pdf'.
    Returns int or None if not found.
    """
    # Work on the stem (without extension)
    stem = Path(filename).stem
    # Normalize spaces
    stem = stem.strip()
    # Pattern: start of string, optional spaces, one or more digits, optional dot, then space or end
    match = re.match(r"^\s*(\d+)(?:\s*\.)?(?:\s+|$)", stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None

def collect_pdfs_in_range(folder_path, start_num, end_num, logger):
    """Scan folder for .pdf whose filename starts with a number in [start_num, end_num].
    Returns list of absolute file paths sorted by that number, preserving original title.
    """
    if not os.path.isdir(folder_path):
        logger.error(f"Folder not found: {folder_path}")
        return []
    candidates = []
    for name in os.listdir(folder_path):
        if not name.lower().endswith('.pdf'):
            continue
        num = parse_leading_number(name)
        if num is None:
            continue
        if start_num <= num <= end_num:
            candidates.append((num, os.path.join(folder_path, name)))
    candidates.sort(key=lambda x: x[0])
    logger.info(f"Selected {len(candidates)} PDFs in range [{start_num}, {end_num}] from {folder_path}")
    return [path for _, path in candidates]

def clean_question_id_ocr(ocr_text):
    """
    Clean question ID OCR text to extract only numeric values
    Examples: 
    - "\\( 13574 \\)" -> "13574"
    - "Question 12345" -> "12345"
    - "ID: 98765" -> "98765"
    """
    if not ocr_text or not str(ocr_text).strip():
        return "0"
    
    # Extract all numbers from the text
    numbers = re.findall(r'\d+', str(ocr_text))
    
    if numbers:
        # Take the longest number sequence (likely the question ID)
        longest_number = max(numbers, key=len)
        return longest_number
    else:
        # If no numbers found, return 0
        return "0"

def process_images_with_ocr(folder_path, block_number, category, image_type, logger):
    """
    Process images (answers or IDs) with Mathpix OCR and return text results
    """
    try:
        # Try to import Mathpix client and config
        from mpxpy.mathpix_client import MathpixClient
        from mathpix_config import get_credentials
        
        # Get credentials from config file
        credentials = get_credentials()
        
        # Initialize Mathpix client with credentials
        client = MathpixClient(
            app_id=credentials["app_id"],
            app_key=credentials["app_key"]
        )
        logger.info("Mathpix client initialized successfully")
        
    except ImportError:
        logger.error("mpxpy package not found. Please install with: py -m pip install mpxpy")
        print("ERROR: mpxpy package not found. Please install with: py -m pip install mpxpy")
        return []
    except Exception as e:
        logger.error(f"Failed to initialize Mathpix client: {str(e)}")
        print(f"ERROR: Failed to initialize Mathpix client: {str(e)}")
        return []
    
    # Find images in the folder (answers or ids)
    images_folder = os.path.join(folder_path, f"{block_number}_{category}_{image_type}")
    
    if not os.path.exists(images_folder):
        logger.error(f"{image_type.title()} folder not found: {images_folder}")
        print(f"ERROR: {image_type.title()} folder not found: {images_folder}")
        return []
    
    # Process each image
    ocr_results = []
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]
    image_files.sort()  # Ensure consistent ordering
    
    logger.info(f"Found {len(image_files)} {image_type} files to process")
    
    if len(image_files) == 0:
        logger.warning(f"No {image_type} files found to process")
        return []
    
    for idx, image_file in enumerate(image_files):
        print(f"Processing {image_type} {idx+1}/{len(image_files)}: {image_file}")
        logger.info(f"Processing {image_type} {idx+1}/{len(image_files)}: {image_file}")
        image_path = os.path.join(images_folder, image_file)
        
        try:
            logger.info(f"Processing {image_file}")
            
            # Process image with Mathpix with timeout protection (Windows compatible)
            def process_mathpix_image():
                return client.image_new(file_path=image_path)
            
            try:
                logger.info(f"Starting Mathpix API call for {image_file} (30s timeout)...")
                image = run_with_timeout(process_mathpix_image, timeout_seconds=30)
                logger.info(f"Image object type: {type(image)}")
            except TimeoutError as e:
                logger.error(f"TIMEOUT: Mathpix API call timed out for {image_file}: {e}")
                print(f"TIMEOUT: Mathpix API call timed out for {image_file}")
                if image_type == "ids":
                    ocr_results.append("0")
                else:
                    ocr_results.append(f"TIMEOUT: API call timed out for {image_file}")
                continue
            except Exception as e:
                logger.error(f"API ERROR for {image_file}: {e}")
                print(f"API ERROR for {image_file}: {e}")
                if image_type == "ids":
                    ocr_results.append("0")
                else:
                    ocr_results.append(f"API ERROR: {e}")
                continue
            
            # Debug: Show available attributes (but don't log all values to avoid hanging)
            attrs = [attr for attr in dir(image) if not attr.startswith('_')]
            logger.info(f"Available image attributes: {attrs}")
            
            # Extract text from the result dictionary
            text_result = None
            
            if hasattr(image, 'result') and isinstance(image.result, dict):
                result_dict = image.result
                # Try different text fields in the result dictionary
                if 'text' in result_dict:
                    text_result = str(result_dict['text'])
                    logger.info(f"SUCCESS: Used result['text']: {text_result[:50]}...")
                elif 'latex' in result_dict:
                    text_result = str(result_dict['latex'])
                    logger.info(f"SUCCESS: Used result['latex']: {text_result[:50]}...")
                else:
                    logger.info(f"Result dictionary keys: {list(result_dict.keys())}")
                    text_result = f"No text found in result. Keys: {list(result_dict.keys())}"
            else:
                logger.error("FAILED: No result dictionary found")
                text_result = f"Could not extract text. Available attributes: {attrs}"
            
            # Clean up the text (remove extra whitespace, newlines)
            cleaned_text = ' '.join(text_result.split())
            
            # Apply additional cleaning for question IDs (extract only numbers)
            if image_type == "ids":
                original_text = cleaned_text
                cleaned_text = clean_question_id_ocr(cleaned_text)
                logger.info(f"Question ID cleaned: '{original_text}' -> '{cleaned_text}'")
                print(f"Cleaned question ID: '{original_text}' -> '{cleaned_text}'")
            
            ocr_results.append(cleaned_text)
            logger.info(f"SUCCESS: Processed {image_file} - {len(cleaned_text)} characters")
            print(f"SUCCESS: Processed {image_file}")
            
        except Exception as e:
            error_msg = f"ERROR processing {image_file}: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
            
            # For question IDs, use default value "0" instead of error message
            if image_type == "ids":
                ocr_results.append("0")
                logger.info(f"Question ID error, using default: 0")
            else:
                ocr_results.append(error_msg)
    
    return ocr_results

def update_csv_with_ocr(csv_path, question_id_results, answer_results, logger):
    """
    Update the existing CSV file with both question ID OCR and answer OCR results
    """
    try:
        # Read existing CSV content
        rows = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)
        
        logger.info(f"Read {len(rows)} rows from CSV")
        if len(rows) == 0:
            logger.error("CSV file is empty")
            return False
            
        # Log current structure
        current_cols = len(rows[0])
        logger.info(f"Current CSV has {current_cols} columns: {rows[0]}")
        
        # Simple, robust column handling
        # Check if we already have the OCR columns
        has_question_id_ocr = 'question_id_ocr' in rows[0]
        has_answer_ocr = 'answer_ocr' in rows[0]
        
        # Add missing OCR columns to header
        if not has_question_id_ocr:
            rows[0].append('question_id_ocr')
            logger.info("Added question_id_ocr column header")
            
        if not has_answer_ocr:
            rows[0].append('answer_ocr')
            logger.info("Added answer_ocr column header")
        
        # Find column indices after adding headers
        question_id_col = rows[0].index('question_id_ocr') if 'question_id_ocr' in rows[0] else -1
        answer_col = rows[0].index('answer_ocr') if 'answer_ocr' in rows[0] else -1
        
        logger.info(f"Column indices - question_id_ocr: {question_id_col}, answer_ocr: {answer_col}")
        
        # Process each data row
        num_results = max(len(question_id_results), len(answer_results))
        logger.info(f"Processing {num_results} OCR results")
        
        for i in range(num_results):
            row_index = i + 1  # Skip header row
            
            # Ensure we have enough data rows
            while len(rows) <= row_index:
                # Create a new empty row with same number of columns as header
                new_row = [""] * len(rows[0])
                rows.append(new_row)
                logger.info(f"Created new row {len(rows)-1}")
            
            # Ensure current row has enough columns
            while len(rows[row_index]) < len(rows[0]):
                rows[row_index].append("")
            
            # Add OCR results
            if i < len(question_id_results) and question_id_col >= 0:
                rows[row_index][question_id_col] = question_id_results[i]
                logger.info(f"Added question_id_ocr for row {row_index}: {question_id_results[i]}")
                
            if i < len(answer_results) and answer_col >= 0:
                rows[row_index][answer_col] = answer_results[i]
                logger.info(f"Added answer_ocr for row {row_index}: truncated({len(answer_results[i])} chars)")
        
        # Write updated CSV back to file
        with open(csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
        
        logger.info(f"Successfully updated CSV with {num_results} OCR results")
        return True
        
    except Exception as e:
        logger.error(f"Error updating CSV: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
        
        logger.info(f"SUCCESS: Updated CSV with OCR results: {csv_path}")
        print(f"SUCCESS: Updated CSV with OCR results: {csv_path}")
        return True
        
    except Exception as e:
        logger.error(f"ERROR updating CSV: {str(e)}")
        print(f"ERROR updating CSV: {str(e)}")
        return False

def main():
    print("=== PYTHON SCRIPT STARTED ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Arguments: {sys.argv}")
    
    parser = argparse.ArgumentParser(description='Mathpix OCR utilities')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Images → CSV workflow (existing)
    img_parser = subparsers.add_parser('images', help='Process answer/id images and update CSV')
    img_parser.add_argument('folder_path', help='Path to the block folder')
    img_parser.add_argument('block_number', help='Block number')
    img_parser.add_argument('category', help='Category name')
    img_parser.add_argument('csv_path', help='Path to the CSV file to update')

    # PDF extraction workflow (new)
    pdf_parser = subparsers.add_parser('pdf', help='Extract text from a PDF to Markdown')
    pdf_parser.add_argument('pdf_path', help='Path to the PDF file')
    pdf_parser.add_argument('--out', dest='output_dir', default='output', help='Directory to write outputs')
    pdf_parser.add_argument('--timeout', type=int, default=300, help='Timeout seconds for PDF processing')

    # PDF bulk extraction workflow (new)
    pdf_bulk_parser = subparsers.add_parser('pdf-bulk', help='Bulk convert PDFs by numeric filename prefix range')
    pdf_bulk_parser.add_argument('--dir', dest='folder', required=True, help='Folder containing PDFs')
    pdf_bulk_parser.add_argument('--start', type=int, required=True, help='Start number (inclusive)')
    pdf_bulk_parser.add_argument('--end', type=int, required=True, help='End number (inclusive)')
    pdf_bulk_parser.add_argument('--out', dest='output_dir', default='output', help='Directory to write outputs')
    pdf_bulk_parser.add_argument('--timeout', type=int, default=300, help='Timeout seconds for each PDF')
    
    try:
        args = parser.parse_args()
        print(f"Arguments parsed successfully: {args}")
    except Exception as e:
        print(f"ERROR parsing arguments: {e}")
        return
    
    if args.command == 'images':
        # Create debug log in the same folder as the block being processed
        debug_log_path = os.path.join(args.folder_path, f"{args.block_number}_{args.category}_debug.log")
        print(f"Block folder path: {args.folder_path}")
        print(f"Debug log will be created at: {debug_log_path}")
        print(f"Debug log folder exists: {os.path.exists(args.folder_path)}")

        # Ensure the directory exists
        if not os.path.exists(args.folder_path):
            print(f"Creating directory: {args.folder_path}")
            os.makedirs(args.folder_path, exist_ok=True)

        logger = setup_logging(debug_log_path)
        print(f"Logger setup completed. Debug log created at: {debug_log_path}")

        logger.info(f"Starting OCR processing for {args.category} block {args.block_number}")
        logger.info(f"Debug log created at: {debug_log_path}")
        logger.info(f"Arguments: folder_path={args.folder_path}, block_number={args.block_number}, category={args.category}, csv_path={args.csv_path}")

        # Process question ID images
        print(f"Processing question ID images in: {args.folder_path}")
        question_id_results = process_images_with_ocr(args.folder_path, args.block_number, args.category, "ids", logger)

        # Process answer images
        print(f"Processing answer images in: {args.folder_path}")
        answer_results = process_images_with_ocr(args.folder_path, args.block_number, args.category, "answers", logger)

        if question_id_results or answer_results:
            # Update CSV with both results
            success = update_csv_with_ocr(args.csv_path, question_id_results, answer_results, logger)
            if success:
                print("OCR processing completed successfully!")
                print(f"Processed {len(question_id_results)} question IDs and {len(answer_results)} answers")
                logger.info("OCR processing completed successfully!")
            else:
                print("ERROR: Failed to update CSV file")
                logger.error("Failed to update CSV file")
        else:
            print("ERROR: No OCR results generated")
            logger.error("No OCR results generated")

    elif args.command == 'pdf':
        # PDF logging
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        log_name = Path(args.pdf_path).stem + "_pdf_debug.log"
        debug_log_path = os.path.join(out_dir, log_name)
        logger = setup_logging(debug_log_path)
        logger.info(f"Starting PDF extraction for: {args.pdf_path}")
        logger.info(f"Outputs will be written to: {out_dir}")

        ok = extract_pdf_with_mathpix(
            args.pdf_path,
            out_dir,
            logger,
            timeout_seconds=args.timeout,
        )
        if ok:
            print("PDF extraction completed successfully")
            logger.info("PDF extraction completed successfully")
        else:
            print("ERROR: PDF extraction failed")
            logger.error("PDF extraction failed")

    elif args.command == 'pdf-bulk':
        folder = args.folder
        start_num = args.start
        end_num = args.end

        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # Bulk run log
        bulk_log_path = os.path.join(out_dir, "pdf_bulk_debug.log")
        logger = setup_logging(bulk_log_path)
        logger.info(f"Starting bulk PDF extraction. Folder={folder}, Range=[{start_num}, {end_num}], Output={out_dir}")

        pdf_paths = collect_pdfs_in_range(folder, start_num, end_num, logger)
        if not pdf_paths:
            print("No PDFs found matching the specified range.")
            logger.warning("No PDFs found in the specified range.")
            return

        total = len(pdf_paths)
        success_count = 0
        for idx, pdf_path in enumerate(pdf_paths, start=1):
            print(f"[{idx}/{total}] Converting: {pdf_path}")
            logger.info(f"Converting ({idx}/{total}): {pdf_path}")
            ok = extract_pdf_with_mathpix(
                pdf_path,
                out_dir,
                logger,
                timeout_seconds=args.timeout,
            )
            if ok:
                success_count += 1
            else:
                logger.error(f"Failed to convert: {pdf_path}")

        print(f"Bulk conversion complete: {success_count}/{total} succeeded")
        logger.info(f"Bulk conversion complete: {success_count}/{total} succeeded")

def _append_history_line(cmd_line: str):
    try:
        history_path = Path(__file__).resolve().parent / "mathpix_commands_history"
        with history_path.open("a", encoding="utf-8") as hf:
            hf.write(cmd_line + "\n")
    except Exception:
        pass


def interactive_session():
    print("No CLI arguments detected. Choose a mode:")
    print("  1) images  (process images and update CSV)")
    print("  2) pdf     (extract a single PDF to Markdown)")
    print("  3) pdf-bulk (bulk convert PDFs by numeric filename prefix range)")

    base_dir = Path(__file__).resolve().parent

    # Select mode
    mode = None
    while mode not in {"1", "2", "3", "images", "pdf", "pdf-bulk"}:
        mode = input("Enter choice [1/2/3 or images/pdf/pdf-bulk]: ").strip().lower()

    if mode in {"1", "images"}:
        # Step-aware prompts
        resume_step = 1
        folder_path = ""
        block_number = ""
        category = ""
        csv_path_val = ""
        while True:
            try:
                if resume_step <= 1:
                    prompt = "1) Enter path to the block folder"
                    if folder_path:
                        prompt += f" [{folder_path}]"
                    prompt += ": "
                    v = input(prompt).strip().strip("\"")
                    if v:
                        folder_path = v
                    if not folder_path:
                        raise ValueError("Block folder path is required")

                if resume_step <= 2:
                    prompt = "2) Enter block number"
                    if block_number:
                        prompt += f" [{block_number}]"
                    prompt += ": "
                    v = input(prompt).strip()
                    if v:
                        block_number = v
                    if not block_number:
                        raise ValueError("Block number is required")

                if resume_step <= 3:
                    prompt = "3) Enter category name"
                    if category:
                        prompt += f" [{category}]"
                    prompt += ": "
                    v = input(prompt).strip()
                    if v:
                        category = v
                    if not category:
                        raise ValueError("Category is required")

                if resume_step <= 4:
                    prompt = "4) Enter path to the CSV to update"
                    if csv_path_val:
                        prompt += f" [{csv_path_val}]"
                    prompt += ": "
                    v = input(prompt).strip().strip("\"")
                    if v:
                        csv_path_val = v
                    if not csv_path_val:
                        raise ValueError("CSV path is required")

                # Create debug log
                debug_log_path = os.path.join(folder_path, f"{block_number}_{category}_debug.log")
                logger = setup_logging(debug_log_path)
                logger.info("Starting interactive 'images' run")

                # Ensure the directory exists
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path, exist_ok=True)

                # Process
                print(f"Processing question ID images in: {folder_path}")
                question_id_results = process_images_with_ocr(folder_path, block_number, category, "ids", logger)
                print(f"Processing answer images in: {folder_path}")
                answer_results = process_images_with_ocr(folder_path, block_number, category, "answers", logger)

                if question_id_results or answer_results:
                    success = update_csv_with_ocr(csv_path_val, question_id_results, answer_results, logger)
                    if success:
                        print("OCR processing completed successfully!")
                    else:
                        print("ERROR: Failed to update CSV file")
                else:
                    print("ERROR: No OCR results generated")

                # Append history
                cmd = (
                    f"py 'pdf_2_problem\\mathpix_processor.py' images "
                    f"'{folder_path}' '{block_number}' '{category}' '{csv_path_val}'"
                )
                _append_history_line(cmd)

                try:
                    choice = input("\nWould you like to run 'md to csv.py'? (y/n, n runs 'pdf_to_csv_orchestrator.py'): ").strip().lower()
                except Exception:
                    choice = "n"
                if choice == "y":
                    try:
                        md_script = Path(__file__).resolve().parent / "md to csv.py"
                        if not md_script.exists():
                            print(f"ERROR: Could not find {md_script}")
                        else:
                            subprocess.run([sys.executable, str(md_script)], check=False)
                    except Exception as e:
                        print(f"ERROR launching md to csv.py: {e}")
                else:
                    try:
                        orch_script = Path(__file__).resolve().parent / "pdf_to_csv_orchestrator.py"
                        if not orch_script.exists():
                            print(f"ERROR: Could not find {orch_script}")
                        else:
                            subprocess.run([sys.executable, str(orch_script)], check=False)
                    except Exception as e:
                        print(f"ERROR launching pdf_to_csv_orchestrator.py: {e}")
                return
            except Exception as exc:
                msg = str(exc)
                if "Block folder path" in msg:
                    resume_step = 1
                elif "Block number" in msg:
                    resume_step = 2
                elif "Category" in msg:
                    resume_step = 3
                elif "CSV path" in msg:
                    resume_step = 4
                else:
                    # Most runtime errors: let user fix and resume from last step
                    resume_step = 4
                print(f"ERROR: {exc}")
                input("\nPress Enter to try again...")
                continue

    elif mode in {"2", "pdf"}:
        # Read PDF path and output directory from config files instead of prompting
        timeout_val = 300
        config_dir = Path(__file__).resolve().parent
        pdf_path_file = config_dir / "pdf_path_convert.txt"
        out_dir_file = config_dir / "destination_for_md_file.txt"

        def _read_single_line(path_obj):
            try:
                with path_obj.open("r", encoding="utf-8") as f:
                    return f.read().strip().strip("\"'")
            except Exception:
                return ""

        pdf_path_val = _read_single_line(pdf_path_file)
        out_dir = _read_single_line(out_dir_file) or "output"

        if not pdf_path_val:
            print(f"ERROR: Missing or empty file: {pdf_path_file}")
            return

        # Resolve relative paths against the config directory
        if not os.path.isabs(pdf_path_val):
            pdf_path_val = str((config_dir / pdf_path_val).resolve())
        if not os.path.isabs(out_dir):
            out_dir = str((config_dir / out_dir).resolve())

        os.makedirs(out_dir, exist_ok=True)
        log_name = Path(pdf_path_val).stem + "_pdf_debug.log"
        debug_log_path = os.path.join(out_dir, log_name)
        logger = setup_logging(debug_log_path)
        logger.info("Starting interactive 'pdf' run (using config files)")

        ok = extract_pdf_with_mathpix(
            pdf_path_val,
            out_dir,
            logger,
            timeout_seconds=timeout_val,
        )
        if ok:
            print("PDF extraction completed successfully")
        else:
            print("ERROR: PDF extraction failed")

        cmd = (
            f"py 'pdf_2_problem\\mathpix_processor.py' pdf "
            f"'{pdf_path_val}' --out '{out_dir}' --timeout {timeout_val}"
        )
        _append_history_line(cmd)

        try:
            choice = input("\nWould you like to run 'md to csv.py'? (y/n, n runs 'pdf_to_csv_orchestrator.py'): ").strip().lower()
        except Exception:
            choice = "n"
        if choice == "y":
            try:
                md_script = Path(__file__).resolve().parent / "md to csv.py"
                if not md_script.exists():
                    print(f"ERROR: Could not find {md_script}")
                else:
                    subprocess.run([sys.executable, str(md_script)], check=False)
            except Exception as e:
                print(f"ERROR launching md to csv.py: {e}")
        else:
            try:
                orch_script = Path(__file__).resolve().parent / "pdf_to_csv_orchestrator.py"
                if not orch_script.exists():
                    print(f"ERROR: Could not find {orch_script}")
                else:
                    subprocess.run([sys.executable, str(orch_script)], check=False)
            except Exception as e:
                print(f"ERROR launching pdf_to_csv_orchestrator.py: {e}")
        return

    else:  # pdf-bulk
        resume_step = 1
        folder = ""
        start_num = None
        end_num = None
        out_dir = "output"
        timeout_val = 300
        while True:
            try:
                if resume_step <= 1:
                    prompt = "1) Enter folder containing PDFs"
                    if folder:
                        prompt += f" [{folder}]"
                    prompt += ": "
                    v = input(prompt).strip().strip("\"")
                    if v:
                        folder = v
                    if not folder:
                        raise ValueError("Folder is required")

                if resume_step <= 2:
                    prompt = "2) Enter start number (inclusive)"
                    prompt += f" [{start_num if start_num is not None else ''}]"
                    prompt += ": "
                    v = input(prompt).strip()
                    if v:
                        try:
                            start_num = int(v)
                        except Exception:
                            raise ValueError("Start must be an integer")
                    if start_num is None:
                        raise ValueError("Start number is required")

                if resume_step <= 3:
                    prompt = "3) Enter end number (inclusive)"
                    prompt += f" [{end_num if end_num is not None else ''}]"
                    prompt += ": "
                    v = input(prompt).strip()
                    if v:
                        try:
                            end_num = int(v)
                        except Exception:
                            raise ValueError("End must be an integer")
                    if end_num is None:
                        raise ValueError("End number is required")

                if resume_step <= 4:
                    prompt = "4) Enter output directory"
                    if out_dir:
                        prompt += f" [{out_dir}]"
                    prompt += ": "
                    v = input(prompt).strip().strip("\"")
                    if v:
                        out_dir = v
                    if not out_dir:
                        raise ValueError("Output directory is required")

                if resume_step <= 5:
                    prompt = "5) Enter timeout seconds per PDF"
                    prompt += f" [{timeout_val}]"
                    prompt += ": "
                    v = input(prompt).strip()
                    if v:
                        try:
                            timeout_val = int(v)
                        except Exception:
                            raise ValueError("Timeout must be an integer")

                os.makedirs(out_dir, exist_ok=True)
                bulk_log_path = os.path.join(out_dir, "pdf_bulk_debug.log")
                logger = setup_logging(bulk_log_path)
                logger.info("Starting interactive 'pdf-bulk' run")

                pdf_paths = collect_pdfs_in_range(folder, int(start_num), int(end_num), logger)
                if not pdf_paths:
                    print("No PDFs found matching the specified range.")
                    logger.warning("No PDFs found in the specified range.")
                else:
                    total = len(pdf_paths)
                    success_count = 0
                    for idx, pdf_path in enumerate(pdf_paths, start=1):
                        print(f"[{idx}/{total}] Converting: {pdf_path}")
                        logger.info(f"Converting ({idx}/{total}): {pdf_path}")
                        ok = extract_pdf_with_mathpix(pdf_path, out_dir, logger, timeout_seconds=timeout_val)
                        if ok:
                            success_count += 1
                        else:
                            logger.error(f"Failed to convert: {pdf_path}")
                    print(f"Bulk conversion complete: {success_count}/{total} succeeded")
                    logger.info(f"Bulk conversion complete: {success_count}/{total} succeeded")

                cmd = (
                    f"py 'pdf_2_problem\\mathpix_processor.py' pdf-bulk "
                    f"--dir '{folder}' --start {int(start_num)} --end {int(end_num)} "
                    f"--out '{out_dir}' --timeout {timeout_val}"
                )
                _append_history_line(cmd)

                try:
                    choice = input("\nWould you like to run 'md to csv.py'? (y/n, n runs 'pdf_to_csv_orchestrator.py'): ").strip().lower()
                except Exception:
                    choice = "n"
                if choice == "y":
                    try:
                        md_script = Path(__file__).resolve().parent / "md to csv.py"
                        if not md_script.exists():
                            print(f"ERROR: Could not find {md_script}")
                        else:
                            subprocess.run([sys.executable, str(md_script)], check=False)
                    except Exception as e:
                        print(f"ERROR launching md to csv.py: {e}")
                else:
                    try:
                        orch_script = Path(__file__).resolve().parent / "pdf_to_csv_orchestrator.py"
                        if not orch_script.exists():
                            print(f"ERROR: Could not find {orch_script}")
                        else:
                            subprocess.run([sys.executable, str(orch_script)], check=False)
                    except Exception as e:
                        print(f"ERROR launching pdf_to_csv_orchestrator.py: {e}")
                return
            except Exception as exc:
                msg = str(exc)
                if "Folder is required" in msg:
                    resume_step = 1
                elif "Start" in msg:
                    resume_step = 2
                elif "End" in msg:
                    resume_step = 3
                elif "Output directory" in msg:
                    resume_step = 4
                elif "Timeout" in msg:
                    resume_step = 5
                else:
                    resume_step = 5
                print(f"ERROR: {exc}")
                input("\nPress Enter to try again...")
                continue


if __name__ == "__main__":
    def _set_completion_state(value: str):
        try:
            state_path = Path(__file__).resolve().parent / "completion_state.txt"
            with state_path.open("w", encoding="utf-8") as f:
                f.write(str(value).strip())
        except Exception:
            # Silently ignore write errors; do not block primary flow
            pass

    try:
        try:
            _set_completion_state("0")
        except Exception:
            pass
        if len(sys.argv) == 1:
            try:
                interactive_session()
            except Exception as e:
                print(f"FATAL ERROR: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                sys.exit(1)
        else:
            try:
                main()
            except Exception as e:
                print(f"FATAL ERROR: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                sys.exit(1)
    finally:
        try:
            _set_completion_state("1")
        except Exception:
            pass