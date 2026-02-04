import os
import sys
import subprocess
from pathlib import Path


def _read_single_line(path_obj: Path) -> str:
    try:
        with path_obj.open("r", encoding="utf-8") as f:
            return f.read().strip().strip("\"'")
    except Exception:
        return ""


def _write_single_line(path_obj: Path, value: str) -> None:
    try:
        with path_obj.open("w", encoding="utf-8") as f:
            f.write((value or "").strip())
    except Exception as e:
        print(f"WARNING: Failed to write {path_obj}: {e}")


def _prompt_with_default(label: str, default_value: str) -> str:
    prompt = f"{label}"
    if default_value:
        prompt += f" [{default_value}]"
    prompt += ": "
    v = input(prompt).strip()
    return v if v else default_value


def main():
    base_dir = Path(__file__).resolve().parent
    processor = base_dir / "mathpix_processor.py"
    if not processor.exists():
        print(f"ERROR: Could not find mathpix_processor.py at {processor}")
        return

    pdf_path_file = base_dir / "pdf_path_convert.txt"
    out_dir_file = base_dir / "destination_for_md_file.txt"

    default_pdf = _read_single_line(pdf_path_file)
    default_out = _read_single_line(out_dir_file) or "output"
    default_timeout = "300"

    # Prompt user for inputs (UI similar to previous behavior)
    while True:
        try:
            pdf_path_val = _prompt_with_default("1) Enter path to the PDF", default_pdf)
            if not pdf_path_val:
                raise ValueError("PDF path is required")

            out_dir = _prompt_with_default("2) Enter output directory", default_out)
            if not out_dir:
                raise ValueError("Output directory is required")

            timeout_raw = _prompt_with_default("3) Enter timeout seconds", default_timeout)
            try:
                timeout_val = int(timeout_raw)
            except Exception:
                raise ValueError("Timeout must be an integer")

            # Persist choices to the helper files
            _write_single_line(pdf_path_file, pdf_path_val)
            _write_single_line(out_dir_file, out_dir)

            # Ensure output directory exists
            try:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                # Resolve relative to script directory if needed
                resolved_out = (base_dir / out_dir).resolve()
                resolved_out.mkdir(parents=True, exist_ok=True)
                out_dir = str(resolved_out)

            # Build and show the command
            cmd = [
                sys.executable,
                str(processor),
                "pdf",
                pdf_path_val,
                "--out",
                out_dir,
                "--timeout",
                str(timeout_val),
            ]
            pretty_cmd = (
                f"py 'pdf_2_problem\\mathpix_processor.py' pdf "
                f"'{pdf_path_val}' --out '{out_dir}' --timeout {timeout_val}"
            )
            print("\nAbout to run:")
            print(pretty_cmd)

            # Execute
            try:
                subprocess.run(cmd, check=False)
            except Exception as e:
                print(f"ERROR: Failed to launch mathpix_processor.py: {e}")

            return
        except Exception as exc:
            msg = str(exc)
            print(f"ERROR: {msg}")
            input("\nPress Enter to try again...")
            continue


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
