# Run (PowerShell): python ".\pdf_2_problem\split_LADR4e_chapters.py" ".\pdf_2_problem\LADR_4e_solns.pdf" ".\pdf_2_problem\LADR4e_solns_chapters.csv" ".\pdf_2_problem\SolutionSections" 3
from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path
import shutil

try:
    # Prefer pypdf (modern), falls back to PyPDF2 if needed
    from pypdf import PdfReader, PdfWriter  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without pypdf
    from PyPDF2 import PdfReader, PdfWriter  # type: ignore


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for Windows while preserving readability.

    Removes characters not allowed on Windows and trims trailing dots/spaces.
    """
    # Remove invalid Windows filename characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "", filename)
    # Replace control characters
    sanitized = re.sub(r"[\x00-\x1f]", "", sanitized)
    # Collapse excessive whitespace
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    # Avoid trailing dot or space
    sanitized = sanitized.rstrip(" .")
    # Reasonable length cap to avoid long path issues
    if len(sanitized) > 140:
        sanitized = sanitized[:140].rstrip()
    # Ensure non-empty
    return sanitized or "chapter"


def strip_leading_index(title: str) -> str:
    """Remove any leading numeric index from the title for cleaner filenames.

    Examples:
    - "10. Exercises 3D - Invertibility" -> "Exercises 3D - Invertibility"
    - "1.A. Rn and Cn" -> "A. Rn and Cn"
    """
    # Strip patterns like "10. " or "10) "
    title_wo_num = re.sub(r"^\s*\d+\s*[\.)]\s*", "", title)
    return title_wo_num.strip()


def preferred_python_invocation() -> str:
    """Return a robust Python invocation command for Windows.

    Prefer the Windows launcher 'py' when available, otherwise fall back to the
    current interpreter path (sys.executable), and as a last resort 'python'.
    """
    # Prefer 'py' on Windows (works even if 'python' App alias is disabled)
    if shutil.which("py"):
        return "py"

    # Fall back to the exact interpreter running this script
    exe = sys.executable or ""
    if exe:
        return f'"{exe}"' if " " in exe else exe

    # Final fallback
    return "python"


def prompt_run_mathpix(base_dir: Path) -> None:
    """Prompt to run mathpix_processor.py or pdf_to_csv_orchestrator.py based on choice.

    - 'y' → run mathpix_processor.py
    - 'n' (or anything else) → run pdf_to_csv_orchestrator.py
    """
    try:
        choice = input("\nRun mathpix_processor.py now? (y/n, n runs pdf_to_csv_orchestrator.py): ").strip().lower()
    except Exception:
        return
    python_cmd = preferred_python_invocation()
    if choice == "y":
        script_path = base_dir / "mathpix_processor.py"
    else:
        script_path = base_dir / "pdf_to_csv_orchestrator.py"
    cmd = f"{python_cmd} \"{str(script_path)}\""
    try:
        os.system(cmd)
    except Exception:
        pass

def read_exercise_sections(csv_path: Path) -> list[tuple[str, int, int]]:
    """Read exercise section titles with 1-based start and end pages from CSV.

    Expects columns: [title, start_page, end_page]. Extra columns are ignored.
    Returns a list of (title, start_1_based, end_1_based) in file order.
    """
    sections: list[tuple[str, int, int]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        # Skip header
        try:
            next(reader)
        except StopIteration:
            raise ValueError("CSV appears to be empty: " + str(csv_path))

        for row in reader:
            if not row or all((c is None or str(c).strip() == "") for c in row):
                continue
            if len(row) < 3:
                raise ValueError(f"CSV row missing columns (need title,start,end): {row}")
            title_raw = str(row[0]).strip()
            start_str = str(row[1]).strip()
            end_str = str(row[2]).strip()
            if not title_raw:
                raise ValueError(f"Empty title in row: {row}")
            if not start_str or not end_str:
                raise ValueError(f"Empty start/end page in row: {row}")
            try:
                start_page_1_based = int(start_str)
                end_page_1_based = int(end_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid page number(s) start='{start_str}', end='{end_str}' in row: {row}"
                ) from exc
            if end_page_1_based < start_page_1_based:
                raise ValueError(
                    f"End page {end_page_1_based} < start page {start_page_1_based} for '{title_raw}'"
                )
            sections.append((title_raw, start_page_1_based, end_page_1_based))

    # Validate monotonic increase of start pages
    for i in range(1, len(sections)):
        prev_title, prev_start, _ = sections[i - 1]
        curr_title, curr_start, _ = sections[i]
        if curr_start <= prev_start:
            raise ValueError(
                "Exercise section start pages must be strictly increasing: "
                f"'{prev_title}' starts at {prev_start}, "
                f"'{curr_title}' starts at {curr_start}"
            )

    return sections


def split_pdf_by_exercises(
    pdf_path: Path,
    exercises_csv_path: Path,
    output_dir: Path,
    page_offset: int = 0,
) -> list[Path]:
    """Split the PDF into per-exercise-section files using explicit start/end pages.

    page_offset allows converting book page numbers to PDF page numbers if needed
    (e.g., if PDF has a front-matter offset). The formula is:
      pdf_zero_based_start = (start_1_based + page_offset) - 1
      pdf_exclusive_end    = (end_1_based   + page_offset)
    """
    sections = read_exercise_sections(exercises_csv_path)
    if not sections:
        raise ValueError("No exercise sections found in CSV.")

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("PDF has no pages: " + str(pdf_path))

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    # Global ordinal across all sections
    for idx, (title, start_1b, end_1b) in enumerate(sections, start=1):
        start_zero = (start_1b + page_offset) - 1
        end_excl = end_1b + page_offset

        if start_zero < 0 or start_zero >= total_pages:
            raise ValueError(
                f"Section {idx} start out of range: {start_1b} (+{page_offset}) -> {start_zero + 1} (1-based); PDF pages={total_pages}"
            )
        if end_excl <= 0 or end_excl > total_pages:
            raise ValueError(
                f"Section {idx} end out of range: {end_1b} (+{page_offset}) -> {end_excl} (1-based exclusive end); PDF pages={total_pages}"
            )
        if end_excl <= start_zero:
            raise ValueError(
                f"Computed empty/invalid range for '{title}': start_zero={start_zero}, end_excl={end_excl}"
            )

        writer = PdfWriter()
        for page_index in range(start_zero, end_excl):
            writer.add_page(reader.pages[page_index])

        # Clean the title of any leading global numbering (e.g., "1.", "10.")
        display_title = strip_leading_index(title)
        # Prefix with the global counter (no zero padding), e.g., "1 Exercises 1A - ..."
        filename = f"{idx} " + sanitize_filename(display_title) + ".pdf"
        out_path = output_dir / filename
        with out_path.open("wb") as out_f:
            writer.write(out_f)
        outputs.append(out_path)

    return outputs


def main(argv: list[str]) -> int:
    base_dir = Path(__file__).resolve().parent
    # Defaults for LADR4e exercises
    csv_path = base_dir / "LADR4e_chapters.csv"
    pdf_path = base_dir / "LADR4e.pdf"
    output_dir = base_dir / "exercise_sections"
    page_offset = 0  # adjust if CSV pages are book pages, not PDF pages
    first_chapter_page_1_based = None

    # Interactive prompts if run without CLI args (e.g., double-click)
    if len(argv) == 1:
        print("No CLI arguments detected. Please answer the following:")
        pdf_in = input("1) Enter full path to the PDF to split: ").strip().strip('"')
        csv_in = input("2) Enter full path to the sections CSV: ").strip().strip('"')
        out_in = input(
            "3) Enter output directory path (it will be created if missing): "
        ).strip().strip('"')
        first_page_in = input(
            "4) Enter the first chapter's page number (>= 1): "
        ).strip().strip('"')

        if not pdf_in or not csv_in or not out_in or not first_page_in:
            print("All four inputs are required.", file=sys.stderr)
            return 1

        pdf_path = Path(pdf_in).expanduser().resolve()
        csv_path = Path(csv_in).expanduser().resolve()
        output_dir = Path(out_in).expanduser().resolve()
        try:
            first_chapter_page_1_based = int(first_page_in)
        except ValueError:
            print(
                f"Invalid first_chapter_page '{first_page_in}', expected integer.",
                file=sys.stderr,
            )
            return 1
        if first_chapter_page_1_based <= 0:
            print(
                f"Invalid first_chapter_page '{first_chapter_page_1_based}', must be >= 1.",
                file=sys.stderr,
            )
            return 1
        page_offset = first_chapter_page_1_based - 1

    # Allow optional CLI overrides: python split_LADR4e_chapters.py [pdf_path] [csv_path] [output_dir] [first_chapter_page]
    if len(argv) >= 2:
        pdf_path = Path(argv[1]).expanduser().resolve()
    if len(argv) >= 3:
        csv_path = Path(argv[2]).expanduser().resolve()
    if len(argv) >= 4:
        output_dir = Path(argv[3]).expanduser().resolve()
    if len(argv) >= 5:
        try:
            first_chapter_page_1_based = int(argv[4])
        except ValueError:
            print(
                f"Invalid first_chapter_page '{argv[4]}', expected integer.",
                file=sys.stderr,
            )
            return 1
        if first_chapter_page_1_based <= 0:
            print(
                f"Invalid first_chapter_page '{first_chapter_page_1_based}', must be >= 1.",
                file=sys.stderr,
            )
            return 1
        # Derive page_offset from the first visible chapter page
        page_offset = first_chapter_page_1_based - 1

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    outputs = split_pdf_by_exercises(pdf_path, csv_path, output_dir, page_offset=page_offset)
    print(
        f"Wrote {len(outputs)} exercise section file(s) to: {output_dir} (offset={page_offset})"
    )
    for p in outputs:
        print(f"- {p.name}")
    # Append a copy-pastable PowerShell command to history when run interactively
    if len(argv) == 1 and first_chapter_page_1_based is not None:
        try:
            python_cmd = preferred_python_invocation()
            history_line = (
                f"{python_cmd} \".\\pdf_2_problem\\split_LADR4e_chapters.py\" "
                f"\"{str(pdf_path)}\" \"{str(csv_path)}\" \"{str(output_dir)}\" {first_chapter_page_1_based}"
            )
            history_path = base_dir / "split_commands_history"
            with history_path.open("a", encoding="utf-8") as hf:
                hf.write(history_line + "\n")
        except Exception:
            # Best-effort; ignore history write failures in interactive mode
            pass
    return 0


def interactive_session() -> int:
    base_dir = Path(__file__).resolve().parent
    pdf_path = None
    csv_path = None
    output_dir = None
    first_chapter_page_1_based = None
    resume_step = 1

    while True:
        try:
            # Step 1: PDF path
            if resume_step <= 1:
                current_pdf = str(pdf_path) if pdf_path else ""
                prompt_pdf = "1) Enter full path to the PDF to split"
                if current_pdf:
                    prompt_pdf += f" [{current_pdf}]"
                prompt_pdf += ": "
                pdf_in = input(prompt_pdf).strip().strip('"')
                if pdf_in:
                    pdf_path = Path(pdf_in).expanduser().resolve()
                if not pdf_path:
                    print("PDF path is required.", file=sys.stderr)
                    resume_step = 1
                    input("\nPress Enter to try again...")
                    continue

            # Step 2: CSV path
            if resume_step <= 2:
                current_csv = str(csv_path) if csv_path else ""
                prompt_csv = "2) Enter full path to the sections CSV"
                if current_csv:
                    prompt_csv += f" [{current_csv}]"
                prompt_csv += ": "
                csv_in = input(prompt_csv).strip().strip('"')
                if csv_in:
                    csv_path = Path(csv_in).expanduser().resolve()
                if not csv_path:
                    print("CSV path is required.", file=sys.stderr)
                    resume_step = 2
                    input("\nPress Enter to try again...")
                    continue

            # Step 3: Output directory
            if resume_step <= 3:
                current_out = str(output_dir) if output_dir else ""
                prompt_out = "3) Enter output directory path (it will be created if missing)"
                if current_out:
                    prompt_out += f" [{current_out}]"
                prompt_out += ": "
                out_in = input(prompt_out).strip().strip('"')
                if out_in:
                    output_dir = Path(out_in).expanduser().resolve()
                if not output_dir:
                    print("Output directory is required.", file=sys.stderr)
                    resume_step = 3
                    input("\nPress Enter to try again...")
                    continue

            # Step 4: First chapter page
            if resume_step <= 4:
                current_first = str(first_chapter_page_1_based) if first_chapter_page_1_based else ""
                prompt_first = "4) Enter the first chapter's page number (>= 1)"
                if current_first:
                    prompt_first += f" [{current_first}]"
                prompt_first += ": "
                first_page_in = input(prompt_first).strip().strip('"')
                if first_page_in:
                    try:
                        first_chapter_page_1_based = int(first_page_in)
                    except ValueError:
                        print(
                            f"Invalid first_chapter_page '{first_page_in}', expected integer.",
                            file=sys.stderr,
                        )
                        resume_step = 4
                        input("\nPress Enter to try again...")
                        continue
                if not first_chapter_page_1_based or first_chapter_page_1_based <= 0:
                    print(
                        f"Invalid first_chapter_page '{first_chapter_page_1_based}', must be >= 1.",
                        file=sys.stderr,
                    )
                    resume_step = 4
                    input("\nPress Enter to try again...")
                    continue

            # Validate existence before running
            if not Path(pdf_path).exists():
                print(f"PDF not found: {pdf_path}", file=sys.stderr)
                resume_step = 1
                input("\nPress Enter to try again...")
                continue
            if not Path(csv_path).exists():
                print(f"CSV not found: {csv_path}", file=sys.stderr)
                resume_step = 2
                input("\nPress Enter to try again...")
                continue

            page_offset = int(first_chapter_page_1_based) - 1
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            outputs = split_pdf_by_exercises(Path(pdf_path), Path(csv_path), Path(output_dir), page_offset=page_offset)
            print(
                f"Wrote {len(outputs)} exercise section file(s) to: {output_dir} (offset={page_offset})"
            )
            for p in outputs:
                print(f"- {p.name}")

            # Append history line
            try:
                python_cmd = preferred_python_invocation()
                history_line = (
                    f"{python_cmd} \".\\pdf_2_problem\\split_LADR4e_chapters.py\" "
                    f"\"{str(Path(pdf_path))}\" \"{str(Path(csv_path))}\" \"{str(Path(output_dir))}\" {int(first_chapter_page_1_based)}"
                )
                history_path = base_dir / "split_commands_history"
                with history_path.open("a", encoding="utf-8") as hf:
                    hf.write(history_line + "\n")
            except Exception:
                pass

            # Offer to run mathpix_processor.py, then exit
            prompt_run_mathpix(base_dir)
            return 0

        except Exception as exc:
            print(f"Unexpected error: {exc}", file=sys.stderr)
            msg = str(exc)
            if "PDF not found" in msg:
                resume_step = 1
            elif "CSV not found" in msg:
                resume_step = 2
            else:
                # Most validation errors relate to content/offset → resume at step 4
                resume_step = 4
            try:
                input("\nPress Enter to try again...")
            except Exception:
                pass
            continue


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(interactive_session())
    else:
        raise SystemExit(main(sys.argv))


