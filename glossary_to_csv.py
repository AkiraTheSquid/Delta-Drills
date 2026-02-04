from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    # Prefer pypdf (modern), falls back to PyPDF2 if needed
    from pypdf import PdfReader, PdfWriter  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without pypdf
    from PyPDF2 import PdfReader, PdfWriter  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
CHATGPT_DIR = BASE_DIR / "chatgpt"
MATHPIX_DIR = BASE_DIR / "mathpix processor"


def _read_pdf_page_texts(pdf_path: Path, max_chars: int, page_window: tuple[int, int] | None) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages == 0:
        return []

    start_idx = 0
    end_idx = total_pages - 1
    if page_window:
        start_idx = max(0, min(total_pages - 1, page_window[0]))
        end_idx = max(0, min(total_pages - 1, page_window[1]))

    items: list[tuple[int, str]] = []
    for i in range(start_idx, end_idx + 1):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        items.append((i + 1, text))
    return items


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _find_title_page_indices(
    pdf_path: Path,
    titles: list[str],
    max_pages: int | None = None,
    start_page: int | None = None,
) -> dict[str, int]:
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    last_idx = total_pages - 1
    if max_pages is not None:
        last_idx = min(last_idx, max_pages - 1)
    start_idx = 0
    if start_page is not None and start_page > 1:
        start_idx = min(total_pages - 1, start_page - 1)

    normalized_titles: dict[str, list[str]] = {}
    for t in titles:
        tokens = []
        norm = _normalize_text(t)
        if norm:
            tokens.append(norm)
        m = re.match(r"^\s*chapter\s+(\d+)", t, flags=re.IGNORECASE)
        if m:
            tokens.append(f"chapter {m.group(1)}")
        normalized_titles[t] = tokens
    hits: dict[str, int] = {}
    for i in range(start_idx, last_idx + 1):
        try:
            page_text = reader.pages[i].extract_text() or ""
        except Exception:
            page_text = ""
        page_text_norm = _normalize_text(page_text)
        if not page_text_norm:
            continue
        for title, norms in normalized_titles.items():
            if title in hits:
                continue
            for norm_title in norms:
                if norm_title and norm_title in page_text_norm:
                    hits[title] = i + 1  # 1-based
                    break
        if len(hits) == len(titles):
            break
    return hits


def _run_chatgpt(prompt: str, model: str | None = None) -> str:
    if not CHATGPT_DIR.exists():
        raise FileNotFoundError(f"ChatGPT folder not found: {CHATGPT_DIR}")
    prompt_path = CHATGPT_DIR / "prompt.txt"
    output_path = CHATGPT_DIR / "output.txt"

    prompt_path.write_text(prompt, encoding="utf-8")

    env = os.environ.copy()
    if model:
        env["OPENAI_MODEL"] = model

    result = subprocess.run(
        [sys.executable, str(CHATGPT_DIR / "ChatGPT.py")],
        cwd=str(CHATGPT_DIR),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ChatGPT.py failed with exit code {result.returncode}")
    if not output_path.exists():
        raise FileNotFoundError(f"ChatGPT output not found: {output_path}")
    return output_path.read_text(encoding="utf-8").strip()


def _build_toc_prompt(page_snippets: list[tuple[int, str]]) -> str:
    lines = [
        "You are identifying TABLE OF CONTENTS (TOC) pages in a PDF.",
        "Given page snippets, return ONLY valid JSON with keys:",
        "  toc_start (int), toc_end (int), page_numbers (array of ints), confidence (0-100).",
        "Choose a contiguous range for toc_start/toc_end if possible.",
        "If unsure, make your best guess. Do not include commentary.",
        "",
        "Page snippets:",
    ]
    for page_num, text in page_snippets:
        snippet = text if text else "(no text extracted)"
        lines.append(f"Page {page_num}: {snippet}")
    return "\n".join(lines)


def _parse_glossary_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: extract first JSON object from text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError("Could not parse glossary JSON from ChatGPT output.")


def _make_glossary_pdf(pdf_path: Path, start_page: int, end_page: int, out_dir: Path) -> Path:
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("PDF has no pages.")

    start_idx = max(1, min(start_page, total_pages)) - 1
    end_idx = max(1, min(end_page, total_pages)) - 1
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"glossary_pages_{start_idx + 1}_to_{end_idx + 1}.pdf"

    writer = PdfWriter()
    for i in range(start_idx, end_idx + 1):
        writer.add_page(reader.pages[i])
    with out_path.open("wb") as f:
        writer.write(f)
    return out_path


def _run_mathpix(pdf_path: Path, out_dir: Path, timeout: int) -> Path:
    processor = MATHPIX_DIR / "mathpix_processor.py"
    if not processor.exists():
        raise FileNotFoundError(f"mathpix_processor.py not found: {processor}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(processor),
        "pdf",
        str(pdf_path),
        "--out",
        str(out_dir),
        "--timeout",
        str(timeout),
    ]
    result = subprocess.run(cmd, cwd=str(MATHPIX_DIR), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"mathpix_processor.py failed with exit code {result.returncode}")
    md_path = out_dir / (pdf_path.stem + ".md")
    if not md_path.exists():
        raise FileNotFoundError(f"Expected Markdown not found: {md_path}")
    return md_path


def _build_csv_prompt(markdown_text: str, schema: list[str], include_header: bool) -> str:
    cols = ", ".join(schema)
    header_line = cols if include_header else "(no header)"
    return "\n".join(
        [
            "Convert the following TABLE OF CONTENTS (TOC) markdown into CSV.",
            f"Output columns (in order): {cols}.",
            f"Header: {header_line}.",
            "Rules:",
            "- One TOC entry per CSV row.",
            "- If a title spans multiple lines, keep it in a single CSV cell.",
            "- The page_number column must be a numeric page reference (digits).",
            "- Do not wrap output in code fences.",
            "- Quote any field containing a comma or leading/trailing spaces.",
            "- Do not output commentary or extra text.",
            "",
            "TOC markdown:",
            markdown_text.strip(),
        ]
    )


def _write_csv_text(csv_text: str, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(csv_text.strip() + "\n", encoding="utf-8")


def _parse_toc_csv(csv_text: str, schema: list[str]) -> list[dict[str, str]]:
    lines = [line for line in csv_text.splitlines() if line.strip()]
    # Remove fenced code block markers if present
    lines = [line for line in lines if not line.strip().startswith("```")]
    rows: list[list[str]] = []
    for line in lines:
        # Basic CSV parsing without importing csv to preserve quoted commas
        fields: list[str] = []
        buf = ""
        in_quotes = False
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == '"' and (i == 0 or line[i - 1] != "\\"):
                if in_quotes and i + 1 < len(line) and line[i + 1] == '"':
                    buf += '"'
                    i += 1
                else:
                    in_quotes = not in_quotes
            elif ch == "," and not in_quotes:
                fields.append(buf.strip())
                buf = ""
            else:
                buf += ch
            i += 1
        fields.append(buf.strip())
        rows.append(fields)

    parsed: list[dict[str, str]] = []
    for row in rows:
        if len(row) < len(schema):
            continue
        item = {schema[i]: row[i].strip().strip('"') for i in range(len(schema))}
        parsed.append(item)
    return parsed


def _safe_int(s: str) -> int | None:
    try:
        return int(re.findall(r"\d+", s)[0])
    except Exception:
        return None


def _compute_offset_from_hits(entries: list[dict[str, str]], hits: dict[str, int]) -> int | None:
    offsets: list[int] = []
    for entry in entries:
        title = entry.get("section_title", "")
        page_raw = entry.get("page_number", "")
        book_page = _safe_int(page_raw or "")
        if not title or book_page is None:
            continue
        pdf_page = hits.get(title)
        if pdf_page is None:
            continue
        offsets.append(pdf_page - book_page)
    if not offsets:
        return None
    offsets.sort()
    return offsets[len(offsets) // 2]


def _write_chapters_csv(
    entries: list[dict[str, str]],
    offset: int,
    total_pdf_pages: int,
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int, int]] = []
    for idx, entry in enumerate(entries):
        title = entry.get("section_title", "").strip()
        if not re.match(r"^\s*chapter\s+\d+", title, flags=re.IGNORECASE):
            continue
        page_raw = entry.get("page_number", "")
        book_page = _safe_int(page_raw or "")
        if not title or book_page is None:
            continue
        start_pdf = book_page + offset
        if start_pdf < 1:
            start_pdf = 1
        rows.append((title, start_pdf, 0))

    # Sort by start page and de-duplicate identical starts
    rows.sort(key=lambda r: r[1])
    deduped: list[tuple[str, int, int]] = []
    last_start = None
    for title, start, end in rows:
        if last_start is not None and start == last_start:
            continue
        deduped.append((title, start, end))
        last_start = start
    rows = deduped

    # Compute end pages
    for i in range(len(rows)):
        start = rows[i][1]
        if i + 1 < len(rows):
            end = rows[i + 1][1] - 1
        else:
            end = total_pdf_pages
        if end < start:
            end = start
        rows[i] = (rows[i][0], start, end)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("title,start_page,end_page\n")
        for title, start, end in rows:
            safe_title = title.replace("\n", " ").strip()
            f.write(f"\"{safe_title}\",{start},{end}\n")


def _run_splitter(pdf_path: Path, chapters_csv: Path, output_dir: Path) -> int:
    splitter = BASE_DIR / "split_LADR4e_chapters.py"
    if not splitter.exists():
        raise FileNotFoundError(f"split_LADR4e_chapters.py not found: {splitter}")
    cmd = [
        sys.executable,
        str(splitter),
        str(pdf_path),
        str(chapters_csv),
        str(output_dir),
        "1",
    ]
    result = subprocess.run(cmd, cwd=str(BASE_DIR), check=False)
    return result.returncode


def _extract_outline_chapters(pdf_path: Path) -> list[tuple[str, int]]:
    reader = PdfReader(str(pdf_path))
    try:
        outlines = reader.outline
    except Exception:
        return []
    if not outlines:
        return []

    chapters: list[tuple[str, int]] = []

    def walk(items):
        for it in items:
            if isinstance(it, list):
                walk(it)
                continue
            title = getattr(it, "title", None)
            if not title:
                continue
            if not re.match(r"^\s*chapter\s+\d+", title, flags=re.IGNORECASE):
                continue
            try:
                page_num = reader.get_destination_page_number(it) + 1
            except Exception:
                continue
            chapters.append((title.strip(), page_num))

    walk(outlines)
    # De-duplicate by title, keep first occurrence, and sort by page
    seen = set()
    unique: list[tuple[str, int]] = []
    for title, page in chapters:
        if title in seen:
            continue
        seen.add(title)
        unique.append((title, page))
    unique.sort(key=lambda x: x[1])
    return unique


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect TOC pages via ChatGPT, convert to Markdown via Mathpix, then to CSV via ChatGPT."
    )
    parser.add_argument("pdf_path", help="Path to the source PDF")
    parser.add_argument("--scan-first", type=int, default=60, help="Scan only the first N pages for TOC detection")
    parser.add_argument("--scan-last", type=int, default=0, help="Scan only the last N pages for TOC detection (overrides --scan-first)")
    parser.add_argument("--snippet-chars", type=int, default=1200, help="Max chars per page snippet for detection prompt")
    parser.add_argument("--mathpix-out", default=str(BASE_DIR / "toc_md"), help="Output folder for Mathpix Markdown")
    parser.add_argument("--csv-path", default=str(BASE_DIR / "toc.csv"), help="Output CSV path")
    parser.add_argument("--schema", default="section_title,page_number", help="Comma-separated CSV columns")
    parser.add_argument("--no-header", action="store_true", help="Do not include a CSV header row")
    parser.add_argument("--model", default=None, help="Override ChatGPT model (OPENAI_MODEL)")
    parser.add_argument("--timeout", type=int, default=600, help="Mathpix timeout seconds")
    parser.add_argument("--chapters-csv", default=str(BASE_DIR / "toc_chapters.csv"), help="Output chapters CSV path")
    parser.add_argument("--split", action="store_true", help="Split PDF into chapter PDFs using TOC")
    parser.add_argument("--chapters-dir", default=None, help="Output folder for chapter PDFs (auto if omitted)")
    parser.add_argument("--no-prefer-outline", action="store_true", help="Disable using PDF outline/bookmarks for chapter pages")
    args = parser.parse_args(argv)

    pdf_path = Path(args.pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages == 0:
        print("PDF has no pages.", file=sys.stderr)
        return 1

    # Limit scan window: default to first N pages (TOC is usually front-matter)
    scan_last = int(args.scan_last)
    scan_first = int(args.scan_first)
    if scan_last and scan_last > 0:
        start_idx = max(0, total_pages - scan_last)
        page_window = (start_idx, total_pages - 1)
    else:
        scan_first = max(1, scan_first)
        start_idx = 0
        end_idx = min(total_pages - 1, scan_first - 1)
        page_window = (start_idx, end_idx)
    snippets = _read_pdf_page_texts(pdf_path, max_chars=args.snippet_chars, page_window=page_window)
    if not snippets:
        print("No text could be extracted from the PDF.", file=sys.stderr)
        return 1

    prompt = _build_toc_prompt(snippets)
    toc_json_text = _run_chatgpt(prompt, model=args.model)
    toc_info = _parse_glossary_json(toc_json_text)

    start_page = int(toc_info.get("toc_start", 0) or 0)
    end_page = int(toc_info.get("toc_end", 0) or 0)
    if start_page <= 0 or end_page <= 0:
        # Fallback: use explicit page_numbers if provided
        page_numbers = toc_info.get("page_numbers") or []
        if page_numbers:
            start_page = int(min(page_numbers))
            end_page = int(max(page_numbers))
    if start_page <= 0 or end_page <= 0:
        print("Could not determine TOC page range from ChatGPT output.", file=sys.stderr)
        return 1

    toc_dir = Path(args.mathpix_out).expanduser().resolve()
    toc_pdf = _make_glossary_pdf(pdf_path, start_page, end_page, toc_dir)
    md_path = _run_mathpix(toc_pdf, toc_dir, args.timeout)

    markdown_text = md_path.read_text(encoding="utf-8")
    schema = [c.strip() for c in args.schema.split(",") if c.strip()]
    if not schema:
        print("CSV schema is empty.", file=sys.stderr)
        return 1
    csv_prompt = _build_csv_prompt(markdown_text, schema, include_header=not args.no_header)
    csv_text = _run_chatgpt(csv_prompt, model=args.model)

    csv_path = Path(args.csv_path).expanduser().resolve()
    _write_csv_text(csv_text, csv_path)

    # Build chapters CSV with inferred offset (TOC-based)
    schema = [c.strip() for c in args.schema.split(",") if c.strip()]
    entries = _parse_toc_csv(csv_text, schema)
    # If the TOC lacks page numbers, fall back to locating titles directly in the PDF.
    with_page_nums = [e for e in entries if _safe_int(e.get("page_number", "") or "") is not None]
    if len(with_page_nums) < 2:
        titles = [e.get("section_title", "") for e in entries if e.get("section_title")]
        title_hits = _find_title_page_indices(
            pdf_path,
            titles,
            max_pages=total_pages,
            start_page=end_page + 1,
        )
        for e in entries:
            title = e.get("section_title", "")
            if title and title in title_hits:
                e["page_number"] = str(title_hits[title])
        with_page_nums = [e for e in entries if _safe_int(e.get("page_number", "") or "") is not None]
        offset = 0
    else:
        sample_titles = [e.get("section_title", "") for e in entries[:8] if e.get("section_title")]
        hits = _find_title_page_indices(
            pdf_path,
            sample_titles,
            max_pages=total_pages,
            start_page=end_page + 1,
        )
        offset = _compute_offset_from_hits(entries, hits)
        if offset is None:
            offset = 0

    chapters_csv = Path(args.chapters_csv).expanduser().resolve()

    outline_chapters: list[tuple[str, int]] = []
    prefer_outline = not args.no_prefer_outline
    if prefer_outline:
        outline_chapters = _extract_outline_chapters(pdf_path)

    if outline_chapters:
        # Use outline data as the source of truth for chapter start pages
        rows = [(title, page, 0) for title, page in outline_chapters]
        rows.sort(key=lambda r: r[1])
        for i in range(len(rows)):
            start = rows[i][1]
            end = rows[i + 1][1] - 1 if i + 1 < len(rows) else total_pages
            if end < start:
                end = start
            rows[i] = (rows[i][0], start, end)
        chapters_csv.parent.mkdir(parents=True, exist_ok=True)
        with chapters_csv.open("w", encoding="utf-8", newline="") as f:
            f.write("title,start_page,end_page\n")
            for title, start, end in rows:
                safe_title = title.replace("\n", " ").strip()
                f.write(f"\"{safe_title}\",{start},{end}\n")
    else:
        _write_chapters_csv(entries, offset, total_pages, chapters_csv)

    print("TOC detection:")
    print(f"- Pages: {start_page} to {end_page}")
    print(f"- Mathpix Markdown: {md_path}")
    print(f"- TOC CSV output: {csv_path}")
    print(f"- Chapters CSV output: {chapters_csv}")
    print(f"- Inferred page offset: {offset}")
    if outline_chapters:
        print(f"- Outline chapters used: {len(outline_chapters)}")

    if args.split:
        if args.chapters_dir:
            chapters_dir = Path(args.chapters_dir).expanduser().resolve()
        else:
            chapters_dir = BASE_DIR / f"{pdf_path.stem}_chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        rc = _run_splitter(pdf_path, chapters_csv, chapters_dir)
        if rc != 0:
            print(f"Splitter failed with exit code {rc}", file=sys.stderr)
            return rc
        print(f"- Chapters folder: {chapters_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
