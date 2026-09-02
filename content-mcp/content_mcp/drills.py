"""The drill bank: reading it, and adding to it without renumbering it.

Reading comes from the compiled `questions_structured.json`. Writing NEVER
touches that file — it is regenerated on every export and hand edits are
erased. A new drill is a row appended to `curated_additions.csv` (the LAST
export source, so ids are minted above every existing one and no tag or stored
attempt id re-points) plus an optional record in `curated_overrides.jsonl`
carrying the function-mode payload. A change to an existing drill is an
override record and nothing else.

Ids are POSITIONAL: id N is the Nth data row across the export's CSV sources in
order. `next_id()` therefore counts rows rather than reading the bank's maximum
id — retired and deleted ids still consume their slot.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import json
import sys

from . import paths

CSV_FIELDS = ["Topic", "Subtopic", "Question", "Answer", "Problem difficulty", "Output"]
OVERRIDE_FIELDS = {
    "function_name", "starter_code", "test_cases", "submission_mode", "question_text",
    "answer_code", "expected_output", "task_type", "expected_artifact_type",
    "supports_visual_output", "difficulty_score", "wrong_examples",
}


def _export_module():
    """The export script is the authority on id assignment — reuse it."""
    if str(paths.PIPELINE) not in sys.path:
        sys.path.insert(0, str(paths.PIPELINE))
    import export_questions_json  # noqa: PLC0415

    return export_questions_json


def bank() -> list[dict]:
    return json.loads(paths.QUESTIONS_STRUCTURED.read_text())


def tags() -> dict:
    return json.loads(paths.QMATRIX.read_text()) if paths.QMATRIX.exists() else {}


def next_id() -> int:
    export = _export_module()
    total = 0
    for source in export.CSV_SOURCES:
        if not source["path"].exists():
            continue
        total += sum(1 for _ in export.iter_csv_rows(source["path"], source["skip_rows"]))
    return total + 1


def _flatten(record: dict, question_tags: dict) -> dict:
    exercise = record.get("exercise", {})
    curriculum = record.get("curriculum", {})
    tag = question_tags.get(str(record["id"]), {})
    return {
        "id": record["id"],
        "topic": curriculum.get("topic"),
        "subtopic": curriculum.get("subtopic_key"),
        "difficulty": curriculum.get("difficulty_label"),
        "question_text": exercise.get("question_text"),
        "function_name": exercise.get("function_name"),
        "submission_mode": exercise.get("submission_mode"),
        "target_kcs": tag.get("target_kcs", []),
        "supporting_kcs": tag.get("supporting_kcs", []),
        "test_case_count": len(exercise.get("test_cases") or []),
    }


def search(query: str | None = None, kc: str | None = None, ids: list[int] | None = None,
           limit: int = 25) -> dict:
    question_tags = tags()
    wanted = set(ids or [])
    needle = (query or "").lower()
    hits = []
    for record in bank():
        if wanted and record["id"] not in wanted:
            continue
        tag = question_tags.get(str(record["id"]), {})
        if kc and kc not in (tag.get("target_kcs", []) + tag.get("supporting_kcs", [])):
            continue
        if needle:
            blob = json.dumps(record).lower()
            if needle not in blob:
                continue
        hits.append(_flatten(record, question_tags))
    return {"total_matches": len(hits), "returned": hits[:limit], "bank_size": len(bank())}


def read(qid: int) -> dict:
    for record in bank():
        if int(record["id"]) == int(qid):
            record = dict(record)
            record["tags"] = tags().get(str(qid), {})
            record["override_layers"] = _override_layers_for(int(qid))
            return record
    raise KeyError(f"No drill with id {qid} in the bank. Retired or deleted ids read as missing.")


def _override_layers_for(qid: int) -> list[str]:
    """Which JSONL layers carry a record for this id.

    Reported by scanning the override directory, NOT by reading the exporter's
    layer list. That list is already duplicated between the exporter and
    `backend/app/questions.py` and has to stay in sync as an ordered sequence;
    a third copy here would be a third thing to forget. Which layer WINS is
    therefore the exporter's answer, not this function's — this only says where
    the id is mentioned.
    """
    found = []
    if not paths.CHATGPT.is_dir():
        return found
    for path in sorted(paths.CHATGPT.glob("*.jsonl")):
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line and json.loads(line).get("id") == qid:
                        found.append(path.name)
                        break
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
    return found


@contextlib.contextmanager
def _bank_lock():
    """Serialize id minting across processes.

    Ids are positional, so two contributors appending at the same moment would
    both compute the same next id and the second row would silently take an id
    the first one's override is already keyed to.
    """
    lock_path = paths.state_dir() / "drill-bank.lock"
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def add(topic: str, subtopic: str, question_text: str, answer_code: str,
        difficulty: int = 50, expected_output: str = "", override: dict | None = None) -> dict:
    """Append one authored drill. Returns the id it will be served under."""
    # Validate BEFORE touching the CSV. An override with a bad field used to
    # raise after the row was already committed, leaving a half-added drill
    # behind a reported failure — and a retry then minted a second one.
    if override:
        _check_fields(override)
    if not str(question_text).strip() or not str(answer_code).strip():
        raise ValueError("A drill needs both a question_text and an answer_code.")

    with _bank_lock():
        return _append_locked(topic, subtopic, question_text, answer_code,
                              difficulty, expected_output, override)


def _append_locked(topic, subtopic, question_text, answer_code,
                   difficulty, expected_output, override) -> dict:
    minted = next_id()
    row = {
        "Topic": topic,
        "Subtopic": subtopic,
        "Question": question_text,
        "Answer": answer_code,
        "Problem difficulty": str(int(difficulty)),
        "Output": expected_output,
    }
    existing_header = []
    if paths.CURATED_CSV.exists():
        with open(paths.CURATED_CSV, newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), [])
    fields = existing_header or CSV_FIELDS
    with open(paths.CURATED_CSV, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not existing_header:
            writer.writeheader()
        writer.writerow(row)

    wrote_override = False
    if override:
        # The id does not exist in the compiled bank yet — it appears only after
        # the next export — so the existence check is deliberately skipped here.
        update(minted, override, _require_existing=False)
        wrote_override = True
    return {
        "id": minted,
        "csv": str(paths.CURATED_CSV.relative_to(paths.REPO)),
        "override_written": wrote_override,
        "next_step": "Run pipeline_export to rebuild the bank, then pipeline_check.",
    }


def _check_fields(fields: dict) -> None:
    unknown = set(fields) - OVERRIDE_FIELDS
    if unknown:
        raise ValueError(
            f"Fields not in the export whitelist are silently ignored: {sorted(unknown)}. "
            f"Allowed: {sorted(OVERRIDE_FIELDS)}"
        )


def exists(qid: int) -> bool:
    return any(int(record["id"]) == int(qid) for record in bank())


def update(qid: int, fields: dict, _require_existing: bool = True) -> dict:
    """Append an override record. Later records win, so this is also an edit."""
    _check_fields(fields)
    if _require_existing and not exists(int(qid)):
        raise KeyError(
            f"No drill {qid} in the bank — an override for an id that does not exist "
            "is silently ignored by the export, so this would report success and "
            "change nothing. Check the id with drill_search."
        )
    record = {"id": int(qid), **fields}
    with open(paths.CURATED_OVERRIDES, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return {
        "id": int(qid),
        "layer": str(paths.CURATED_OVERRIDES.relative_to(paths.REPO)),
        "fields": sorted(fields),
        "next_step": "Run pipeline_export to rebuild the bank.",
    }


def retire(qid: int, reason: str = "") -> dict:
    """Stop serving a drill without renumbering anything behind it."""
    if not exists(int(qid)):
        raise KeyError(
            f"No drill {qid} in the bank. Retiring an id that does not exist yet "
            "would silently kill whichever drill is authored into that slot later, "
            "because ids are positional."
        )
    current = json.loads(paths.RETIRED_IDS.read_text()) if paths.RETIRED_IDS.exists() else {}
    if isinstance(current, list):
        payload, bucket = None, current
    else:
        payload, bucket = current, current.setdefault("ids", [])
    if int(qid) in bucket:
        return {"id": int(qid), "already_retired": True}
    bucket.append(int(qid))
    bucket.sort()
    if payload is not None and reason:
        payload.setdefault("notes", {})[str(qid)] = reason
    indent = paths.json_indent_of(paths.RETIRED_IDS)
    paths.RETIRED_IDS.write_text(
        json.dumps(payload if payload is not None else bucket, indent=indent) + "\n"
    )
    return {"id": int(qid), "retired_total": len(bucket),
            "file": str(paths.RETIRED_IDS.relative_to(paths.REPO))}
