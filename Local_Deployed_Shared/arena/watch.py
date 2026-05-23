"""watch.py — health checks for arena

This folder is JavaScript (loaded as plain <script> tags, not ES modules),
so we use file-existence + text-grep checks rather than Python imports.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

JS_FILES = ['manifest.js', 'stage1.js', 'exercises.js']

# Curriculum spec must declare these chapter keys, and each must have a
# matching ARENA_CHAPTER_DEFAULTS entry — otherwise buildArenaProblem
# crashes at script load.
EXPECTED_CHAPTERS = [
    'chapter0_fundamentals',
    'chapter1_transformer_interp',
    'chapter2_rl',
    'chapter3_llm_evals',
    'chapter4_alignment_science',
]

# Public contract — every problem object must have these fields, since
# both stage1.js (ARENA tab) and stats/predicted.js read them.
REQUIRED_PROBLEM_FIELDS = [
    'id', 'chapterId', 'chapterLabel', 'sectionLabel', 'title', 'summary',
    'readinessScore', 'readinessLabel', 'readinessNote', 'prerequisiteTags',
    'skillWeights', 'lessonPath', 'notebookPath', 'backupNotebookPath',
    'launchPath', 'executionMode',
]


def _read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ── Import checks ──────────────────────────────
def check_imports():
    for name in JS_FILES:
        path = os.path.join(HERE, name)
        assert os.path.isfile(path), f"missing JS file: {name}"
        assert os.path.getsize(path) > 0, f"empty JS file: {name}"


# ── Public API checks ─────────────────────────
def check_public_api():
    manifest = _read(os.path.join(HERE, 'manifest.js'))
    assert 'window.ARENA_STAGE1_PROBLEMS' in manifest, \
        "manifest.js must expose window.ARENA_STAGE1_PROBLEMS"
    assert 'ARENA_CURRICULUM' in manifest, \
        "manifest.js must define ARENA_CURRICULUM"
    assert 'ARENA_CHAPTER_DEFAULTS' in manifest, \
        "manifest.js must define ARENA_CHAPTER_DEFAULTS"
    assert 'buildArenaProblem' in manifest, \
        "manifest.js must define buildArenaProblem"

    # NOTE: stage1.js is kept on disk but NOT loaded by index.html anymore —
    # the dedicated ARENA tab was removed 2026-05-16. These assertions exist
    # so the file's contract stays internally consistent in case it's
    # re-attached later. See README "Recent Changes" 2026-05-16.
    stage1 = _read(os.path.join(HERE, 'stage1.js'))
    assert 'window.ARENA_STAGE1_PROBLEMS' in stage1, \
        "stage1.js must consume window.ARENA_STAGE1_PROBLEMS"
    assert 'ARENA_EXERCISES_BY_NOTEBOOK' in stage1, \
        "stage1.js must consume window.ARENA_EXERCISES_BY_NOTEBOOK"
    assert 'computeArenaReadiness' in stage1, \
        "stage1.js must consume window.computeArenaReadiness for per-exercise pills"

    exercises = _read(os.path.join(HERE, 'exercises.js'))
    assert 'window.ARENA_EXERCISES_BY_NOTEBOOK' in exercises, \
        "exercises.js must expose window.ARENA_EXERCISES_BY_NOTEBOOK"

    assert 'window.computeArenaReadiness' in manifest, \
        "manifest.js must expose window.computeArenaReadiness for predicted-scores + arena-tab consumers"
    assert 'ARENA_SKILL_TO_TOPIC_ALIASES' in manifest, \
        "manifest.js must define ARENA_SKILL_TO_TOPIC_ALIASES (skill→topic mapping for readiness compute)"

    # Every required problem field must be referenced somewhere in
    # the manifest (either set by the builder or the rich-overrides).
    for field in REQUIRED_PROBLEM_FIELDS:
        assert field in manifest, \
            f"manifest.js: required problem field `{field}` not found"


# ── Invariant checks ──────────────────────────
def check_invariants():
    manifest = _read(os.path.join(HERE, 'manifest.js'))

    # Every chapter referenced in ARENA_CURRICULUM must have a matching
    # ARENA_CHAPTER_DEFAULTS entry, otherwise buildArenaProblem crashes.
    chapters_in_curriculum = set(re.findall(r'chapter:\s*"([^"]+)"', manifest))
    chapters_in_defaults = set(
        re.findall(r'(chapter\d+_\w+):\s*\{', manifest)
    )
    missing = chapters_in_curriculum - chapters_in_defaults
    assert not missing, \
        f"chapters used in ARENA_CURRICULUM with no ARENA_CHAPTER_DEFAULTS entry: {missing}"

    # All four canonical ARENA chapters must be represented.
    for ch in EXPECTED_CHAPTERS:
        assert ch in chapters_in_curriculum, \
            f"expected chapter `{ch}` missing from ARENA_CURRICULUM"

    # Every curriculum entry must declare id/section/title/notebookPath/lessonPath.
    entries = re.findall(r'\{\s*id:\s*"([^"]+)"', manifest)
    assert len(entries) == len(set(entries)), \
        f"duplicate ids in ARENA_CURRICULUM: {[e for e in entries if entries.count(e) > 1]}"
    assert len(entries) >= 32, \
        f"expected at least 32 curriculum entries, found {len(entries)}"

    # ── exercises.js (auto-generated catalog) invariants ──────────
    # The catalog is the source of truth for the Targeted Practice search.
    # We can't import JS, but it's a JSON-ish literal — strip the JS prologue
    # and parse the assignment value as JSON to validate shape.
    import json
    exercises_raw = _read(os.path.join(HERE, 'exercises.js'))

    # Line 1 must still carry the auto-generator marker. Manual edits would
    # silently drift from the script's output — targeted-practice catalog
    # would still work, but the next deploy would clobber the edits.
    first_line = exercises_raw.splitlines()[0]
    assert 'AUTO-GENERATED' in first_line, (
        "exercises.js line 1 lost the AUTO-GENERATED marker — someone may "
        "have hand-edited this file. Re-run extract_arena_exercises.py instead."
    )

    # Pull the object literal after `window.ARENA_EXERCISES_BY_NOTEBOOK = `.
    body_match = re.search(
        r'window\.ARENA_EXERCISES_BY_NOTEBOOK\s*=\s*(\{.*?\});\s*$',
        exercises_raw, re.DOTALL,
    )
    assert body_match, "exercises.js: couldn't locate ARENA_EXERCISES_BY_NOTEBOOK literal"
    catalog = json.loads(body_match.group(1))
    assert isinstance(catalog, dict) and catalog, "ARENA_EXERCISES_BY_NOTEBOOK is empty"

    # Every value must be a list; every entry inside must be a dict with
    # exactly the {title, anchor} keys the targeted-practice flattener
    # expects (an extra key is fine; missing keys break the search).
    bad_entries = []
    for nbpath, exs in catalog.items():
        if not isinstance(exs, list):
            bad_entries.append(f"{nbpath}: value is not a list")
            continue
        for i, ex in enumerate(exs):
            if not isinstance(ex, dict) or 'title' not in ex or 'anchor' not in ex:
                bad_entries.append(f"{nbpath}[{i}]: missing title/anchor")
    assert not bad_entries, "exercises.js shape violations: " + "; ".join(bad_entries[:5])

    # Every notebookPath in the catalog should correspond to a problem in
    # ARENA_CURRICULUM — otherwise the targeted-practice readiness lookup
    # (which keys problemsByPath by notebookPath) returns null and shows
    # an empty bar for that exercise.
    curriculum_paths = set(re.findall(r'notebookPath:\s*"([^"]+)"', manifest))
    orphan_paths = [p for p in catalog.keys() if p not in curriculum_paths]
    assert not orphan_paths, (
        "exercises.js notebooks with no matching ARENA_CURRICULUM entry "
        f"(targeted-practice readiness will be null): {orphan_paths[:3]}"
    )


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
