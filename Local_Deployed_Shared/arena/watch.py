"""watch.py — health checks for arena

This folder is JavaScript (loaded as plain <script> tags, not ES modules),
so we use file-existence + text-grep checks rather than Python imports.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

JS_FILES = ['manifest.js', 'stage1.js']

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

    stage1 = _read(os.path.join(HERE, 'stage1.js'))
    assert 'window.ARENA_STAGE1_PROBLEMS' in stage1, \
        "stage1.js must consume window.ARENA_STAGE1_PROBLEMS"

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


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
