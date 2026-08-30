"""watch.py — health checks for retired-content-2026-08-30

A retirement has two halves: the KC leaves the registry, and the page comes
here. Doing only one is silent — the graph either points at a page nobody can
reach, or serves a concept whose page is gone. These checks assert the halves
agree.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
REGISTRY = os.path.join(REPO, 'Local_Deployed_Shared', 'lessons', 'kc_registry.json')
LIVE_LESSONS = os.path.join(REPO, 'Local_Deployed_Shared', 'lessons')
MANIFEST = os.path.join(
    REPO, 'Local_Deployed_Shared', 'pipeline', 'retired_question_ids.json')


def _pages():
    return sorted(f for f in os.listdir(HERE) if f.startswith('kp-') and f.endswith('.md'))


def _kc_of(path):
    head = open(path, encoding='utf-8').read(2000)
    m = re.search(r'^kc:\s*(\S+)', head, re.M)
    return m.group(1) if m else None


def check_imports():
    """The archive is not empty and every page still declares what it taught."""
    pages = _pages()
    assert pages, 'no kp-*.md here — an archive with nothing in it is a lie'
    missing = [p for p in pages if not _kc_of(os.path.join(HERE, p))]
    assert not missing, f'archived pages with no kc: in frontmatter: {missing}'


def check_data_dependencies():
    """The manifest and the pages here describe the SAME retirement.

    Checking only the pages that happen to exist cannot see the failure that
    actually costs something: a concept named as retired whose page was never
    moved. It is then in neither tree — gone from the registry and gone from
    the archive — and the only record of what it taught is the git history.
    """
    manifest = json.load(open(MANIFEST, encoding='utf-8'))
    declared = set(manifest.get('retired_kcs') or [])
    assert declared, (
        'retired_question_ids.json declares no retired_kcs — the manifest and '
        'this archive are the two halves of one record; a manifest that names '
        'no concepts cannot cross-check anything'
    )
    archived = {kc for kc in (_kc_of(os.path.join(HERE, p)) for p in _pages()) if kc}
    lost = sorted(declared - archived)
    assert not lost, (
        'retired but not archived: ' + ', '.join(lost)
        + '. These concepts left kc_registry.json with no page filed here, so '
        'nothing but git history says what they taught. Move the page in, or '
        'drop the id from retired_kcs.'
    )
    stray = sorted(archived - declared)
    assert not stray, (
        'archived but not in the manifest: ' + ', '.join(stray)
        + '. Add them to retired_kcs so the retirement has one list, or move '
        'the page back to Local_Deployed_Shared/lessons/.'
    )


def check_public_api():
    """Nothing here may be a LIVE concept. That is the whole contract."""
    registry = json.load(open(REGISTRY, encoding='utf-8'))
    live = {kc['id'] for kc in registry['kcs']}
    resurrected = sorted(
        kc for kc in (_kc_of(os.path.join(HERE, p)) for p in _pages()) if kc in live
    )
    assert not resurrected, (
        'these concepts are in kc_registry.json but their page is archived: '
        + ', '.join(resurrected)
        + '. The graph would serve a concept whose page nothing can load. Move the '
        'page back to Local_Deployed_Shared/lessons/, or drop the KC again.'
    )


def check_invariants():
    """A page is here OR live, never both, and never edited in place."""
    archived = {p for p in _pages()}
    live = set()
    for topic in sorted(os.listdir(LIVE_LESSONS)):
        d = os.path.join(LIVE_LESSONS, topic)
        if os.path.isdir(d):
            live |= {f for f in os.listdir(d) if f.startswith('kp-') and f.endswith('.md')}
    both = sorted(archived & live)
    assert not both, (
        'these pages exist in BOTH the archive and the live lessons: ' + ', '.join(both)
        + '. One of them is a stale copy, and the compiler reads the live one — so '
        'the archive would quietly stop being the record of what was retired.'
    )


if __name__ == '__main__':
    checks = [check_imports, check_data_dependencies,
              check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
