"""watch.py — health checks for concept-graph

Guards the route from a graded answer to a coloured bubble. Every failure this
folder has had was silent: the graph kept drawing 63 bubbles, in plausible
colours, that no amount of practice moved. Nothing threw, nothing 500'd, and
the numbers were priors.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def check_crosswalk_is_present_and_tiered():
    """The join between concept ids and the atoms belief is held under.

    Everything downstream reads as "this learner has done nothing" without it,
    on both sides: `kc_graph.kc_mastery` returns the bare prior, and the
    browser's own read declines every concept.
    """
    path = os.path.join(HERE, 'kc_atom_crosswalk.json')
    assert os.path.exists(path), (
        'kc_atom_crosswalk.json is missing — every concept will report the '
        'prior. Regenerate: This-Directory-Only/scripts/export_kc_atom_crosswalk.py'
    )
    kcs = (json.loads(_read(path)) or {}).get('kcs') or {}
    assert len(kcs) >= 60, f'crosswalk covers only {len(kcs)} KCs — expected all 63'
    tiers = {row.get('tier') for row in kcs.values()}
    assert 'measured' in tiers, (
        'no KC is in the `measured` tier — with every concept a topic proxy, '
        'nothing in the graph is ever a per-concept measurement'
    )


def check_the_backend_image_carries_the_crosswalk():
    """The Fly image must COPY it, and .dockerignore must let it through.

    This is the 2026-08-06 bug, verbatim: the folder is gitignored, the
    Dockerfile copied `lessons/` but not this, and production served
    `tier: unmapped` for all 63 concepts for a month. The only symptom was one
    WARNING line. Both halves are asserted because either one alone silently
    drops the file back out of the image.
    """
    dockerfile = _read(os.path.join(REPO, 'This-Directory-Only', 'Dockerfile'))
    assert 'concept-graph/kc_atom_crosswalk.json' in dockerfile, (
        'the Dockerfile no longer COPYs kc_atom_crosswalk.json — the backend '
        'lattice will report every concept unmapped in production while '
        'passing every test locally'
    )
    ignore = _read(os.path.join(REPO, '.dockerignore'))
    assert '!Local_Deployed_Shared/concept-graph/kc_atom_crosswalk.json' in ignore, (
        '.dockerignore excludes everything by default; without the ! line the '
        'COPY above fails the build or ships nothing'
    )


def check_the_graph_reads_the_live_learner_model():
    """Not `localStorage`, which backend mode never writes.

    `saveAdaptiveState` is reached only by the Pyodide paths, so for a
    signed-in learner the persisted key is absent — or holds a stale
    `adaptive_state_guest` blob from before they signed in, which is worse: it
    reports month-old numbers as current.
    """
    src = _read(os.path.join(HERE, 'lesson-graph.js'))
    readiness = src[src.index('const kcReadinessInfo'):src.index('const kcReadiness =')]
    assert '_learnerState()' in readiness, (
        'kcReadinessInfo must read the freshest state (_learnerState), not the '
        'persisted copy — backend mode never writes localStorage'
    )
    assert '_persistedState()' not in readiness, (
        'kcReadinessInfo is back on the persisted state; a signed-in learner '
        "will see a stale guest snapshot or nothing"
    )


def check_the_server_reading_is_preferred():
    """The colour and the gate must come from the same number.

    `/api/practice/kc-lattice` is `kc_graph.kc_report` — the code that decides
    what practice serves next. Recomputing mastery in the browser instead lets
    a bubble disagree with the queue that draws its ring.
    """
    src = _read(os.path.join(HERE, 'lesson-graph.js'))
    assert 'window.kcLatticeReadiness(kc, lattice, s, _decay)' in src, (
        'lesson-graph.js no longer routes the concept read through '
        'kc_lattice_read.js — the server lattice would go unread'
    )
    assert 'window.kcLatticeNote(data)' in src, (
        'the lattice response must pass through kcLatticeNote, or a lattice '
        'that can measure nothing will be drawn as if it could'
    )
    index = _read(os.path.join(REPO, 'Local_Deployed_Shared', 'index.html'))
    assert index.index('kc_lattice_read.js') < index.index('lesson-graph.js'), (
        'kc_lattice_read.js must load before lesson-graph.js'
    )


def check_the_measured_tier_is_never_widened():
    """A topic proxy must not be shown as a concept's own measurement.

    40 of the 63 concepts sit on atoms coarser than themselves. Reading one is
    reporting the topic's number under the concept's name, for a dozen
    siblings at once — the exact overclaim the tiering exists to prevent.
    """
    src = _read(os.path.join(HERE, 'kc_lattice_read.js'))
    assert 'row.tier === "measured"' in src, (
        'kc_lattice_read.js must gate the server reading on the measured tier'
    )
    assert 'row.evidenced' in src, (
        'the server\'s covered-weight test must gate the reading too, or one '
        'lightly-weighted atom stands in for a whole concept'
    )


if __name__ == '__main__':
    checks = [
        check_crosswalk_is_present_and_tiered,
        check_the_backend_image_carries_the_crosswalk,
        check_the_graph_reads_the_live_learner_model,
        check_the_server_reading_is_preferred,
        check_the_measured_tier_is_never_widened,
    ]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
