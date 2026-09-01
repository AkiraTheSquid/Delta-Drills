#!/usr/bin/env python3
"""Compile KP markdown sources into Local_Deployed_Shared/lessons/lessons_structured.json.

Usage: python3 scripts/compile_lessons.py
Run scripts/validate_lessons.py first (or with --gate) — compilation does not re-check code.
"""
import json
import re
import sys
from lesson_lib import (
    LESSONS_DIR,
    all_kp_paths,
    attach_example_run,
    blank_new_syntax,
    code_fences,
    derive_faded_starter,
    load_bank,
    load_registry,
    parse_kp,
    split_items,
)


NOTES_DIR = LESSONS_DIR / "notes"
_NOTE_RE = re.compile(r"\A---\s*\nkc:\s*(\S+)\s*\n---\s*\n?(.*)\Z", re.S)


def load_notes():
    """The metadata layer: notes/<kc>.md bodies, keyed by KC id.

    Filename and front matter must agree, and every note must name a concept
    the registry knows — enforced with a hard exit in compile_lessons(),
    because the failure this guards against is silent: a KC rename that
    leaves its note behind would otherwise just stop showing the note, and
    nobody reads what stopped being shown."""
    notes = {}
    if not NOTES_DIR.is_dir():
        return notes
    for path in sorted(NOTES_DIR.glob("*.md")):
        if path.name == "README.md":
            continue
        m = _NOTE_RE.match(path.read_text())
        if not m:
            sys.exit(f"notes/{path.name}: missing `kc:` front matter — every note names its concept")
        kc, body = m.group(1), m.group(2).strip()
        if path.stem != kc:
            sys.exit(f"notes/{path.name}: filename disagrees with its front matter kc ({kc})")
        notes[kc] = body
    return notes


_ID_UNSAFE = re.compile(r"[^a-z0-9]+")


def _concept_id(authored, index, title):
    """A stable id for one concept of a KP.

    This is what the learner's exposure map is keyed on — `<kc>#<concept_id>`
    is how the gate remembers that concept 2 of 3 has been taught and concept 3
    has not — so it has to survive a recompile unchanged. Four KPs name their
    concepts in frontmatter (`concepts: [...]`); the other 27 segmented ones do
    not, and waiting for them to be annotated would leave the loop switched off
    for almost every KP it exists for.

    So: the authored id when there is one, otherwise position plus title slug.
    The position makes collisions impossible (two segments may legitimately
    share a title) and the slug makes the id readable in a stored map that
    someone will one day have to debug. Re-titling or reordering a segment
    mints a new id and re-teaches that one concept once, which is the right
    failure: the alternative is a bare index that silently credits a rewritten
    concept as already read.
    """
    if index < len(authored) and str(authored[index]).strip():
        return str(authored[index]).strip()
    return f"s{index}-{_ID_UNSAFE.sub('-', title.lower()).strip('-')[:48]}"


def _derived_faded(qid, new_syntax, bank):
    """A faded starter built from a question's own canonical answer, or "".

    The answer with every NEW symbol blanked is a completion problem by
    construction: the structure is the real solution's, and the only thing
    missing is the move being taught. The question's demo block goes back on
    the end so the learner can run it and read the output.

    Returns "" when nothing was blanked — that case is not a scaffold, it is
    the ANSWER, and handing it over would be the worst possible version of this
    rung.
    """
    exercise = (bank.get(qid) or {}).get("exercise") or {}
    # The structured bank calls it `canonical_solution`; the flat export
    # calls it `answer_code`. Read both — a silent "" here disables the
    # whole derivation and looks exactly like "no drill needed one".
    answer = exercise.get("answer_code") or exercise.get("canonical_solution") or ""
    fn = exercise.get("function_name") or "solve"
    blanked = derive_faded_starter(answer, fn, new_syntax)
    if not blanked:
        return ""
    return attach_example_run(blanked, exercise.get("starter_code") or "", fn)


def compile_lessons():
    registry = load_registry()
    bank = load_bank()
    notes = load_notes()
    kc_by_id = {kc["id"]: kc for kc in registry["kcs"]}
    lessons = {l["id"]: {**l, "kps": []} for l in registry["lessons"]}

    for path in all_kp_paths():
        kp = parse_kp(path)
        kc = kc_by_id[kp["kc"]]

        def _faded_items(section_text):
            items = []
            for qid, content in split_items(section_text).items():
                starters = code_fences(content, "python starter")
                solutions = code_fences(content, "python solution")
                prose = content.split("```", 1)[0].strip()
                # The authored starter is the function alone. Its question's own
                # starter ends with a fixture and a print, and without that the
                # learner has nothing to compare the expected output against —
                # so graft it on here, once, for every route that serves this
                # record (see lesson_lib.attach_example_run).
                exercise = (bank.get(qid) or {}).get("exercise") or {}
                fn = exercise.get("function_name") or "solve"
                # Hide the concept, keep the scaffold. An authored starter that
                # blanks only the ARGUMENT prints the call the drill exists to
                # test — q67's `z.clamp(_____=0.0)` on a KP whose whole subject
                # is `clamp`. See lesson_lib.blank_new_syntax.
                starter = blank_new_syntax(
                    starters[0] if starters else "", kp["new_syntax"], fn
                )
                items.append({
                    "question_id": qid,
                    "prompt": prose,
                    "starter_code": attach_example_run(
                        starter, exercise.get("starter_code") or "", fn
                    ),
                    "solution": solutions[0] if solutions else "",
                })
            return items

        # One page per single-concept segment (the in-app player steps
        # through these); legacy aggregate fields kept for viewer.html.
        segments = [{
            "concept_id": _concept_id(kp["concepts"], i, seg["title"]),
            "title": seg["title"],
            "concept_markdown": seg["concept"],
            "watch_out_markdown": seg["watch_out"],
            "worked_example_markdown": seg["worked"],
            "worked_example_code": code_fences(seg["worked"], "python")[0],
            "faded_items": _faded_items(seg["faded"]),
        } for i, seg in enumerate(kp["segments"])]
        faded_items = [item for seg in segments for item in seg["faded_items"]]
        guided_items = []
        for qid, content in split_items(kp["sections"].get("Guided practice", "")).items():
            # A guided drill is served at the same ladder stage as a faded one
            # (kc_graph._STAGE_TO_RANKS puts both on the supported rungs), so it
            # needs the same thing above it: a solved example of the move, not
            # only hints hidden behind a <details>. An optional ```python worked
            # fence carries it. It is split OUT of the hints so the example is
            # visible without opening anything — the whole point is that the
            # learner reads it first.
            worked = code_fences(content, "python worked")
            hints = re.sub(
                r"```python worked\n.*?```\n?", "", content, flags=re.S
            ).strip()
            guided_items.append({
                "question_id": qid,
                "hints_markdown": hints,
                "worked_example_code": worked[0] if worked else "",
                # A guided drill is served on the SAME rung as a faded one
                # (kc_graph._STAGE_TO_RANKS gives `faded` ranks 0 and 1), and
                # the rung's whole promise is that most of the solution is
                # written and you supply the rest. Nothing wrote one: guided
                # items carry hints, the backend's mechanical backward fade
                # gives up on a one-statement body, and the learner got q487 —
                # a bare `def solve(x)` under a strip reading "Faded". Derive
                # one by blanking the canonical answer's new syntax, which is
                # the same rule the authored starters are now held to.
                "starter_code": _derived_faded(qid, kp["new_syntax"], bank),
            })
        # Applied practice is the THIRD rung of the ladder: the drill is an
        # independent one — no blanks, the learner writes the whole function —
        # but an example of the same move sits above it. That is what separates
        # it from `solo`, where the same kind of drill arrives with nothing to
        # read first. The question ids stay in the `independent` frontmatter
        # list, so the rung the backend derives is unchanged; what this section
        # adds is the example, and having one is what routes the drill to
        # `partial` rather than `solo` (see kc_graph.questions_at_stage).
        applied_items = []
        for qid, content in split_items(kp["sections"].get("Applied practice", "")).items():
            worked = code_fences(content, "python worked")
            applied_items.append({
                "question_id": qid,
                "prompt": content.split("```", 1)[0].strip(),
                "worked_example_code": worked[0] if worked else "",
            })
        # Solo practice — the ladder's THIRD rung. The learner writes the whole
        # function unaided; an item may carry a ```python worked``` fence, and
        # when it does that example is shown ABOVE the problem the way the
        # lesson's is. Most items do not carry one, and that is the fade: the
        # queue serves the example-bearing items of a rung first (see
        # kc_graph.questions_at_stage), so examples thin out on their own as the
        # learner works through the rung rather than stopping on a cutoff.
        #
        # Supersedes `## Applied practice`, which is still parsed for the 62
        # KPs that have not been rewritten.
        solo_items = []
        for qid, content in split_items(kp["sections"].get("Solo practice", "")).items():
            worked = code_fences(content, "python worked")
            solo_items.append({
                "question_id": qid,
                "prompt": content.split("```", 1)[0].strip(),
                "worked_example_code": worked[0] if worked else "",
            })
        # Integrated practice — the FOURTH rung. Whole-KP problems, and never an
        # example: this is the rung the examples have faded out of entirely.
        integrated_items = []
        for qid, content in split_items(kp["sections"].get("Integrated practice", "")).items():
            integrated_items.append({
                "question_id": qid,
                "prompt": content.split("```", 1)[0].strip(),
            })
        lessons[kc["lesson"]]["kps"].append({
            "kc": kp["kc"],
            "title": kp["title"] or kc["title"],
            "supporting_kcs": kp["supporting"],
            "new_syntax": kp["new_syntax"],
            "concept_markdown": kp["sections"].get("Concept", ""),
            "worked_example_markdown": kp["sections"].get("Worked example", ""),
            "segments": segments,
            "faded_items": faded_items,
            "guided_items": guided_items,
            "applied_items": applied_items,
            "solo_items": solo_items,
            "integrated_items": integrated_items,
            "independent_items": kp["independent"],
            "misconceptions_markdown": kp["sections"].get("Misconceptions", ""),
            # The metadata layer (notes/<kc>.md) — global audit findings and
            # decision records, rendered by the advanced-mode Metadata tab.
            "notes_markdown": notes.get(kp["kc"], ""),
        })

    # Order KPs within each lesson by registry order (authoring source of truth).
    kc_order = {kc["id"]: i for i, kc in enumerate(registry["kcs"])}
    for lesson in lessons.values():
        lesson["kps"].sort(key=lambda kp: kc_order[kp["kc"]])

    known = {kp["kc"] for lesson in lessons.values() for kp in lesson["kps"]}
    orphans = sorted(set(notes) - known)
    if orphans:
        sys.exit(f"orphan notes for unknown KCs: {orphans} — a renamed concept must take its note with it")

    out = {
        "version": registry["version"],
        "lessons": [lessons[l["id"]] for l in registry["lessons"]],
    }
    out_path = LESSONS_DIR / "lessons_structured.json"
    out_path.write_text(json.dumps(out, indent=1))
    n_kps = sum(len(l["kps"]) for l in lessons.values())
    print(f"compiled {n_kps} KPs across {len(lessons)} lessons -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(compile_lessons())
