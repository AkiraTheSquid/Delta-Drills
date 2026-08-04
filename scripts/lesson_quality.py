#!/usr/bin/env python3
"""Quality rules for worked examples and the problems they scaffold.

Every rule here came from a learner reading a real page and saying what was
wrong with it. That is why they are worth encoding: each one is a defect that
passed the structural validator, produced a page that looked finished, and was
only caught by someone working through it.

The rules
---------

INTRO       A worked example opens with prose, before any code. An example that
            starts with a fence never says what it is about to demonstrate, so
            the learner has to reverse-engineer the point from the code — which
            is the thing they do not know yet.

INTERLEAVE  Code arrives in short blocks with prose between them. One long
            uncommented block is the shape learners skip, and it is what the
            "one fence per segment" rule used to force.

PRINTS      A block that asserts must also print. `assert` is silent on success:
            the learner is shown a claim about a value they never see. Keep the
            assert — it is what makes the example self-checking — and print
            alongside it.

CASES       When a problem's answer changes with its input, one demonstrated
            case is misinformation. The learner reads a single expected output
            and takes it for THE answer, when the honest statement is "run these
            two checks". Detected from the bank: a question whose own test cases
            disagree on their expected output needs an example showing more than
            one case.

GIVEAWAY    The example must not hand over the problem's answer. Two ways it
            does: reproducing the expected output verbatim, or building from the
            same input literals so the solution transcribes. Passing a drill you
            copied is evidence of nothing, and the ladder promotes on it.

PROMPT_LEAK A problem must not name the answer in its own prompt. A prompt that
            says which dtype to compare against, or spells the call the learner
            is supposed to choose, is a reading exercise wearing a drill's
            clothes — and it is worst on the `solo` rung, which exists precisely
            to be unaided.

Scope
-----

`strict_for` decides which KPs these are ERRORS for. A KP that has written an
`## Applied practice` section has been through this pass; everything else is
legacy and is reported as a to-do (see audit_ladder_pairing.py) rather than
failing the build. That is deliberate: turning 63 pages red at once would mean
turning the rules off, and rules that are off catch nothing.
"""
from __future__ import annotations

import re

# A block longer than this stops being an example and becomes a program. The
# number is a judgement call, set from the pages that read well after the
# ndarray-model rewrite: those blocks run 6-12 lines.
MAX_FENCE_LINES = 16

# Below this there is nothing to interleave — a two-line block does not need
# prose in the middle of it.
MIN_LINES_TO_SPLIT = 8

_FENCE = re.compile(r"```([^\n]*)\n(.*?)```", re.S)
_NUM = re.compile(r"-?\d+\.?\d*")


def fences(text: str, info: str = "python"):
    """(code, start, end) for each fence with exactly this info string."""
    out = []
    for m in _FENCE.finditer(text or ""):
        if m.group(1).strip() == info:
            out.append((m.group(2), m.start(), m.end()))
    return out


def _prose_between(text: str, start: int, end: int) -> str:
    """Non-blank, non-fence text in a span. Comments inside code do not count —
    the whole point of INTERLEAVE is prose the learner reads OUTSIDE the block."""
    return (text or "")[start:end].strip()


def check_example_shape(example_md: str, label: str, info: str = "python") -> list[str]:
    """INTRO, INTERLEAVE and PRINTS for one worked example's markdown.

    `info` is the fence tag to read: segment examples use plain ```python,
    while guided and applied items carry theirs as ```python worked so the
    compiler can tell an example from a starter.
    """
    problems = []
    blocks = fences(example_md, info)
    if not blocks:
        return [f"{label}: no Python worked example"]

    if not _prose_between(example_md, 0, blocks[0][1]):
        problems.append(
            f"{label}: INTRO — the example opens with code. Say what it is "
            f"about to demonstrate, in prose, before the first block."
        )

    for i, (code, start, end) in enumerate(blocks):
        lines = [l for l in code.strip().splitlines() if l.strip()]
        n = len(lines)
        if n > MAX_FENCE_LINES:
            problems.append(
                f"{label}: INTERLEAVE — block {i + 1} is {n} lines "
                f"(max {MAX_FENCE_LINES}). Split it and explain between the pieces."
            )
        if i > 0:
            prev_end = blocks[i - 1][2]
            if not _prose_between(example_md, prev_end, start):
                problems.append(
                    f"{label}: INTERLEAVE — blocks {i} and {i + 1} are adjacent "
                    f"with no prose between them."
                )
        if "assert" in code and "print(" not in code:
            problems.append(
                f"{label}: PRINTS — block {i + 1} asserts but never prints. "
                f"assert is silent on success; show the value too."
            )

    if len(blocks) == 1 and len(
        [l for l in blocks[0][0].strip().splitlines() if l.strip()]
    ) >= MIN_LINES_TO_SPLIT:
        problems.append(
            f"{label}: INTERLEAVE — one block carries the whole example. "
            f"Alternate prose and short blocks."
        )
    return problems


def _expected_outputs(question: dict) -> list[str]:
    """Every expected output the bank records for a question."""
    out = []
    for case in question.get("test_cases") or []:
        if isinstance(case, dict):
            # `expected_expr` is the bank's actual field name; the others are
            # older shapes still present on a few rows.
            for key in ("expected_expr", "expected_output", "expected", "output"):
                if case.get(key) is not None:
                    out.append(str(case[key]).strip())
                    break
    if question.get("expected_output"):
        out.append(str(question["expected_output"]).strip())
    return [o for o in out if o]


def answer_varies(question: dict) -> bool:
    """Does this question's answer depend on which input it is given?

    Read off the bank's own test cases rather than guessed: if two cases expect
    different outputs, then no single expected output describes the question,
    and an example that demonstrates one case is teaching a wrong invariant.
    """
    return len(set(_expected_outputs(question))) > 1


def check_pairing(example_md: str, question: dict, label: str,
                  info: str = "python") -> list[str]:
    """CASES and GIVEAWAY for one example/problem pair."""
    problems = []
    code = "\n".join(c for c, _, _ in fences(example_md, info))
    if not code:
        return problems

    if answer_varies(question):
        # Count demonstrated cases by how many times the example checks or shows
        # a result. One assert/print pair is one case.
        shown = len(re.findall(r"^\s*(?:assert|print)\b", code, re.M))
        if shown < 2:
            problems.append(
                f"{label}: CASES — this problem's answer changes with its input, "
                f"but the example demonstrates {shown} case(s). Show at least two, "
                f"with their different results."
            )

    flat_code = " ".join(code.split())

    # Reusing the graded input is transcription however many cases are shown:
    # the learner can read the answer off the example and type it back.
    starter = str(question.get("starter_code") or "")
    for literal in _input_literals(question, starter):
        if len(literal) >= 8 and " ".join(literal.split()) in flat_code:
            problems.append(
                f"{label}: GIVEAWAY — the example is run on {literal[:40]!r}, which "
                f"is one of the inputs this problem is graded on. Use different data."
            )
            break

    # Showing an expected output is only a giveaway when the example demonstrates
    # ONE case. An example that shows two inputs producing two different results
    # is teaching the variation — which is what CASES asks for — and the numbers
    # it prints are not "the answer" to anything.
    if not _demonstrates_variation(code):
        for expected in set(_expected_outputs(question)):
            flat = " ".join(expected.split())
            if len(flat) >= 4 and flat in flat_code:
                problems.append(
                    f"{label}: GIVEAWAY — the example shows one case and it is the "
                    f"problem's expected output ({flat[:40]!r}). Either change the "
                    f"data or demonstrate more than one case."
                )
                break
    return problems


def _demonstrates_variation(code: str) -> bool:
    """Does the example show at least two DIFFERENT results?

    Two asserts against the same value is one case repeated. Comparing the
    right-hand sides is what distinguishes "here are two inputs and their two
    answers" from "here is the answer, twice".
    """
    shown = re.findall(r"^\s*assert\s+.*?==\s*(.+?)\s*$", code, re.M)
    return len({" ".join(s.split()) for s in shown}) >= 2


def _input_literals(question: dict, starter: str) -> list[str]:
    """The literal input expressions this question is actually run on."""
    out = []
    for source in (starter, str(question.get("answer_code") or "")):
        for m in re.finditer(r"^\s*example\s*=\s*(.+)$", source, re.M):
            out.append(m.group(1).strip())
    for case in question.get("test_cases") or []:
        if isinstance(case, dict) and case.get("setup_code"):
            for m in re.finditer(r"^\s*\w+\s*=\s*(\[.+?\])\s*$", str(case["setup_code"]), re.M):
                out.append(m.group(1).strip())
    return out


def _example_input(question: dict, starter: str) -> str:
    """The literal inputs the problem is actually run on.

    Both the `example = ...` line the starter prints and the graded test cases'
    setup, because either one appearing in the example makes the drill a copy.
    """
    parts = []
    for source in (starter, str(question.get("answer_code") or "")):
        m = re.search(r"^\s*example\s*=\s*(.+)$", source, re.M)
        if m:
            parts.append(m.group(1))
    for case in question.get("test_cases") or []:
        if isinstance(case, dict) and case.get("setup_code"):
            parts.append(str(case["setup_code"]))
            break  # the first case is the one a learner would eyeball
    return "\n".join(parts)


def check_prompt_leak(question: dict, rung: str, label: str) -> list[str]:
    """PROMPT_LEAK — the prompt must not name what the learner has to choose.

    Only the symbols the ANSWER needs and the starter does not already show
    count: a prompt is allowed to name what it hands you. Reported for the
    unaided rungs, where the leak is the whole difference between a drill and a
    reading comprehension question.
    """
    if rung not in ("independent", "solo", "applied"):
        return []
    text = str(question.get("question_text") or "")
    answer = str(question.get("answer_code") or "")
    starter = str(question.get("starter_code") or "")
    leaked = []
    for sym in sorted(set(re.findall(r"\b(?:t|torch)\.([A-Za-z_][A-Za-z0-9_]*)", answer))):
        # Exempt only when the starter actually CALLS it. A substring test
        # exempts a starter whose docstring merely names the thing — which is
        # itself a leak, and was why this rule silently passed the worst
        # offender it exists to catch.
        if re.search(rf"\b(?:t|torch)\.{re.escape(sym)}\b", starter):
            continue
        if re.search(rf"\b(?:t|torch)\.{re.escape(sym)}\b", text):
            leaked.append(f"torch.{sym}")
    if leaked:
        return [
            f"{label}: PROMPT_LEAK — the prompt names {', '.join(leaked)}, which "
            f"is what the learner is supposed to choose. Describe the goal instead."
        ]
    return []


def strict_for(kp: dict) -> bool:
    """Is this KP held to these rules as errors?

    Opting in by having written `## Applied practice` — the section that only
    exists on pages taken through this pass — is what lets the rules be strict
    without failing every legacy page on the day they land.
    """
    return bool((kp.get("sections") or {}).get("Applied practice", "").strip())
