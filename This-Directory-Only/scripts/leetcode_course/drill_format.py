"""Per-drill presentation for the LeetCode export: the answer clock and the
statement's markdown.

Clock. A LeetCode problem is timed by its own size, not a flat table row: the
practice path clamps concept clocks to 5:00 (question_pick.PRACTICE_MAX_SECS),
which no LeetCode medium fits. Each drill's `secs_allowed` is
    base(LeetCode difficulty) x pattern weight x TypeSafe spread (+ design bump)
rounded to the minute and kept in [MIN_SECS, MAX_SECS]. The backend serves it
as the question's own clock (leetcode_questions.clock_secs); the learner's
multiplier scales it like any other.

Markdown. LeetCodeDataset statements are plain text ("Example 1:", "Input: ...",
"Constraints:" followed by one constraint per line). The prompt renderer
(practice/prompt-markdown.js) reads markdown, so those markers become headings,
bold labels and a list. Already-markdown statements (the generated problems)
pass through: none of the patterns match `### Example 1` or `**Input:**`.
"""
from __future__ import annotations

import re

# Minutes a solver needs for a problem of this LeetCode difficulty, before the
# pattern weight. Interview-pace numbers: an easy in 15, a medium in 25, a hard in 40.
BASE_MINUTES = {"Easy": 15, "Medium": 25, "Hard": 40}
# Patterns whose problems take longer (state design, search spaces) or shorter
# (one pass over an array) than the difficulty label alone says.
HEAVY = {"graphs", "topological-sort", "union-find", "shortest-paths", "mst", "backtracking", "trie",
         "two-heaps", "k-way-merge", "dp-1d", "knapsack-dp", "dp-grid", "dp-strings", "dp-lis", "dp-intervals"}
LIGHT = {"arrays-strings", "hash-maps", "prefix-sums", "two-pointers", "stack", "math", "bit-manipulation"}
PATTERN_WEIGHT = {"heavy": 1.2, "light": 0.85, "plain": 1.0}
DESIGN_BUMP_MINUTES = 5  # a class to design (LRU Cache, MyStack) is several methods, not one
MIN_SECS, MAX_SECS = 5 * 60, 60 * 60


def lc_difficulty(p: dict) -> str:
    """LeetCode's own label; problems without one (Striver / generated rows)
    are placed by their TypeSafe score, which is on the same 0-100 scale."""
    label = p.get("difficulty_label") or ""
    if label in BASE_MINUTES:
        return label
    score = int(p.get("difficulty_score", 50))
    return "Easy" if score < 40 else "Medium" if score < 65 else "Hard"


def clock_secs(p: dict) -> int:
    pattern = p["concept"].split(".", 1)[-1]
    weight = PATTERN_WEIGHT["heavy" if pattern in HEAVY else "light" if pattern in LIGHT else "plain"]
    # TypeSafe ranks problems within a concept: +-15% around the label's base.
    spread = 0.85 + 0.3 * int(p.get("difficulty_score", 50)) / 100
    minutes = BASE_MINUTES[lc_difficulty(p)] * weight * spread
    if "class Solution" not in p["starter_code"]:
        minutes += DESIGN_BUMP_MINUTES
    return max(MIN_SECS, min(MAX_SECS, round(minutes) * 60))


_HEADING = re.compile(r"^(Example \d+|Constraints|Note|Notes):\s*$")
_LABEL = re.compile(r"^(Input|Output|Explanation|Note|Follow[- ]up)\s*:\s*")
_FORMULA = re.compile(r"<=|>=|<|>|\*|\^|==")
# The dataset scraped LeetCode's superscripts flat: 10<sup>5</sup> -> "105",
# 2<sup>31</sup> -> "231". In a constraint those are bounds, so restore them.
_POWER = [(re.compile(r"(?<![\d.])10([2-9])(?!\d)"), r"10^\1"), (re.compile(r"(?<![\d.])2(31|32|63)(?!\d)"), r"2^\1")]


def _constraint(line: str) -> str:
    line = line.strip()
    if "`" in line or not _FORMULA.search(line):
        return f"- {line}"
    for pat, rep in _POWER:
        line = pat.sub(rep, line)
    # A formula, not prose: a code span keeps `*` and `_` (n * m, nums[i])
    # from reading as emphasis.
    return f"- `{line}`"


def statement_markdown(text: str) -> str:
    out: list[str] = []
    in_constraints = False
    seen_item = False
    for raw in text.replace("\u00a0", " ").split("\n"):
        line = raw.rstrip()
        m = _HEADING.match(line.strip())
        if m:
            out += ["", f"### {m.group(1)}", ""]
            in_constraints, seen_item = m.group(1) == "Constraints", False
            continue
        if not line.strip():
            # The blank after the heading opens the list; one after an item ends it.
            in_constraints = in_constraints and not seen_item
            out.append("")
            continue
        if in_constraints and not line.lstrip().startswith(("-", "*", "#")):
            out.append(_constraint(line))
            seen_item = True
            continue
        out.append(_LABEL.sub(lambda m: f"**{m.group(1)}:** ", line))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
