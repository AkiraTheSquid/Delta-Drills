"""Shared parsing for the first-encounter lesson pipeline.

KP markdown format (see Local_Deployed_Shared/lessons/AUTHORING.md):
  - flat YAML-subset frontmatter between --- lines: scalars and inline [a, b] lists
  - segment sections: ## Concept, ## Watch out, ## Worked example,
    ## Faded practice
  - KP-level tail: ## Guided practice, ## Independent practice,
    ## Misconceptions (legacy single-segment fallback)
  - Faded/Guided practice contain ### q<id> subsections; faded ones carry
    ```python starter``` and ```python solution``` fences.
Used by scripts/compile_lessons.py and scripts/validate_lessons.py.
"""
import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LESSONS_DIR = REPO / "Local_Deployed_Shared" / "lessons"
REGISTRY_PATH = LESSONS_DIR / "kc_registry.json"
BANK_PATH = REPO / "Local_Deployed_Shared" / "questions_structured.json"

SECTIONS = [
    "Concept",
    "Watch out",
    "Worked example",
    "Faded practice",
    "Guided practice",
    # The ladder's third rung: an independent-rung drill with an example above
    # it. Listed here so `## Applied practice` parses; `compile_lessons.py`
    # turns it into `applied_items`.
    "Applied practice",
    "Independent practice",
    "Misconceptions",
]


def load_registry():
    return json.loads(REGISTRY_PATH.read_text())


def load_bank():
    bank = json.loads(BANK_PATH.read_text())
    return {q["id"]: q for q in bank}


def _parse_value(raw):
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(part) for part in inner.split(",")]
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw.strip("'\"")


def parse_frontmatter(text, path):
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing frontmatter")
    end = text.index("\n---", 3)
    meta = {}
    for line in text[3:end].strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, raw = line.partition(":")
        meta[key.strip()] = _parse_value(raw)
    body = text[end + 4:]
    return meta, body


def split_sections(body, path):
    """Return ordered [(section_title, subtitle, content)] for ## headings.

    Concept / Watch out / Worked example / Faded practice may repeat — each
    `## Concept` starts a new single-concept segment (see AUTHORING.md). A
    section heading may carry a segment subtitle: `## Concept: np.diag`.
    """
    parts = re.split(r"^## (.+)$", body, flags=re.M)
    ordered = []
    for title, content in zip(parts[1::2], parts[2::2]):
        title = title.strip()
        subtitle = ""
        if ":" in title:
            title, _, subtitle = title.partition(":")
            title = title.strip()
            subtitle = subtitle.strip()
        if title not in SECTIONS:
            raise ValueError(f"{path}: unknown section '## {title}'")
        ordered.append((title, subtitle, content.strip()))
    return ordered


def build_segments(ordered, path):
    """Group ordered sections into single-concept segments plus KP-level tail.

    Each `## Concept` opens a segment; its `## Watch out`,
    `## Worked example`, and `## Faded practice` belong to it.
    Guided/Independent/Misconceptions are KP-level tail sections.
    """
    segments = []
    tail = {}
    current = None
    for title, subtitle, content in ordered:
        if title == "Concept":
            current = {
                "title": subtitle,
                "concept": content,
                "watch_out": "",
                "worked": "",
                "faded": "",
            }
            segments.append(current)
        elif title in ("Watch out", "Worked example", "Faded practice"):
            if current is None:
                raise ValueError(f"{path}: '## {title}' before any '## Concept'")
            key = {
                "Watch out": "watch_out",
                "Worked example": "worked",
                "Faded practice": "faded",
            }[title]
            if current[key]:
                raise ValueError(f"{path}: duplicate '## {title}' in one segment")
            current[key] = content
        else:
            if title in tail:
                raise ValueError(f"{path}: duplicate '## {title}'")
            tail[title] = content
    if not segments:
        raise ValueError(f"{path}: no '## Concept' section")
    return segments, tail


def split_items(section_text):
    """Return {qid: content} for ### q<id> subsections."""
    parts = re.split(r"^### q(\d+)\s*$", section_text, flags=re.M)
    return {int(qid): content.strip() for qid, content in zip(parts[1::2], parts[2::2])}


def code_fences(text, info=None):
    """Return code blocks; info filters the fence info string exactly."""
    blocks = re.findall(r"```([^\n]*)\n(.*?)```", text, flags=re.S)
    out = []
    for fence_info, code in blocks:
        fence_info = fence_info.strip()
        if info is None or fence_info == info:
            out.append(code)
    return out


_IMPORT_LINE = re.compile(r"^\s*(import|from)\s")


def _calls_at_top_level(src, fn_name):
    """Does `src` call `fn_name` outside any function body?

    Line-based rather than AST-based on purpose: this is asked of an AUTHORED
    faded starter, whose blanks (`t._____(z)`) happen to parse today but are a
    text convention, not a promise. An unindented line mentioning `fn_name(` is
    a top-level call in every starter in the bank.
    """
    call = re.compile(r"\b" + re.escape(fn_name) + r"\s*\(")
    return any(
        line and not line[0].isspace() and call.search(line)
        and not line.lstrip().startswith(("def ", "#"))
        for line in (src or "").splitlines()
    )


def _function_span(src, fn_name):
    """(first_line, last_line) of `fn_name`'s def, 1-based inclusive, or None."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            end = max(getattr(n, "end_lineno", n.lineno) for n in node.body)
            return node.lineno, end
    return None


def example_run_block(starter_code, fn_name="solve"):
    """The demo lines a question's own starter carries, without the function.

    Every generated starter ends with a fixture and a `print(solve(fixture))`,
    and that block is the only thing on the page that shows the learner what
    their function PRINTS. Comments are kept — the block explains itself ("the
    grader calls solve() with several different inputs").

    Returns "" when the starter has no such block, or will not parse.
    """
    span = _function_span(starter_code or "", fn_name)
    if not span:
        return ""
    first, last = span
    kept = []
    for i, line in enumerate(starter_code.splitlines(), start=1):
        if first <= i <= last:
            continue
        if _IMPORT_LINE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept).strip("\n")


def attach_example_run(faded_starter, question_starter, fn_name="solve"):
    """Give an authored faded starter the demo block its question already has.

    THE BUG THIS FIXES. All 250 hand-cut faded starters are the function and
    nothing else — `def solve(z): return t._____(z)`. The question's own
    starter ends with a fixture and a print, so the learner runs it, reads the
    output and compares it against the expected output above. Served the
    authored starter instead, they get a cell that defines a function and
    prints nothing: filling the blank correctly and filling it wrongly look
    exactly the same, and the expected-output block has nothing to be compared
    against. "It didn't do the thing where it has something right after the
    function like a tensor that makes it such that you can see the printed
    output."

    Grafted at COMPILE time rather than served at runtime because the same
    starter reaches the learner by three routes that do not share code — the
    backend's `faded` rung (prioritization.ladder_starter), the published Colab
    notebook (generate_colab_notebooks.problem_cells), and the client-side
    single-KC ladder (practice/kc-practice.js). One of them fixed is two of
    them still broken.

    Left alone when the authored starter already calls the function at the top
    level: the author wrote their own demo and it is the better one.
    """
    if not faded_starter or not question_starter:
        return faded_starter
    if _calls_at_top_level(faded_starter, fn_name):
        return faded_starter
    block = example_run_block(question_starter, fn_name)
    if not block:
        return faded_starter
    # Imports the block needs and the faded starter does not already make. The
    # fixture line is usually `t.tensor(...)`, so a missing `import torch as t`
    # turns a helpful demo into a NameError.
    have = {l.strip() for l in faded_starter.splitlines() if _IMPORT_LINE.match(l)}
    missing = [
        l for l in question_starter.splitlines()
        if _IMPORT_LINE.match(l) and l.strip() not in have
    ]
    lead = "\n".join(missing) + "\n" if missing else ""
    return faded_starter.rstrip("\n") + "\n\n\n" + lead + block + "\n"


BLANK = "_____"


def body_span(src, fn_name):
    """(start, end) line indices of `fn_name`'s def block, 0-based half-open.

    Line-based rather than AST-based because a faded starter is not required to
    parse — the blanks are a text convention, and 7 of the authored ones are
    already syntactically invalid. Also the reason this cannot just regex the
    whole file: the example-run block below the function uses the same calls
    (`t.tensor(...)`) and blanking THOSE would break the demo it was grafted on
    to provide.
    """
    lines = src.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^\s*def\s+{re.escape(fn_name)}\s*\(", line):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].strip() and not lines[j][0].isspace():
            end = j
            break
    return start, end


def new_syntax_patterns(symbol):
    """Regexes matching what a KP's `new_syntax` entry looks like in code.

    The vocabulary is the registry's, and each form names a different thing:

        torch.floor      a module-level call     →  t.floor(…)   / torch.floor(…)
        Tensor.clamp     a method on a tensor    →  z.clamp(…)
        Tensor.clamp#min a keyword ARGUMENT of one  →  clamp(min=…)
        syntax.matmul    an operator (`@`)       →  no identifier to blank

    `syntax.*` yields nothing on purpose: an operator has no name to hide, and
    replacing `@` with a blank produces code that cannot even be read as a
    sentence. Those are reported by the FADE_LEAK rule instead of rewritten.
    """
    base, _, kwarg = symbol.partition("#")
    owner, _, name = base.rpartition(".")
    if kwarg:
        # A keyword argument: blank the NAME, keep the value. `min=0.0` is the
        # concept ("clamp takes a floor"); `0.0` is this problem's specifics.
        return [(re.compile(rf"\b{re.escape(kwarg)}(?=\s*=[^=])"), BLANK)]
    if owner == "syntax":
        return []
    if owner == "Tensor":
        return [(re.compile(rf"(?<=\.){re.escape(name)}\b"), BLANK)]
    # torch.* — keep the module, blank the call.
    return [(re.compile(rf"(?<=\b)(t|torch)\.{re.escape(name)}\b"), rf"\1.{BLANK}")]


def blank_new_syntax(starter, new_syntax, fn_name="solve"):
    """Hide the concept a faded starter is supposed to be teaching.

    THE RULE, in Seth's words: "the fading should not give away any part of the
    solution that's being learned that's new, but it's okay if it specifies the
    parts that the learner has seen before."

    Faded practice on q67 was `return z.clamp(_____=0.0)`. The KP teaches
    `Tensor.clamp` and `Tensor.clamp#min`; the starter handed over `clamp` and
    blanked only the argument, so the one recall the drill existed to test was
    printed on the page. Blanking both gives `return z._____(_____=0.0)`, which
    still carries everything the learner HAS seen — a method call on the tensor,
    one keyword argument, the value 0.0 — and nothing they have not.

    Applied to the function body only, and only to symbols the KP itself
    declares as new. A symbol taught by an earlier KP is exactly the supporting
    structure that is supposed to stay visible.
    """
    if not starter or not new_syntax:
        return starter
    span = body_span(starter, fn_name)
    if not span:
        return starter
    start, end = span
    lines = starter.splitlines()
    for symbol in new_syntax:
        for pattern, replacement in new_syntax_patterns(symbol):
            for i in range(start, end):
                lines[i] = pattern.sub(replacement, lines[i])
    return "\n".join(lines) + ("\n" if starter.endswith("\n") else "")


def parse_kp(path):
    meta, body = parse_frontmatter(path.read_text(), path)
    ordered = split_sections(body, path)
    segments, tail = build_segments(ordered, path)
    # Aggregate view for consumers that don't care about segment boundaries
    # (qmatrix builder, coverage checks). Content order is preserved per kind.
    sections = dict(tail)
    sections["Concept"] = "\n\n".join(s["concept"] for s in segments if s["concept"])
    sections["Watch out"] = "\n\n".join(s["watch_out"] for s in segments if s["watch_out"])
    sections["Worked example"] = "\n\n".join(s["worked"] for s in segments if s["worked"])
    sections["Faded practice"] = "\n\n".join(s["faded"] for s in segments if s["faded"])
    kp = {
        "segments": segments,
        "file": str(path.relative_to(REPO)),
        "kc": meta.get("kc"),
        "title": meta.get("title"),
        "supporting": meta.get("supporting", []),
        "new_syntax": meta.get("new_syntax", []),
        "previews": meta.get("previews", []),
        "concepts": meta.get("concepts", []),
        "faded": meta.get("faded", []),
        "guided": meta.get("guided", []),
        "independent": meta.get("independent", []),
        "sections": sections,
    }
    return kp


def all_kp_paths():
    return sorted(LESSONS_DIR.glob("*/kp-*.md"))
