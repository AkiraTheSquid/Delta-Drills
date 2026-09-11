"""Deterministic checks for learner-facing answer leaks.

These checks catch implementation fragments and common prose recipes. They
cannot certify that arbitrary prose gives no hint; the author must also review
the complete task, including the lesson context. No model calls are made here.
"""
from __future__ import annotations

import ast
import re

UNAIDED = {"independent", "partial", "solo", "integrated", "applied"}
# Pages whose every faded id is at or above this were authored after the
# ARENA-style rule (whole-function stubs, unaided rungs guarded as errors).
NEW_PAGE_FLOOR = 993
RECIPE = re.compile(
    r"\b(?:use|using)\s+(?:windows?\s+and\s+(?:einsum|reductions)|"
    r"as_strided\s+to\s+align)|"
    r"\b(?:insert|add|place)\s+(?:the\s+)?(?:length[- ]1|singleton)\s+ax(?:is|es)|"
    r"\b(?:build|compute|reduce|reshape|stack|pile|broadcast)\b[^.!?\n]{0,100}"
    r"\b(?:first[^.!?\n]{0,40}then|then)\b|"
    r"\b(?:use|call|apply)\s+`?(?:t|torch|np|numpy|einops|F)\.[\w.]+",
    re.I,
)


def _tree(code):
    try:
        return ast.parse(code)
    except (SyntaxError, TypeError):
        return None


def _dump(node):
    return ast.dump(node, include_attributes=False)


def is_stub(code, function_name="solve"):
    """Only imports and an empty function contract; no computation or hints."""
    tree = _tree(code)
    if tree is None:
        return False
    found = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if not isinstance(node, ast.FunctionDef) or node.name != function_name:
            return False
        found = True
        for statement in node.body:
            if isinstance(statement, ast.Pass):
                continue
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                if isinstance(statement.value.value, str) or statement.value.value is Ellipsis:
                    continue
            if isinstance(statement, ast.Raise) and isinstance(statement.exc, ast.Call):
                if isinstance(statement.exc.func, ast.Name) and statement.exc.func.id == "NotImplementedError":
                    continue
            return False
    return found


def leak_findings(question, rung, label):
    if rung not in UNAIDED:
        return []
    answer = question.get("answer_code") or question.get("canonical_solution") or ""
    starter = question.get("starter_code") or ""
    prompt = question.get("question_text") or question.get("prompt") or ""
    findings = []
    # Comments and docstrings are included deliberately. An import in the
    # starter must never exempt the same API from scrutiny in its comments.
    surfaces = {"prompt": prompt, "starter": starter}
    for surface, text in surfaces.items():
        for match in RECIPE.finditer(text):
            findings.append(f"{label}: PROSE_RECIPE ({surface}) — {match.group(0)!r}")
    tree = _tree(answer)
    if tree is None:
        return findings + [f"{label}: invalid reference solution"]
    exprs = {
        _dump(n): ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Call, ast.BinOp, ast.Subscript, ast.Compare, ast.IfExp))
        and sum(1 for _ in ast.walk(n)) >= 5
    }
    calls = {ast.unparse(n.func) for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for surface, text in surfaces.items():
        snippets = re.findall(r"`([^`\n]+)`", text)
        # Raw comment text can carry code without Markdown backticks.
        snippets += [line.strip().lstrip("# ").removeprefix("return ")
                     for line in text.splitlines() if line.strip().startswith("#")]
        for snippet in snippets:
            candidate = _tree(snippet)
            if candidate:
                for node in ast.walk(candidate):
                    if _dump(node) in exprs:
                        findings.append(f"{label}: SOLUTION_FRAGMENT ({surface}) — {snippet!r}")
                        break
        # Catch unquoted calls, kwargs and einsum patterns, including in prose.
        for call in calls:
            if re.search(r"(?<![\w.])" + re.escape(call) + r"\s*\(", text):
                findings.append(f"{label}: SOLUTION_CALL ({surface}) — {call}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "->" in node.value:
                if node.value in text:
                    findings.append(f"{label}: SOLUTION_PATTERN ({surface}) — {node.value!r}")
    if starter and not is_stub(starter, question.get("function_name") or "solve"):
        findings.append(f"{label}: STARTER_IMPLEMENTATION — unaided starter contains executable work")
    return sorted(set(findings))


# `ast.walk` counts Load/Mult/keyword nodes, so `a*b` is already six. Count
# expression nodes only: `x*self.weight` is four (not an algorithm, and the
# whole point of a Scale module), `z.exp()/z.exp().sum(dim=1)` is nine.
MIN_EXPR_NODES = 6


def _size(node):
    return sum(1 for n in ast.walk(node) if isinstance(n, ast.expr))


def _big_exprs(tree):
    return {
        _dump(n)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Call, ast.BinOp, ast.Subscript, ast.Compare, ast.IfExp))
        and _size(n) >= MIN_EXPR_NODES
    }


def lesson_giveaway(question, lesson_code, label):
    """The integrated rung has no example above it; the lesson's own fences
    must not carry its solution's returned expression either.

    Only the RETURN expression is compared, as a normalized AST: the concept
    fences necessarily show the idioms a solo drill applies, but an integrated
    drill combines them, and its final expression appearing verbatim in the
    lesson is the whole answer on the previous page."""
    tree = _tree(question.get("answer_code") or question.get("canonical_solution") or "")
    lesson = _tree(lesson_code)
    if tree is None or lesson is None:
        return []
    shown = _big_exprs(lesson)
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            if _size(node.value) >= MIN_EXPR_NODES and _dump(node.value) in shown:
                return [f"{label}: LESSON_GIVEAWAY — the lesson shows {ast.unparse(node.value)!r}, "
                        f"which is this integrated drill's returned expression"]
    return []
