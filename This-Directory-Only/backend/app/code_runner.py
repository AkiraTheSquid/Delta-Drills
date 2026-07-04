"""
Sandboxed Python code execution.

Runs user-submitted code in a subprocess with:
  - A hard timeout (default 20 seconds)
  - numpy available
  - No direct use of exec()/eval() — uses subprocess instead
  - Captures stdout, stderr, and success flag
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 20

TORCH_COLAB_MESSAGE = "This drill uses PyTorch, which the in-app sandbox can't run. Open it in Colab (use Show Answer / the solution notebook), complete it there, then self-rate your result."


def code_uses_torch(code: str) -> bool:
    """Return True if the code imports torch (``import torch`` or ``from torch ...``)."""
    return re.search(r"(?m)^\s*(?:import\s+torch\b|from\s+torch[\s.])", code or "") is not None


ARENA_NUMBERS_PATH = (
    Path(__file__).resolve().parents[3]
    / "Local_Deployed_Shared"
    / "content"
    / "ARENA_3.0-main"
    / "chapter0_fundamentals"
    / "exercises"
    / "part0_prereqs"
    / "numbers.npy"
)

# Preamble injected before user code to ensure numpy (and einops) are available
# and results are reproducible. einops import is guarded so numpy/einsum problems
# continue to work even if einops is not installed in this environment.
CODE_PREAMBLE = (
    "import numpy as np\n"
    "np.random.seed(0)\n"
    "_delta_original_np_load = np.load\n"
    "def _delta_np_load(file, *args, **kwargs):\n"
    f"    if str(file) == '/delta_numbers.npy':\n"
    f"        file = r'{ARENA_NUMBERS_PATH.as_posix()}'\n"
    "    return _delta_original_np_load(file, *args, **kwargs)\n"
    "np.load = _delta_np_load\n"
    "_delta_einsum = None\n"
    "try:\n"
    "    import einops\n"
    "    from einops import einsum as _delta_einsum, rearrange, reduce, repeat\n"
    "except ImportError:\n"
    "    pass\n"
    "def einsum(*args):\n"
    "    if _delta_einsum is not None:\n"
    "        return _delta_einsum(*args)\n"
    "    if len(args) < 2:\n"
    "        raise TypeError('einsum expects one or more arrays plus a pattern string')\n"
    "    *arrays, pattern = args\n"
    "    return np.einsum(pattern.replace(' ', ''), *arrays)\n"
    f"try:\n"
    f"    arr = np.load(r'{ARENA_NUMBERS_PATH.as_posix()}')\n"
    f"except Exception:\n"
    f"    pass\n"
    "def display_array_as_img(*args, **kwargs):\n"
    "    return None\n"
)


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    success: bool


@dataclass
class TestCaseResult:
    passed: bool
    actual: str
    expected: str
    error: str = ""


def run_code(code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> ExecutionResult:
    """
    Execute Python code in a sandboxed subprocess.

    The code is written to a temporary file and executed with the same
    Python interpreter. numpy is automatically imported as np.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        ExecutionResult with stdout, stderr, and success flag.
    """
    # Torch/GPU drills are Colab-only: the in-app sandbox can't import torch
    # within a sane timeout. Return an honest message instead of hanging.
    if code_uses_torch(code):
        return ExecutionResult(stdout="", stderr=TORCH_COLAB_MESSAGE, success=False)

    full_code = CODE_PREAMBLE + code

    # Write to a temp file so we avoid shell injection issues
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix="practice_",
    ) as tmp:
        tmp.write(full_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Don't inherit parent environment variables that could leak info,
            # but keep PATH and common env so numpy/libs can be found
            env=_safe_env(),
        )
        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            success=result.returncode == 0,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            stdout="",
            stderr=f"Execution timed out after {timeout} seconds",
            success=False,
        )
    except Exception as exc:
        logger.exception("Unexpected error running user code")
        return ExecutionResult(
            stdout="",
            stderr=f"Internal error: {exc}",
            success=False,
        )
    finally:
        # Clean up temp file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


def _safe_env() -> dict[str, str]:
    """
    Build a minimal environment for the subprocess.
    Keep PATH and basic vars so that numpy/system libraries work,
    but strip secrets like JWT keys, API keys, etc.
    """
    import os

    safe_keys = {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "TMPDIR",
        "TEMP",
        "TMP",
    }
    return {k: v for k, v in os.environ.items() if k in safe_keys}


def compare_output(actual: str, expected_code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> tuple[bool, str, str]:
    """
    Run the expected answer code, then compare its stdout to actual output.

    Returns (match, actual_output_stripped, expected_output_stripped).
    """
    expected_result = run_code(expected_code, timeout=timeout)
    expected_output = expected_result.stdout.strip()
    actual_stripped = actual.strip()

    # Exact match after stripping whitespace
    match = actual_stripped == expected_output
    return match, actual_stripped, expected_output


def run_function_tests(
    user_code: str,
    test_cases: list[dict],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[list[TestCaseResult], ExecutionResult]:
    payload = json.dumps(test_cases)
    harness = f"""
import json
import numpy as np

def _delta_to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_delta_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_delta_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {{k: _delta_to_jsonable(v) for k, v in value.items()}}
    return value

def _delta_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_delta_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)

_delta_results = []
{user_code}
for _delta_case in json.loads({payload!r}):
    try:
        if _delta_case.get("setup_code"):
            np.random.seed(0)
            exec(_delta_case["setup_code"], globals())
        _delta_actual = eval(_delta_case["call"], globals())
        _delta_expected_setup = _delta_case.get("expected_setup_code") or _delta_case.get("setup_code")
        if _delta_expected_setup:
            np.random.seed(0)
            exec(_delta_expected_setup, globals())
        _delta_expected = eval(_delta_case["expected_expr"], globals())
        _delta_results.append({{
            "passed": bool(_delta_equal(_delta_actual, _delta_expected)),
            "actual": repr(_delta_to_jsonable(_delta_actual)),
            "expected": repr(_delta_to_jsonable(_delta_expected)),
            "error": "",
        }})
    except Exception as _delta_exc:
        _delta_results.append({{
            "passed": False,
            "actual": "",
            "expected": "",
            "error": f"{{type(_delta_exc).__name__}}: {{_delta_exc}}",
        }})
print("__DELTA_TESTS__" + json.dumps(_delta_results))
"""
    execution = run_code(harness, timeout=timeout)
    marker = "__DELTA_TESTS__"
    results: list[TestCaseResult] = []
    if marker in execution.stdout:
        prefix, _, suffix = execution.stdout.partition(marker)
        execution.stdout = prefix.rstrip()
        try:
            parsed = json.loads(suffix.strip())
            results = [TestCaseResult(**item) for item in parsed]
        except Exception as exc:
            results = [TestCaseResult(passed=False, actual="", expected="", error=f"Invalid test payload: {exc}")]
    elif not execution.success:
        error_text = execution.stderr.strip() or execution.stdout.strip() or "Test harness execution failed."
        results = [TestCaseResult(passed=False, actual="", expected="", error=error_text)]
    elif execution.success:
        results = [TestCaseResult(passed=False, actual="", expected="", error="Missing test results.")]
    return results, execution
