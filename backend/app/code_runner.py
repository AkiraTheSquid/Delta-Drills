"""
Sandboxed Python code execution.

Runs user-submitted code in a subprocess with:
  - A hard timeout (default 5 seconds)
  - numpy available
  - No direct use of exec()/eval() — uses subprocess instead
  - Captures stdout, stderr, and success flag
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 5

# Preamble injected before user code to ensure numpy (and einops) are available
# and results are reproducible. einops import is guarded so numpy/einsum problems
# continue to work even if einops is not installed in this environment.
CODE_PREAMBLE = (
    "import numpy as np\n"
    "np.random.seed(0)\n"
    "try:\n"
    "    import einops\n"
    "    from einops import rearrange, reduce, repeat\n"
    "except ImportError:\n"
    "    pass\n"
)


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    success: bool


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
