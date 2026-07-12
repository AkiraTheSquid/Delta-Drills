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
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 20

TORCH_COLAB_MESSAGE = "This drill uses PyTorch, which the in-app sandbox can't run. Open it in Colab (use Show Answer / the solution notebook), complete it there, then self-rate your result."

# --- warm torch fork runner ---------------------------------------------------
# A cold `import torch` per submission (the subprocess path) costs seconds on a
# small VM and OOM'd the old 512mb Fly box outright. Instead the API process
# preimports torch ONCE at startup (preload_torch, called from app.main); torch
# submissions then run in an os.fork() child that sees the already-imported
# module via copy-on-write — per-run cost is milliseconds. Non-torch code keeps
# the proven subprocess path.
_torch_preloaded = False


def preload_torch() -> bool:
    """Import torch into this process (call once at app startup). Returns
    availability; safe to call in environments without torch installed."""
    global _torch_preloaded
    if _torch_preloaded:
        return True
    try:
        import torch  # noqa: F401
        # Single-thread torch BEFORE any fork: OMP/intra-op thread pools do not
        # survive fork() and can deadlock the child (observed as rare 20s
        # timeouts on F.cross_entropy / BatchNorm in forked grading). Drill
        # tensors are tiny — one thread is also the fast path.
        torch.set_num_threads(1)
        _torch_preloaded = True
        logger.info("torch preloaded for fork runner (%s)", torch.__version__)
    except Exception as exc:
        _torch_preloaded = False
        logger.warning("torch preload failed — torch drills stay Colab-only: %s", exc)
    return _torch_preloaded


def torch_available() -> bool:
    """True when torch submissions can be graded in-process (fork runner)."""
    return _torch_preloaded


def code_uses_torch(code: str) -> bool:
    """Return True if the code imports torch (``import torch`` or ``from torch ...``)."""
    return re.search(r"(?m)^\s*(?:import\s+torch\b|from\s+torch[\s.])", code or "") is not None


def _resolve_arena_numbers_path() -> Path:
    """First existing copy of the ARENA digits fixture. The vendored copy in
    app/data ships in the Docker image (the ARENA content tree is gitignored
    and absent from the deploy build context — visual fixtures could never
    grade on Fly until it was vendored, 2026-07-12)."""
    candidates = (
        Path(__file__).resolve().parent / "data" / "numbers.npy",
        Path(__file__).resolve().parents[3]
        / "Local_Deployed_Shared" / "content" / "ARENA_3.0-main"
        / "chapter0_fundamentals" / "exercises" / "part0_prereqs" / "numbers.npy",
    )
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


ARENA_NUMBERS_PATH = _resolve_arena_numbers_path()

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


def _forked_child_main(code: str, conn) -> None:
    """Body of the forked grading child. Runs the (preamble-prefixed) user
    code in a fresh globals dict and ships the result back over the pipe.

    Hardening vs the parent API process it was forked from:
      - new session (own process group → parent can SIGKILL the whole group)
      - environment cleared to a minimal PATH (the subprocess path strips env
        via _safe_env; forks inherit os.environ, so scrub it here)
      - app.* modules dropped from sys.modules and the backend dir removed
        from sys.path so user code can't `import app.config` for secrets
    """
    import contextlib
    import io
    import traceback

    try:
        os.setsid()
    except OSError:
        pass
    keep_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    os.environ.clear()
    os.environ["PATH"] = keep_path
    backend_dir = Path(__file__).resolve().parents[1]

    def _points_at_backend(entry: str) -> bool:
        try:
            return Path(entry or os.getcwd()).resolve() == backend_dir
        except OSError:
            return False

    sys.path[:] = [p for p in sys.path if not _points_at_backend(p)]
    sys.path_importer_cache.clear()
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        sys.modules.pop(name, None)
    # Poison the package name outright — even a surviving path entry can't
    # resurrect `import app` when sys.modules maps it to None.
    sys.modules["app"] = None

    buf_out, buf_err = io.StringIO(), io.StringIO()
    success = True
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            exec(compile(code, "<practice>", "exec"), {"__name__": "__main__"})
    except SystemExit as exc:
        success = exc.code in (0, None)
    except BaseException:
        buf_err.write(traceback.format_exc())
        success = False
    try:
        conn.send({"stdout": buf_out.getvalue(), "stderr": buf_err.getvalue(),
                   "success": success})
        conn.close()
    except Exception:
        os._exit(1)
    os._exit(0)


def _run_code_forked(code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> ExecutionResult:
    """Execute code in an os.fork() child of this process (torch preimported
    → no per-run import cost). Same contract as the subprocess path."""
    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_forked_child_main, args=(code, child_conn), daemon=True)
    try:
        proc.start()
        child_conn.close()
        result = None
        if parent_conn.poll(timeout):
            try:
                result = parent_conn.recv()
            except EOFError:
                result = None
        if result is not None:
            return ExecutionResult(
                stdout=result["stdout"], stderr=result["stderr"],
                success=bool(result["success"]),
            )
        # timeout or child died without reporting — kill the whole group
        timed_out = proc.is_alive()
        if proc.pid:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if timed_out:
            return ExecutionResult(
                stdout="", stderr=f"Execution timed out after {timeout} seconds",
                success=False,
            )
        return ExecutionResult(
            stdout="", stderr="Execution failed (grading process died)", success=False,
        )
    except Exception as exc:
        logger.exception("Fork runner error")
        return ExecutionResult(stdout="", stderr=f"Internal error: {exc}", success=False)
    finally:
        parent_conn.close()
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()


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
    # Torch code: fork runner when torch is preloaded (milliseconds); honest
    # Colab-routing refusal when it isn't (e.g. an env without torch).
    if code_uses_torch(code):
        if torch_available():
            return _run_code_forked(CODE_PREAMBLE + code, timeout=timeout)
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
import sys
import numpy as np

def _delta_torch_tensor(value):
    # torch only if the question already imported it — numpy questions must
    # not pay the torch import. Guests never reach here for torch questions
    # (Pyodide has no torch; they get Colab routing), so backend-only is fine.
    _torch = sys.modules.get("torch")
    return _torch is not None and isinstance(value, _torch.Tensor)

def _delta_to_jsonable(value):
    if _delta_torch_tensor(value):
        return value.detach().cpu().tolist()
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
    if _delta_torch_tensor(a) or _delta_torch_tensor(b):
        try:
            a2 = a.detach().cpu().numpy() if _delta_torch_tensor(a) else np.asarray(a)
            b2 = b.detach().cpu().numpy() if _delta_torch_tensor(b) else np.asarray(b)
            return bool(np.array_equal(a2, b2))
        except Exception:
            # dtypes numpy can't hold (bfloat16, conj views): equal tensors must
            # not grade as unequal — fall back to torch's own comparison.
            _torch = sys.modules.get("torch")
            if _torch is not None and isinstance(a, _torch.Tensor) and isinstance(b, _torch.Tensor):
                try:
                    return bool(_torch.equal(a.detach().cpu().resolve_conj(),
                                             b.detach().cpu().resolve_conj()))
                except Exception:
                    return False
            return False
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_delta_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)

def _delta_seed():
    # Same seed before the actual-side and expected-side setup runs, for BOTH
    # rngs — setup executes twice, so unseeded torch.rand in setup would give
    # the two sides different fixtures and fail honest answers.
    np.random.seed(0)
    _torch = sys.modules.get("torch")
    if _torch is not None:
        _torch.manual_seed(0)

_delta_results = []
{user_code}
for _delta_case in json.loads({payload!r}):
    try:
        if _delta_case.get("setup_code"):
            _delta_seed()
            exec(_delta_case["setup_code"], globals())
        _delta_actual = eval(_delta_case["call"], globals())
        _delta_expected_setup = _delta_case.get("expected_setup_code") or _delta_case.get("setup_code")
        if _delta_expected_setup:
            _delta_seed()
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
    # Torch may hide inside the JSON payload's setup/expected strings (where
    # the line-anchored code_uses_torch can't see it) — e.g. a non-torch
    # submission against a torch question. Route those through the fork
    # runner too rather than paying a cold torch import in a subprocess.
    if torch_available() and ("torch" in payload or code_uses_torch(user_code)):
        execution = _run_code_forked(CODE_PREAMBLE + harness, timeout=timeout)
    else:
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
