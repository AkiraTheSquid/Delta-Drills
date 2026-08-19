"""
Persistent notebook kernels.

`code_runner` runs ONE block of code in a hardened `os.fork()` child that exits
as soon as it has answered. That is right for grading a submission, and wrong
for a notebook: cells in Colab share a live session, so `a` bound in cell 6 is
still bound in cell 8.

A kernel here is that same forked child WITH THE EXIT REMOVED. It keeps a
globals dict and a duplex pipe, and answers one cell at a time until it is
interrupted, evicted, or idles out. Everything the grading child does to fence
itself off from the API process it forked from is done once at kernel start:
own session, environment scrubbed to PATH, backend directory dropped from
`sys.path`, `app.*` poisoned out of `sys.modules`. Persistence changes how LONG
that child lives, not what it can reach.

Why this is safe to keep in-process: Fly runs one machine
(`min_machines_running = 1`, `auto_stop_machines = false`) and the Dockerfile
starts one uvicorn worker, so a module-level registry is the whole truth. Two
workers would need a real kernel manager; there is one.

The server deliberately knows NOTHING about cell semantics. Last-expression
echo, `<cell N>` tracebacks and the bound-name summary live in
`practice/notebook.js` and are shipped here as `bootstrap` code, exec'd only
when a kernel was created fresh. One copy of that harness, and installing it
cannot race a kernel that was evicted between two clicks.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.code_runner import CODE_PREAMBLE, ExecutionResult

DEFAULT_TIMEOUT_SECONDS = 30

# A cell that was interrupted still has to hand its traceback back. Short: the
# child only has to unwind and send.
INTERRUPT_GRACE_SECONDS = 3.0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# The box is shared-cpu-1x/2gb and torch already lives in it, so the caps are
# deliberately small. Tunable without a code change because the right numbers
# are a property of the machine, not of the design.
MAX_KERNELS = _env_int("DD_KERNEL_MAX", 4)
IDLE_SECONDS = _env_int("DD_KERNEL_IDLE_SECONDS", 1200)
# Copy-on-write means every kernel's RSS counts the torch pages it SHARES with
# the parent, so this over-reports by roughly one torch image per kernel. It is
# a runaway-allocation backstop, not an accounting figure — keep it generous.
MAX_RSS_MB = _env_int("DD_KERNEL_RSS_MB", 512)

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

# The registry's RSS check runs between requests and SKIPS a kernel that is
# mid-cell, so on its own it cannot stop the allocation that matters: one
# `t.ones(10**10)` inside a running cell reaches the box's OOM killer long
# before any other request looks. The kernel therefore watches its OWN memory
# from a thread — ask the cell to stop, then take the process down if it will
# not. Losing one session beats losing the API for everybody on the machine.
MEMORY_POLL_SECONDS = 0.25
MEMORY_GRACE_SECONDS = 1.5


# --- the child ---------------------------------------------------------------

def _harden_child() -> None:
    """Fence the forked child off from the API process. Same measures as
    `code_runner._forked_child_main`, applied once at kernel start instead of
    once per run."""
    try:
        os.setsid()
    except OSError:
        pass
    keep_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    os.environ.clear()
    os.environ["PATH"] = keep_path

    backend_dir = Path(__file__).resolve().parent.parent

    def _points_at_backend(entry: str) -> bool:
        try:
            return Path(entry or os.getcwd()).resolve() == backend_dir
        except OSError:
            return False

    sys.path[:] = [p for p in sys.path if not _points_at_backend(p)]
    sys.path_importer_cache.clear()
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        sys.modules.pop(name, None)
    sys.modules["app"] = None


# Frames the learner should never see. `kernel_runner.py` is the `exec` that
# launched their cell, and `<harness>` is the client's cell wrapper — showing
# either means a traceback opens with a server path instead of their own code,
# which is not what a notebook does.
_INTERNAL_FRAMES = {__file__, "<harness>", "<preamble>"}


def _user_traceback(traceback_mod) -> str:
    """The current exception, with this module's frames stripped.

    Chained causes are dropped with them — a `raise ... from ...` in a lesson
    cell loses its "during handling" section. Worth it: every traceback the
    learner sees otherwise starts inside the server.
    """
    exc_type, exc, tb = sys.exc_info()
    frames = [f for f in traceback_mod.extract_tb(tb) if f.filename not in _INTERNAL_FRAMES]
    lines = []
    if frames:
        lines.append("Traceback (most recent call last):\n")
        lines.extend(traceback_mod.format_list(frames))
    lines.extend(traceback_mod.format_exception_only(exc_type, exc))
    return "".join(lines)


def _child_rss_mb() -> float:
    """This process's resident size, read from inside the kernel."""
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):
        return 0.0
    return resident_pages * _PAGE_SIZE / (1024 * 1024)


_MEMORY_KILL_NOTE = (
    "MemoryError: the cell went over the {cap} MB the kernel is allowed and did "
    "not stop when interrupted, so the kernel was restarted. Variables are gone "
    "— re-run the cells you need."
)


def _memory_watchdog(conn, send_lock, state) -> None:
    """Kill this kernel before it can OOM the machine.

    Two stages on purpose. A Python-level allocation loop unwinds cleanly on
    `interrupt_main()` and the learner keeps their session; a single C-level
    allocation never checks for the interrupt, so after a grace period the
    process goes. The note is sent under the same lock the cell's own reply
    uses, so the parent reads exactly one payload either way.
    """
    import _thread

    over_since = None
    while True:
        time.sleep(MEMORY_POLL_SECONDS)
        if _child_rss_mb() <= MAX_RSS_MB:
            over_since = None
            continue
        now = time.monotonic()
        if over_since is None:
            over_since = now
            if state.get("running"):
                try:
                    _thread.interrupt_main()
                except Exception:
                    pass
            continue
        if now - over_since < MEMORY_GRACE_SECONDS:
            continue
        if send_lock.acquire(timeout=0.5):
            if state.get("running"):
                try:
                    conn.send({"stdout": "", "success": False,
                               "stderr": _MEMORY_KILL_NOTE.format(cap=MAX_RSS_MB)})
                except Exception:
                    pass
        os._exit(137)


def _kernel_child_main(conn) -> None:
    """Serve cells until the parent hangs up.

    Deliberately silent: no logging. This process was forked from a threaded
    uvicorn worker, and a logging lock held by a thread that does not exist in
    the child would deadlock it.
    """
    import contextlib
    import io
    import traceback

    _harden_child()
    # Default SIGINT handling — the parent interrupts a runaway cell with
    # SIGINT so the KeyboardInterrupt unwinds the cell and leaves the session.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    # `running` tells the watchdog whether there is a cell to blame; the lock
    # makes the watchdog's note and the cell's own reply mutually exclusive.
    send_lock = threading.Lock()
    state = {"running": False}
    threading.Thread(target=_memory_watchdog, args=(conn, send_lock, state),
                     daemon=True).start()

    namespace: dict = {"__name__": "__main__"}
    try:
        exec(compile(CODE_PREAMBLE, "<preamble>", "exec"), namespace)
    except BaseException:
        try:
            conn.send({"ready": False, "error": traceback.format_exc()})
        except Exception:
            pass
        os._exit(1)
    try:
        conn.send({"ready": True})
    except Exception:
        os._exit(1)

    while True:
        # A SIGINT can land after a cell has already answered — the parent
        # timed out on a cell that was finishing. Swallow it here rather than
        # letting it kill the kernel between cells.
        try:
            message = conn.recv()
        except KeyboardInterrupt:
            continue
        except (EOFError, OSError):
            os._exit(0)
        if not isinstance(message, dict) or message.get("op") == "shutdown":
            os._exit(0)

        code = message.get("code") or ""
        filename = message.get("filename") or "<cell>"
        buf_out, buf_err = io.StringIO(), io.StringIO()
        success = True
        state["running"] = True
        try:
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                exec(compile(code, filename, "exec"), namespace)
        except SystemExit as exc:
            success = exc.code in (0, None)
        except KeyboardInterrupt:
            buf_err.write("KeyboardInterrupt: cell interrupted (the session is still alive)\n")
            success = False
        except BaseException:
            buf_err.write(_user_traceback(traceback))
            success = False
        payload = {"stdout": buf_out.getvalue(), "stderr": buf_err.getvalue(),
                   "success": success}
        with send_lock:
            state["running"] = False
            while True:
                try:
                    conn.send(payload)
                    break
                except KeyboardInterrupt:
                    continue  # late SIGINT for the cell just finished
                except Exception:
                    os._exit(1)


# --- the parent-side handle --------------------------------------------------

@dataclass
class KernelSession:
    """One learner's live Python session."""

    session_id: str
    proc: multiprocessing.process.BaseProcess
    conn: object
    created_at: float
    last_used: float
    # Which notebook this kernel belongs to. A learner gets ONE kernel, the way
    # a person gets one Colab runtime; opening a different notebook restarts it
    # rather than letting the last lesson's names linger in this one.
    context: str = ""
    exec_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def alive(self) -> bool:
        return bool(self.proc.is_alive())

    def rss_mb(self) -> float:
        """Resident size of the kernel, or 0.0 when it cannot be read."""
        try:
            with open(f"/proc/{self.proc.pid}/statm", "r", encoding="ascii") as handle:
                resident_pages = int(handle.read().split()[1])
        except (OSError, IndexError, ValueError):
            return 0.0
        return resident_pages * _PAGE_SIZE / (1024 * 1024)

    def execute(self, code: str, filename: str = "<cell>",
                timeout: int = DEFAULT_TIMEOUT_SECONDS) -> ExecutionResult:
        """Run one cell. On timeout the cell is interrupted, not the kernel:
        SIGINT unwinds the cell and leaves the bindings intact, which is what
        Colab's stop button does. SIGKILL of the process group is the fallback
        for code that ignores or outruns the interrupt (a tight C loop that
        never returns to the interpreter), and it costs the session."""
        if not self.alive:
            return ExecutionResult(stdout="", stderr="Kernel is not running.", success=False)
        try:
            self.conn.send({"op": "run", "code": code, "filename": filename})
        except (BrokenPipeError, OSError) as exc:
            self.shutdown()
            return ExecutionResult(stdout="", stderr=f"Kernel is gone: {exc}", success=False)

        if self.conn.poll(timeout):
            return self._recv_result()

        # Cell overran. Interrupt it and give it a moment to report back.
        try:
            os.kill(self.proc.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass
        if self.conn.poll(INTERRUPT_GRACE_SECONDS):
            result = self._recv_result()
            note = f"Cell timed out after {timeout} seconds and was interrupted."
            stderr = f"{note}\n{result.stderr}" if result.stderr else note
            return ExecutionResult(stdout=result.stdout, stderr=stderr, success=False)

        self.shutdown()
        return ExecutionResult(
            stdout="",
            stderr=(f"Cell timed out after {timeout} seconds and did not respond to an "
                    "interrupt, so the kernel was restarted. Variables are gone — "
                    "re-run the cells you need."),
            success=False,
        )

    def _recv_result(self) -> ExecutionResult:
        try:
            payload = self.conn.recv()
        except (EOFError, OSError) as exc:
            self.shutdown()
            return ExecutionResult(stdout="", stderr=f"Kernel died mid-cell: {exc}",
                                   success=False)
        self.exec_count += 1
        self.last_used = time.monotonic()
        return ExecutionResult(
            stdout=str(payload.get("stdout", "")),
            stderr=str(payload.get("stderr", "")),
            success=bool(payload.get("success")),
        )

    def shutdown(self) -> None:
        """Stop the kernel. Kills the whole process group — the child called
        `setsid()`, so anything it spawned goes with it."""
        try:
            self.conn.send({"op": "shutdown"})
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass
        pid = self.proc.pid
        self.proc.join(timeout=1)
        if self.proc.is_alive() and pid:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self.proc.join(timeout=1)


def _spawn_kernel(session_id: str, context: str = "") -> KernelSession:
    """Fork a kernel off this process. torch is already imported here
    (`code_runner.preload_torch` at app startup), so the child sees it through
    copy-on-write instead of paying seconds to import it."""
    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    proc = ctx.Process(target=_kernel_child_main, args=(child_conn,), daemon=True)
    proc.start()
    child_conn.close()
    now = time.monotonic()
    session = KernelSession(session_id=session_id, proc=proc, conn=parent_conn,
                            created_at=now, last_used=now, context=context)
    # Wait for the preamble to finish before anyone can send a cell.
    if not parent_conn.poll(30):
        session.shutdown()
        raise RuntimeError("Kernel did not start within 30 seconds.")
    hello = parent_conn.recv()
    if not hello.get("ready"):
        session.shutdown()
        raise RuntimeError(f"Kernel preamble failed: {hello.get('error', 'unknown')}")
    return session


# --- registry ----------------------------------------------------------------

_kernels: dict[str, KernelSession] = {}
_registry_lock = threading.Lock()

# How long to wait for a session's own lock before calling it busy. A kernel
# runs one cell at a time; a second click while a cell is running should be
# told so, not queued behind a 30-second timeout.
_BUSY_WAIT_SECONDS = 1.0


def _reap_locked() -> None:
    """Drop kernels that are dead, idle, or over their memory budget. Called
    with `_registry_lock` held."""
    now = time.monotonic()
    for session_id, session in list(_kernels.items()):
        # A session holding its own lock is running a cell right now. Killing
        # it from under the thread that is reading its pipe would turn a
        # healthy cell into "Kernel died mid-cell", so a busy kernel is left
        # alone and reaped on the next request.
        if session.lock.locked() and session.alive:
            continue
        reason = None
        if not session.alive:
            reason = "dead"
        elif now - session.last_used > IDLE_SECONDS:
            reason = "idle"
        elif session.rss_mb() > MAX_RSS_MB:
            reason = "memory"
        if reason:
            _kernels.pop(session_id, None)
            session.shutdown()


def _evict_lru_locked() -> None:
    """Make room for one more kernel. Called with `_registry_lock` held."""
    while len(_kernels) >= MAX_KERNELS:
        idle = [s for s in _kernels.values() if not s.lock.locked()]
        if not idle:
            # Every kernel is mid-cell. Refuse rather than kill someone's
            # running cell to seat a new learner.
            raise RuntimeError("All kernels are busy — try again in a moment.")
        oldest = min(idle, key=lambda s: s.last_used)
        _kernels.pop(oldest.session_id, None)
        oldest.shutdown()


@dataclass
class KernelRunResult:
    session_id: str
    result: ExecutionResult
    fresh: bool
    exec_count: int


def run_cell(session_id: str | None, code: str, bootstrap: str = "",
             filename: str = "<cell>", context: str = "",
             timeout: int = DEFAULT_TIMEOUT_SECONDS,
             skip_on_fresh: bool = False) -> KernelRunResult:
    """Run one cell in the named session, creating the kernel if needed.

    `fresh` is True when this call created the kernel — which is the client's
    signal that whatever state it thought it had is gone. `bootstrap` is exec'd
    ONLY on a fresh kernel, so the client's cell harness is installed exactly
    once per kernel and cannot be skipped by an eviction between two clicks.

    `skip_on_fresh` says: if you had to build the kernel, do NOT run my code —
    just report `fresh` and let me send the whole prefix instead. Without it a
    client that replays cells 1..N after a fresh reply runs cell N twice, and a
    cell that appends to a list or writes a file is not idempotent.

    Raises RuntimeError when the session is busy running another cell.
    """
    # A session can be evicted or restarted between choosing it under the
    # registry lock and reserving it under its own. Holding the registry lock
    # across a one-second wait would block every other learner, so the handoff
    # is checked instead: if the session we reserved is no longer the registry's
    # session, it was shut down under us and the whole choice is made again.
    for _attempt in range(2):
        outcome = _reserve_and_run(session_id, code, bootstrap, filename, context,
                                   timeout, skip_on_fresh)
        if outcome is not None:
            return outcome
    raise RuntimeError("The kernel was restarted mid-request — run the cell again.")


def _reserve_and_run(session_id, code, bootstrap, filename, context, timeout,
                     skip_on_fresh) -> KernelRunResult | None:
    """One attempt at `run_cell`. None when the session was replaced under us."""
    fresh = False
    with _registry_lock:
        _reap_locked()
        session = _kernels.get(session_id) if session_id else None
        if session is not None and session.context != context:
            if session.lock.locked():
                raise RuntimeError("This kernel is already running a cell.")
            _kernels.pop(session.session_id, None)
            session.shutdown()
            session = None
        if session is None:
            _evict_lru_locked()
            session_id = session_id or uuid.uuid4().hex
            session = _spawn_kernel(session_id, context=context)
            _kernels[session_id] = session
            fresh = True

    if not session.lock.acquire(timeout=_BUSY_WAIT_SECONDS):
        raise RuntimeError("This kernel is already running a cell.")
    try:
        with _registry_lock:
            if _kernels.get(session.session_id) is not session:
                return None  # shut down between the two locks — start over
        if fresh and bootstrap.strip():
            boot = session.execute(bootstrap, filename="<harness>", timeout=timeout)
            if not boot.success:
                return KernelRunResult(session_id=session.session_id, result=boot,
                                       fresh=True, exec_count=session.exec_count)
        if fresh and skip_on_fresh:
            return KernelRunResult(
                session_id=session.session_id,
                result=ExecutionResult(stdout="", stderr="", success=True),
                fresh=True, exec_count=session.exec_count)
        result = session.execute(code, filename=filename, timeout=timeout)
        return KernelRunResult(session_id=session.session_id, result=result,
                               fresh=fresh, exec_count=session.exec_count)
    finally:
        session.lock.release()
        if not session.alive:
            with _registry_lock:
                if _kernels.get(session.session_id) is session:
                    _kernels.pop(session.session_id, None)


def reset_kernel(session_id: str) -> bool:
    """Kill a session's kernel. True when there was one to kill."""
    with _registry_lock:
        session = _kernels.pop(session_id, None)
    if session is None:
        return False
    session.shutdown()
    return True


def kernel_status(session_id: str | None = None) -> dict:
    """What the registry is holding.

    `session_id` narrows `sessions` to that one — which is what the endpoint
    passes, because a learner has no business reading which lesson another
    learner is on, how long they have been there, or how much memory they are
    using. `count` stays whole-box: it is a number, and how full the machine is
    is exactly what a client needs to know before asking for a kernel.
    """
    with _registry_lock:
        _reap_locked()
        now = time.monotonic()
        total = len(_kernels)
        shown = [s for s in _kernels.values()
                 if session_id is None or s.session_id == session_id]
        sessions = [
            {
                "session_id": s.session_id,
                "context": s.context,
                "alive": s.alive,
                "exec_count": s.exec_count,
                "idle_seconds": round(now - s.last_used, 1),
                "age_seconds": round(now - s.created_at, 1),
                "rss_mb": round(s.rss_mb(), 1),
            }
            for s in shown
        ]
    return {
        "count": total,
        "max_kernels": MAX_KERNELS,
        "idle_seconds": IDLE_SECONDS,
        "max_rss_mb": MAX_RSS_MB,
        "sessions": sessions,
    }


def shutdown_all() -> None:
    """Stop every kernel (app shutdown)."""
    with _registry_lock:
        sessions = list(_kernels.values())
        _kernels.clear()
    for session in sessions:
        session.shutdown()
