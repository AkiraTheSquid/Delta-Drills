"""
Notebook kernels on Modal — one sandbox per learner, a real IPython inside.

Same public surface as `kernel_runner` (`run_cell`, `reset_kernel`,
`kernel_status`, `shutdown_all`), chosen over it by `kernel_backend` when Modal
credentials are present. What changes underneath:

  * The kernel is not a fork of the API process. It is `modal_kernel_shim.py`
    running in a `modal.Sandbox` — its own container, CPU and memory, on a
    box that is not ours. `%pip`, `!wget`, `!sudo apt-get` work, because the
    thing executing cells is ipykernel, not `exec()`. The fork runner's
    fences (env scrub, sys.path trim, `app` poisoning) have nothing to fence:
    the sandbox has no DATABASE_URL to read and no `app` package to import.
  * The image is Colab-shaped. ARENA's own setup cell probes for
    `/root/<chapter>` and downloads the repo only when it is missing, so a
    sparse checkout baked at `/root` makes that cell a no-op WITHOUT editing
    it. Pinned to a commit: the compiled notebooks and the image must be the
    same edition or `import part0_prereqs.tests` tests a different file than
    the one the prose describes.
  * Rich output. The shim forwards display_data mimebundles, so a plotly
    figure reaches the page instead of vanishing into a headless `fig.show()`.

Lifetimes: Modal's `idle_timeout` reaps a sandbox nobody has spoken to; its
`timeout` is a hard ceiling that stops a forgotten sandbox billing forever if
this process dies holding the only handle. A sandbox whose handle this process
lost (redeploy) is unreachable by design — its stdin belonged to the old
process — so startup sweeps and terminates every sandbox under the app.

The registry is module-level for the same reason `kernel_runner`'s is: Fly
runs one machine and one uvicorn worker. Two workers would each think they
own the learner's sandbox, and each would restart the other's.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.code_runner import ExecutionResult
from app.kernel_runner import KernelRunResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30
APP_NAME = os.environ.get("DD_MODAL_APP", "delta-drills-kernels")

# The ARENA edition the notebooks under lessons/notebooks/ were compiled from.
# Bump both together (scripts/compile_arena_notebooks.py reads the checkout at
# content/ARENA_5.0-main, so `git -C … rev-parse HEAD` is the value to paste).
ARENA_REPO = "https://github.com/callummcdougall/ARENA_3.0.git"
ARENA_SHA = "527f9376b40ad9a12ecd80490884b0009b54dd55"
ARENA_CHAPTERS = (
    "chapter0_fundamentals",
    "chapter1_transformer_interp",
    "chapter2_rl",
    "chapter3_llm_evals",
    "chapter4_alignment_science",
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# Caps are per learner-session now, not per box. MAX_KERNELS is a spend guard:
# every live sandbox bills by the second.
MAX_KERNELS = _env_int("DD_KERNEL_MAX", 20)
IDLE_SECONDS = _env_int("DD_KERNEL_IDLE_SECONDS", 1800)
SANDBOX_LIFETIME_SECONDS = _env_int("DD_KERNEL_LIFETIME_SECONDS", 4 * 3600)
SANDBOX_CPU = float(os.environ.get("DD_KERNEL_CPU", "2"))
SANDBOX_MEMORY_MB = _env_int("DD_KERNEL_MEMORY_MB", 4096)
# Seconds to wait for the shim's `{"ready": true}` — image build on a cold
# cache included, which is why it is generous.
BOOT_SECONDS = _env_int("DD_KERNEL_BOOT_SECONDS", 600)
_BUSY_WAIT_SECONDS = 1.0

_SHIM_PATH = Path(__file__).with_name("modal_kernel_shim.py")
_SHIM_REMOTE = "/opt/delta/kernel_shim.py"


def _import_modal():
    import modal  # noqa: WPS433 — optional dependency, imported on first use
    return modal


_app = None
_image = None
_setup_lock = threading.Lock()


def _ensure_app():
    """The Modal app + image, built once per process. The image definition is
    hashed by Modal, so an unchanged definition costs one lookup, not a
    rebuild."""
    global _app, _image
    with _setup_lock:
        if _app is not None:
            return _app, _image
        modal = _import_modal()
        _app = modal.App.lookup(APP_NAME, create_if_missing=True)
        sparse = " ".join(f"{c}/exercises" for c in ARENA_CHAPTERS)
        _image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install("wget", "unzip", "sudo", "git")
            .pip_install("torch", "torchvision",
                         extra_index_url="https://download.pytorch.org/whl/cpu")
            .pip_install(
                "ipykernel", "jupyter_client", "ipython", "jupyter",
                "einops", "jaxtyping", "numpy", "plotly", "pandas", "tqdm", "rich",
                "torchinfo", "datasets", "pillow", "matplotlib", "wandb", "eindex-callum",
            )
            # Colab-shaped: /root/<chapter>/exercises exists, so ARENA's own
            # setup cell skips its download branch. Sparse + blobless keeps
            # the 100 MB repo down to the ~15 MB the exercises need.
            .run_commands(
                "cd /tmp && git clone --filter=blob:none --no-checkout --depth 1 "
                f"{ARENA_REPO} arena || git clone --filter=blob:none --no-checkout {ARENA_REPO} arena",
                f"cd /tmp/arena && git fetch --depth 1 origin {ARENA_SHA} && "
                f"git sparse-checkout init --cone && git sparse-checkout set {sparse} && "
                f"git checkout {ARENA_SHA}",
                *(f"mv /tmp/arena/{c} /root/{c}" for c in ARENA_CHAPTERS),
                "rm -rf /tmp/arena",
            )
            .add_local_file(_SHIM_PATH, remote_path=_SHIM_REMOTE, copy=True)
        )
        return _app, _image


# --- one learner's session ---------------------------------------------------

@dataclass
class ModalSession:
    session_id: str
    sandbox: object
    proc: object
    created_at: float
    last_used: float
    context: str = ""
    exec_count: int = 0
    dead: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    _chunks: object = None
    _buf: str = ""

    @property
    def alive(self) -> bool:
        return not self.dead

    def rss_mb(self) -> float:
        return 0.0  # the sandbox's memory is Modal's to report

    def _readline(self, timeout: float) -> str | None:
        """One line from the shim, or None when nothing arrived in time. The
        stream reader yields chunks, not lines, and blocks — so it is drained
        on a helper thread and waited on with a deadline.

        A pull that outlives its deadline keeps blocking on `next()` and will
        swallow the chunk that eventually arrives. That is fine ONLY because
        every caller treats None as fatal: the session is marked dead and the
        sandbox terminated, so nothing reads this stream again."""
        if self._chunks is None:
            self._chunks = iter(self.proc.stdout)
        deadline = time.monotonic() + timeout
        while "\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            holder: dict = {}

            def _pull():
                try:
                    holder["chunk"] = next(self._chunks)
                except StopIteration:
                    holder["eof"] = True
                except Exception as exc:  # stream torn down under us
                    holder["error"] = exc

            worker = threading.Thread(target=_pull, daemon=True)
            worker.start()
            worker.join(remaining)
            if worker.is_alive():
                return None
            if holder.get("eof") or holder.get("error"):
                return None
            self._buf += holder.get("chunk", "")
        line, self._buf = self._buf.split("\n", 1)
        return line

    def execute(self, code: str, filename: str = "<cell>",
                timeout: int = DEFAULT_TIMEOUT_SECONDS) -> ExecutionResult:
        """Run one cell. The shim owns the per-cell timeout and the interrupt;
        this side only guards against the shim itself going silent."""
        self.last_used = time.monotonic()
        try:
            self.proc.stdin.write(json.dumps({"op": "exec", "code": code, "timeout": timeout}) + "\n")
            self.proc.stdin.drain()
        except Exception as exc:
            self.dead = True
            return ExecutionResult(stdout="", stderr=f"The Python session was lost ({exc}).\n",
                                   success=False)
        line = self._readline(timeout + INTERRUPT_GRACE + 15)
        self.last_used = time.monotonic()
        self.exec_count += 1
        if line is None:
            self.dead = True
            logger.warning("modal kernel: %s went silent; shim stderr: %s",
                           self.session_id, self._stderr_tail())
            return ExecutionResult(
                stdout="", success=False,
                stderr=f"Cell timed out after {timeout} seconds and the session did not "
                       "come back; it was restarted.\n")
        try:
            reply = json.loads(line)
        except ValueError:
            self.dead = True
            return ExecutionResult(stdout="", stderr="The Python session sent an unreadable "
                                   "reply and was restarted.\n", success=False)
        if reply.get("dead"):
            self.dead = True
        return ExecutionResult(
            stdout=reply.get("stdout", ""),
            stderr=reply.get("stderr", ""),
            success=bool(reply.get("success")),
            outputs=list(reply.get("outputs") or []),
        )

    def _stderr_tail(self) -> str:
        """What the shim wrote to stderr — read without blocking for long, and
        only for the log. On the happy path nobody reads this."""
        try:
            holder: dict = {}

            def _pull():
                try:
                    holder["text"] = "".join(list(self.proc.stderr))
                except Exception as exc:
                    holder["text"] = f"<unreadable: {exc}>"

            worker = threading.Thread(target=_pull, daemon=True)
            worker.start()
            worker.join(3)
            return (holder.get("text") or "<nothing>")[-1500:]
        except Exception as exc:
            return f"<unreadable: {exc}>"

    def shutdown(self) -> None:
        self.dead = True
        try:
            self.proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
            self.proc.stdin.drain()
        except Exception:
            pass
        try:
            self.sandbox.terminate()
        except Exception as exc:
            logger.warning("modal kernel: terminate %s failed: %s", self.session_id, exc)


INTERRUPT_GRACE = 5


def _spawn(session_id: str, context: str = "") -> ModalSession:
    modal = _import_modal()
    app, image = _ensure_app()
    sandbox = modal.Sandbox.create(
        "sleep", "infinity",
        app=app, image=image, workdir="/root",
        cpu=SANDBOX_CPU, memory=SANDBOX_MEMORY_MB,
        timeout=SANDBOX_LIFETIME_SECONDS, idle_timeout=IDLE_SECONDS,
        tags={"session": session_id, "context": context[:60]},
    )
    now = time.monotonic()
    # From here on the sandbox bills. Every failure path below must terminate
    # it, or a bad exec / torn stream leaves a container running until its
    # lifetime cap.
    try:
        proc = sandbox.exec("python", "-u", _SHIM_REMOTE)
        session = ModalSession(session_id=session_id, sandbox=sandbox, proc=proc,
                               created_at=now, last_used=now, context=context)
        ready = session._readline(BOOT_SECONDS)
        try:
            is_ready = bool(ready) and bool(json.loads(ready).get("ready"))
        except ValueError:
            is_ready = False
        if not is_ready:
            raise RuntimeError("The Python session could not start — try again in a moment.")
    except BaseException:
        try:
            sandbox.terminate()
        except Exception as exc:
            logger.warning("modal kernel: terminate after failed start: %s", exc)
        raise
    logger.info("modal kernel: %s up as %s (%.1fs)", session_id, sandbox.object_id,
                time.monotonic() - now)
    return session


# --- the registry -------------------------------------------------------------

_kernels: dict[str, ModalSession] = {}
_registry_lock = threading.Lock()


def _reap_locked() -> None:
    now = time.monotonic()
    for sid, session in list(_kernels.items()):
        if session.lock.locked():
            continue
        if session.dead or now - session.last_used > IDLE_SECONDS:
            _kernels.pop(sid, None)
            session.shutdown()


def _evict_lru_locked() -> None:
    while len(_kernels) >= MAX_KERNELS:
        idle = [s for s in _kernels.values() if not s.lock.locked()]
        if not idle:
            raise RuntimeError("Every kernel is busy right now — try again in a moment.")
        victim = min(idle, key=lambda s: s.last_used)
        _kernels.pop(victim.session_id, None)
        victim.shutdown()


def run_cell(session_id: str | None, code: str, bootstrap: str = "",
             filename: str = "<cell>", context: str = "",
             timeout: int = DEFAULT_TIMEOUT_SECONDS,
             skip_on_fresh: bool = False) -> KernelRunResult:
    """Same contract as `kernel_runner.run_cell` — see its docstring."""
    for _attempt in range(2):
        outcome = _reserve_and_run(session_id, code, bootstrap, filename, context,
                                   timeout, skip_on_fresh)
        if outcome is not None:
            return outcome
    raise RuntimeError("The kernel was restarted mid-request — run the cell again.")


def _reserve_and_run(session_id, code, bootstrap, filename, context, timeout,
                     skip_on_fresh) -> KernelRunResult | None:
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
            session_id = session_id or os.urandom(8).hex()
            # Spawning can take a minute on a cold image; the registry lock
            # is not held across it. A placeholder keeps a second click from
            # spawning a twin.
            placeholder = ModalSession(session_id=session_id, sandbox=None, proc=None,
                                       created_at=time.monotonic(), last_used=time.monotonic(),
                                       context=context)
            placeholder.lock.acquire()
            _kernels[session_id] = placeholder
            fresh = True
    if fresh:
        try:
            session = _spawn(session_id, context=context)
        except Exception as exc:
            with _registry_lock:
                _kernels.pop(session_id, None)
            placeholder.lock.release()
            logger.exception("modal kernel: spawn failed for %s", session_id)
            raise RuntimeError(str(exc) or "The Python session could not start.") from exc
        with _registry_lock:
            if _kernels.get(session_id) is not placeholder:
                session.shutdown()
                placeholder.lock.release()
                return None
            _kernels[session_id] = session
        placeholder.lock.release()

    if not session.lock.acquire(timeout=_BUSY_WAIT_SECONDS):
        raise RuntimeError("This kernel is already running a cell.")
    try:
        with _registry_lock:
            if _kernels.get(session.session_id) is not session:
                return None
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
            session.shutdown()


def reset_kernel(session_id: str) -> bool:
    with _registry_lock:
        session = _kernels.pop(session_id, None)
    if session is None:
        return False
    session.shutdown()
    return True


def kernel_status(session_id: str | None = None) -> dict:
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
                "rss_mb": 0.0,
                "sandbox_id": getattr(s.sandbox, "object_id", None),
            }
            for s in shown
        ]
    return {
        "backend": "modal",
        "count": total,
        "max_kernels": MAX_KERNELS,
        "idle_seconds": IDLE_SECONDS,
        "max_rss_mb": SANDBOX_MEMORY_MB,
        "sessions": sessions,
    }


def shutdown_all() -> None:
    with _registry_lock:
        sessions = list(_kernels.values())
        _kernels.clear()
    for session in sessions:
        session.shutdown()


def sweep_orphans() -> int:
    """Terminate every sandbox under the app that this process does not hold.
    Called at startup: a sandbox from a previous process still bills, and its
    stdin belonged to a process that no longer exists."""
    modal = _import_modal()
    try:
        app, _ = _ensure_app()
        held = {getattr(s.sandbox, "object_id", None) for s in _kernels.values()}
        count = 0
        for sandbox in modal.Sandbox.list(app_id=app.app_id):
            if sandbox.object_id in held:
                continue
            try:
                sandbox.terminate()
                count += 1
            except Exception as exc:
                logger.warning("modal kernel: orphan %s not terminated: %s",
                               sandbox.object_id, exc)
        if count:
            logger.info("modal kernel: swept %d orphan sandbox(es)", count)
        return count
    except Exception as exc:
        logger.warning("modal kernel: orphan sweep skipped: %s", exc)
        return 0
