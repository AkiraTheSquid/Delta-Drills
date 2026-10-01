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

Lifetimes: Modal's `idle_timeout` never fires here — the shim is a live
`sandbox.exec` for the sandbox's whole life, and Modal counts a running exec
as activity. So idle sandboxes are closed by `start_reaper`'s thread, which
checks every minute; without it a learner who ran two cells and left billed
until the lifetime cap (Sept 2026: every sandbox ran the full 4 h, $56). The
`timeout` is a hard ceiling that stops a forgotten sandbox billing forever if
this process dies holding the only handle. A sandbox whose handle this process
lost (redeploy) is unreachable by design — its stdin belonged to the old
process — so startup sweeps and terminates every sandbox under the app.

GPU sections (`modal_gpu`): a context ending in `#gpu` for a section that has
a GPU part gets an L4 sandbox on the CUDA image, the prefetched weights, and a
shorter idle close. Its own admission (GPUs free, the learner's monthly GPU
hours) is checked BEFORE the learner's CPU sandbox is closed, so a refused GPU
cell leaves the kernel they had.

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

from app import modal_gpu, modal_image, modal_spend
from app.code_runner import ExecutionResult
from app.kernel_runner import KernelRunResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30
APP_NAME = os.environ.get("DD_MODAL_APP", "delta-drills-kernels")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# Caps are per learner-session now, not per box. MAX_KERNELS is a spend guard:
# every live sandbox bills by the second.
MAX_KERNELS = _env_int("DD_KERNEL_MAX", 20)
IDLE_SECONDS = _env_int("DD_KERNEL_IDLE_SECONDS", 900)
REAP_INTERVAL_SECONDS = max(5, _env_int("DD_KERNEL_REAP_SECONDS", 60))
SANDBOX_LIFETIME_SECONDS = _env_int("DD_KERNEL_LIFETIME_SECONDS", 2 * 3600)
# Requests, not caps. Modal bills max(request, usage) at sandbox rates
# ($0.142/core/h, $0.024/GiB/h), and a sandbox bursts above its CPU request,
# so a small request costs little while a learner reads and still gets the
# cores when a cell trains. The old 2 cores + 4 GiB billed $0.38/h idle.
SANDBOX_CPU = float(os.environ.get("DD_KERNEL_CPU", "0.25"))
SANDBOX_MEMORY_MB = _env_int("DD_KERNEL_MEMORY_MB", 1024)
# Modal sets OMP_NUM_THREADS to ceil(cpu request) — 1 at 0.25 — and torch
# sizes its pool from it, which made a matmul 2.4x slower than on the old
# 2-core sandbox. Pinning the threads gets the burst back; it bills as used.
SANDBOX_THREADS = str(_env_int("DD_KERNEL_THREADS", 2))
# Seconds to wait for the shim's `{"ready": true}` — image build on a cold
# cache included, which is why it is generous.
BOOT_SECONDS = _env_int("DD_KERNEL_BOOT_SECONDS", 600)
# Seconds a spawn may take END TO END before the request gives up on it.
# `modal.Sandbox.create` has no timeout of its own: on 2026-09-12 the prod
# process's spawn threads never returned (the box was starved; ssh hung too),
# and because the placeholder held the session lock the whole time, every
# later click was answered "already running a cell" until the machine was
# restarted. Above BOOT_SECONDS is meaningless — the shim wait is inside it.
SPAWN_SECONDS = _env_int("DD_KERNEL_SPAWN_SECONDS", 120)
_BUSY_WAIT_SECONDS = 1.0


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
        _image = modal_image.build(modal)
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
    gpu: bool = False
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
        modal_spend.record(getattr(self.sandbox, "object_id", None), self.session_id,
                           self.context, _wall(self.created_at), gpu=self.gpu)
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
    gpu = modal_gpu.wants_gpu(context)
    options = {"cpu": SANDBOX_CPU, "memory": SANDBOX_MEMORY_MB}
    env = {"OMP_NUM_THREADS": SANDBOX_THREADS, "MKL_NUM_THREADS": SANDBOX_THREADS}
    if gpu:
        image = modal_gpu.image(modal)
        options = modal_gpu.sandbox_options(modal)
        env.update(modal_gpu.SANDBOX_ENV)
    sandbox = modal.Sandbox.create(
        "sleep", "infinity",
        app=app, image=image, workdir="/root",
        timeout=SANDBOX_LIFETIME_SECONDS, idle_timeout=_idle_for(gpu),
        env=env, tags={"session": session_id, "context": context[:60]},
        **options,
    )
    now = time.monotonic()
    # From here on the sandbox bills. Every failure path below must terminate
    # it, or a bad exec / torn stream leaves a container running until its
    # lifetime cap.
    try:
        if gpu:
            modal_gpu.link_weights(sandbox)
        proc = sandbox.exec("python", "-u", modal_image.SHIM_REMOTE)
        session = ModalSession(session_id=session_id, sandbox=sandbox, proc=proc,
                               created_at=now, last_used=now, context=context, gpu=gpu)
        ready = session._readline(BOOT_SECONDS)
        try:
            is_ready = bool(ready) and bool(json.loads(ready).get("ready"))
        except ValueError:
            is_ready = False
        if not is_ready:
            raise RuntimeError("The Python session could not start — try again in a moment.")
    except BaseException:
        # It billed from create to here; a GPU start that keeps failing must
        # still count against the month and the learner's GPU hours.
        modal_spend.record(getattr(sandbox, "object_id", None), session_id, context,
                           _wall(now), gpu=gpu)
        try:
            sandbox.terminate()
        except Exception as exc:
            logger.warning("modal kernel: terminate after failed start: %s", exc)
        raise
    logger.info("modal kernel: %s up as %s%s (%.1fs)", session_id, sandbox.object_id,
                " on " + modal_gpu.GPU_TYPE if gpu else "", time.monotonic() - now)
    return session


def _spawn_with_deadline(session_id: str, context: str, deadline: float) -> ModalSession:
    """`_spawn`, but the caller stops waiting after `deadline` seconds.

    The spawn itself keeps running on its own thread — there is no way to
    cancel a `Sandbox.create` mid-flight — so a sandbox that arrives AFTER the
    caller gave up is terminated on the spot: nobody holds it, and it bills.
    The gate makes "did the caller give up" and "did the spawn land" one
    decision, so a late arrival can neither leak nor be handed to no one."""
    gate = threading.Lock()
    outcome: dict = {}
    finished = threading.Event()

    def _work():
        try:
            session = _spawn(session_id, context=context)
        except BaseException as exc:  # surfaced to the caller below
            outcome["error"] = exc
            finished.set()
            return
        with gate:
            if outcome.get("abandoned"):
                logger.warning("modal kernel: %s spawned after the %ss deadline; terminating",
                               session_id, deadline)
                session.shutdown()
            else:
                outcome["session"] = session
        finished.set()

    threading.Thread(target=_work, daemon=True, name=f"modal-spawn-{session_id}").start()
    finished.wait(deadline)
    with gate:
        if "session" in outcome:
            return outcome["session"]
        if "error" in outcome:
            raise outcome["error"]
        outcome["abandoned"] = True
    raise RuntimeError(f"The Python session did not start within {int(deadline)} seconds "
                       "— try again in a moment.")


# --- the registry -------------------------------------------------------------

_kernels: dict[str, ModalSession] = {}
_registry_lock = threading.Lock()


def _wall(monotonic_ts: float) -> float:
    return time.time() - (time.monotonic() - monotonic_ts)


def _idle_for(gpu: bool) -> int:
    return modal_gpu.GPU_IDLE_SECONDS if gpu else IDLE_SECONDS


def _spend_locked() -> dict:
    """The month's spend, counting the sandboxes open right now."""
    live = [(_wall(s.created_at), s.gpu) for s in _kernels.values() if s.sandbox is not None]
    return modal_spend.summary(live, modal_spend.request_rate(SANDBOX_CPU, SANDBOX_MEMORY_MB),
                               modal_gpu.usd_per_hour())


def _gpu_starts_locked(session_id: str | None) -> list[float]:
    """Wall-clock starts of this learner's open GPU sandboxes."""
    return [_wall(s.created_at) for s in _kernels.values()
            if s.gpu and s.sandbox is not None and s.session_id == session_id]


def _admit_locked(session_id: str | None, context: str, replacing: ModalSession | None) -> None:
    """Raise with the learner-facing reason when a new sandbox for `context`
    may not start. Runs BEFORE `replacing` is closed, so a refusal costs the
    learner nothing: the kernel they had is still there."""
    if _spend_locked()["capped"]:
        raise RuntimeError(modal_spend.CAP_MESSAGE)
    if not modal_gpu.wants_gpu(context):
        return
    # The sandbox being replaced frees its GPU, but its hours this month count.
    live_gpu = sum(1 for s in _kernels.values() if s.gpu and s is not replacing)
    modal_gpu.admit(session_id or "", live_gpu, _gpu_starts_locked(session_id))


def _pop_stale_locked() -> list[ModalSession]:
    now = time.monotonic()
    stale = []
    for sid, session in list(_kernels.items()):
        if session.lock.locked():
            continue
        if session.dead or now - session.last_used > _idle_for(session.gpu):
            _kernels.pop(sid, None)
            stale.append(session)
    return stale


def _reap_locked() -> None:
    for session in _pop_stale_locked():
        session.shutdown()


_reaper_started = False


def start_reaper() -> None:
    """Close idle sandboxes on a timer. The request path reaps too, but only
    when the NEXT request arrives — after the last learner leaves there is no
    next request, and Modal's own idle_timeout cannot see the idleness (see
    the module docstring). Terminating happens outside the registry lock: it
    is a network call, and requests must not queue behind it."""
    global _reaper_started
    with _registry_lock:
        if _reaper_started:
            return
        _reaper_started = True
    # The month's bill is read on its own thread: a stalled billing call here
    # would stop the reaping, and idle sandboxes would bill on. Startup waits a
    # bounded time for the first read so the cap starts from the real bill.
    modal_spend.start_refresher()

    def _loop() -> None:
        while True:
            time.sleep(REAP_INTERVAL_SECONDS)
            try:
                with _registry_lock:
                    stale = _pop_stale_locked()
            except Exception:
                logger.exception("modal kernel: reaper pass failed")
                continue
            for session in stale:
                try:
                    logger.info("modal kernel: reaping idle %s", session.session_id)
                    session.shutdown()
                except Exception:
                    logger.exception("modal kernel: reaping %s failed", session.session_id)

    threading.Thread(target=_loop, daemon=True, name="modal-reaper").start()


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
            _admit_locked(session_id, context, session)
            _kernels.pop(session.session_id, None)
            session.shutdown()
            session = None
        elif session is None:
            _admit_locked(session_id, context, None)
        if session is None:
            _evict_lru_locked()
            session_id = session_id or os.urandom(8).hex()
            # Spawning can take a minute on a cold image; the registry lock
            # is not held across it. A placeholder keeps a second click from
            # spawning a twin.
            # It carries `gpu` so a second learner's admission counts it.
            placeholder = ModalSession(session_id=session_id, sandbox=None, proc=None,
                                       created_at=time.monotonic(), last_used=time.monotonic(),
                                       context=context, gpu=modal_gpu.wants_gpu(context))
            placeholder.lock.acquire()
            _kernels[session_id] = placeholder
            fresh = True
    if fresh:
        try:
            deadline = modal_gpu.SPAWN_SECONDS if placeholder.gpu else SPAWN_SECONDS
            session = _spawn_with_deadline(session_id, context, deadline)
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
        # A placeholder (no sandbox yet) is a spawn in flight, not a cell.
        # Saying "running a cell" here is what made a hung spawn read as a
        # runaway cell — nothing was running.
        if session.sandbox is None:
            raise RuntimeError("Your Python session is still starting — give it a moment.")
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
                "gpu": s.gpu,
                "alive": s.alive,
                "exec_count": s.exec_count,
                "idle_seconds": round(now - s.last_used, 1),
                "age_seconds": round(now - s.created_at, 1),
                "rss_mb": 0.0,
                "sandbox_id": getattr(s.sandbox, "object_id", None),
            }
            for s in shown
        ]
        spend = _spend_locked()
    return {
        "backend": "modal",
        "count": total,
        "max_kernels": MAX_KERNELS,
        "idle_seconds": IDLE_SECONDS,
        "max_rss_mb": SANDBOX_MEMORY_MB,
        "sessions": sessions,
        "spend": spend,
        "gpu": {"type": modal_gpu.GPU_TYPE, "max": modal_gpu.GPU_MAX,
                "idle_seconds": modal_gpu.GPU_IDLE_SECONDS,
                "live": sum(1 for s in _kernels.values() if s.gpu)},
    }


def compute_view(session_id: str | None) -> dict:
    """What a notebook needs to place its compute banner and GPU switch."""
    with _registry_lock:
        starts = _gpu_starts_locked(session_id)
    return modal_gpu.learner_view(session_id, starts, enabled=True)


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
