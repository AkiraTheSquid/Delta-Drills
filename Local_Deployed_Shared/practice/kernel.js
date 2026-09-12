/* ================================================================
   NOTEBOOK KERNEL CLIENT — one live Python session per learner
   ================================================================

   Grading uses fresh processes. Notebook Run uses this client instead: cell 8
   is entitled to names cell 6 bound, until learner restarts runtime or server
   evicts idle kernel.

   This talks to `/api/practice/kernel/exec`, which keeps a forked Python
   process alive between calls on the backend. The rules:

     * Signed-in learners only. A kernel is a real process on a small box, so
       the endpoint is authenticated; a guest keeps the stateless path.
     * The server holds the session, the client holds the CELL SEMANTICS. The
       `_delta_cell` harness (last-expression echo, `<cell N>` tracebacks) is
       sent as `bootstrap` and installed by the server only on a kernel it had
       to create, so there is one copy of it and installing it cannot race an
       eviction between two clicks.
     * `fresh: true` in the reply means the server had no kernel and made one —
       the client's cue that everything it thought was defined is gone.
     * A backend that does not know these endpoints (404) turns the kernel off
       for the page rather than failing every cell. Transient failures do not:
       one bad network moment should cost one cell, not the session.
*/

const DeltaKernel = (() => {
  const EXEC_PATH = "/api/practice/kernel/exec";
  const RESET_PATH = "/api/practice/kernel/reset";
  const STATUS_PATH = "/api/practice/kernel/status";
  /* 🔴 Mirrors `MAX_TIMEOUT_SECONDS` in backend/app/practice/kernel_router.py.
     The server 422s a larger value, and a 422 used to come back as
     `unavailable` — so an ARENA setup cell asking for 300 s against a 60 s cap
     opened every notebook with "Python unavailable" (2026-09-11). Clamp here
     so no caller can trip that, and keep the two numbers equal. */
  const MAX_TIMEOUT = 300;

  /* Set only when the backend has told us it has no kernel endpoints. A
     network blip must NOT set this — see the header. */
  let unsupported = false;

  const available = () =>
    !unsupported &&
    typeof apiFetch === "function" &&
    typeof practiceMode !== "undefined" &&
    practiceMode === "backend";

  /* Run one cell in the learner's session.

     Returns { ok, stdout, stderr, fresh, busy, detail, unavailable }. `unavailable`
     is the caller's signal to fall back to the stateless runner — it never
     carries output, and it is never an error the learner should read.

     `skipOnFresh` asks the server not to run this code if it had to CREATE the
     kernel: the reply comes back `fresh` with no output, and the caller sends
     the prefix instead. Without it the clicked cell runs once here and again
     inside the prefix, which is wrong for anything that is not idempotent. */
  const runCell = async ({ code, bootstrap = "", filename = "<cell>", context = "", timeout = 30, skipOnFresh = false } = {}) => {
    if (!available()) return { unavailable: true };
    const seconds = Math.min(MAX_TIMEOUT, Math.max(1, Math.round(Number(timeout) || 0)));
    let res;
    try {
      res = await apiFetch(EXEC_PATH, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code, bootstrap, filename, context, timeout: seconds,
          skip_on_fresh: !!skipOnFresh,
        }),
      });
    } catch (_networkErr) {
      return { unavailable: true };
    }
    if (res.status === 404) {
      // Older backend. Stop asking for the rest of the page.
      unsupported = true;
      return { unavailable: true };
    }
    // 409 is the kernel saying it is mid-cell, the box saying it is full, or
    // (2026-09-12) a spawn that is still in flight or gave up. All are "in a
    // moment", not a failure of this code — but they are DIFFERENT moments,
    // and the server's `detail` says which. Carry it: collapsing every 409
    // into "still running a cell" told the learner a runaway cell was the
    // problem on a day when nothing was running at all.
    if (res.status === 409) return { busy: true, detail: await _detail(res) };
    // Signed out, or a token the server no longer honours: no kernel to be
    // had, and the stateless path is the right answer.
    if (res.status === 401 || res.status === 403) return { unavailable: true };
    // A 4xx the server sent about THIS request (422 above all: a body the
    // contract did not allow) is a real answer about this cell, not "no
    // Python" — saying so sent the learner to "retry when connected" for a bug
    // no reconnect could fix. A 5xx, a 429 or a proxy failure is the outage
    // the header describes, and keeps the stateless fallback (codex, 2026-09-11).
    if (res.status >= 400 && res.status < 500 && res.status !== 429) {
      return { ok: false, stdout: "", stderr: await _refusal(res), outputs: [], fresh: false, execCount: 0 };
    }
    if (!res.ok) return { unavailable: true };
    let data;
    try {
      data = await res.json();
    } catch (_parseErr) {
      return { unavailable: true };
    }
    return {
      ok: !!data.success,
      stdout: data.stdout || "",
      stderr: data.stderr || "",
      // display_data mimebundles — see practice/cell-outputs.js. Empty on a
      // backend that has no display channel.
      outputs: Array.isArray(data.outputs) ? data.outputs : [],
      fresh: !!data.fresh,
      execCount: data.exec_count || 0,
    };
  };

  /* One line the learner can read AND report: the status, and the server's
     own `detail` when it sent one (FastAPI's validation errors do). */
  // The server's `detail` string, or "" when the body carried none.
  const _detail = async (res) => {
    try {
      const data = await res.json();
      return typeof data?.detail === "string" ? data.detail : JSON.stringify(data?.detail ?? data);
    } catch (_err) {
      return "";
    }
  };

  const _refusal = async (res) => {
    const detail = await _detail(res);
    return `The kernel server refused this cell (HTTP ${res.status})${detail ? `: ${detail}` : "."}\n`;
  };

  /* Throw the session away — the notebook's "Restart runtime". The next cell
     comes back `fresh`, which is how the caller learns to rebuild state. */
  const reset = async () => {
    if (!available()) return false;
    try {
      const res = await apiFetch(RESET_PATH, { method: "POST" });
      return res.ok;
    } catch (_err) {
      return false;
    }
  };

  /* What the server is holding for THIS learner, without running anything.

     `sessions[]` carries the `context` each live kernel was last used for and
     whether it is `alive`, which is the one question a notebook needs on the
     way in: are the names my saved outputs came from still bound, or is this a
     photograph of a session that idled out? (practice/arena-notebook-outputs.js
     — a restored output over a dead kernel is the misleading state.)

     Returns null for every unhappy answer — signed out, older backend, network
     blip. 🔴 A 404 here does NOT set `unsupported`: a build that has exec but
     not status must keep running cells. Only /exec may make that call. */
  const status = async () => {
    if (!available()) return null;
    try {
      const res = await apiFetch(STATUS_PATH);
      if (!res.ok) return null;
      return await res.json();
    } catch (_err) {
      return null;
    }
  };

  /* Is there a live kernel bound to this context right now? `null` when the
     question could not be asked at all — which is NOT the same as "no", and
     the caller must not paint a warning on it. */
  const contextAlive = async (context) => {
    const info = await status();
    if (!info || !Array.isArray(info.sessions)) return null;
    return info.sessions.some((s) => s && s.alive && s.context === String(context || ""));
  };

  return { available, runCell, reset, status, contextAlive, MAX_TIMEOUT };
})();

window.DeltaKernel = DeltaKernel;
