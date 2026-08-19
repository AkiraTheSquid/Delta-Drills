/* ================================================================
   NOTEBOOK KERNEL CLIENT — one live Python session per learner
   ================================================================

   `DeltaRunner.runSnippet` runs a block of code and forgets it: Pyodide is
   reset per call and the backend's `/run-code` forks a fresh process per
   submission. That is right for grading and wrong for a notebook, where cell 8
   is entitled to the `a` that cell 6 bound.

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

  /* Set only when the backend has told us it has no kernel endpoints. A
     network blip must NOT set this — see the header. */
  let unsupported = false;

  const available = () =>
    !unsupported &&
    typeof apiFetch === "function" &&
    typeof practiceMode !== "undefined" &&
    practiceMode === "backend";

  /* Run one cell in the learner's session.

     Returns { ok, stdout, stderr, fresh, busy, unavailable }. `unavailable`
     is the caller's signal to fall back to the stateless runner — it never
     carries output, and it is never an error the learner should read.

     `skipOnFresh` asks the server not to run this code if it had to CREATE the
     kernel: the reply comes back `fresh` with no output, and the caller sends
     the prefix instead. Without it the clicked cell runs once here and again
     inside the prefix, which is wrong for anything that is not idempotent. */
  const runCell = async ({ code, bootstrap = "", filename = "<cell>", context = "", timeout = 30, skipOnFresh = false } = {}) => {
    if (!available()) return { unavailable: true };
    let res;
    try {
      res = await apiFetch(EXEC_PATH, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code, bootstrap, filename, context, timeout,
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
    // 409 is the kernel saying it is mid-cell, or the box saying it is full.
    // Both are "in a moment", not a failure of this code.
    if (res.status === 409) return { busy: true };
    if (res.status === 401 || !res.ok) return { unavailable: true };
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
      fresh: !!data.fresh,
      execCount: data.exec_count || 0,
    };
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

  return { available, runCell, reset };
})();

window.DeltaKernel = DeltaKernel;
