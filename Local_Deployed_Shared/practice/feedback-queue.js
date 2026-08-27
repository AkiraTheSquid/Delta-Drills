/* ================================================================
   FEEDBACK QUEUE — the reports that have not reached the server yet

   A report written while signed out, or while the backend was unreachable, is
   kept here and pushed later. Split out of api.js because it is a delivery
   problem, not a practice one, and because every bug found in it so far was a
   lifecycle bug — a drain that overwrote what arrived during it, a latch that
   only covered one tab, an unbounded retry — which are easier to see, and to
   pin, on their own.

   Loaded BEFORE practice/api.js. `apiFetch` and `authEmail` are app.js
   top-level consts, visible to any later classic script but not on `window`.
   ================================================================ */

const DDFeedbackQueue = (() => {
  /* One drain at a time, per queue.

     🔴 The naive version — read the whole queue, await N posts, setItem(rest) —
     loses feedback two ways. Anything the learner queues DURING the drain is
     absent from the snapshot and is erased by that final write, and two
     overlapping drains send the same entries twice. So: a per-key in-flight
     latch, and the write-back re-reads storage and removes only the entries this
     drain actually delivered, one occurrence each. Whatever arrived meanwhile
     stays. */
  const _flushing = new Set();

  /* Whose feedback is this? A queue lives in ONE browser, which more than one
     person can sign into. Stamping the account at write time and refusing to
     send another account's entries is what stops a report typed by the previous
     person being filed — and shown back — under the next one's name. An unstamped
     entry was written signed OUT, which is the same person about to sign in. */
  const _currentAccount = () =>
    (typeof authEmail === "string" ? authEmail : localStorage.getItem("auth_email") || "").trim();

  const _sendableHere = (entry) => {
    const owner = (entry && entry.account) || "";
    return !owner || owner === _currentAccount();
  };

  /* Minted where the report is WRITTEN, so a retry carries the id of the report
     it is retrying. The backend ignores a client_id it has already stored, which
     is what makes a lost response, or a tab closed mid-drain, harmless. */
  const _clientId = () => {
    try {
      if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
    } catch (_) {
      /* fall through */
    }
    return "c-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
  };

  const _readQueue = (key) => {
    try {
      const parsed = JSON.parse(localStorage.getItem(key) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch (_) {
      return [];
    }
  };

  /* Returns true only if the entry is genuinely on disk. Quota exhaustion and
     disabled storage both land here, and the panel must not call either one
     "saved". */
  const _queueFeedback = (key, entry) => {
    try {
      const queue = _readQueue(key);
      queue.push({
        ...entry,
        account: _currentAccount(),
        timestamp: new Date().toISOString(),
      });
      localStorage.setItem(key, JSON.stringify(queue));
      return true;
    } catch (_) {
      return false;
    }
  };

  /* Rounds, not recursion without end. 5 x 50 = 250 reports per trigger: enough
     that a long-offline learner empties a realistic queue in one go, bounded so
     a corrupt queue that never shrinks cannot turn one boot into thousands of
     requests. What is left waits for the next report or the next load. */
  const _FLUSH_BATCH = 50;
  const _FLUSH_ROUNDS = 5;

  const _flushFeedbackQueue = async (key, path, round = 0) => {
    if (_flushing.has(key)) return;
    const queue = _readQueue(key).filter(_sendableHere);
    if (!queue.length) return;
    _flushing.add(key);
    const delivered = [];
    // A queue longer than one batch has to come back for the rest, or a learner
    // who was offline for a week strands everything past entry 50 until they
    // reload AGAIN (codex, round 4).
    let more = false;
    try {
      // Bounded: a long-offline learner drains 50 per successful report, so one
      // send never turns into a hundred requests.
      for (const entry of queue.slice(0, _FLUSH_BATCH)) {
        const { timestamp: _ts, account: _acct, ...body } = entry || {};
        try {
          const res = await apiFetch(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });
          if (res.ok) delivered.push(JSON.stringify(entry));
        } catch (_) {
          /* keep it queued */
        }
      }
      if (!delivered.length) return;
      more = queue.length > _FLUSH_BATCH && round + 1 < _FLUSH_ROUNDS;
      // Re-read: this is the merge. Drop one occurrence per delivered entry.
      const current = _readQueue(key);
      const pending = delivered.slice();
      const remaining = current.filter((entry) => {
        const at = pending.indexOf(JSON.stringify(entry));
        if (at === -1) return true;
        pending.splice(at, 1);
        return false;
      });
      try {
        localStorage.setItem(key, JSON.stringify(remaining));
      } catch (_) {
        /* ignore storage errors */
      }
    } finally {
      _flushing.delete(key);
    }
    if (more) await _flushFeedbackQueue(key, path, round + 1);
  };

  const FEEDBACK_QUEUES = [
    ["problem_feedback_queue", "/api/practice/problem-feedback"],
    ["lesson_feedback_queue", "/api/practice/lesson-feedback"],
  ];

  /* Drain whatever is still on this device. Signing in RELOADS the app into
     backend mode (app.js::setAuthState), so a boot-time call is the sign-in
     trigger — without it a report only leaves the browser when the learner
     happens to file a second one on the same surface, which the panel's
     "it sends when you are signed in" promises it does not require. */
  const flushAll = () => {
    if (typeof practiceMode !== "undefined" && practiceMode !== "backend") return;
    FEEDBACK_QUEUES.forEach(([key, path]) => {
      _flushFeedbackQueue(key, path).catch(() => {});
    });
  };

  return {
    queue: _queueFeedback,
    flush: _flushFeedbackQueue,
    flushAll,
    clientId: _clientId,
    QUEUES: FEEDBACK_QUEUES,
  };
})();

window.DDFeedbackQueue = DDFeedbackQueue;
