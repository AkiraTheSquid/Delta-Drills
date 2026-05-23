/* ================================================================
   PRACTICE — ARENA UNLOCK TIMER (in-card stopwatch + auto-rating)

   Lightweight stopwatch that lives inside the ARENA unlock card. The
   student presses Start when they begin the exercise (in Colab), Stop
   when they finish. Their elapsed time vs. the per-exercise target
   becomes a feedback rating that flows back to the Delta Drills
   adaptive engine for all of that exercise's prereq subtopics.

   Module boundaries:
     - Controller: this file. No DOM ownership outside #arena-unlock-timer.
     - Mount: <div id="arena-unlock-timer"> inside the unlock card
       (see index.html, sibling of arena-unlock-heading-block).
     - Styles: practice/arena-unlock-timer.css.
     - Caller: practice/arena-unlock.js — calls resetForExercise on show,
       getRating on Continue, then posts to /api/practice/arena-rating.

   Public API (window.ArenaUnlockTimer):
     resetForExercise(targetSeconds)
       Stops + clears the stopwatch, displays "0:00", and sets the
       target-time label. Call from arena-unlock.js#showCard.
     getRating() → { feedback: "not_much"|"somewhat"|"a_lot",
                     elapsedSeconds: number, targetSeconds: number,
                     ratio: number, started: bool }
       Snapshot the current state. If the student never pressed Start,
       returns started:false and feedback defaults to "somewhat" (caller
       can decide to skip the backend bump in that case).

   Rating mapping (elapsed / target):
     ≤ 0.66   → "a_lot"     (fast, high confidence boost)
     ≤ 1.50   → "somewhat"  (around expected)
     > 1.50   → "not_much"  (slow — small bump, room to grow)
   ================================================================ */

(function () {
  const root = document.getElementById("arena-unlock-timer");
  if (!root) return;

  // displayEl shows the COUNTDOWN remaining (target - elapsed, floored at 0).
  // The element id is still "arena-unlock-timer-elapsed" for historical reasons.
  const displayEl = document.getElementById("arena-unlock-timer-elapsed");
  const startBtn = document.getElementById("arena-unlock-timer-start");
  const stopBtn = document.getElementById("arena-unlock-timer-stop");
  const resetBtn = document.getElementById("arena-unlock-timer-reset");
  const hintEl = document.getElementById("arena-unlock-timer-hint");
  const autoSubmitWrongEl = document.getElementById("arena-unlock-auto-submit-wrong");
  const autoStartEl = document.getElementById("arena-unlock-auto-start");
  if (!displayEl || !startBtn || !stopBtn || !resetBtn) return;

  // Persisted-preference toggles. Keys are namespaced under arena-unlock so
  // they don't collide with anything else in localStorage. Default both OFF.
  const PREF_AUTO_SUBMIT_WRONG = "arena-unlock:auto-submit-wrong";
  const PREF_AUTO_START = "arena-unlock:auto-start";
  const loadPref = (key) => {
    try { return window.localStorage?.getItem(key) === "1"; } catch (_) { return false; }
  };
  const savePref = (key, on) => {
    try { window.localStorage?.setItem(key, on ? "1" : "0"); } catch (_) {}
  };
  if (autoSubmitWrongEl) {
    autoSubmitWrongEl.checked = loadPref(PREF_AUTO_SUBMIT_WRONG);
    autoSubmitWrongEl.addEventListener("change", () => savePref(PREF_AUTO_SUBMIT_WRONG, autoSubmitWrongEl.checked));
  }
  if (autoStartEl) {
    autoStartEl.checked = loadPref(PREF_AUTO_START);
    autoStartEl.addEventListener("change", () => savePref(PREF_AUTO_START, autoStartEl.checked));
  }

  let targetSeconds = 300;
  let startTimestamp = null;        // ms, when Start was pressed
  let accumulatedMs = 0;            // ms, total elapsed across start/stop cycles
  let tickHandle = null;
  let everStarted = false;
  let overFired = false;            // one-shot per resetForExercise — guards onOver callback
  let overCallback = null;          // optional: fires once when elapsed crosses target

  const fmtClock = (totalSeconds) => {
    const s = Math.max(0, Math.floor(totalSeconds));
    const m = Math.floor(s / 60);
    const sec = (s % 60).toString().padStart(2, "0");
    return `${m}:${sec}`;
  };

  const elapsedMsNow = () => {
    if (startTimestamp == null) return accumulatedMs;
    return accumulatedMs + (Date.now() - startTimestamp);
  };

  const refreshDisplay = () => {
    const elapsed = elapsedMsNow() / 1000;
    const remaining = targetSeconds - elapsed;
    // Floor at 0 so the display never shows negative time.
    displayEl.textContent = fmtClock(Math.max(0, remaining));
    // Color the display once we hit 0, and fire the one-shot over callback.
    if (remaining <= 0) {
      root.classList.add("arena-unlock-timer--over");
      if (!overFired) {
        overFired = true;
        if (typeof overCallback === "function") {
          try { overCallback(); } catch (_) {}
        }
      }
    } else {
      root.classList.remove("arena-unlock-timer--over");
    }
  };

  const startTick = () => {
    stopTick();
    tickHandle = window.setInterval(refreshDisplay, 250);
  };
  const stopTick = () => {
    if (tickHandle != null) {
      window.clearInterval(tickHandle);
      tickHandle = null;
    }
  };

  const setButtons = (mode /* "ready" | "running" | "stopped" */) => {
    startBtn.hidden = mode === "running";
    stopBtn.hidden = mode !== "running";
    resetBtn.hidden = mode === "ready";
  };

  startBtn.addEventListener("click", () => {
    if (startTimestamp != null) return;
    startTimestamp = Date.now();
    everStarted = true;
    setButtons("running");
    refreshDisplay();
    startTick();
  });

  stopBtn.addEventListener("click", () => {
    if (startTimestamp == null) return;
    accumulatedMs += Date.now() - startTimestamp;
    startTimestamp = null;
    stopTick();
    refreshDisplay();
    setButtons("stopped");
    if (hintEl) hintEl.textContent = "Click Continue when you're done — your rating will be auto-set from the time.";
  });

  resetBtn.addEventListener("click", () => {
    stopTick();
    startTimestamp = null;
    accumulatedMs = 0;
    refreshDisplay();
    setButtons("ready");
    if (hintEl) hintEl.textContent = "";
  });

  const ratingFromRatio = (ratio) => {
    if (ratio <= 0.66) return "a_lot";
    if (ratio <= 1.50) return "somewhat";
    return "not_much";
  };

  window.ArenaUnlockTimer = {
    resetForExercise(seconds) {
      const s = Number(seconds);
      targetSeconds = Number.isFinite(s) && s > 0 ? s : 300;
      stopTick();
      startTimestamp = null;
      accumulatedMs = 0;
      everStarted = false;
      overFired = false;
      refreshDisplay();
      setButtons("ready");
      if (hintEl) hintEl.textContent = "Start the timer when you open the exercise in Colab.";
      // Auto-start: if the user opted in, programmatically click Start so the
      // stopwatch begins the moment the unlock card appears.
      if (autoStartEl?.checked) {
        // Defer one tick — lets caller (arena-unlock.js#showCard) finish wiring.
        setTimeout(() => { if (!everStarted) startBtn.click(); }, 0);
      }
    },
    onOver(cb) {
      // Register a one-shot callback that fires the first time elapsed
      // crosses targetSeconds within the current resetForExercise cycle.
      overCallback = (typeof cb === "function") ? cb : null;
    },
    isAutoSubmitWrong() {
      // Read by arena-unlock.js inside its onOver callback to decide whether
      // to auto-click the "Looked up solution" choice when the timer expires.
      return !!autoSubmitWrongEl?.checked;
    },
    getRating() {
      // If still running, freeze without mutating state (snapshot only).
      const ms = elapsedMsNow();
      const elapsedSeconds = ms / 1000;
      const ratio = targetSeconds > 0 ? elapsedSeconds / targetSeconds : 0;
      return {
        feedback: everStarted ? ratingFromRatio(ratio) : "somewhat",
        elapsedSeconds,
        targetSeconds,
        ratio,
        started: everStarted,
      };
    },
  };
})();
