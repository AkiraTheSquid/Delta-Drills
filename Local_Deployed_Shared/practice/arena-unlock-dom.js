/* ================================================================
   ARENA UNLOCK — DOM MOUNT

   Injects the ARENA-unlock interstitial markup into #page-practice as
   the last child (sibling of .practice-container). Lives in its own
   file so the entire DOM block + its three controllers (arena-unlock,
   arena-unlock-timer, and the targeted-practice "Practice this problem"
   handoff) can be ripped out together when the real concept-graph
   backend ships and replaces the temp scaffold.

   LOAD ORDER: this script MUST be loaded BEFORE practice/arena-unlock-timer.js
   and practice/arena-unlock.js — both query DOM ids inside the injected
   block at IIFE-eval time. See Local_Deployed_Shared/watch.py for the
   load-order assertion.
   ================================================================ */

(function () {
  const mount = document.getElementById("page-practice");
  if (!mount) {
    console.warn("[arena-unlock-dom] #page-practice not in DOM — interstitial unavailable");
    return;
  }
  // Sentinel — if the page already contains this block (e.g. injected
  // twice via a hot-reload), don't duplicate.
  if (document.getElementById("arena-unlock-page")) return;

  mount.insertAdjacentHTML("beforeend", `
    <!-- ARENA unlock view — sibling of .practice-container inside page-practice.
         When showing, .practice-container is hidden and this view fills the
         practice tab's content area. Header / tabs / footer stay visible (they
         live outside #page-practice). Its own module: practice/arena-unlock.js
         + practice/arena-unlock.css. Toggled by window.ArenaUnlock.tryShow(). -->
    <div id="arena-unlock-page" class="arena-unlock-page hidden" aria-live="polite" aria-labelledby="arena-unlock-title">
      <div class="arena-unlock-card" id="arena-unlock-card">
        <div class="arena-unlock-banner" data-dd-info="arena-unlock">🎯 New ARENA exercise unlocked</div>
        <div class="arena-unlock-title" id="arena-unlock-title">—</div>
        <div class="arena-unlock-sub">Your subtopic scores now meet every prereq for this exercise. Go practice it before the next Delta Drills question.</div>
        <div class="arena-unlock-stuck-hint hidden" id="arena-unlock-stuck-hint">⏱️ Time's up. If you're stuck, give up and check the solution — you can still rate yourself below.</div>
        <div class="arena-unlock-why" id="arena-unlock-why"></div>
        <div class="arena-unlock-heading-block" id="arena-unlock-heading-block">
          <div class="arena-unlock-heading-label" id="arena-unlock-heading-label">Section heading inside the notebook — copy it, then press Ctrl+F in Colab to jump straight to the right cell:</div>
          <div class="arena-unlock-heading-row">
            <code class="arena-unlock-heading" id="arena-unlock-heading">—</code>
            <button type="button" class="ghost arena-unlock-copy-btn" id="arena-unlock-copy-btn">📋 Copy</button>
          </div>
        </div>
        <!-- ARENA unlock timer — Start when you open the exercise in Colab,
             Stop when you're done. Elapsed-vs-target time becomes the
             feedback rating that bumps your prereq subtopic scores when
             you click Continue. Module: practice/arena-unlock-timer.js
             + practice/arena-unlock-timer.css. -->
        <div class="arena-unlock-timer" id="arena-unlock-timer">
          <div class="arena-unlock-timer-row">
            <div class="arena-unlock-timer-elapsed" id="arena-unlock-timer-elapsed" aria-label="Time remaining">5:00</div>
            <span data-dd-info="arena-unlock-timer"></span>
          </div>
          <div class="arena-unlock-timer-controls">
            <button type="button" class="arena-unlock-timer-btn arena-unlock-timer-start" id="arena-unlock-timer-start">▶ Start</button>
            <button type="button" class="arena-unlock-timer-btn" id="arena-unlock-timer-stop" hidden>⏸ Stop</button>
            <button type="button" class="arena-unlock-timer-btn" id="arena-unlock-timer-reset" hidden>↺ Reset</button>
            <label class="arena-unlock-timer-toggle" title="When the elapsed time crosses the target, automatically pick 'Looked up solution'.">
              <input type="checkbox" id="arena-unlock-auto-submit-wrong" />
              <span>Auto-submit as wrong when timer ends</span>
            </label>
            <label class="arena-unlock-timer-toggle" title="When this card opens, automatically start the timer instead of waiting for the Start button.">
              <input type="checkbox" id="arena-unlock-auto-start" />
              <span>Auto-start timer for exercises like these</span>
            </label>
          </div>
          <div class="arena-unlock-timer-hint" id="arena-unlock-timer-hint"></div>
        </div>
        <div class="arena-unlock-actions">
          <button type="button" class="ghost arena-unlock-btn" id="arena-unlock-hint-btn">Show hint</button>
          <button type="button" class="ghost arena-unlock-btn" id="arena-unlock-answer-btn">Show answer</button>
          <a class="primary arena-unlock-btn arena-unlock-colab-btn" id="arena-unlock-colab-btn" href="#" target="_blank" rel="noreferrer">Open in Colab ↗</a>
        </div>
        <div class="arena-unlock-placeholder hidden" id="arena-unlock-placeholder"></div>
        <!-- 2-stage rating block: 4-option self-rating → score deltas + Continue.
             Controller: practice/arena-unlock.js. Data-attrs on each choice button
             encode {correct, feedback} for the apply_feedback backend call. -->
        <div class="arena-unlock-rating" id="arena-unlock-rating">
          <div class="arena-unlock-rating-stage" id="arena-unlock-stage-choice">
            <div class="arena-unlock-rating-prompt" data-dd-info="arena-unlock-rating">How did you do on this exercise?</div>
            <div class="arena-unlock-choice-buttons">
              <button type="button" class="arena-unlock-choice-btn arena-unlock-choice--best"
                      data-correct="true" data-feedback="a_lot"
                      id="arena-unlock-choice-best">
                <span class="arena-unlock-choice-mark">✓</span>
                <span class="arena-unlock-choice-main">Solved in target time</span>
                <span class="arena-unlock-choice-sub">no help</span>
              </button>
              <button type="button" class="arena-unlock-choice-btn arena-unlock-choice--good"
                      data-correct="true" data-feedback="somewhat"
                      id="arena-unlock-choice-good">
                <span class="arena-unlock-choice-mark">✓</span>
                <span class="arena-unlock-choice-main">Solved in target time</span>
                <span class="arena-unlock-choice-sub">with a hint</span>
              </button>
              <button type="button" class="arena-unlock-choice-btn arena-unlock-choice--okay"
                      data-correct="true" data-feedback="not_much"
                      id="arena-unlock-choice-okay">
                <span class="arena-unlock-choice-mark">✓</span>
                <span class="arena-unlock-choice-main">Solved correctly</span>
                <span class="arena-unlock-choice-sub">over target time</span>
              </button>
              <button type="button" class="arena-unlock-choice-btn arena-unlock-choice--bad"
                      data-correct="false" data-feedback="a_lot"
                      id="arena-unlock-choice-bad">
                <span class="arena-unlock-choice-mark">✗</span>
                <span class="arena-unlock-choice-main">Looked up</span>
                <span class="arena-unlock-choice-sub">the solution</span>
              </button>
            </div>
          </div>
          <div class="arena-unlock-rating-stage hidden" id="arena-unlock-stage-result">
            <div class="arena-unlock-rating-prompt">Score updates</div>
            <div class="arena-unlock-result-list" id="arena-unlock-result-list"></div>
            <div class="arena-unlock-continue-row">
              <button type="button" class="primary arena-unlock-continue-btn" id="arena-unlock-continue-btn">Continue to next question →</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `);
})();
