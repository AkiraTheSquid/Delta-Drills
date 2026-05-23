/* ================================================================
   TARGETED PRACTICE — DOM MOUNT

   Injects the Targeted Practice page (#page-targeted-practice) into
   the body just before <footer.footer>, so it sits between the other
   <main class="page"> elements and the footer. Keeps the SPA's
   visual DOM order intact even though the HTML lives in this module.

   LOAD ORDER: this script MUST be loaded BEFORE
   targeted-practice/targeted-practice.js, whose IIFE queries
   #page-targeted-practice at eval time.
   ================================================================ */

(function () {
  if (document.getElementById("page-targeted-practice")) return;

  const html = `
    <!-- Targeted Practice — search ARENA exercises by heading, check to add to a
         practice queue, then submit. Controller: targeted-practice/targeted-practice.js
         + targeted-practice/targeted-practice.css. -->
    <main class="page hidden" id="page-targeted-practice">
      <div class="container">
        <div class="section-header">
          <div>
            <h1>Targeted Practice</h1>
            <p class="subtitle">Search ARENA exercises by heading. Check the ones you want to drill, then submit to start a focused session.</p>
          </div>
        </div>

        <div class="tp-card tp-search-card">
          <label class="tp-search-label" for="tp-search-input">Search exercises</label>
          <div class="tp-search-row">
            <input
              type="search"
              id="tp-search-input"
              class="tp-search-input"
              placeholder="Start typing an exercise heading (e.g. 'softmax', 'rearrange', 'ray tracing')…"
              autocomplete="off"
              spellcheck="false"
            />
            <button type="button" class="tp-search-clear" id="tp-search-clear" title="Clear">✕</button>
          </div>
          <div class="tp-results-wrap">
            <div class="tp-results-hint" id="tp-results-hint">Type at least 2 characters to search.</div>
            <ul class="tp-results-list" id="tp-results-list" role="listbox" aria-label="Matching exercises"></ul>
          </div>
        </div>

        <div class="tp-card tp-selected-card">
          <div class="tp-selected-header">
            <span class="tp-selected-title" id="tp-selected-title">Selected to practice</span>
            <span class="tp-selected-count" id="tp-selected-count">0</span>
          </div>
          <ul class="tp-selected-list" id="tp-selected-list"></ul>
          <div class="tp-selected-empty" id="tp-selected-empty">No exercises selected yet. Use the search above to add some.</div>
          <div class="tp-submit-row">
            <button type="button" class="primary tp-submit-btn" id="tp-submit-btn" disabled>Submit to start practicing</button>
            <button type="button" class="primary tp-back-btn hidden" id="tp-back-btn">← Back to search</button>
          </div>
        </div>
      </div>
    </main>
  `;

  // Insert before <footer> if present (preserves SPA DOM ordering);
  // otherwise append at body end.
  const footer = document.querySelector("footer.footer");
  if (footer) footer.insertAdjacentHTML("beforebegin", html);
  else document.body.insertAdjacentHTML("beforeend", html);
})();
