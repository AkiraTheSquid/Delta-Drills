/* ================================================================
   STATS — DOM MOUNT

   Injects the Statistics page (#page-statistics) into the body just
   before <footer.footer>, preserving the SPA's visual DOM order even
   though the markup now lives in this module.

   LOAD ORDER: this script MUST be loaded BEFORE any stats/ controller
   (stats/dom.js, stats/init.js, etc.) — those scripts query
   #stats-table-body / #adv-table-body / #predicted-table-body / etc.
   at IIFE-eval time and would null-out if the DOM isn't yet there.
   ================================================================ */

(function () {
  if (document.getElementById("page-statistics")) return;

  const html = `
    <main class="page hidden" id="page-statistics">
      <div class="container stats-container">
        <div class="section-header">
          <div>
            <h1>Statistics</h1>
            <p class="subtitle">Priority and learning-rate breakdown by area and sub-area.</p>
          </div>
        </div>

        <div class="stats-tabs">
          <button class="stats-tab active" type="button" data-stats-tab="areas">Areas</button>
          <button class="stats-tab" type="button" data-stats-tab="graph">Graph</button>
          <button class="stats-tab" type="button" data-stats-tab="advanced">Advanced</button>
          <button class="stats-tab" type="button" data-stats-tab="predicted">Predicted course scores</button>
        </div>

        <section class="stats-panel" data-stats-panel="areas">
          <div class="stats-card">
            <div class="stats-card-header">
              <span class="stats-title">Delta Analysis</span>
              <span class="stats-subtitle">Baseline Score = running average of the difficulty of questions you solve correctly (hardest = 100, half as hard = 50)</span>
              <span class="stats-subtitle">Delta = Weight × Learning Rate</span>
            </div>
            <div class="stats-table-wrap">
              <table class="stats-table">
                <thead>
                  <tr>
                    <th class="stats-col-toggle"></th>
                    <th class="stats-col-check">Use</th>
                    <th>Rank</th>
                    <th class="stats-col-area">Area</th>
                    <th class="stats-col-weight">Weight</th>
                    <th class="stats-col-score">Baseline Score</th>
                    <th class="stats-col-solved">Solved</th>
                    <th class="stats-col-lr">Learning Rate</th>
                    <th class="stats-col-delta">Delta Score</th>
                  </tr>
                </thead>
                <tbody id="stats-table-body"></tbody>
              </table>
            </div>
          </div>
        </section>

        <section class="stats-panel hidden" data-stats-panel="advanced">
          <div class="stats-card">
            <div class="stats-card-header">
              <span class="stats-title">Advanced Statistics</span>
              <span class="stats-subtitle">% Correct = smoothed correctness rate p(n): how often you're solving questions right, weighted toward recent attempts</span>
              <span class="stats-subtitle">Target Difficulty = Baseline × Difficulty Multiplier — the difficulty level the algorithm aims to serve next</span>
              <span class="stats-subtitle">Difficulty Multiplier = f(p): &lt;1× when struggling (pushes easier), ~1× near p=0.85, up to 2.5× when consistently correct</span>
            </div>
            <div class="stats-table-wrap">
              <table class="stats-table stats-table-adv">
                <thead>
                  <tr>
                    <th class="stats-col-toggle"></th>
                    <th class="stats-col-check">Use</th>
                    <th class="stats-col-rank">Rank</th>
                    <th class="stats-col-area">Area</th>
                    <th class="stats-col-score">Baseline Score</th>
                    <th class="stats-col-solved">Solved</th>
                    <th class="stats-col-lr">Learning Rate</th>
                    <th class="stats-col-delta">Delta</th>
                    <th class="stats-col-p">% Correct</th>
                    <th class="stats-col-target">Target Diff.</th>
                    <th class="stats-col-mult">Diff. Mult.</th>
                  </tr>
                </thead>
                <tbody id="adv-table-body"></tbody>
              </table>
            </div>
          </div>
        </section>

        <section class="stats-panel hidden" data-stats-panel="graph">
          <div class="stats-card stats-graph-card">
            <div class="stats-card-header">
              <span class="stats-title">Priority Graph</span>
              <span class="stats-subtitle">Data visualization will appear here.</span>
            </div>
            <div class="stats-graph-controls">
              <button class="stats-tab stats-tab-compact active" type="button" data-graph-range="day">Day</button>
              <button class="stats-tab stats-tab-compact" type="button" data-graph-range="week">Week</button>
              <button class="stats-tab stats-tab-compact" type="button" data-graph-range="month">Month</button>
            </div>
            <div class="stats-graph" id="stats-graph"></div>
            <div class="stats-tree">
              <div class="stats-tree-title">Display areas</div>
              <label class="stats-tree-row">
                <input class="stats-check" type="checkbox" checked />
                <span>NumPy</span>
              </label>
              <div class="stats-tree-children">
                <label class="stats-tree-row">
                  <input class="stats-check" type="checkbox" checked />
                  <span>Core array literacy</span>
                </label>
                <label class="stats-tree-row">
                  <input class="stats-check" type="checkbox" checked />
                  <span>Indexing and selection</span>
                </label>
                <label class="stats-tree-row">
                  <input class="stats-check" type="checkbox" checked />
                  <span>Vectorization and broadcasting</span>
                </label>
                <label class="stats-tree-row">
                  <input class="stats-check" type="checkbox" checked />
                  <span>Applied patterns and advanced</span>
                </label>
              </div>
            </div>
          </div>
        </section>

        <section class="stats-panel hidden" data-stats-panel="predicted">
          <div class="stats-card">
            <div class="stats-card-header">
              <span class="stats-title">Predicted course scores</span>
              <span class="stats-subtitle">Per-chapter predicted accuracy on the ARENA main notebook. Score = predicted % of questions answered correctly; Gap = remainder. Listed in curriculum order — chapter 0 → 3.</span>
              <span class="stats-subtitle">Source: <code>ARENA_5.0-main</code> (upstream mirror at <code>callummcdougall/ARENA_3.0</code>). For Colab progress to persist across sessions, fork the upstream repo and paste your GitHub username in <strong>Account</strong> — the <em>Colab</em> pills will then open your fork.</span>
              <span class="stats-subtitle"><strong>[TEMP — frontend pipeline test]</strong> Only the <strong>0.0 Prerequisites</strong> exercises are currently tagged with <strong>Delta Drills prerequisite concepts</strong> (hardcoded). Open the ▸ toggle on any 0.0 exercise row to see your current subtopic scores vs. the target % you should reach before tackling it. Other chapters render normally — their per-exercise prereq tags will fill in once the concept graph ships and this temp scaffold (<code>stats/predicted-prereqs-temp.js</code>) is deleted.</span>
            </div>
            <div class="stats-table-wrap">
              <table class="stats-table">
                <thead>
                  <tr>
                    <th class="stats-col-toggle"></th>
                    <th class="stats-col-check">Use</th>
                    <th class="stats-col-section">Section</th>
                    <th class="stats-col-area">Chapter / Problem</th>
                    <th class="stats-col-weight">Weight</th>
                    <th class="stats-col-score">Predicted % Correct</th>
                    <th class="stats-col-solved">Problems</th>
                    <th class="stats-col-lr">Top Skill</th>
                    <th class="stats-col-delta">Gap to 100%</th>
                    <th class="stats-col-open">Open</th>
                  </tr>
                </thead>
                <tbody id="predicted-table-body"></tbody>
              </table>
            </div>
          </div>
        </section>
      </div>
    </main>
  `;

  const footer = document.querySelector("footer.footer");
  if (footer) footer.insertAdjacentHTML("beforebegin", html);
  else document.body.insertAdjacentHTML("beforeend", html);
})();
