/* "See in knowledge graph" — jump from the question you are answering to the
 * concept node the tutor believes it teaches.
 *
 * This exists to be AUDITABLE. The queue picks a question; the graph claims a
 * concept; without a way to cross the two, neither claim can be checked. So
 * the button maps the live question through the same q-matrix the ITS itself
 * uses (`lessons/qmatrix_tags.json` target_kcs) rather than through anything
 * computed here — if the mapping is wrong, this shows the wrong node, which is
 * exactly the signal worth having.
 *
 * A question can legitimately teach several concepts. The first target is the
 * primary one by convention in that file; the button's tooltip names them all
 * so a multi-KC question is visible as such instead of silently collapsing.
 */
(function () {
  "use strict";

  var tagsPromise = null;
  var currentKcs = [];

  function loadTags() {
    if (!tagsPromise) {
      tagsPromise = fetch("lessons/qmatrix_tags.json", { cache: "no-cache" })
        .then(function (r) { return r.json(); })
        .catch(function () { return {}; });
    }
    return tagsPromise;
  }

  function kcsForQuestion(tags, qid) {
    var row = tags && (tags[String(qid)] || tags[qid]);
    var targets = row && row.target_kcs;
    return Array.isArray(targets) ? targets : [];
  }

  /* Called from renderQuestion. Hides the button when the question carries no
   * KC tag at all: offering a jump that lands nowhere is worse than no button,
   * and an untagged question is itself worth noticing (it means the ITS is
   * serving something outside its own map). */
  function updateGraphJump(q) {
    var btn = document.getElementById("practice-graph-jump");
    if (!btn) return;
    var qid = q && (q.question_id != null ? q.question_id : q.id);
    if (qid == null) { btn.hidden = true; return; }
    loadTags().then(function (tags) {
      currentKcs = kcsForQuestion(tags, qid);
      btn.hidden = currentKcs.length === 0;
      if (currentKcs.length) {
        btn.title = currentKcs.length > 1
          ? "This question is tagged to " + currentKcs.length + " concepts: " +
            currentKcs.join(", ") + ". Opens the first."
          : "Open “" + currentKcs[0] + "” in the knowledge graph";
      }
    });
  }

  function onClick() {
    if (!currentKcs.length) return;
    if (typeof switchTab === "function") switchTab("knowledge-graph");
    // The graph can only size itself once its page is visible, and on a first
    // visit it still has to build — deltaFocusConceptGraphKc waits for both.
    requestAnimationFrame(function () {
      if (typeof window.deltaFocusConceptGraphKc === "function") {
        window.deltaFocusConceptGraphKc(currentKcs[0]);
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("practice-graph-jump");
    if (btn) btn.addEventListener("click", onClick);
  });

  window.updateGraphJump = updateGraphJump;
})();
