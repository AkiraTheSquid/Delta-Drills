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

  /* Called from renderQuestion. The button is ALWAYS shown for a real question.
   *
   * It used to hide itself whenever the question carried no KC tag, on the
   * reasoning that a jump landing nowhere is worse than no button. That was
   * backwards for the one job this control has. Its purpose is auditing — "the
   * queue picked this; the graph claims that; do they agree?" — and a control
   * that silently disappears on exactly the questions the tutor cannot place is
   * a control you cannot audit with. Worse, absence is ambiguous: the learner
   * cannot tell "this question is off the map" from "the button is broken".
   *
   * 75 of the 449 bank questions carry no tag (the CNN/PyTorch-Fundamentals
   * rows no KP references), and placement probes can serve any of them. So the
   * untagged case is common, not exotic. It now says so out loud. */
  function updateGraphJump(q) {
    var btn = document.getElementById("practice-graph-jump");
    if (!btn) return;
    var qid = q && (q.question_id != null ? q.question_id : q.id);
    if (qid == null) { btn.hidden = true; return; }
    btn.hidden = false;
    btn.textContent = "See in knowledge graph";
    btn.classList.remove("is-untagged");
    btn.title = "Looking up this question's concept…";
    loadTags().then(function (tags) {
      currentKcs = kcsForQuestion(tags, qid);
      if (!currentKcs.length) {
        btn.textContent = "Not on the map";
        btn.classList.add("is-untagged");
        btn.title =
          "Question " + qid + " carries no concept tag, so the tutor cannot say " +
          "what it teaches. Opens the knowledge graph anyway.";
        return;
      }
      btn.textContent = "See in knowledge graph";
      btn.classList.remove("is-untagged");
      btn.title = currentKcs.length > 1
        ? "This question is tagged to " + currentKcs.length + " concepts: " +
          currentKcs.join(", ") + ". Opens the first."
        : "Open “" + currentKcs[0] + "” in the knowledge graph";
    });
  }

  function onClick() {
    if (typeof switchTab === "function") switchTab("knowledge-graph");
    // Untagged question: open the map with nothing focused. Seeing the graph
    // and finding no node for what you were just asked IS the audit result.
    if (!currentKcs.length) return;
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
