/* About page editing — visible only to the server-authorized owner.

   The actual authorization lives in the Fly API. This client-side email check
   only decides whether to show editing controls; a forged browser request is
   still rejected by GET/PUT /site-content/about on the server. */
(() => {
  "use strict";

  const EDITOR_EMAIL = "sethbgibson@gmail.com";
  const content = document.getElementById("about-page-content");
  const controls = document.getElementById("about-editor-controls");
  const status = document.getElementById("about-editor-status");
  const start = document.getElementById("about-editor-start");
  const save = document.getElementById("about-editor-save");
  const discard = document.getElementById("about-editor-discard");
  if (!content || !controls || !status || !start || !save || !discard) return;

  // Live figures own their DOM once they start (Cytoscape, SVG
  // charts, sliders). They are never editable and never serialized from their
  // live state: the static shell from index.html is what gets saved, and it is
  // put back when a saved copy is shown, because the server's sanitizer strips
  // <svg>, <input>, <label>, <output> and inline styles out of them.
  // `[data-about-runtime]` (keyed by id) marks them; `.wta-graph` is the old
  // page's concept map, kept so an older saved copy still round-trips.
  const RUNTIME = "[data-about-runtime][id], .wta-graph";
  const runtimeKey = (node, index) => node.id || `#${index}`;
  const shellsFrom = (root) => {
    const map = new Map();
    root.querySelectorAll(RUNTIME).forEach((node, index) => map.set(runtimeKey(node, index), node.innerHTML));
    return map;
  };
  const shells = shellsFrom(content);
  const restoreShells = (root) => {
    root.querySelectorAll(RUNTIME).forEach((node, index) => {
      const shell = shells.get(runtimeKey(node, index));
      if (shell !== undefined) node.innerHTML = shell;
    });
  };
  let editing = false;

  const ownsEditor = () =>
    window.DDIdentity?.isSignedIn?.() === true &&
    String(window.DDIdentity?.email?.() || "").trim().toLowerCase() === EDITOR_EMAIL;

  const protectRuntime = () => {
    content.querySelectorAll(RUNTIME).forEach((node) => {
      node.contentEditable = "false";
    });
  };

  const serializedContent = () => {
    const clone = content.cloneNode(true);
    restoreShells(clone);
    return clone.innerHTML;
  };

  const refreshControls = () => {
    const allowed = ownsEditor();
    controls.hidden = !allowed;
    if (!allowed && editing) window.location.reload();
  };

  const setEditing = (next) => {
    editing = next;
    content.contentEditable = next ? "true" : "false";
    content.classList.toggle("dd-about-editing", next);
    protectRuntime();
    start.hidden = next;
    save.hidden = !next;
    discard.hidden = !next;
    status.textContent = next ? "Editing locally — save to publish." : "";
  };

  const load = async () => {
    try {
      const response = await window.apiFetch("/site-content/about");
      if (!response.ok) throw new Error(`Could not load saved content (${response.status}).`);
      const saved = await response.json();
      // A copy saved before the AISC write-up replaced the page (no
      // #aisc-root) would put the old page back; the shipped HTML wins until
      // the page is saved again.
      if (saved?.html && saved.html.includes('id="aisc-root"')) {
        content.innerHTML = saved.html;
        restoreShells(content);
      }
    } catch (error) {
      // Static HTML remains fully usable if the API is offline.
      console.warn("[about-page]", error);
    } finally {
      protectRuntime();
      refreshControls();
    }
  };

  // why-graph.js waits for this before it mounts Cytoscape into the page.
  window.DDAboutContentReady = load();

  start.addEventListener("click", () => setEditing(true));
  discard.addEventListener("click", () => window.location.reload());
  save.addEventListener("click", async () => {
    if (!ownsEditor()) {
      window.location.reload();
      return;
    }
    save.disabled = true;
    status.textContent = "Saving…";
    try {
      const response = await window.apiFetch("/site-content/about", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ html: serializedContent() }),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.detail || `Save failed (${response.status}).`);
      setEditing(false);
      status.textContent = "Published.";
    } catch (error) {
      status.textContent = error.message || "Save failed.";
    } finally {
      save.disabled = false;
    }
  });

  window.addEventListener("delta:auth-state-changed", refreshControls);
})();
