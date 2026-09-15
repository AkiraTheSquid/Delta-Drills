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

  // Cytoscape owns this DOM after it starts. It is intentionally not editable
  // or serialized from its live state; the initial static shell is saved back.
  const mapShells = [...content.querySelectorAll(".wta-graph")].map((node) => node.innerHTML);
  let editing = false;

  const ownsEditor = () =>
    window.DDIdentity?.isSignedIn?.() === true &&
    String(window.DDIdentity?.email?.() || "").trim().toLowerCase() === EDITOR_EMAIL;

  const protectRuntime = () => {
    content.querySelectorAll(".wta-graph").forEach((node) => {
      node.contentEditable = "false";
    });
  };

  const serializedContent = () => {
    const clone = content.cloneNode(true);
    clone.querySelectorAll(".wta-graph").forEach((node, index) => {
      node.innerHTML = mapShells[index] || "";
    });
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
      if (saved?.html) content.innerHTML = saved.html;
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
