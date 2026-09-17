/* ================================================================
   COLAB_NO_AI.JS — turn Gemini's inline completions off inside Colab's editor.

   THE PROBLEM
     Colab ships Gemini code completion on by default (its own setting is
     `inlineCompletions`, "Show AI-powered inline completions", under Settings →
     AI Assistance). It renders as grey "shadow" text ahead of the caret, and on
     a Delta Drills notebook the thing it completes is the answer. A learner
     who has been routed to a problem the course believes they cannot do yet is
     shown a solution before they have typed the second line of it.

   WHY THIS IS NOT A CSS RULE
     The obvious fix is to hide the ghost text — Monaco draws it as
     `.ghost-text-decoration` and friends, and one `display: none` makes it
     invisible. That fix is worse than nothing. Monaco's Tab handler acts on the
     suggestion in the MODEL, not on the pixels: with the text hidden, Tab still
     accepts a completion the learner cannot see, so the answer lands in their
     cell out of nowhere. Suppression has to happen where the suggestion is
     produced, and only then is the CSS in `colab_dd.css` safe to add as a
     backstop for the frames before this runs.

   WHY IT IS A SEPARATE FILE, IN THE MAIN WORLD
     Content scripts run in an isolated world, which has the page's DOM but not
     its JavaScript — `window.monaco` there is undefined. Reaching the editor
     means `"world": "MAIN"` (see the manifest), and a MAIN-world script has no
     `chrome.*` at all: no storage, no messaging. So this file holds no policy.
     It listens for two DOM events that `colab_focus.js` dispatches from the
     isolated world, and does as it is told:

       dd:gemini-off   suppress inline completions   (the default)
       dd:gemini-on    hand them back

     The flag rides in the event NAME rather than in `detail`, because a
     `CustomEvent`'s detail is created in the sending world and reading it back
     across the boundary is exactly the kind of thing that works until it does
     not. An event name is a string, and a string always crosses.

   DEFAULT SUPPRESSED, THEN CORRECTED
     This starts suppressing the moment Monaco exists, before the isolated
     script has read `chrome.storage`. That order is deliberate: storage is
     async, and the other order leaves a window in which the editor is live and
     completing. Erring towards "no suggestions for a moment on a notebook that
     did not want suppression" is a flicker; erring the other way hands over an
     answer, permanently, and silently.

   GIVING THEM BACK MEANS GIVING BACK EXACTLY WHAT WE TOOK
     Only editors this file actually turned off are turned back on. Colab's own
     setting may already have inline completions disabled — a blanket
     "enabled: true" on release would then switch on a feature the user had
     chosen to switch off, using our toggle as the lever. The WeakSet is the
     whole memory of what we changed.
   ================================================================ */

(() => {
  const POLL_MS = 400;         // waiting for Monaco to load
  const SWEEP_MS = 3000;       // ...and the slow backstop once it has
  const GIVE_UP_MS = 120000;   // Monaco is lazy-loaded; a notebook can be slow

  let suppressed = true;
  let hooked = false;

  // Editors whose inline completions WE switched off. Weak so that a disposed
  // editor is collectable — a notebook lives for hours and mints an editor per
  // cell mounted.
  const ours = new WeakSet();

  function monacoNow() {
    const m = window.monaco;
    return m && m.editor && typeof m.editor.getEditors === "function" ? m : null;
  }

  /** Is inline completion currently on for this editor? `null` if unreadable. */
  function inlineOn(editor, monaco) {
    try {
      const option = editor.getOption(monaco.editor.EditorOption.inlineSuggest);
      return option ? Boolean(option.enabled) : null;
    } catch (_) {
      return null;   // a disposed editor, mid-teardown
    }
  }

  /**
   * Bring one editor in line with the current policy.
   *
   * Reads before it writes, and that read is load-bearing rather than tidy:
   * this also runs from `onDidChangeConfiguration`, so an unconditional write
   * would fire the event that called it. Writing only on a real change is what
   * makes the loop terminate.
   */
  function align(editor, monaco) {
    const on = inlineOn(editor, monaco);
    if (on === null) return;
    try {
      if (suppressed && on) {
        editor.updateOptions({ inlineSuggest: { enabled: false } });
        ours.add(editor);
      } else if (!suppressed && !on && ours.has(editor)) {
        editor.updateOptions({ inlineSuggest: { enabled: true } });
        ours.delete(editor);
      }
    } catch (_) {
      /* disposed between the read and the write; the next pass covers it */
    }
  }

  function alignAll() {
    const monaco = monacoNow();
    if (!monaco) return false;
    monaco.editor.getEditors().forEach((editor) => align(editor, monaco));
    return true;
  }

  /**
   * Watch every editor, including the ones that do not exist yet.
   *
   * Two hooks, and both are needed. `onDidCreateEditor` catches the cells Colab
   * mounts as the learner scrolls — without it only the cells on screen at load
   * are covered, which is the majority of a notebook still completing.
   * `onDidChangeConfiguration` catches Colab re-applying its own option set to
   * an editor we had already dealt with, which puts the suggestion back without
   * creating anything for the first hook to see.
   */
  function install(monaco) {
    if (hooked) return;
    hooked = true;
    try {
      monaco.editor.onDidCreateEditor((editor) => {
        align(editor, monaco);
        try {
          editor.onDidChangeConfiguration(() => align(editor, monaco));
        } catch (_) { /* not every editor exposes it */ }
      });
    } catch (_) {
      hooked = false;   // fall back to the poll below, which never stops
    }
    monaco.editor.getEditors().forEach((editor) => {
      try {
        editor.onDidChangeConfiguration(() => align(editor, monaco));
      } catch (_) { /* as above */ }
    });
  }

  function setSuppressed(next) {
    if (suppressed === next) return;
    suppressed = next;
    alignAll();
  }

  document.addEventListener("dd:gemini-off", () => setSuppressed(true));
  document.addEventListener("dd:gemini-on", () => setSuppressed(false));

  // Monaco arrives long after this script does. Poll fast until it is there,
  // install the hooks, then drop to a slow sweep — the hooks cover everything
  // they can see, and the sweep is the answer to anything they cannot (an
  // editor created before the hook landed, or one whose configuration event we
  // could not subscribe to). Fast forever would mean walking every editor in a
  // fifty-cell notebook twice a second, for a case the hooks already handle.
  const started = Date.now();
  const waiting = setInterval(() => {
    const monaco = monacoNow();
    if (monaco) {
      clearInterval(waiting);
      install(monaco);
      alignAll();
      setInterval(alignAll, SWEEP_MS);
      return;
    }
    if (Date.now() - started > GIVE_UP_MS) clearInterval(waiting);
  }, POLL_MS);
})();
