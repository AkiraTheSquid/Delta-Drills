/* ================================================================
   THEME.JS — the three-way theme switch behind the Account tab.

   WHY IT IS A <script> IN <head> AND NOT A MODULE AT THE BOTTOM
   The theme is an attribute on <html>, and CSS resolves it at first paint.
   Loaded with the rest of the app's scripts (bottom of <body>, after the
   stylesheets have already painted) a light-theme user would get a full
   frame of the dark palette before the swap — the white flash this file
   exists to avoid. So: linked SYNCHRONOUSLY in <head>, above the stylesheet
   links, exactly like the two inline pre-paint scripts already there
   (`dd-solo`, `kg-embed`). It must therefore not touch the DOM at load —
   there isn't one yet. Everything that does is deferred to DOMContentLoaded.

   WHAT A THEME IS
   `styles/variables.css` defines every design token three times, once per
   theme block, keyed on `html[data-theme="..."]`. This file's whole job is
   to decide which of those three strings goes on <html> and to keep the
   Account tab's radio group in sync with it. There is no per-theme
   JavaScript and no re-render on switch: nothing in this app reads a colour
   out of the CSS (verified — no getComputedStyle/getPropertyValue on a
   custom property anywhere), so flipping the attribute repaints everything
   including the Knowledge Graph pane, in one frame.

   Stored per-browser in localStorage, like `dd_advanced_mode` and the other
   display preferences on that tab — NOT on the account. A theme is a
   property of the screen you are looking at, and syncing it to the backend
   would drag a dark-room choice onto a daylit laptop.
   ================================================================ */
(function () {
  "use strict";

  var STORAGE_KEY = "dd_theme";
  var DEFAULT_THEME = "blue";

  // The order here is the order the Account tab renders them in.
  var THEMES = [
    {
      id: "light",
      label: "Light",
      note: "White surfaces, near-black text.",
    },
    {
      id: "dark",
      label: "Dark",
      note: "Neutral Colab greys — no hue, low glare.",
    },
    {
      id: "blue",
      label: "Blue",
      note: "The original navy palette. Default.",
    },
  ];

  var VALID = THEMES.map(function (t) { return t.id; });

  function isValid(id) {
    return VALID.indexOf(id) !== -1;
  }

  // localStorage throws, not returns null, in a partitioned/blocked context
  // (Safari private mode, an embedded frame with third-party storage off).
  // The Chrome extension's side panel frames this page, so this is a real
  // path, not a defensive habit — and a theme read is not worth a broken app.
  function read() {
    try {
      var v = window.localStorage.getItem(STORAGE_KEY);
      return isValid(v) ? v : DEFAULT_THEME;
    } catch (e) {
      return DEFAULT_THEME;
    }
  }

  function write(id) {
    try {
      window.localStorage.setItem(STORAGE_KEY, id);
    } catch (e) {
      /* Non-fatal: the theme still applies for this page load. */
    }
  }

  // The one line that actually does the work.
  function stamp(id) {
    document.documentElement.setAttribute("data-theme", id);
  }

  function get() {
    var attr = document.documentElement.getAttribute("data-theme");
    return isValid(attr) ? attr : read();
  }

  function set(id, opts) {
    if (!isValid(id)) return get();
    stamp(id);
    if (!opts || opts.persist !== false) write(id);
    syncControls(id);
    window.dispatchEvent(
      new CustomEvent("delta:theme-changed", { detail: { theme: id } })
    );
    return id;
  }

  // ── Pre-paint ───────────────────────────────────────────────────────────
  // Runs at parse time, before any stylesheet has been applied.
  stamp(read());

  // ── The Account tab control ─────────────────────────────────────────────

  function syncControls(id) {
    var inputs = document.querySelectorAll('input[name="dd-theme"]');
    for (var i = 0; i < inputs.length; i++) {
      inputs[i].checked = inputs[i].value === id;
    }
  }

  function wire() {
    var group = document.getElementById("account-theme-options");
    if (!group) return;

    // The radios are rendered here rather than written into index.html so the
    // THEMES list above is the single place a theme is named. Adding a fourth
    // one is: a block in variables.css, an entry in THEMES, and nothing else.
    var html = "";
    for (var i = 0; i < THEMES.length; i++) {
      var t = THEMES[i];
      html +=
        '<label class="account-theme-option" data-theme-preview="' + t.id + '">' +
        '<input type="radio" name="dd-theme" value="' + t.id + '" />' +
        '<span class="account-theme-swatch" aria-hidden="true">' +
        '<span class="account-theme-swatch-bar"></span>' +
        '<span class="account-theme-swatch-card"></span>' +
        '<span class="account-theme-swatch-line"></span>' +
        '<span class="account-theme-swatch-line account-theme-swatch-line--short"></span>' +
        "</span>" +
        '<span class="account-theme-name">' + t.label + "</span>" +
        '<span class="account-theme-note">' + t.note + "</span>" +
        "</label>";
    }
    group.innerHTML = html;

    // Applies on `change`, with no Save step — same reasoning as the
    // advanced-mode toggle right above it in the markup: the thing it changes
    // is the page the user is looking at, and a colour scheme that only lands
    // after a form submit reads as a broken control.
    group.addEventListener("change", function (event) {
      var input = event.target;
      if (!input || input.name !== "dd-theme") return;
      set(input.value);
    });

    syncControls(get());
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire, { once: true });
  } else {
    wire();
  }

  window.DDTheme = {
    get: get,
    set: set,
    themes: THEMES,
    DEFAULT: DEFAULT_THEME,
    STORAGE_KEY: STORAGE_KEY,
  };
})();
