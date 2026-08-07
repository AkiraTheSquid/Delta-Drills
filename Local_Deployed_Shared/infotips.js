/* ================================================================
   INFOTIPS — one ⓘ per tab and per feature, click for a short panel.

   WHAT THIS IS
     A tester can see every control in this app and guess what about
     half of them do. The `title=""` attributes that used to carry the
     explanations only appear on hover, never on touch, and are
     invisible until you happen to rest the pointer on the right pixel.
     This replaces them with a control you can see: a small dot beside
     the thing, and a panel when you click it.

   HOW A FEATURE GETS ONE
     Put `data-dd-info="<key>"` on the element and write the copy under
     the same key in infotips-registry.js. Nothing else. Placement:

       (default)                     the dot is appended INSIDE the element
       data-dd-info-place="after"    inserted as the next sibling
       data-dd-info-place="before"   inserted as the previous sibling

     A <button>/<a> anchor is forced to "after" — nesting a button in a
     button is invalid, and the click would fight the anchor's own.

   WHY IT RESCANS
     Half this app's DOM is written at runtime (targeted-practice-dom.js,
     arena-unlock-dom.js) or re-rendered mid-session (the stage dots and
     the competency bar are innerHTML-replaced on every question). Dots
     are therefore re-derived from the DOM on a debounced MutationObserver
     rather than injected once at load: an anchor that comes back gets its
     dot back, and an anchor that never appears costs nothing.

   THE TOPBAR IS THE EXCEPTION
     Tab dots are hand-written in index.html, not injected here. They
     have to exist before app.js runs, because app.js captures
     `.auth-only` / `.guest-only` into static NodeLists at eval time and
     drives tab visibility off them — a dot injected later would stay
     visible next to a tab that had been hidden. Giving each dot the same
     auth class and `data-tab` as its tab is what keeps the two in step,
     and it is why `.tab` (the click target) and `.tab-info` are
     deliberately different class names.
   ================================================================ */

(function initInfotips() {
  const REGISTRY = window.DD_INFOTIPS || {};
  const ANCHOR_SELECTOR = "[data-dd-info]";
  const NO_NEST = new Set(["BUTTON", "A", "INPUT", "TEXTAREA", "SELECT", "OL", "UL"]);

  // Set while we are the ones touching the DOM, so our own insertions
  // don't feed the observer back into another scan.
  let mutating = false;
  let scanQueued = false;
  // Anchors whose dot could not be placed where the next scan would look
  // for it. Weak so a re-rendered anchor gets a clean second chance.
  const unservable = new WeakSet();

  /* ---- The panel --------------------------------------------------- */

  let pop = null;
  let popTitle = null;
  let popBody = null;
  let openFor = null;

  const buildPop = () => {
    if (pop) return pop;
    pop = document.createElement("div");
    pop.className = "dd-infopop";
    pop.id = "dd-infopop";
    pop.setAttribute("role", "dialog");
    pop.hidden = true;
    pop.innerHTML =
      '<div class="dd-infopop-head">' +
      '<span class="dd-infopop-title"></span>' +
      '<button type="button" class="dd-infopop-close" aria-label="Close">✕</button>' +
      "</div>" +
      '<div class="dd-infopop-body"></div>';
    popTitle = pop.querySelector(".dd-infopop-title");
    popBody = pop.querySelector(".dd-infopop-body");
    pop.querySelector(".dd-infopop-close").addEventListener("click", closePop);
    mutating = true;
    document.body.appendChild(pop);
    mutating = false;
    return pop;
  };

  // Anchored below the dot, left-aligned to it, clamped into the viewport;
  // flipped above when there isn't room below. position:fixed on purpose —
  // the topbar, the practice split and the graph pane all clip.
  const placePop = (trigger) => {
    const rect = trigger.getBoundingClientRect();
    const gap = 8;
    pop.style.left = "0px";
    pop.style.top = "0px";
    const box = pop.getBoundingClientRect();
    let left = rect.left;
    left = Math.min(left, window.innerWidth - box.width - 12);
    left = Math.max(12, left);
    let top = rect.bottom + gap;
    if (top + box.height > window.innerHeight - 12) {
      const above = rect.top - gap - box.height;
      top = above >= 12 ? above : Math.max(12, window.innerHeight - box.height - 12);
    }
    pop.style.left = `${Math.round(left)}px`;
    pop.style.top = `${Math.round(top)}px`;
  };

  function closePop() {
    if (!pop || pop.hidden) return;
    pop.hidden = true;
    if (openFor) openFor.setAttribute("aria-expanded", "false");
    openFor = null;
  }

  const openPop = (trigger) => {
    const entry = REGISTRY[trigger.dataset.ddInfo];
    if (!entry) return;
    buildPop();
    popTitle.textContent = entry.title || "";
    popBody.innerHTML = entry.body || "";
    pop.hidden = false;
    if (openFor && openFor !== trigger) openFor.setAttribute("aria-expanded", "false");
    openFor = trigger;
    trigger.setAttribute("aria-expanded", "true");
    placePop(trigger);
  };

  /* ---- Dots -------------------------------------------------------- */

  const makeDot = (key, label) => {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = "dd-info";
    dot.dataset.ddInfo = key;
    dot.dataset.ddInfoGenerated = "1";
    dot.setAttribute("aria-expanded", "false");
    dot.setAttribute("aria-label", label ? `What is ${label}?` : "What is this?");
    dot.textContent = "i";
    return dot;
  };

  // Find the dot this anchor already owns, or null.
  //
  // Both lookups have to match on the KEY, not just on "is a dot". Anchors
  // cluster: .kg2-controls carries one and holds the Fit button, which
  // carries another, so the first `.dd-info` child belongs to Fit. Matching
  // on class alone found the wrong dot, concluded this anchor had none,
  // minted a second one — and since minting mutates the DOM, the observer
  // re-scanned and did it again. It ran to ~1800 dots before the scan was
  // read. Same trap on the sibling side, hence the walk.
  const dotFor = (anchor, place) => {
    const key = anchor.dataset.ddInfo;
    const matches = (el) =>
      !!el && el.classList.contains("dd-info") && el.dataset.ddInfo === key;
    if (place === "in") {
      for (let i = 0; i < anchor.children.length; i += 1) {
        if (matches(anchor.children[i])) return anchor.children[i];
      }
      return null;
    }
    const step = place === "before" ? "previousElementSibling" : "nextElementSibling";
    let el = anchor[step];
    while (el && el.classList.contains("dd-info")) {
      if (el.dataset.ddInfo === key) return el;
      el = el[step];
    }
    return null;
  };

  // An anchor that is itself hidden should not leave a dot floating beside
  // it. Only meaningful for "before"/"after" dots — an inner one is hidden
  // by its own parent already.
  const syncHidden = (anchor, dot) => {
    const off = anchor.hidden || anchor.classList.contains("hidden");
    dot.classList.toggle("hidden", off);
  };

  const scan = () => {
    scanQueued = false;
    mutating = true;
    try {
      document.querySelectorAll(ANCHOR_SELECTOR).forEach((anchor) => {
        // The dots themselves carry data-dd-info (that's what the click
        // handler reads); they are not anchors for further dots.
        if (anchor.classList.contains("dd-info")) return;
        const key = anchor.dataset.ddInfo;
        if (!REGISTRY[key]) return;

        let place = anchor.dataset.ddInfoPlace || "in";
        if (place === "in" && NO_NEST.has(anchor.tagName)) place = "after";

        if (unservable.has(anchor)) return;

        let dot = dotFor(anchor, place);
        if (!dot) {
          dot = makeDot(key, (REGISTRY[key].title || "").toLowerCase());
          if (place === "before") anchor.parentNode.insertBefore(dot, anchor);
          else if (place === "after") anchor.parentNode.insertBefore(dot, anchor.nextSibling);
          else anchor.appendChild(dot);
          // Backstop against the failure mode that cost ~1800 dots once: if
          // the dot we just placed is not the dot the next scan will find,
          // this anchor mints a fresh one every pass, forever. Drop it after
          // one bad attempt rather than let it run.
          if (!dotFor(anchor, place)) {
            unservable.add(anchor);
            dot.remove();
            console.warn(`[infotips] cannot place a dot for "${key}" — skipping`);
          }
        }
        if (place !== "in") syncHidden(anchor, dot);
      });
    } finally {
      mutating = false;
    }
  };

  const queueScan = () => {
    if (scanQueued || mutating) return;
    scanQueued = true;
    requestAnimationFrame(scan);
  };

  /* ---- Wiring ------------------------------------------------------- */

  // Delegated, so dots created three re-renders from now still work and
  // nothing has to be re-bound after a scan.
  document.addEventListener("click", (e) => {
    if (!e.target || typeof e.target.closest !== "function") return;
    const dot = e.target.closest(".dd-info");
    if (dot) {
      e.preventDefault();
      e.stopPropagation();
      if (openFor === dot && pop && !pop.hidden) closePop();
      else openPop(dot);
      return;
    }
    if (pop && !pop.hidden && !e.target.closest(".dd-infopop")) closePop();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closePop();
  });

  // The panel is fixed to a point on screen, so any movement under it
  // orphans it. Cheaper to close than to chase.
  window.addEventListener("resize", closePop);
  window.addEventListener("scroll", closePop, true);

  const start = () => {
    scan();
    new MutationObserver(queueScan).observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["class", "hidden"],
    });
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();

  window.DDInfotips = { refresh: scan, close: closePop };
})();
