/* The node field in the About page's side gutters (Seth, 2026-09-24: the
   write-up should read as one column of paragraphs, with "the floating nodes
   that connect to your mouse" on the left and right and an extremely subtle
   fade between the two, after uchicagoaisafety.com/about-us).

   Ported from the standalone site's js/bg-field.js. Floating nodes link when
   closer than the sum of their reaches, and the link strengthens as they
   close in; the cursor is one more, larger node that links to everything
   near it. What changed for the app:
   - The canvas is fixed under the topbar and lives on #page-learn-about-app,
     outside #about-page-content, so the About editor never saves it.
   - Since 2026-09-25 (Seth: back to the card style, "remove the gradient
     thing") the canvas is drawn unmasked across the whole page. The extra
     gutter nodes read `--aisc-clear` (aisc.css: the cards' 1180px measure)
     for where the side bands start.
   - The cards are see-through wireframes, so the field erases its drawing
     inside the boxes of the cards on screen (CARDS, the same list aisc.css
     outlines): it reads as running behind opaque cards, as on the original
     site, and links never cross the text. While a figure is full screen
     there is nothing to keep clear.
   - The motion is the original site's, unchanged (Seth, 2026-09-25: "bring
     back the old graph background … just a normal background graph"). The
     cards push and pull nothing, and the cursor drags nothing; it only
     draws the nodes near it in, gently, as before.
   - The card under the cursor is marked .is-hot (aisc.css lights its outline
     in the cursor's colour).
   - It runs only while #aisc-root is on screen. The app keeps every page in
     the DOM, and a `display: none` page never intersects, so the animation
     stops the moment the learner leaves this page.
   - Colours come from #aisc-root's --field-node / --field-edge ("r, g, b"),
     like every other figure token. */
(function () {
  "use strict";

  var host = document.getElementById("aisc-root");
  // #page-learn-about-app, reached from the write-up rather than by id
  var page = host && host.closest(".page");
  // the canvas is made here, not in index.html, so it is never in a saved copy
  if (!host || !page || page.querySelector("#aisc-field")) return;

  var canvas = document.createElement("canvas");
  canvas.id = "aisc-field";
  canvas.setAttribute("aria-hidden", "true");
  page.insertBefore(canvas, page.firstChild);
  var ctx = canvas.getContext("2d");
  var motionQ = matchMedia("(prefers-reduced-motion: reduce)");
  var reduced = motionQ.matches;

  var REACH = 36;        // px of link range per px of radius
  var MOUSE_REACH = 200; // cursor link range
  var PULL = 0.00004;    // spring along a link toward REST x range, scaled by strength
  var REST = 0.6;
  var PUSH = 0.5;        // short-range repulsion so clusters don't collapse
  var WANDER = 0.012;    // random drift per frame
  var DAMP = 0.985;
  var VMAX = 0.7;
  var SIDE_AREA = 4500;  // px^2 per extra node in each side gutter
  var HOME = 0.0006;     // soft pull that keeps a gutter node in its gutter
  var CARDS = ".tile, .hero > div:first-child, .chain, .timeline, .runway, .table-wrap," +
    " .note, .stat, .modes .hand, .modes .lane, figure.fig, .status, .card, .qs li, .aisc-footer";
  var PAD = 6;           // px of clear space kept round every card's outline

  var W = 0, H = 0, top = 0, dpr = 1, nodes = [], mouse = null, col = {};
  var hot = null;        // the card under the cursor
  var band = 0;          // width of each side gutter the extra nodes live in
  var onScreen = false;
  var cards = [], boxes = []; // card elements; their boxes on screen, canvas px
  var laid = [], stale = true, age = 0; // the cards' boxes in page px, re-read on a layout change

  function tokens() {
    var cs = getComputedStyle(host);
    col.node = cs.getPropertyValue("--field-node").trim() || "120,130,160";
    col.edge = cs.getPropertyValue("--field-edge").trim() || "109,93,252";
    var p = cs.getPropertyValue("--paper").trim().replace("#", "");
    col.dark = p.length === 6 && parseInt(p.slice(0, 2), 16) < 80;
  }

  // side 0 roams the whole width; side -1 / 1 are the extra nodes that live
  // in the left / right gutter beside the text column
  function makeNode(side) {
    var r = 1.1 + Math.pow(Math.random(), 1.8) * 2.6; // mostly small, a few hubs
    var x = side ? Math.random() * band : Math.random() * W;
    return { x: side > 0 ? W - x : x, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.5, vy: (Math.random() - 0.5) * 0.5, r: r, side: side };
  }

  // add or drop nodes so each group matches its target for this viewport
  function fill() {
    var extra = Math.min(50, Math.round(band * H / SIDE_AREA));
    [[0, Math.max(36, Math.min(150, Math.round(W * H / 11000)))], [-1, extra], [1, extra]].forEach(function (g) {
      var side = g[0], want = g[1], have = 0, i;
      for (i = 0; i < nodes.length; i++) if (nodes[i].side === side) have++;
      for (; have < want; have++) nodes.push(makeNode(side));
      for (i = nodes.length - 1; i >= 0 && have > want; i--) if (nodes[i].side === side) { nodes.splice(i, 1); have--; }
    });
  }

  // gutter = what the clear band leaves on each side, plus a little overlap
  // into the fade
  function measureBand() {
    var clear = parseFloat(getComputedStyle(canvas).getPropertyValue("--aisc-clear")) || 800;
    band = Math.max(80, Math.min(W / 3, (W - clear) / 2 + 60));
  }

  // The cards' boxes are measured in page coordinates only when the layout
  // may have moved (resize, the write-up changing size, a reveal settling,
  // and once a second besides); scrolling just shifts them.
  function layout() {
    laid = [];
    var y = window.scrollY;
    for (var i = 0; i < cards.length; i++) {
      var r = cards[i].getBoundingClientRect();
      if (r.width) laid.push({ l: r.left, r: r.right, t: r.top + y, b: r.bottom + y });
    }
    stale = false; age = 0;
  }
  function restale() { stale = true; }

  // the cards on screen now, in canvas coordinates
  function readBoxes() {
    boxes = [];
    var b = document.body.classList;
    if (b.contains("aisc-fig-max-open") || b.contains("wta-max-open")) return;
    if (stale || ++age > 60) layout();
    var off = window.scrollY + top;
    for (var i = 0; i < laid.length; i++) {
      var c = laid[i];
      if (c.b - off < 0 || c.t - off > H) continue;
      boxes.push({ l: c.l, r: c.r, t: c.t - off, b: c.b - off });
    }
  }

  function resize() {
    var oldW = W, oldH = H;
    var box = canvas.getBoundingClientRect();
    if (!box.width || !box.height) { W = H = 0; return; }
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = box.width; H = box.height; top = box.top;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    measureBand();
    cards = host.querySelectorAll(CARDS);
    stale = true;
    if (oldW && oldH) nodes.forEach(function (a) { a.x *= W / oldW; a.y *= H / oldH; });
    fill();
  }

  function step(dt) {
    var i, j, a, b, dx, dy, d2, d, range, s, f;
    for (i = 0; i < nodes.length; i++) {
      a = nodes[i];
      for (j = i + 1; j < nodes.length; j++) {
        b = nodes[j];
        dx = b.x - a.x; dy = b.y - a.y; d2 = dx * dx + dy * dy;
        range = REACH * (a.r + b.r);
        if (d2 > range * range || d2 < 0.01) continue;
        d = Math.sqrt(d2); s = 1 - d / range;
        // spring toward a rest length (heavier = stiffer), push apart when very close
        f = PULL * s * a.r * b.r * (d - REST * range) - PUSH * Math.max(0, 1 - d / (14 * (a.r + b.r))) / d;
        dx /= d; dy /= d;
        a.vx += dx * f / a.r * dt; a.vy += dy * f / a.r * dt;
        b.vx -= dx * f / b.r * dt; b.vy -= dy * f / b.r * dt;
      }
      if (mouse) {
        dx = mouse.x - a.x; dy = mouse.y - a.y; d2 = dx * dx + dy * dy;
        if (d2 < MOUSE_REACH * MOUSE_REACH) {
          d = Math.sqrt(d2); s = 1 - d / MOUSE_REACH;
          if (d2 > 400) { a.vx += dx / d * s * 0.002 * dt; a.vy += dy / d * s * 0.002 * dt; }
        }
      }
    }
    for (i = 0; i < nodes.length; i++) {
      a = nodes[i];
      if (a.side) {
        var over = a.side < 0 ? a.x - band : W - band - a.x;
        if (over > 0) a.vx += a.side * over * HOME * dt;
      }
      a.vx = (a.vx + (Math.random() - 0.5) * WANDER * dt) * Math.pow(DAMP, dt);
      a.vy = (a.vy + (Math.random() - 0.5) * WANDER * dt) * Math.pow(DAMP, dt);
      var v = Math.hypot(a.vx, a.vy);
      if (v > VMAX) { a.vx *= VMAX / v; a.vy *= VMAX / v; }
      else if (v < 0.08) { a.vx += (Math.random() - 0.5) * 0.05; a.vy += (Math.random() - 0.5) * 0.05; }
      a.x += a.vx * dt; a.y += a.vy * dt;
      if (a.x < -20) { a.x = -20; a.vx = Math.abs(a.vx); } else if (a.x > W + 20) { a.x = W + 20; a.vx = -Math.abs(a.vx); }
      if (a.y < -20) { a.y = -20; a.vy = Math.abs(a.vy); } else if (a.y > H + 20) { a.y = H + 20; a.vy = -Math.abs(a.vy); }
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    var edgeMax = col.dark ? 0.42 : 0.3, i, j, a, b, dx, dy, d2, range, s;
    ctx.lineCap = "round";
    for (i = 0; i < nodes.length; i++) {
      a = nodes[i];
      for (j = i + 1; j < nodes.length; j++) {
        b = nodes[j];
        dx = b.x - a.x; dy = b.y - a.y; d2 = dx * dx + dy * dy;
        range = REACH * (a.r + b.r);
        if (d2 > range * range) continue;
        s = 1 - Math.sqrt(d2) / range;
        ctx.strokeStyle = "rgba(" + col.edge + "," + (s * s * edgeMax).toFixed(3) + ")";
        ctx.lineWidth = 0.4 + s * Math.min(a.r, b.r) * 0.7;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }
    if (mouse) {
      for (i = 0; i < nodes.length; i++) {
        a = nodes[i];
        dx = a.x - mouse.x; dy = a.y - mouse.y; d2 = dx * dx + dy * dy;
        if (d2 > MOUSE_REACH * MOUSE_REACH) continue;
        s = 1 - Math.sqrt(d2) / MOUSE_REACH;
        ctx.strokeStyle = "rgba(" + col.edge + "," + (s * (col.dark ? 0.7 : 0.55)).toFixed(3) + ")";
        ctx.lineWidth = 0.5 + s * 1.4;
        ctx.beginPath(); ctx.moveTo(mouse.x, mouse.y); ctx.lineTo(a.x, a.y); ctx.stroke();
      }
      ctx.fillStyle = "rgba(" + col.edge + ",0.85)";
      ctx.beginPath(); ctx.arc(mouse.x, mouse.y, 3, 0, Math.PI * 2); ctx.fill();
    }
    for (i = 0; i < nodes.length; i++) {
      a = nodes[i];
      ctx.fillStyle = "rgba(" + col.node + "," + (col.dark ? 0.55 : 0.5) + ")";
      ctx.beginPath(); ctx.arc(a.x, a.y, a.r, 0, Math.PI * 2); ctx.fill();
    }
    // nothing of the field inside a card: not a node drifting behind it, not
    // a link across it, not the cursor's links while it reads
    if (boxes.length) {
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "#000";
      for (i = 0; i < boxes.length; i++) {
        a = boxes[i];
        ctx.fillRect(a.l - PAD, a.t - PAD, a.r - a.l + 2 * PAD, a.b - a.t + 2 * PAD);
      }
      ctx.globalCompositeOperation = "source-over";
    }
  }

  // Animates only while the page is on screen and the tab is visible.
  // Reduced motion: one still frame, no cursor links, redrawn only when the
  // canvas itself changes (resize, theme). The preference is watched live.
  var last = 0, raf = 0;
  function frame(t) {
    var dt = last ? Math.min((t - last) / 16.67, 3) : 1;
    last = t;
    readBoxes(); step(dt); draw();
    raf = requestAnimationFrame(frame);
  }
  function stop() { cancelAnimationFrame(raf); raf = 0; }
  function sync() {
    reduced = motionQ.matches;
    if (!onScreen || document.hidden) { stop(); forget(); return; }
    resize();
    if (!W) { stop(); return; } // canvas hidden (narrow screen): nothing to draw
    if (reduced) { stop(); mouse = null; still(); }
    else if (!raf) { last = 0; raf = requestAnimationFrame(frame); }
  }
  // the still frame, erased inside the cards as they sit now
  function still() { readBoxes(); draw(); }
  // the cursor is gone (left the window, or the page is off screen): no
  // links to it, no card left lit
  function forget() { mouse = null; at = null; setHot(null); }
  function lose() { forget(); if (reduced) draw(); }

  // the innermost card under the cursor; a note inside a tile is part of
  // that tile (it has no outline of its own)
  function setHot(el) {
    while (el && el.matches(".tile .note")) el = el.parentElement.closest(CARDS);
    if (el === hot) return;
    if (hot) hot.classList.remove("is-hot");
    hot = el;
    if (hot) hot.classList.add("is-hot");
  }
  var at = null; // the cursor, page-relative, for re-reading the card under it on scroll
  function hotAt(x, y) {
    var el = document.elementFromPoint(x, y);
    setHot(el && host.contains(el) ? el.closest(CARDS) : null);
  }
  // the cards move under a still frame as the page scrolls
  var stillRaf = 0, hotRaf = 0;
  window.addEventListener("scroll", function () {
    // a card scrolls under a cursor that stays put (once a frame)
    if (at && onScreen && !hotRaf) hotRaf = requestAnimationFrame(function () {
      hotRaf = 0;
      if (at) hotAt(at.x, at.y);
    });
    if (!reduced || !onScreen || !W || stillRaf) return;
    stillRaf = requestAnimationFrame(function () { stillRaf = 0; still(); });
  }, { passive: true });

  window.addEventListener("pointermove", function (e) {
    if (e.pointerType === "touch") return;
    at = { x: e.clientX, y: e.clientY };
    setHot(onScreen && host.contains(e.target) ? e.target.closest(CARDS) : null);
    if (raf) mouse = { x: e.clientX, y: e.clientY - top };
  }, { passive: true });
  document.documentElement.addEventListener("pointerleave", lose);
  window.addEventListener("blur", lose);
  window.addEventListener("resize", function () { if (onScreen) sync(); });
  document.addEventListener("visibilitychange", sync);
  if (motionQ.addEventListener) motionQ.addEventListener("change", sync);
  if (window.AISC && AISC.onTheme) AISC.onTheme(function () { tokens(); if (reduced) still(); });

  if (window.ResizeObserver) new ResizeObserver(restale).observe(host);
  host.addEventListener("transitionend", restale);

  new IntersectionObserver(function (ents) {
    onScreen = ents.some(function (e) { return e.isIntersecting; });
    sync();
  }).observe(host);

  tokens();
})();
