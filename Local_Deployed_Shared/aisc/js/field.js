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
   - The cards are see-through wireframes (Seth, 2026-09-25: "an invisible
     card there … the nodes get pushed out of the way so that it's easier to
     read"), so the field keeps out of them itself: every frame it reads the
     boxes of the cards on screen (CARDS, the same list aisc.css outlines),
     pushes any node inside one out through its nearest side, and erases its
     drawing inside them, so links never cross the text either. While a
     figure is full screen there is nothing to keep clear.
   - The cursor drags the nodes it is linked to as it moves, each by the
     same share its link is drawn at (Seth, 2026-09-25: "a strength
     according to … how solid the lines are"): a node right at the cursor
     follows it, one at the edge of its reach barely stirs.
   - The card under the cursor is marked .is-hot (aisc.css lights its outline
     in the cursor's colour) and pulls the nodes in round its outline. A
     section's title card (a .tile opening on a .kicker) is always lit and
     always pulls, much harder. Pulled nodes still stop at the card's edge.
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
  var PAD = 12;          // px of clear space kept round every card
  var EJECT = 0.14;      // share of its depth a node inside a card moves out per frame
  var FALL = 48;         // px outside a card over which it still pushes, fading out
  var SHOVE = 0.03;      // that push at the card's edge
  var DRAG = 1;          // share of the cursor's move a node takes, times its link strength
  var DRAG_MAX = 20;     // px a drag moves a node per frame: under half the thinnest padded card, so a flick can't carry one through
  var HOT_PULL = 0.03, HOT_REACH = 260;     // the card under the cursor pulls nodes in
  var TITLE_PULL = 0.05, TITLE_REACH = 420; // a title card, always; under the cursor, twice that

  var W = 0, H = 0, top = 0, dpr = 1, nodes = [], mouse = null, col = {};
  var moved = null;      // the cursor's move since the last frame
  var hot = null;        // the card under the cursor
  var band = 0;          // width of each side gutter the extra nodes live in
  var onScreen = false;
  var cards = [], boxes = []; // card elements; their padded boxes on screen, canvas px
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

  function isTitle(el) { return !!el.querySelector(":scope > .kicker"); }

  // The cards' boxes are measured in page coordinates only when the layout
  // may have moved (resize, the write-up changing size, a reveal settling,
  // and once a second besides); scrolling just shifts them.
  function layout() {
    laid = [];
    var y = window.scrollY;
    for (var i = 0; i < cards.length; i++) {
      var r = cards[i].getBoundingClientRect();
      if (r.width) laid.push({ l: r.left, r: r.right, t: r.top + y, b: r.bottom + y,
        el: cards[i], title: isTitle(cards[i]) });
    }
    stale = false; age = 0;
  }
  function restale() { stale = true; }

  // the cards on screen now, in canvas coordinates, grown by PAD, each with
  // how hard and how far it pulls nodes in (0 for most). A pulling card just
  // off screen still draws in the nodes near that edge.
  function readBoxes() {
    boxes = [];
    var b = document.body.classList;
    if (b.contains("aisc-fig-max-open") || b.contains("wta-max-open")) return;
    if (stale || ++age > 60) layout();
    var off = window.scrollY + top;
    for (var i = 0; i < laid.length; i++) {
      var c = laid[i], pull = 0, reach = 0;
      if (c.title) { pull = TITLE_PULL; reach = TITLE_REACH; }
      else if (c.el === hot) { pull = HOT_PULL; reach = HOT_REACH; }
      if (c.title && c.el === hot) pull *= 2;
      var edge = Math.max(PAD, reach);
      if (c.b - off < -edge || c.t - off > H + edge) continue;
      boxes.push({ l: c.l - PAD, r: c.r + PAD, t: c.t - off - PAD, b: c.b - off + PAD, pull: pull, reach: reach });
    }
  }

  // a node inside a card leaves through the nearest side, fast when deep in
  // (a card scrolled over it) and never slower than a drift, losing only the
  // part of its speed that points in. One just outside a card is shoved
  // gently away, so the nodes don't stack up into a wall on the edge, unless
  // the card pulls: then it is drawn toward the edge instead.
  function eject(a, dt) {
    for (var k = 0; k < boxes.length; k++) {
      var c = boxes[k];
      if (a.x <= c.l || a.x >= c.r || a.y <= c.t || a.y >= c.b) {
        var ox = a.x < c.l ? a.x - c.l : a.x > c.r ? a.x - c.r : 0;
        var oy = a.y < c.t ? a.y - c.t : a.y > c.b ? a.y - c.b : 0;
        var od = Math.sqrt(ox * ox + oy * oy);
        if (c.pull) {
          if (od > 0 && od < c.reach) {
            var pl = (1 - od / c.reach) * c.pull * dt / od;
            a.vx -= ox * pl; a.vy -= oy * pl;
          }
        } else if (od > 0 && od < FALL) {
          var sh = (1 - od / FALL) * SHOVE * dt / od;
          a.vx += ox * sh; a.vy += oy * sh;
        }
        continue;
      }
      var dl = a.x - c.l, dr = c.r - a.x, du = a.y - c.t, dd = c.b - a.y;
      var m = Math.min(dl, dr, du, dd), mv = Math.min(m, (1 + m * EJECT) * dt);
      if (m === dl) { a.x -= mv; a.vx = Math.min(a.vx, 0); }
      else if (m === dr) { a.x += mv; a.vx = Math.max(a.vx, 0); }
      else if (m === du) { a.y -= mv; a.vy = Math.min(a.vy, 0); }
      else { a.y += mv; a.vy = Math.max(a.vy, 0); }
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
          // carried along by the cursor's move, as strongly as its link is drawn
          if (moved) { a.x += moved.x * s * DRAG; a.y += moved.y * s * DRAG; }
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
      eject(a, dt);
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
    // nothing of the field inside a card: not a node on its way out, not a
    // link across it, not the cursor's links while it reads
    if (boxes.length) {
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "#000";
      for (i = 0; i < boxes.length; i++) {
        a = boxes[i];
        ctx.fillRect(a.l + PAD / 2, a.t + PAD / 2, a.r - a.l - PAD, a.b - a.t - PAD);
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
    if (moved) {
      var mm = Math.hypot(moved.x, moved.y);
      if (mm > DRAG_MAX) { moved.x *= DRAG_MAX / mm; moved.y *= DRAG_MAX / mm; }
    }
    readBoxes(); step(dt); draw();
    moved = null;
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
  // the still frame: nodes already out of the cards as they sit now
  function still() {
    readBoxes();
    for (var i = 0; i < nodes.length; i++) for (var n = 0; n < 40; n++) eject(nodes[i], 3);
    draw();
  }
  // the cursor is gone (left the window, or the page is off screen): no
  // links to it, no drag pending, no card left lit
  function forget() { mouse = null; moved = null; at = null; setHot(null); }
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
    if (!raf) return;
    var next = { x: e.clientX, y: e.clientY - top };
    // a jump (the cursor coming back in) is not a drag
    if (mouse && Math.abs(next.x - mouse.x) + Math.abs(next.y - mouse.y) < 120) {
      moved = moved || { x: 0, y: 0 };
      moved.x += next.x - mouse.x; moved.y += next.y - mouse.y;
    }
    mouse = next;
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
