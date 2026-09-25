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
     thing") the canvas is drawn unmasked across the whole page, as on the
     standalone site: the write-up's opaque cards hide it under the text and
     it shows between them and in the gutters. The extra gutter nodes read
     `--aisc-clear` (aisc.css: the cards' 1180px measure) for where the side
     bands start.
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

  var W = 0, H = 0, top = 0, dpr = 1, nodes = [], mouse = null, col = {};
  var band = 0;          // width of each side gutter the extra nodes live in
  var onScreen = false;

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

  function resize() {
    var oldW = W, oldH = H;
    var box = canvas.getBoundingClientRect();
    if (!box.width || !box.height) { W = H = 0; return; }
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = box.width; H = box.height; top = box.top;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    measureBand();
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
        if (d2 < MOUSE_REACH * MOUSE_REACH && d2 > 400) {
          d = Math.sqrt(d2); s = 1 - d / MOUSE_REACH;
          a.vx += dx / d * s * 0.002 * dt; a.vy += dy / d * s * 0.002 * dt;
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
  }

  // Animates only while the page is on screen and the tab is visible.
  // Reduced motion: one still frame, no cursor links, redrawn only when the
  // canvas itself changes (resize, theme). The preference is watched live.
  var last = 0, raf = 0;
  function frame(t) {
    var dt = last ? Math.min((t - last) / 16.67, 3) : 1;
    last = t;
    step(dt); draw();
    raf = requestAnimationFrame(frame);
  }
  function stop() { cancelAnimationFrame(raf); raf = 0; }
  function sync() {
    reduced = motionQ.matches;
    if (!onScreen || document.hidden) { stop(); return; }
    resize();
    if (!W) { stop(); return; } // canvas hidden (narrow screen): nothing to draw
    if (reduced) { stop(); mouse = null; draw(); }
    else if (!raf) { last = 0; raf = requestAnimationFrame(frame); }
  }
  function lose() { mouse = null; if (reduced) draw(); }

  window.addEventListener("pointermove", function (e) {
    if (!raf || e.pointerType === "touch") return;
    mouse = { x: e.clientX, y: e.clientY - top };
  }, { passive: true });
  document.documentElement.addEventListener("pointerleave", lose);
  window.addEventListener("blur", lose);
  window.addEventListener("resize", function () { if (onScreen) sync(); });
  document.addEventListener("visibilitychange", sync);
  if (motionQ.addEventListener) motionQ.addEventListener("change", sync);
  if (window.AISC && AISC.onTheme) AISC.onTheme(function () { tokens(); if (reduced) draw(); });

  new IntersectionObserver(function (ents) {
    onScreen = ents.some(function (e) { return e.isIntersecting; });
    sync();
  }).observe(host);

  tokens();
})();
