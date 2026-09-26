/* concept-graph/kg-layout.js — places the Knowledge Graph's nodes and routes
 * its edges to cut crossings, starting from dagre's layout.
 *
 * Seth, 2026-09-25: "make it such that it moves the nodes up and down and
 * around … the edges … go down and around, not just as simple up right up …
 * an algorithm to reduce the number of edge overlaps … node positioning to
 * reduce node distances away from each other (but not too much) and the
 * lowest total edge overlaps."
 *
 * Two passes, both deterministic (seeded by the graph itself, so a reload
 * draws the same map):
 *
 *   1. PLACE — simulated annealing over node positions. dagre's rows are the
 *      start, not a rule: a node may move sideways and up or down, or trade
 *      places with a node of its row, as long as it overlaps nothing and every
 *      prerequisite stays below what it unlocks (`gapY` clear). The cost it
 *      lowers is, heaviest first: straight-line crossings between edges, lines
 *      running through a node they don't touch, total edge length (keeps the
 *      map tight), and a weak pull to the centre (keeps loose nodes from
 *      drifting). Moves are scored incrementally: only the moved nodes' edges.
 *
 *   2. ROUTE — every edge is a shortest path on a grid (A*, state = cell +
 *      heading) from its source's top to its target's bottom, around every
 *      node box. Paths may go any way — right, down, back up, left — and pay
 *      for length, for each turn, for hugging a node, and heavily for sharing
 *      a cell with an edge it has no endpoint in common with (a crossing or an
 *      overlap). Edges that share an endpoint may share cells: that is a
 *      bundle, not a crossing. Then negotiated congestion (PathFinder): the
 *      edges still in conflict are ripped up and rerouted, with the contested
 *      cells made dearer each round.
 *
 * A route comes back as control points with rounded corners, in model
 * coordinates; kg-look.js turns them into Cytoscape's unbundled-bezier.
 *
 * Pure: no DOM, no Cytoscape. Also loads under Node for offline tuning
 * (`module.exports`). */
(function (root) {
  "use strict";

  /* ---------------- small tools ---------------------------------------- */
  const hashStr = (s) => {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  };
  const rng = (seed) => {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  const orient = (ax, ay, bx, by, cx, cy) => (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  // Proper crossing of segments ab and cd (touching doesn't count).
  const segX = (ax, ay, bx, by, cx, cy, dx, dy) => {
    if (Math.max(ax, bx) < Math.min(cx, dx) || Math.max(cx, dx) < Math.min(ax, bx) ||
        Math.max(ay, by) < Math.min(cy, dy) || Math.max(cy, dy) < Math.min(ay, by)) return false;
    const d1 = orient(ax, ay, bx, by, cx, cy), d2 = orient(ax, ay, bx, by, dx, dy);
    if (d1 * d2 >= 0) return false;
    const d3 = orient(cx, cy, dx, dy, ax, ay), d4 = orient(cx, cy, dx, dy, bx, by);
    return d3 * d4 < 0;
  };
  // Does segment ab pass through the box (Liang–Barsky)?
  const segBox = (ax, ay, bx, by, x1, y1, x2, y2) => {
    let t0 = 0, t1 = 1;
    const dx = bx - ax, dy = by - ay;
    const p = [-dx, dx, -dy, dy], q = [ax - x1, x2 - ax, ay - y1, y2 - ay];
    for (let i = 0; i < 4; i++) {
      if (p[i] === 0) { if (q[i] < 0) return false; continue; }
      const r = q[i] / p[i];
      if (p[i] < 0) { if (r > t1) return false; if (r > t0) t0 = r; }
      else { if (r < t0) return false; if (r < t1) t1 = r; }
    }
    return t1 - t0 > 1e-6;
  };

  /* ---------------- the model ------------------------------------------ */
  // nodes: [{ id, x, y, w, h, ox, oy, cw, ch }]
  //   x, y   the node's position (Cytoscape's), seeded from dagre
  //   w, h   its whole box, label included; the box's centre is (x+ox, y+oy)
  //   cw, ch the drawn shape alone (a dot is smaller than its box)
  // edges: [{ id, s, t }] — s is the prerequisite, drawn BELOW t.
  const model = (nodes, edges) => {
    const idx = new Map();
    nodes.forEach((n, i) => idx.set(n.id, i));
    const E = [];
    edges.forEach((e) => {
      const s = idx.get(e.s), t = idx.get(e.t);
      if (s == null || t == null || s === t) return;
      E.push({ id: e.id, s, t });
    });
    const inc = nodes.map(() => []);
    E.forEach((e, k) => { inc[e.s].push(k); inc[e.t].push(k); });
    return { idx, E, inc };
  };

  /* ---------------- 1. place ------------------------------------------- */
  const PLACE = {
      // Separations are box-to-box. 40/40/50 packed every node at the
      // minimum (median gap 60 px; Seth 2026-09-25: "too jumbled and close
      // together"); these give a median of ~137 px at the same crossings.
      sepX: 130, sepY: 110, gapY: 120, wCross: 200, wThrough: 40, wLen: 0.2, wGrav: 0.02, slope: 1.5,
      sweeps: 220, gap: 140, polish: 30, t0: 70, t1: 0.4, sigma: 220, seed: 1, ov0: 0.0005, ov1: 2,
  };
  const place = (nodes, edges, opt) => {
    const o = Object.assign({}, PLACE, opt || {});
    const n = nodes.length;
    const { E, inc } = model(nodes, edges);
    const X = new Float64Array(n), Y = new Float64Array(n);
    const hw = new Float64Array(n), hh = new Float64Array(n), ox = new Float64Array(n), oy = new Float64Array(n);
    nodes.forEach((d, i) => { X[i] = d.x; Y[i] = d.y; hw[i] = d.w / 2; hh[i] = d.h / 2; ox[i] = d.ox || 0; oy[i] = d.oy || 0; });
    const up = nodes.map(() => []), down = nodes.map(() => []);
    E.forEach((e) => { up[e.s].push(e.t); down[e.t].push(e.s); });
    let cx = 0, cy = 0;
    for (let i = 0; i < n; i++) { cx += X[i]; cy += Y[i]; }
    cx /= n || 1; cy /= n || 1;

    // Hard rules: no overlap, and every prerequisite clear below.
    // Room for the grid snap in layout(), which moves a node up to half a
    // cell each way: the rules are kept a cell wider here so they still hold.
    const sepX = o.sepX + (o.snap || 0), sepY = o.sepY + (o.snap || 0), gapY = o.gapY + (o.snap || 0);
    const overlaps = (i, x, y, skip) => {
      const bx = x + ox[i], by = y + oy[i];
      for (let j = 0; j < n; j++) {
        if (j === i || j === skip) continue;
        if (Math.abs(bx - X[j] - ox[j]) < hw[i] + hw[j] + sepX &&
            Math.abs(by - Y[j] - oy[j]) < hh[i] + hh[j] + sepY) return true;
      }
      return false;
    };
    // The band of y a node may take: under all it unlocks, over all it needs.
    const band = (i) => {
      let lo = -Infinity, hi = Infinity;
      const top = (j) => Y[j] + oy[j] - hh[j], bot = (j) => Y[j] + oy[j] + hh[j];
      up[i].forEach((t) => { lo = Math.max(lo, bot(t) + gapY + hh[i] - oy[i]); });
      down[i].forEach((s) => { hi = Math.min(hi, top(s) - gapY - hh[i] - oy[i]); });
      return [lo, hi];
    };

    // Edges run centre to centre of the boxes for scoring.
    const ex = (k, end) => { const i = end ? E[k].t : E[k].s; return X[i] + ox[i]; };
    const ey = (k, end) => { const i = end ? E[k].t : E[k].s; return Y[i] + oy[i]; };
    const share = (a, b) => a.s === b.s || a.s === b.t || a.t === b.s || a.t === b.t;
    const crosses = (a, b) => !share(E[a], E[b]) &&
      segX(ex(a, 0), ey(a, 0), ex(a, 1), ey(a, 1), ex(b, 0), ey(b, 0), ex(b, 1), ey(b, 1));
    const PAD = 4;
    const through = (k, j) => j !== E[k].s && j !== E[k].t &&
      segBox(ex(k, 0), ey(k, 0), ex(k, 1), ey(k, 1),
        X[j] + ox[j] - hw[j] - PAD, Y[j] + oy[j] - hh[j] - PAD, X[j] + ox[j] + hw[j] + PAD, Y[j] + oy[j] + hh[j] + PAD);
    // Length, plus a charge for running more sideways than up: a route has
    // to leave upward and arrive from below, so a wide edge becomes a loop.
    const len = (k) => {
      const dx = Math.abs(ex(k, 1) - ex(k, 0)), dy = Math.abs(ey(k, 1) - ey(k, 0));
      return Math.hypot(dx, dy) + o.slope * Math.max(0, dx - dy);
    };

    // Cost of everything the moved nodes M touch. `mark` flags their edges.
    const mark = new Uint8Array(E.length);
    const local = (M, C) => {
      let c = 0;
      C.forEach((k) => {
        c += o.wLen * len(k);
        for (let f = 0; f < E.length; f++) {
          if (f === k || (mark[f] && f < k)) continue;
          if (crosses(k, f)) c += o.wCross;
        }
        for (let j = 0; j < n; j++) if (through(k, j)) c += o.wThrough;
      });
      M.forEach((j) => {
        for (let f = 0; f < E.length; f++) if (!mark[f] && through(f, j)) c += o.wThrough;
        c += o.wGrav * Math.hypot(X[j] - cx, Y[j] - cy);
      });
      return c;
    };
    const total = () => {
      let c = 0;
      for (let k = 0; k < E.length; k++) {
        c += o.wLen * len(k);
        for (let f = k + 1; f < E.length; f++) if (crosses(k, f)) c += o.wCross;
        for (let j = 0; j < n; j++) if (through(k, j)) c += o.wThrough;
      }
      for (let j = 0; j < n; j++) c += o.wGrav * Math.hypot(X[j] - cx, Y[j] - cy);
      return c;
    };

    const rand = rng(o.seed);
    const gauss = () => { let u = 0, v = 0; while (!u) u = rand(); while (!v) v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const before = total();
    let xs0 = 0;
    for (let k = 0; k < E.length; k++) for (let f = k + 1; f < E.length; f++) if (crosses(k, f)) xs0++;

    // Overlap as a price, not a wall, while the search is hot: a node can
    // pass through a crowded row to a better place, and the price climbs
    // until nothing overlaps (`legalize` settles what's left).
    const ovPair = (i, j) => {
      const ax = hw[i] + hw[j] + sepX - Math.abs(X[i] + ox[i] - X[j] - ox[j]);
      if (ax <= 0) return 0;
      const ay = hh[i] + hh[j] + sepY - Math.abs(Y[i] + oy[i] - Y[j] - oy[j]);
      return ay <= 0 ? 0 : ax * ay;
    };
    let wOv = 0;
    const ovLocal = (M) => {
      if (!wOv) return 0;
      let v = 0;
      M.forEach((i) => { for (let j = 0; j < n; j++) if (j !== i && !M.includes(j)) v += ovPair(i, j); });
      if (M.length === 2) v += ovPair(M[0], M[1]);
      return v * wOv;
    };

    let accepted = 0;
    const anneal = (sweeps, t0, t1, sigma, soft) => {
      const steps = Math.max(1, Math.round(sweeps * n));
      for (let step = 0; step < steps; step++) {
        const prog = step / steps;
        const T = t0 * Math.pow(t1 / t0, prog);
        const sig = sigma * (1 - prog) + 12;
        wOv = soft ? o.ov0 * Math.pow(o.ov1 / o.ov0, prog) : 0;
        const i = Math.floor(rand() * n);
        const r = rand();
        let j = -1, nx, ny;
        if (r < 0.2) {
          // Trade places with a node near the same height.
          const cand = [];
          for (let q = 0; q < n; q++) if (q !== i && Math.abs(Y[q] - Y[i]) < 90 && Math.abs(X[q] - X[i]) < 4 * sig + 120) cand.push(q);
          if (!cand.length) continue;
          j = cand[Math.floor(rand() * cand.length)];
        } else if (r < 0.45 && inc[i].length) {
          // Toward the middle of its neighbours.
          let sx = 0, sy = 0;
          inc[i].forEach((k) => { const q = E[k].s === i ? E[k].t : E[k].s; sx += X[q]; sy += Y[q]; });
          nx = sx / inc[i].length + gauss() * sig * 0.3;
          ny = (rand() < 0.5 ? Y[i] : sy / inc[i].length) + gauss() * sig * 0.3;
        } else {
          nx = X[i] + gauss() * sig;
          ny = Y[i] + gauss() * sig * 0.6;
        }

        const M = j < 0 ? [i] : [i, j];
        const Cset = new Set(inc[i]);
        if (j >= 0) inc[j].forEach((k) => Cset.add(k));
        const C = [...Cset];
        C.forEach((k) => { mark[k] = 1; });
        const oldI = [X[i], Y[i]], oldJ = j >= 0 ? [X[j], Y[j]] : null;
        const was = local(M, C) + ovLocal(M);
        let ok = true;
        if (j < 0) {
          const [lo, hi] = band(i);
          if (lo > hi) ok = false;
          else {
            ny = Math.min(hi, Math.max(lo, ny));
            if (!soft && overlaps(i, nx, ny, -1)) ok = false;
            else { X[i] = nx; Y[i] = ny; }
          }
        } else {
          // Swap the boxes' centres; the hierarchy must still hold for both.
          const ai = X[i] + ox[i], bi = Y[i] + oy[i], aj = X[j] + ox[j], bj = Y[j] + oy[j];
          X[i] = aj - ox[i]; Y[i] = bj - oy[i]; X[j] = ai - ox[j]; Y[j] = bi - oy[j];
          const [li, hi_] = band(i), [lj, hj] = band(j);
          if (Y[i] < li || Y[i] > hi_ || Y[j] < lj || Y[j] > hj) ok = false;
          else if (!soft && (overlaps(i, X[i], Y[i], j) || overlaps(j, X[j], Y[j], i) || ovPair(i, j) > 0)) ok = false;
        }
        if (ok) {
          const d = local(M, C) + ovLocal(M) - was;
          if (d <= 0 || rand() < Math.exp(-d / T)) accepted++;
          else ok = false;
        }
        if (!ok) {
          X[i] = oldI[0]; Y[i] = oldI[1];
          if (oldJ) { X[j] = oldJ[0]; Y[j] = oldJ[1]; }
        }
        C.forEach((k) => { mark[k] = 0; });
      }
      wOv = 0;
    };
    // Pull overlapping boxes apart sideways (sideways never breaks the
    // hierarchy), a little at a time, until none overlap.
    const legalize = () => {
      for (let it = 0; it < 400; it++) {
        let moved = false;
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
          if (!ovPair(i, j)) continue;
          const dx = X[j] + ox[j] - X[i] - ox[i];
          const need = hw[i] + hw[j] + sepX - Math.abs(dx) + 0.5;
          const dir = dx > 0 || (dx === 0 && i < j) ? 1 : -1;
          X[i] -= dir * need / 2; X[j] += dir * need / 2;
          moved = true;
        }
        if (!moved) return it;
      }
      return -1;
    };

    anneal(o.sweeps, o.t0, o.t1, o.sigma, o.soft !== false);
    const legal = legalize();
    // A cool pass with hard walls tidies whatever legalizing shoved.
    anneal(o.polish, o.t1 * 4, o.t1 / 4, 40, false);
    const steps = Math.round((o.sweeps + o.polish) * n);

    // Unconnected pieces (a course with no link to the rest) side by side,
    // largest first, bottoms level: nothing between them to route round.
    const comp = new Int32Array(n).fill(-1);
    let nc = 0;
    for (let i = 0; i < n; i++) {
      if (comp[i] >= 0) continue;
      const stack = [i];
      comp[i] = nc;
      while (stack.length) {
        const v = stack.pop();
        inc[v].forEach((k) => { const q = E[k].s === v ? E[k].t : E[k].s; if (comp[q] < 0) { comp[q] = nc; stack.push(q); } });
      }
      nc++;
    }
    if (nc > 1) {
      const bb = Array.from({ length: nc }, () => [Infinity, Infinity, -Infinity, -Infinity, 0]);
      for (let i = 0; i < n; i++) {
        const b = bb[comp[i]];
        b[0] = Math.min(b[0], X[i] + ox[i] - hw[i]); b[1] = Math.min(b[1], Y[i] + oy[i] - hh[i]);
        b[2] = Math.max(b[2], X[i] + ox[i] + hw[i]); b[3] = Math.max(b[3], Y[i] + oy[i] + hh[i]);
        b[4]++;
      }
      const ord = bb.map((b, k) => k).sort((a, b) => bb[b][4] - bb[a][4]);
      const shift = [];
      let at = bb[ord[0]][2] + o.gap;
      const floor = bb[ord[0]][3];
      shift[ord[0]] = [0, 0];
      ord.slice(1).forEach((k) => { shift[k] = [at - bb[k][0], floor - bb[k][3]]; at += bb[k][2] - bb[k][0] + o.gap; });
      for (let i = 0; i < n; i++) { X[i] += shift[comp[i]][0]; Y[i] += shift[comp[i]][1]; }
    }
    const after = total();
    let xs = 0;
    for (let k = 0; k < E.length; k++) for (let f = k + 1; f < E.length; f++) if (crosses(k, f)) xs++;
    const pos = {};
    nodes.forEach((d, i) => { pos[d.id] = { x: X[i], y: Y[i] }; });
    return { pos, stats: { before: Math.round(before), after: Math.round(after), straightCrossings0: xs0, straightCrossings: xs, accepted, steps, legal, pieces: nc } };
  };

  /* ---------------- 2. route ------------------------------------------- */
  // A binary min-heap of (key, value) in parallel arrays.
  const heap = () => {
    let K = new Float64Array(1024), V = new Int32Array(1024), size = 0;
    return {
      get size() { return size; },
      clear() { size = 0; },
      push(k, v) {
        if (size === K.length) {
          const K2 = new Float64Array(size * 2), V2 = new Int32Array(size * 2);
          K2.set(K); V2.set(V); K = K2; V = V2;
        }
        let i = size++;
        while (i > 0) {
          const p = (i - 1) >> 1;
          if (K[p] <= k) break;
          K[i] = K[p]; V[i] = V[p]; i = p;
        }
        K[i] = k; V[i] = v;
      },
      pop() {
        const top = V[0], k = K[--size], v = V[size];
        let i = 0;
        for (;;) {
          let c = 2 * i + 1;
          if (c >= size) break;
          if (c + 1 < size && K[c + 1] < K[c]) c++;
          if (K[c] >= k) break;
          K[i] = K[c]; V[i] = V[c]; i = c;
        }
        K[i] = k; V[i] = v;
        return top;
      },
    };
  };

  // Routing grid. 16 px keeps parallel edges a lane apart that reads as two
  // lines, not a smear, and is no slower than 12 (fewer cells, longer steps).
  const CELL = 16;
  // Eight headings, clockwise from up; odd ones are diagonal.
  const DX = [0, 1, 1, 1, 0, -1, -1, -1], DY = [-1, -1, 0, 1, 1, 1, 0, -1];
  const SQ2 = Math.SQRT2;
  const ROUTE = {
      cell: CELL, pad: 6, margin: 180, bend: 3, near: 0.8, cross: 100, bundle: 0.15, down: 0.8,
      hist: 0.6, rounds: 4, radius: 60, lane: 4, laneRel: 1, lane2: 0.25, entry: "bottom", window: 28, wall: 400, greed: 1.25,
  };
  const route = (nodes, edges, pos, opt) => {
    const o = Object.assign({}, ROUTE, opt || {});
    const { E } = model(nodes, edges);
    const c = o.cell;
    const P = nodes.map((d) => pos[d.id] || { x: d.x, y: d.y });
    const box = nodes.map((d, i) => {
      const bx = P[i].x + (d.ox || 0), by = P[i].y + (d.oy || 0);
      return [bx - d.w / 2, by - d.h / 2, bx + d.w / 2, by + d.h / 2];
    });
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    box.forEach((b) => { x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]); x1 = Math.max(x1, b[2]); y1 = Math.max(y1, b[3]); });
    if (!isFinite(x0)) return { routes: {}, stats: {} };
    // Origin on a multiple of the cell, so a snapped node sits on a centre line.
    x0 = Math.floor((x0 - o.margin) / c) * c; y0 = Math.floor((y0 - o.margin) / c) * c;
    x1 += o.margin; y1 += o.margin;
    const GW = Math.ceil((x1 - x0) / c) + 1, GH = Math.ceil((y1 - y0) / c) + 1, NC = GW * GH;
    const col = (x) => Math.max(0, Math.min(GW - 1, Math.floor((x - x0) / c)));
    const row = (y) => Math.max(0, Math.min(GH - 1, Math.floor((y - y0) / c)));
    const cxOf = (q) => x0 + (q % GW) * c + c / 2, cyOf = (q) => y0 + Math.floor(q / GW) * c + c / 2;

    // Node boxes are walls; the ring around them costs a little (clearance).
    const wall = new Int32Array(NC);
    const near = new Uint8Array(NC);
    box.forEach((b, i) => {
      const cA = col(b[0] - o.pad), cB = col(b[2] + o.pad), rA = row(b[1] - o.pad), rB = row(b[3] + o.pad);
      for (let r = rA; r <= rB; r++) for (let q = cA; q <= cB; q++) wall[r * GW + q] = i + 1;
      for (let r = Math.max(0, rA - 1); r <= Math.min(GH - 1, rB + 1); r++)
        for (let q = Math.max(0, cA - 1); q <= Math.min(GW - 1, cB + 1); q++) near[r * GW + q] = 1;
    });

    // Ports: out of the source's top centre; into the target's bottom centre
    // (or, for dots whose label hangs underneath, either side of the dot).
    const cw = (i) => nodes[i].cw || nodes[i].w, ch = (i) => nodes[i].ch || nodes[i].h;
    const startCell = (i) => (row(box[i][1] - o.pad) - 1) * GW + col(P[i].x);
    const goalCells = (i) => {
      if (o.entry === "side") {
        const r = row(P[i].y);
        return [r * GW + col(box[i][0] - o.pad) - 1, r * GW + col(box[i][2] + o.pad) + 1];
      }
      return [(row(box[i][3] + o.pad) + 1) * GW + col(P[i].x)];
    };
    // The heading a goal cell is entered with: up into a bottom port,
    // sideways into a side one.
    const goalDir = (i, q) => (o.entry === "side" ? (cxOf(q) < P[i].x ? 2 : 6) : 0);
    const turnOf = (a, b) => { const t = Math.abs(a - b) & 7; return t > 4 ? 8 - t : t; };
    const TURN = [0, o.bend * 0.5, o.bend * 1.3, o.bend * 4, 1e6];
    // Heading pairs a step may take (nothing sharper than a right angle)
    // and what the turn costs, flattened for the inner loop.
    const TCOST = new Float64Array(64);
    for (let a = 0; a < 8; a++) for (let b = 0; b < 8; b++) TCOST[a * 8 + b] = turnOf(a, b) > 2 ? -1 : TURN[turnOf(a, b)];

    const occ = new Array(NC);         // cell -> edges through it
    const nbr = new Array(NC);         // cell -> edges through one of its 8 neighbours
    const nbr2 = new Array(NC);        // cell -> edges two cells away (the next ring)
    const docc = new Map();            // 2x2 block + slant -> edges crossing it diagonally
    const hist = new Float32Array(NC); // negotiated-congestion history
    const paths = new Array(E.length);
    const shares = (a, b) => a.s === b.s || a.s === b.t || a.t === b.s || a.t === b.t;
    // Running next to an unrelated edge (one lane over) costs `lane` a cell:
    // two lines 16 px apart read as one smear. Bundles stay free.
    // An edge counts once, in the nearest ring it is in: one that runs
    // THROUGH the cell is a shared trunk (priced by `price`), not a lane over.
    const lanePrice = (L, k, f, A, B) => {
      let v = 0;
      for (let m = 0; m < L.length; m++) {
        const e = L[m];
        if (e === k || (A && A.includes(e)) || (B && B.includes(e))) continue;
        v += shares(E[e], E[k]) ? o.laneRel : o.lane;
      }
      return v * f;
    };
    const price = (L, k, q) => {
      let v = 0;
      if (!L) return 0;
      for (let m = 0; m < L.length; m++) {
        const f = L[m];
        if (f !== k) v += shares(E[f], E[k]) ? o.bundle : o.cross * (1 + hist[q]);
      }
      return v;
    };
    // Two diagonals through one 2x2 block cross without sharing a cell.
    const dkey = (qa, qb) => {
      const ax = qa % GW, ay = (qa - ax) / GW, bx = qb % GW, by = (qb - bx) / GW;
      return (Math.min(ay, by) * GW + Math.min(ax, bx)) * 2 + ((bx - ax) * (by - ay) > 0 ? 0 : 1);
    };
    const lay = (k, add) => {
      const p = paths[k];
      if (!p) return;
      const seen = new Set();
      let into = occ;
      const put = (L, map, key) => {
        if (add) { if (!L) { L = []; if (map) map.set(key, L); else into[key] = L; } L.push(k); }
        else if (L) { const at = L.indexOf(k); if (at >= 0) L.splice(at, 1); }
      };
      const around = new Set(), around2 = new Set();
      p.forEach((q, m) => {
        if (!seen.has(q)) {
          seen.add(q); put(occ[q], null, q);
          {
            const x = q % GW;
            for (let d = 0; d < 8; d++) {
              const nx = x + DX[d], n = q + DY[d] * GW + DX[d];
              if (nx < 0 || nx >= GW || n < 0 || n >= NC || around.has(n)) continue;
              around.add(n); into = nbr; put(nbr[n], null, n); into = occ;
            }
            if (o.lane2) {
              for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
                if (Math.max(Math.abs(dx), Math.abs(dy)) !== 2) continue;
                const nx = x + dx, n = q + dy * GW + dx;
                if (nx < 0 || nx >= GW || n < 0 || n >= NC || around2.has(n)) continue;
                around2.add(n); into = nbr2; put(nbr2[n], null, n); into = occ;
              }
            }
          }
        }
        if (m && q - p[m - 1] !== 1 && p[m - 1] - q !== 1 && q - p[m - 1] !== GW && p[m - 1] - q !== GW) {
          const key = dkey(p[m - 1], q);
          put(docc.get(key), docc, key);
        }
      });
    };

    const NS = NC * 8;
    const g = new Float64Array(NS), stamp = new Int32Array(NS), from = new Int32Array(NS);
    let epoch = 0, pops = 0, searches = 0;
    const H = heap();
    const DONE = 1 << 30;
    const astar = (k, win, soft) => {
      const e = E[k];
      const s0 = startCell(e.s);
      const goals = goalCells(e.t);
      const gA = goals[0], gB = goals.length > 1 ? goals[1] : -1;
      const dA = goalDir(e.t, gA), dB = gB >= 0 ? goalDir(e.t, gB) : 0;
      const gx = goals.map((q) => q % GW), gy = goals.map((q) => Math.floor(q / GW));
      const h = (q) => {
        const x = q % GW, y = (q - x) / GW;
        let m = Infinity;
        for (let a = 0; a < gx.length; a++) {
          const dx = Math.abs(x - gx[a]), dy = Math.abs(y - gy[a]);
          m = Math.min(m, Math.max(dx, dy) + (SQ2 - 1) * Math.min(dx, dy));
        }
        return m;
      };
      let wx0 = 0, wy0 = 0, wx1 = GW - 1, wy1 = GH - 1;
      if (win) {
        const xs = [s0 % GW, ...gx], ys = [Math.floor(s0 / GW), ...gy];
        wx0 = Math.max(0, Math.min(...xs) - win); wx1 = Math.min(GW - 1, Math.max(...xs) + win);
        wy0 = Math.max(0, Math.min(...ys) - win); wy1 = Math.min(GH - 1, Math.max(...ys) + win);
      }
      // Node boxes stop the search. Only if that leaves no way at all (a
      // start boxed in by close neighbours) is the search re-run `soft`,
      // where another node's box costs `wall` a cell instead; the edge's own
      // two boxes always stop it.
      const own1 = e.s + 1, own2 = e.t + 1, WALL = soft ? o.wall : 1e6;
      const wallCost = (q) => {
        const w = wall[q];
        if (!w || q === gA || q === gB) return 0;
        return w === own1 || w === own2 ? 1e6 : WALL;
      };
      epoch++;
      searches++;
      H.clear();
      const st0 = s0 * 8;
      g[st0] = 0; stamp[st0] = epoch; from[st0] = -1;
      H.push(h(s0), st0);
      let found = -1;
      while (H.size) {
        const st = H.pop();
        pops++;
        if (st >= DONE) { found = st - DONE; break; }
        const q = st >> 3, d = st & 7, gq = g[st];
        const x = q % GW, y = (q - x) / GW;
        for (let nd = 0; nd < 8; nd++) {
          const tc = TCOST[d * 8 + nd];
          if (tc < 0) continue;
          const nx = x + DX[nd], ny = y + DY[nd];
          if (nx < wx0 || nx > wx1 || ny < wy0 || ny > wy1) continue;
          const nq = ny * GW + nx;
          const w = wallCost(nq);
          if (w >= 1e6) continue;
          const diag = nd & 1;
          let step = diag ? SQ2 : 1, cost = step + w + tc + (near[nq] ? o.near : 0);
          if (DY[nd] > 0) cost += o.down * step;     // against the flow
          if (diag) {
            // No squeezing between two boxes' corners.
            if (wallCost(y * GW + nx) >= WALL || wallCost(ny * GW + x) >= WALL) continue;
            cost += price(docc.get(dkey(q, nq) ^ 1), k, nq);
          }
          if (occ[nq]) cost += price(occ[nq], k, nq);
          if (nbr[nq]) cost += lanePrice(nbr[nq], k, step, occ[nq]);
          if (nbr2[nq]) cost += lanePrice(nbr2[nq], k, step * o.lane2, occ[nq], nbr[nq]);
          const gd = nq === gA ? dA : nq === gB ? dB : undefined;
          if (gd != null) cost += TURN[Math.min(2, turnOf(nd, gd))] + (turnOf(nd, gd) > 2 ? o.bend * 4 : 0);
          const ns = nq * 8 + nd, ng = gq + cost;
          if (stamp[ns] === epoch && g[ns] <= ng) continue;
          stamp[ns] = epoch; g[ns] = ng; from[ns] = st;
          if (gd != null) H.push(ng, DONE + ns);
          else H.push(ng + o.greed * h(nq), ns);
        }
      }
      if (found < 0) return null;
      const cells = [];
      for (let st = found; st >= 0; st = from[st]) cells.push(st >> 3);
      cells.reverse();
      return cells;
    };

    // Short edges first: they have the fewest ways round.
    const span = (k) => Math.abs(P[E[k].s].x - P[E[k].t].x) + Math.abs(P[E[k].s].y - P[E[k].t].y);
    const order = E.map((e, k) => k).sort((a, b) => span(a) - span(b));
    const routeOne = (k) => { paths[k] = astar(k, o.window, false) || astar(k, 0, false) || astar(k, 0, true); lay(k, true); };
    order.forEach(routeOne);

    // Pairs of unrelated edges that share a cell or cross in a block.
    const clashes = (bump) => {
      const bad = new Set(), pairs = new Set();
      const scan = (A, B, q) => {
        for (let a = 0; a < A.length; a++) for (let b = (B === A ? a + 1 : 0); b < B.length; b++) {
          if (A[a] === B[b] || shares(E[A[a]], E[B[b]])) continue;
          bad.add(A[a]); bad.add(B[b]);
          pairs.add(Math.min(A[a], B[b]) * 100003 + Math.max(A[a], B[b]));
          if (bump) hist[q] += o.hist;
        }
      };
      for (let q = 0; q < NC; q++) if (occ[q] && occ[q].length > 1) scan(occ[q], occ[q], q);
      docc.forEach((L, key) => {
        if (!(key & 1) && L.length) { const M = docc.get(key | 1); if (M && M.length) scan(L, M, key >> 1); }
      });
      return { bad, pairs };
    };
    let rounds = 0;
    for (; rounds < o.rounds; rounds++) {
      const { bad } = clashes(true);
      if (!bad.size) break;
      order.forEach((k) => { if (bad.has(k)) { lay(k, false); routeOne(k); } });
    }
    const left = clashes(false).pairs.size;
    // Edge closeness: cells of a route with an unrelated edge one lane over
    // (a route's own neighbours don't count, nor the cells it shares).
    let closeCells = 0;
    paths.forEach((p, k) => {
      if (p) p.forEach((q) => {
        const L = nbr[q];
        if (L && L.some((f) => f !== k && !shares(E[f], E[k]) && !(occ[q] && occ[q].includes(f)))) closeCells++;
      });
    });

    // Cells -> turning points -> control points.
    const routes = {};
    E.forEach((e, k) => {
      const p = paths[k];
      if (!p) return;
      const pts = [{ x: P[e.s].x, y: P[e.s].y - ch(e.s) / 2 }];
      for (let m = 0; m < p.length; m++) {
        if (m > 0 && m < p.length - 1 && p[m] - p[m - 1] === p[m + 1] - p[m]) continue;
        pts.push({ x: cxOf(p[m]), y: cyOf(p[m]) });
      }
      const gq = p[p.length - 1];
      pts.push(o.entry === "side"
        ? { x: P[e.t].x + (cxOf(gq) < P[e.t].x ? -1 : 1) * cw(e.t) / 2, y: P[e.t].y }
        : { x: P[e.t].x, y: P[e.t].y + ch(e.t) / 2 });
      const clean = [pts[0]];
      for (let m = 1; m < pts.length - 1; m++) {
        const a = clean[clean.length - 1], b = pts[m], z = pts[m + 1];
        if (Math.abs(orient(a.x, a.y, b.x, b.y, z.x, z.y)) < 1 && (b.x - a.x) * (z.x - b.x) + (b.y - a.y) * (z.y - b.y) >= 0) continue;
        clean.push(b);
      }
      clean.push(pts[pts.length - 1]);
      // Control points: every corner, plus a guide `radius` in from each end
      // of a long run. Cytoscape's curve passes half-way between consecutive
      // points, so a long run stays straight and turns in a rounded corner,
      // and a short jog between two corners (no guides) becomes an S-curve.
      const cps = [], R = o.radius;
      for (let m = 0; m + 1 < clean.length; m++) {
        const a = clean[m], b = clean[m + 1];
        if (m > 0) cps.push(a);
        const L = Math.hypot(b.x - a.x, b.y - a.y);
        if (L <= 2 * R) continue;
        const ux = (b.x - a.x) / L, uy = (b.y - a.y) / L;
        if (m > 0) cps.push({ x: a.x + ux * R, y: a.y + uy * R });
        if (m + 2 < clean.length) cps.push({ x: b.x - ux * R, y: b.y - uy * R });
      }
      routes[e.id] = { pts: clean, cps };
    });
    return { routes, stats: { grid: GW + "x" + GH, rounds, clashPairs: left, closeCells, searches, pops } };
  };

  /* ---------------- both ----------------------------------------------- */
  const layout = (nodes, edges, opt) => {
    const o = opt || {};
    const seed = hashStr(nodes.map((d) => d.id).join("|") + "#" + edges.map((e) => e.s + ">" + e.t).join("|"));
    const t0 = Date.now();
    // Snap to the routing grid so a port sits on a cell's centre line.
    const cell = (o.route && o.route.cell) || CELL;
    const placed = place(nodes, edges, Object.assign({ seed, snap: cell }, o.place));
    Object.keys(placed.pos).forEach((id) => {
      const p = placed.pos[id];
      p.x = Math.round((p.x - cell / 2) / cell) * cell + cell / 2;
      p.y = Math.round((p.y - cell / 2) / cell) * cell + cell / 2;
    });
    const t1 = Date.now();
    const routed = route(nodes, edges, placed.pos, o.route);
    const t2 = Date.now();
    return { pos: placed.pos, routes: routed.routes,
      stats: Object.assign({ placeMs: t1 - t0, routeMs: t2 - t1 }, placed.stats, routed.stats) };
  };

  /* ---------------- off the main thread ------------------------------- */
  // The whole graph takes a few seconds, so it runs in a Worker built from
  // this very file, and the result is cached by what went in (memory, then
  // localStorage): a second visit, a reload or a look switched back is
  // instant. A newer request cancels an older one still running (a view
  // switched mid-layout); the older promise rejects with "superseded".
  const IN_WORKER = typeof window === "undefined" && typeof importScripts === "function";
  if (IN_WORKER) {
    root.onmessage = (ev) => {
      const { id, nodes, edges, opt } = ev.data;
      try { root.postMessage({ id, ok: true, out: layout(nodes, edges, opt) }); }
      catch (err) { root.postMessage({ id, ok: false, error: String(err && err.stack || err) }); }
    };
    return;
  }

  const VERSION = 2;
  const r1 = (v) => Math.round((v || 0) * 10) / 10;
  const key = (nodes, edges, opt) => VERSION + "." + hashStr(JSON.stringify([
    nodes.map((d) => [d.id, r1(d.x), r1(d.y), r1(d.w), r1(d.h), r1(d.ox), r1(d.oy), r1(d.cw), r1(d.ch)]),
    edges.map((e) => [e.id, e.s, e.t]), opt || null,
  ])).toString(36);

  const LS = "dd_kg_layout:", LS_INDEX = "dd_kg_layout_index", KEEP = 8;
  const mem = new Map();
  // What's kept: positions and control points only, one decimal.
  const slim = (out) => {
    const pos = {}, routes = {};
    Object.keys(out.pos).forEach((id) => { pos[id] = { x: r1(out.pos[id].x), y: r1(out.pos[id].y) }; });
    Object.keys(out.routes).forEach((id) => { routes[id] = { cps: out.routes[id].cps.map((p) => ({ x: r1(p.x), y: r1(p.y) })) }; });
    return { pos, routes, stats: out.stats };
  };
  const cached = (k) => {
    if (mem.has(k)) return mem.get(k);
    try {
      const raw = localStorage.getItem(LS + k);
      if (raw) { const v = JSON.parse(raw); mem.set(k, v); return v; }
    } catch (_) {}
    return null;
  };
  const store = (k, out) => {
    const v = slim(out);
    mem.set(k, v);
    while (mem.size > KEEP) mem.delete(mem.keys().next().value);
    try {
      let idx = [];
      try { idx = JSON.parse(localStorage.getItem(LS_INDEX) || "[]"); } catch (_) {}
      idx = idx.filter((x) => x !== k);
      idx.push(k);
      while (idx.length > KEEP) { try { localStorage.removeItem(LS + idx.shift()); } catch (_) {} }
      localStorage.setItem(LS + k, JSON.stringify(v));
      localStorage.setItem(LS_INDEX, JSON.stringify(idx));
    } catch (_) {}
    return v;
  };

  const SRC = (typeof document !== "undefined" && document.currentScript && document.currentScript.src) || "";
  let worker = null, seq = 0;
  const pending = new Map();
  const failAll = (why) => { pending.forEach((p) => p.reject(new Error(why))); pending.clear(); };
  const getWorker = () => {
    if (worker !== null) return worker;
    try {
      worker = new Worker(SRC);
      worker.onmessage = (ev) => {
        const p = pending.get(ev.data.id);
        if (!p) return;
        pending.delete(ev.data.id);
        if (ev.data.ok) p.resolve(ev.data.out); else p.reject(new Error(ev.data.error));
      };
      // A worker that can't load (blocked, offline, 404) is not tried again
      // this page; its jobs go to the main thread only if they're small.
      worker.onerror = (ev) => {
        if (ev && ev.preventDefault) ev.preventDefault();
        try { worker.terminate(); } catch (_) {}
        worker = false;
        const jobs = [...pending.values()];
        pending.clear();
        jobs.forEach((p) => inline(p.nodes, p.edges, p.opt).then(p.resolve, p.reject));
      };
    } catch (_) { worker = false; }
    return worker;
  };
  // Without a worker, only a small graph is worth the main thread (well
  // under a tenth of a second); a big one would freeze the page for seconds,
  // so it keeps dagre's layout instead.
  const INLINE_MAX = 40;
  const inline = (nodes, edges, opt) => new Promise((resolve, reject) => {
    if (nodes.length > INLINE_MAX) { reject(new Error("no worker; graph too big for the main thread")); return; }
    setTimeout(() => { try { resolve(layout(nodes, edges, opt)); } catch (err) { reject(err); } }, 0);
  });
  // → Promise of { pos, routes: { edgeId: { cps } }, stats, cached }.
  const run = (nodes, edges, opt) => {
    const k = key(nodes, edges, opt);
    const hit = cached(k);
    if (hit) return Promise.resolve(Object.assign({ cached: true }, hit));
    if (pending.size && worker) { worker.terminate(); worker = null; failAll("superseded"); }
    const w = SRC && typeof Worker === "function" ? getWorker() : false;
    const job = w
      ? new Promise((resolve, reject) => { const id = ++seq; pending.set(id, { resolve, reject, nodes, edges, opt }); w.postMessage({ id, nodes, edges, opt }); })
      : inline(nodes, edges, opt);
    return job.then((out) => store(k, out));
  };

  // The defaults, for kg-tune.js's sliders (copies: nothing may change them).
  const defaults = () => ({ place: Object.assign({}, PLACE), route: Object.assign({}, ROUTE) });
  const api = { layout, place, route, defaults, hashStr, key, cached: (nodes, edges, opt) => cached(key(nodes, edges, opt)), run };
  root.DeltaKgLayout = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
