/* ================================================================
   DOMAIN MOVE — delta-drills.vercel.app → deltadrills.com (2026-09-24)
   ================================================================

   The app moved to https://deltadrills.com. A plain Vercel redirect would
   strand everything a browser keeps in localStorage — XP, answer history,
   notebook edits and outputs, timers, theme, sidebar widths — because
   localStorage belongs to one ORIGIN, and the new domain is a new origin.
   Seth chose to carry that data across first, so for a transition period
   this script does the redirect instead of Vercel:

   On delta-drills.vercel.app (top window only):
     • First visit from this browser: copy every localStorage key into the
       URL fragment (#dd-move=…) and replace the page with the same path on
       deltadrills.com. The fragment never reaches a server.
     • Every later visit: the same hop, WITHOUT data. Once a browser has
       moved, the old origin's copy is stale; sending it again could only
       fill gaps with old values, never help.

   On deltadrills.com, when the fragment is present: write each key that
   the new origin does NOT already have (what is already here is newer and
   wins), strip the fragment, and reload so the app boots on the imported
   data. The import is async (DecompressionStream), and the app's own
   scripts read localStorage synchronously the moment they load, which is
   why this stops the page and reloads rather than importing mid-boot.

   🔴 NOT carried: the sign-in token and API keys (SECRET_KEYS). A URL with
   a fragment is written to browser history, and browser history syncs to
   the user's Google/Brave account. The learner signs in once on the new
   domain; their `auth_email` IS carried, so XP and practice state keyed by
   email line up again the moment they do. Nor `api_base` (NEVER_IMPORT):
   app.js sends every API call, sign-in included, to whatever it names.

   🔴 A fragment is attacker-writable: anyone can send a link to
   deltadrills.com/#dd-move=…. So the .com imports only when document.referrer
   is the OLD origin — only this script on that origin navigates here with a
   payload — and the old origin never forwards a `#dd-move` fragment it was
   handed. A failed import (bad payload, quota) bounces once to the old
   origin with #dd-move-retry, which clears its sent flag so the NEXT visit
   sends again; the bounce itself carries nothing, so it cannot loop.

   🔴 Loaded FIRST in <head>, synchronously, before theme.js: on either
   domain it may call window.stop(), and anything that ran before it would
   already have read (or written) the wrong origin's storage.

   Retire it by switching delta-drills.vercel.app to a Vercel redirect
   (project Domains → Redirect to deltadrills.com), then delete this file
   and its <script> tag.
   ================================================================ */
(function () {
  "use strict";

  const OLD_HOST = "delta-drills.vercel.app";
  const NEW_HOST = "deltadrills.com";
  const NEW_ORIGIN = "https://" + NEW_HOST;
  const HASH_PREFIX = "#dd-move=";
  // Set on the OLD origin once its data has been sent.
  const SENT_FLAG = "dd_domain_move_sent";
  // Set on the NEW origin after an import — a record, read by nobody.
  const IMPORTED_FLAG = "dd_domain_move_imported";
  const SECRET_KEYS = new Set([
    "auth_token",
    "account_openai_key",
    "account_mathpix_key",
    "account_mathpix_id",
  ]);
  const NEVER_IMPORT = new Set([...SECRET_KEYS, "api_base", SENT_FLAG, IMPORTED_FLAG]);
  const RETRY_HASH = "#dd-move-retry";
  // URL budget. Chrome allows 2 MB URLs, Firefox ~1 MB; stay under both.
  // Over budget, the largest values are dropped first (named in the
  // payload's `x`) so the small, valuable keys — XP, history — still land.
  const MAX_URL_CHARS = 900000;
  const isMoveHash = (h) => typeof h === "string" && h.indexOf("#dd-move") === 0;

  let isTop = true;
  try { isTop = window.top === window; } catch (_) { isTop = false; }
  if (!isTop) return;

  const host = location.hostname;

  // ── bytes ⇄ base64url ────────────────────────────────────────────
  const toB64url = (bytes) => {
    let bin = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  };
  const fromB64url = (s) => {
    const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/"));
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  };
  const pipe = async (bytes, stream) =>
    new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(stream)).arrayBuffer());

  // "z" + gzip when the browser has CompressionStream, else "p" + plain.
  const encode = async (obj) => {
    const raw = new TextEncoder().encode(JSON.stringify(obj));
    if (typeof CompressionStream === "function") {
      return "z" + toB64url(await pipe(raw, new CompressionStream("gzip")));
    }
    return "p" + toB64url(raw);
  };
  const decode = async (s) => {
    const kind = s.charAt(0);
    let bytes = fromB64url(s.slice(1));
    if (kind === "z") bytes = await pipe(bytes, new DecompressionStream("gzip"));
    else if (kind !== "p") throw new Error("unknown payload kind " + kind);
    return JSON.parse(new TextDecoder().decode(bytes));
  };

  const halt = () => {
    try { window.stop(); } catch (_) {}
    try { document.documentElement.style.visibility = "hidden"; } catch (_) {}
  };

  // ── old domain: send the data (once), then hop ───────────────────
  if (host === OLD_HOST) {
    halt();
    const target = NEW_ORIGIN + location.pathname + location.search;
    // Never forward a #dd-move… fragment: the .com trusts navigations FROM
    // here, so passing one through would launder a forged payload.
    const ownHash = isMoveHash(location.hash) ? "" : location.hash;
    const hop = (fragment) => location.replace(target + (fragment || ownHash));

    if (location.hash === RETRY_HASH) {
      try { localStorage.removeItem(SENT_FLAG); } catch (_) {}
      hop();
      return;
    }

    let alreadySent = false;
    try { alreadySent = !!localStorage.getItem(SENT_FLAG); } catch (_) {}
    if (alreadySent) { hop(); return; }

    (async () => {
      // Null prototype: a key literally named "__proto__" stays a key.
      const data = Object.create(null);
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k === null || NEVER_IMPORT.has(k)) continue;
        data[k] = localStorage.getItem(k);
      }
      const payload = { v: 1, t: Date.now(), h: ownHash, d: data, x: [] };
      const urlLength = (e) => target.length + HASH_PREFIX.length + e.length;
      let enc = await encode(payload);
      const bySize = Object.keys(data).sort((a, b) => data[b].length - data[a].length);
      while (urlLength(enc) > MAX_URL_CHARS && bySize.length) {
        const k = bySize.shift();
        delete data[k];
        payload.x.push(k);
        enc = await encode(payload);
      }
      // Still over with every value gone (a huge path or hash): send
      // nothing, and leave the flag unset so a plainer visit can retry.
      if (urlLength(enc) > MAX_URL_CHARS) { hop(); return; }
      try { localStorage.setItem(SENT_FLAG, String(Date.now())); } catch (_) {}
      hop(HASH_PREFIX + enc);
    })().catch(() => hop()); // no data beats no app: go anyway, retry next visit
    return;
  }

  // ── new domain: import what arrived, then reload clean ───────────
  if (host === NEW_HOST && location.hash.indexOf(HASH_PREFIX) === 0) {
    halt();
    const fragment = location.hash.slice(HASH_PREFIX.length);
    const clean = (h) => location.replace(location.pathname + location.search + (h || ""));

    let fromOld = false;
    try { fromOld = new URL(document.referrer).host === OLD_HOST; } catch (_) {}
    // Not from the old origin = a link someone made. Import nothing.
    if (!fromOld) { clean(); return; }

    // Ask the old origin to send again on its next visit. Carries no data.
    const retry = () =>
      location.replace("https://" + OLD_HOST + location.pathname + location.search + RETRY_HASH);

    (async () => {
      const payload = await decode(fragment);
      const d = payload && payload.d;
      if (!payload || payload.v !== 1 || !d || typeof d !== "object" || Array.isArray(d)) {
        throw new Error("bad payload");
      }
      let imported = 0;
      let kept = 0;
      let failed = 0;
      for (const [k, v] of Object.entries(d)) {
        if (typeof v !== "string" || NEVER_IMPORT.has(k)) continue;
        if (localStorage.getItem(k) !== null) { kept++; continue; }
        try { localStorage.setItem(k, v); imported++; } catch (_) { failed++; }
      }
      try {
        localStorage.setItem(IMPORTED_FLAG, JSON.stringify({
          at: Date.now(), sentAt: payload.t, imported, kept, failed,
          dropped: Array.isArray(payload.x) ? payload.x : [],
        }));
      } catch (_) {}
      if (failed) { retry(); return; }
      clean(typeof payload.h === "string" && !isMoveHash(payload.h) ? payload.h : "");
    })().catch(retry);
  }
})();
