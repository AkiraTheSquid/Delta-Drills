/**
 * quackback_widget.js — the bottom-right feedback bubble.
 *
 * Learners open it to post and vote on feature requests and read the roadmap
 * and changelog. All of that lives in Delta Drills' OWN self-hosted Quackback
 * (separate from Delta Note's, so the boards never mix); this file loads its
 * SDK and keeps the widget's identity in step with the app's sign-in.
 *
 * Signed-in learners are identified with a SERVER-SIGNED `ssoToken`: the
 * backend (`app/quackback_sso.py`, `POST /api/quackback/sso`) checks the app
 * token and signs `{ sub: user id, email, exp: 5m }` with the widget secret.
 * Quackback rejects an unverified `identify({ id, email })`, and the secret
 * must never be in this file or anywhere served. Guests stay signed out; the
 * board asks them to sign in before posting.
 *
 * Sign-out (and a guest boot) calls `logout`, so a shared browser never keeps
 * the previous person's Quackback session. Costs one backend request per
 * sign-in / page load, none for guests.
 */
(function () {
  // Self-hosted Quackback for Delta Drills. Empty string = inert (nothing fetched or drawn).
  const QUACKBACK_URL = 'https://drills.163-192-116-252.sslip.io';

  function installSdk(url) {
    if (window.Quackback) return;
    window.Quackback = function () {
      (window.Quackback.q = window.Quackback.q || []).push(arguments);
    };
    const script = document.createElement('script');
    script.async = true;
    script.src = `${url.replace(/\/+$/, '')}/api/widget/sdk.js`;
    document.head.appendChild(script);
  }

  // '' = signed out / guest, null = not yet known. Only a change of person
  // matters: identify once per page load per person.
  let current = null;

  function currentPerson() {
    const id = window.DDIdentity;
    if (!id || !id.isSignedIn()) return '';
    return String(id.email() || '');
  }

  async function identify(person) {
    let result = null;
    try {
      const res = await window.apiFetch('/api/quackback/sso', { method: 'POST' });
      result = res.ok ? await res.json() : { ok: false, error: `HTTP ${res.status}` };
    } catch (error) {
      result = { ok: false, error: String((error && error.message) || error) };
    }
    // They may have signed out (or switched) while the token was in flight.
    if (current !== person) return;
    if (result && result.ok && result.ssoToken) {
      window.Quackback('identify', { ssoToken: result.ssoToken });
    } else {
      // Forget them, so the next sync (auth event or the tab coming back into
      // view) tries again instead of treating this person as already done.
      current = null;
      console.warn('[feedback-widget] feedback sign-in unavailable:', (result && result.error) || 'no token');
    }
  }

  function sync() {
    const next = currentPerson();
    if (next === current) return;
    if (next && typeof window.apiFetch !== 'function') return; // app.js not ready; a later sync retries
    // The first signed-out read logs out too, in case the SDK kept an
    // identity from an earlier session in this browser.
    if (!next || current) window.Quackback('logout');
    current = next;
    if (next) identify(next);
  }

  function boot() {
    if (!QUACKBACK_URL) return;
    try {
      installSdk(QUACKBACK_URL);
      window.Quackback('init', { placement: 'right' });
      sync();
      const safeSync = () => {
        try { sync(); } catch (error) { console.warn('[feedback-widget] auth sync failed', error); }
      };
      window.addEventListener('delta:auth-state-changed', safeSync);
      document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') safeSync();
      });
    } catch (error) {
      // A third-party widget must never take the app down with it.
      console.warn('[feedback-widget] Quackback failed to start', error);
    }
  }

  boot();
})();
