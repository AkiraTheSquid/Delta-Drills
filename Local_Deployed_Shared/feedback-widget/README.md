# feedback-widget

## Purpose
The bottom-right feedback bubble. A learner opens it to post and vote on feature
requests and read the roadmap and changelog, without leaving the app. It is the
same setup as Delta Note's bubble, pointed at Delta Drills' OWN Quackback, so the
two apps' feedback never mixes.

## Owns
- Loading the Drills Quackback widget SDK and calling `init` (bubble on the right).
- Signing a learner into the widget: `POST /api/quackback/sso` through
  `window.apiFetch` → `identify({ ssoToken })`, once per sign-in / page load.
- Calling `logout` on sign-out, on a switch of account, and on a guest boot.

## Does NOT own
- The feedback itself: posts, votes, roadmap and changelog live in the Drills
  Quackback and are managed from its admin (`<QUACKBACK_URL>/admin`).
- Minting the token: `This-Directory-Only/backend/app/quackback_sso.py` does,
  with the widget secret. Guests get no token there.
- Auth. It reads `window.DDIdentity` and listens for `delta:auth-state-changed`,
  both from `app.js`; it never signs anyone in or out.
- The per-drill / per-lesson "Provide feedback" panels (`practice/feedback-panel.js`).
  Those report a broken drill to the backend and can queue a rewrite; this bubble
  is for product feedback.

## Key Files
- `quackback_widget.js`: the whole module. A classic script loaded from
  `index.html` after `app.js`, `guest-session.js` and `account-menu.js`.

## Data & External Dependencies
- Self-hosted Quackback v0.13.2 at `https://drills.163-192-116-252.sslip.io`, on
  the same Oracle VM as Delta Note's (`~/quackback-drills/compose.drills.yml` on
  the VM: its own app, Postgres and Dragonfly; TLS by Delta Note's Caddy). No S3
  (no image uploads) and no SMTP yet (emails only reach
  `docker logs quackback-drills-app`).
- The widget signing secret is Fly `delta-drills-backend`'s
  `QUACKBACK_SIGNING_SECRET`; the source copy is
  `~/secrets/quackback/drills-widget-signing-secret` on Seth's machine.
- `window.apiFetch` (adds the app token), `window.DDIdentity.isSignedIn()` (false
  for guests), `window.DDIdentity.email()`.

## How It Works (Flow)
1. `QUACKBACK_URL` empty → return. Nothing is fetched or drawn.
2. Install the SDK queue stub + async script tag, then `Quackback('init', { placement: 'right' })`.
3. Read who is signed in now. Guest or signed out → `logout`. A learner → fetch
   the ssoToken and `identify`, unless the person changed while it was in flight.
4. Re-run step 3 on every `delta:auth-state-changed`; only a change of person acts.

## Invariants & Constraints
- **Inert while `QUACKBACK_URL` is empty.**
- **No imports, IIFE-scoped, and `boot()` in try/catch**: a third-party widget must
  never break app boot or leak globals besides `window.Quackback`.
- **The Widget Secret never appears in this folder or anywhere served.** Everything
  in `Local_Deployed_Shared/` ships to every browser.
- **Never `identify({ id, email })`.** Quackback rejects unverified identity; only
  the server-signed `{ ssoToken }` may identify. `watch.py` enforces this.
- `sub` = the Drills user id (uuid), not the email.
- **Accepted risk (codex critic, 2026-09-28): the SDK runs with the page's full
  authority.** `sdk.js` executes in the app document, so whoever controls the
  Quackback host could call `window.apiFetch` as the learner. That is how
  Quackback's widget is built (Delta Note's bubble is the same); it is bounded by
  the host being our own VM with the image pinned (`QUACKBACK_TAG=0.13.2`), not a
  third-party CDN. Keep that VM patched and never point `QUACKBACK_URL` at a host
  we do not run.
- A failed token fetch forgets the person, so the next auth event or the tab
  becoming visible again retries; it never sticks signed-out until a reload.

## Extension Points
- A menu item that opens the widget: `Quackback('open')`.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Regenerating the widget secret signs everyone out of the bubble** — `ACTIVE`
  - When it happens: someone clicks Regenerate in the Drills Quackback admin → Widget.
  - Symptom: learners open the bubble signed out; the console says
    `feedback sign-in unavailable` only if the token fetch fails, otherwise nothing.
  - Root cause: Fly still signs with the old secret.
  - Prevention/fix: after a regenerate, set the new secret on Fly
    `delta-drills-backend` (`QUACKBACK_SIGNING_SECRET`) in the same sitting.

## Recent Changes
- 2026-09-28: Created. Same bubble as Delta Note's, pointed at Delta Drills' own
  self-hosted Quackback; identity signed by the backend's `/api/quackback/sso`.
