/* ================================================================
   AUTH CONFIG — Sign in with Google
   ================================================================

   Set GOOGLE_CLIENT_ID to the OAuth 2.0 *Web application* Client ID from
   Google Cloud Console (APIs & Services → Credentials). It looks like:
       1234567890-abc123def456.apps.googleusercontent.com

   In that OAuth client, add these to "Authorized JavaScript origins":
       https://delta-drills.vercel.app
       https://delta-drills-colab.vercel.app   (the Colab edition — same client)
       http://localhost:5174      (local dev — what delta_drills_local serves)
       http://localhost:5173
       http://localhost              (no port — see below)

   Those five are what is actually registered as of 2026-09-01. The two
   localhost entries were added that day; before then the comment here named
   :8770, which nothing has served for a long time, so the button had never
   worked on localhost at all.

   🔴 The host matters as much as the port: http://127.0.0.1:5174 is a
   DIFFERENT origin to Google and is not registered, even though app.js treats
   it as local and points API_BASE at the local backend. Serve from `localhost`.

   ⏱️ A new origin is NOT live when the console says "Saved". Google's own note
   on that page is "5 minutes to a few hours", and it took 12 minutes on
   2026-09-01 — during which the button 403s exactly as if the origin were
   never added. Worse, the rollout is not atomic: Google's edges disagreed
   with each other for a while, so a single probe answering 403 after another
   answered 200 is normal mid-rollout and is not evidence the save was lost.
   Check with the request Google itself makes, rather than guessing:

       curl -s -o /dev/null -w '%{http_code}\n' \
         -H 'Origin: http://localhost:5174' \
         "https://accounts.google.com/gsi/button?client_id=<id>&iframe_id=probe"

   200 = it will draw the button here; 403 = it will not. It answers 200 for
   the production origin and 403 for an unregistered one, so it is a real
   oracle. delta_drills_local runs this at startup and warns.

   ⚠️ EVERY NEW ORIGIN IS A NEW REGISTRATION. Sign-in is
   `google.accounts.id.initialize`, an ID-token flow keyed on the JavaScript
   origin — so a deploy at a new hostname gets
   "Error 400: origin_mismatch" the first time anyone presses the button, no
   matter that the client id and the backend are unchanged. Vercel's per-deploy
   preview URLs (delta-drills-colab-<hash>-<scope>.vercel.app) are new origins
   too and are deliberately NOT registered: sign in on the stable alias.
   There is no redirect URI to add — this flow has none — and no backend change:
   the server verifies tokens for this same client id.

   The SAME client ID must be set on the backend as the GOOGLE_CLIENT_ID env
   var / Fly secret, so the server verifies tokens minted for this client.

   While this is empty, the Sign in with Google button shows a "not configured
   yet" notice and the email/password form remains available as a fallback.
   ================================================================ */

window.GOOGLE_CLIENT_ID = "974393489971-95s4svmsafosm4uj4ihrp36v91j1vh73.apps.googleusercontent.com";
