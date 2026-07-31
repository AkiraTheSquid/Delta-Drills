/* ================================================================
   AUTH CONFIG — Sign in with Google
   ================================================================

   Set GOOGLE_CLIENT_ID to the OAuth 2.0 *Web application* Client ID from
   Google Cloud Console (APIs & Services → Credentials). It looks like:
       1234567890-abc123def456.apps.googleusercontent.com

   In that OAuth client, add these to "Authorized JavaScript origins":
       https://delta-drills.vercel.app
       https://delta-drills-colab.vercel.app   (the Colab edition — same client)
       http://localhost:8770      (only if testing the static server locally)

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
