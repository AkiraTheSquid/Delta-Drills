/* ================================================================
   AUTH CONFIG — Sign in with Google
   ================================================================

   Set GOOGLE_CLIENT_ID to the OAuth 2.0 *Web application* Client ID from
   Google Cloud Console (APIs & Services → Credentials). It looks like:
       1234567890-abc123def456.apps.googleusercontent.com

   In that OAuth client, add these to "Authorized JavaScript origins":
       https://delta-drills.vercel.app
       http://localhost:8770      (only if testing the static server locally)

   The SAME client ID must be set on the backend as the GOOGLE_CLIENT_ID env
   var / Fly secret, so the server verifies tokens minted for this client.

   While this is empty, the Sign in with Google button shows a "not configured
   yet" notice and the email/password form remains available as a fallback.
   ================================================================ */

window.GOOGLE_CLIENT_ID = "974393489971-95s4svmsafosm4uj4ihrp36v91j1vh73.apps.googleusercontent.com";
