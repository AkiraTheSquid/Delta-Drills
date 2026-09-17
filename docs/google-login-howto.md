# Sign in with Google — portable how-to

How Google login is wired in Delta Drills, written so it can be copied into
another app. Google is used only to **prove identity once**; the app then runs
on its own session JWT, so the rest of the API never depends on Google.

## The stack

| Layer | What we used |
|---|---|
| **Frontend** | Plain HTML/JS (no framework) + **Google Identity Services (GIS)** — Google's `https://accounts.google.com/gsi/client` library that renders the "Continue with Google" button and returns an **ID token** (a JWT signed by Google). |
| **Backend** | **FastAPI** (Python) on Fly.io, **SQLAlchemy** + **Postgres** (Neon). Verifies Google's ID token with **`google-auth`**, then mints its **own** session JWT with **PyJWT** (HS256). |
| **Auth model** | Google proves identity once → backend exchanges the Google token for the app's own JWT → that JWT is the session from then on. |

## The flow (key mental model)

```
Browser                          Your backend                 Google
  │  click "Continue with Google"
  │ ───────────────────────────────────────────────────────────▶ popup
  │  ◀─────────────────── ID token (JWT signed by Google) ───────
  │  POST /auth/google { credential: <ID token> }
  │ ──────────────────────────▶
  │                          verify token against Google's public certs
  │                          (audience == your client id, issuer, email_verified)
  │                          find-or-create user by email
  │                          mint YOUR OWN app JWT (HS256)
  │  ◀──── { access_token } ──┤
  │  store token in localStorage; send as Bearer on every request
```

The Google token is exchanged immediately for your own token, so existing
email/password JWT auth keeps working unchanged — Google is just another way to
obtain that same session token.

---

## 1. Google Cloud setup (one-time, manual)

1. console.cloud.google.com → create/select a project.
2. **APIs & Services → OAuth consent screen** — set up (External; add test users or publish).
3. **APIs & Services → Credentials → Create credentials → OAuth client ID → Web application.**
   - **Authorized JavaScript origins:** your site origin(s), e.g. `https://yourapp.com`, `http://localhost:PORT` for local testing.
   - No redirect URI needed (GIS uses a popup, not a redirect).
4. Copy the **Client ID** (`NNN-xxxx.apps.googleusercontent.com`). **Not a secret** — it ships in your frontend JS.

The same Client ID goes in **two places**: the frontend (renders the button)
and the backend (verifies tokens were minted for *your* app).

---

## 2. Backend (FastAPI)

**Dependencies** (`requirements.txt`):
```
google-auth==2.38.0   # verify Google ID tokens
requests==2.32.3      # transport google-auth uses to fetch Google's certs
PyJWT==2.10.1         # mint your own session JWT
```

**Config** — read the client id from an env var (Fly secret / `.env`):
```python
# config.py  (pydantic-settings)
class Settings(BaseSettings):
    jwt_secret: str = "change-me"          # signs YOUR tokens
    google_client_id: str = ""             # from env GOOGLE_CLIENT_ID
    ...
settings = Settings()
```

**Verify the Google token** (`auth.py`):
```python
from fastapi import HTTPException, status

def verify_google_id_token(credential: str) -> dict[str, str]:
    if not settings.google_client_id:
        raise HTTPException(503, "Google sign-in is not configured.")
    # lazy import so the app still boots if the lib is absent
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests

    try:
        info = google_id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            settings.google_client_id,     # <-- audience check
            clock_skew_in_seconds=10,
        )
    except ValueError as exc:
        raise HTTPException(401, "Invalid Google token") from exc

    if info.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        raise HTTPException(401, "Invalid Google token issuer")
    email = (info.get("email") or "").lower().strip()
    if not email or not info.get("sub"):
        raise HTTPException(401, "Google token missing email")
    if info.get("email_verified") not in (True, "true"):
        raise HTTPException(401, "Google email not verified")
    return {"sub": info["sub"], "email": email}
```
`verify_oauth2_token` fetches Google's public signing certs (cached), checks the
JWT signature + expiry + that `aud` equals your client id. You add issuer and
`email_verified` checks on top.

**Mint your own token** (reuse whatever your email auth already uses):
```python
def create_access_token(subject: str) -> str:
    payload = {"sub": subject, "exp": datetime.utcnow() + timedelta(minutes=720)}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
```

**The endpoint** (`auth_router.py`):
```python
class GoogleAuthRequest(BaseModel):
    credential: str

@router.post("/google", response_model=Token)
def google_login(payload: GoogleAuthRequest, db: Session = Depends(get_db)) -> Token:
    info = verify_google_id_token(payload.credential)
    email = info["email"]
    user = db.query(User).filter(User.email == email).first()
    if user is None:                       # first sign-in = sign-up
        user = User(email=email, password_hash=hash_password(secrets.token_urlsafe(32)))
        db.add(user)
        try:
            db.commit()
        except IntegrityError:             # race: another tab created it
            db.rollback()
            user = db.query(User).filter(User.email == email).first()
        else:
            db.refresh(user)
    return Token(access_token=create_access_token(str(user.id)))
```
Notes:
- **Find-or-create by email** → one button does both login *and* signup.
- Google users get a random unusable `password_hash` (column is NOT NULL; never used to log in).
- Set the secret: `flyctl secrets set GOOGLE_CLIENT_ID="NNN-xxxx.apps.googleusercontent.com" -a <app>` (or any env mechanism).

---

## 3. Frontend (vanilla JS — adapt trivially to React/etc.)

**Load GIS + your config** (`<head>`):
```html
<script src="auth-config.js"></script>
<script src="https://accounts.google.com/gsi/client" async defer></script>
```
```js
// auth-config.js  — client id is public, safe to commit
window.GOOGLE_CLIENT_ID = "NNN-xxxx.apps.googleusercontent.com";
```

**A mount point** wherever you want the button:
```html
<div id="google-signin-banner"></div>
<span id="google-signin-message"></span>
```

**The JS** (init + callback):
```js
// Display-only decode so we can show "Logged in as <email>". Backend verifies for real.
const decodeJwtPayload = (jwt) => {
  try { return JSON.parse(atob(jwt.split(".")[1].replace(/-/g,"+").replace(/_/g,"/"))); }
  catch { return {}; }
};

// Exchange Google credential for YOUR app token
async function handleGoogleCredential(response) {
  const res = await fetch(`${API_BASE}/auth/google`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ credential: response.credential }),
  });
  const data = await res.json();
  if (!res.ok) { /* show data.detail */ return; }
  const email = decodeJwtPayload(response.credential).email || "";
  setAuthState(data.access_token, email);   // store token, flip UI to logged-in
}

// Render the button once GIS has loaded
let _inited = false;
function initGoogleSignIn() {
  if (_inited) return;
  const el = document.getElementById("google-signin-banner");
  if (!el || !window.GOOGLE_CLIENT_ID) return;
  if (!(window.google && google.accounts && google.accounts.id)) return; // not loaded yet
  google.accounts.id.initialize({
    client_id: window.GOOGLE_CLIENT_ID,
    callback: handleGoogleCredential,
  });
  google.accounts.id.renderButton(el, { theme: "filled_blue", size: "large", text: "continue_with", shape: "pill" });
  _inited = true;
}

// GIS loads async, so poll briefly until it's ready
(function poll(n=0){ initGoogleSignIn(); if (!_inited && n < 40) setTimeout(() => poll(n+1), 250); })();
```

**`setAuthState`** is your existing function: store `access_token` in
`localStorage`, then send it on every API call:
```js
headers.Authorization = `Bearer ${authToken}`;
```

---

## Gotchas that bit us

- **Authorized JavaScript origins must exactly match the serving origin** (scheme + host + port). The button *renders* even with a wrong/missing origin, but the popup fails. `localhost:3000` ≠ `localhost:8770`.
- **GIS loads with `async defer`**, so `google.accounts` isn't defined immediately — the poll loop handles that.
- **Client id in two places, identical.** Frontend mismatch → popup works but backend 401s (audience check). Backend missing → 503.
- **`verify_oauth2_token` needs outbound HTTPS** to fetch Google's certs — backend must have network egress.
- The decoded-on-frontend email is **display only** — never trust it; the backend re-verifies the signature.

---

## Where this lives in Delta Drills (reference)

- Backend: `This-Directory-Only/backend/app/auth.py` (`verify_google_id_token`),
  `auth_router.py` (`POST /auth/google`), `config.py` (`google_client_id`),
  `schemas.py` (`GoogleAuthRequest`), `requirements.txt`.
- Frontend: `Local_Deployed_Shared/auth-config.js`, `index.html` (GIS script +
  `#google-signin-banner` in the guest banner), `app.js` (the Google section).
