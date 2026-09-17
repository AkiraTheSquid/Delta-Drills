# Running Delta Drills on this machine

```bash
delta_drills_local        # frontend :5174, backend :8000, database :54322
delta_drills_dev          # the same, plus a debug-controlled Chrome window
```

`delta_drills_local` is a symlink to `This-Directory-Only/scripts/delta_drills_local.sh`.
It is in the repo now; it used to be a loose copy in `~/.local/bin` that nothing
tracked. The pre-2026-09-01 copy is kept at `~/.local/bin/delta_drills_local.pre-159acd.bak`.

Open **http://localhost:5174**. Not `0.0.0.0`, not the LAN address: `app.js` only
points `API_BASE` at the local backend when the page's own hostname is
`localhost`/`127.0.0.1`/`0.0.0.0`, and from anywhere else the local page talks to
the Fly backend instead — which looks like it works, and is not what you are
trying to test.

You do **not** need to sign in. `guest-session.js` mints a throwaway account
against the local backend on first load, and that is enough to put the session
in backend mode.

## Why local is worth using: the code actually runs here

In backend mode `practice/runner.js` posts Run and Submit to
`/api/practice/run-code`, which executes in a fork of the local uvicorn process:
the `.venv`'s torch, this box's 16 CPUs. Pyodide is not involved. Signed out or
signed in, torch drills run — and they run at native speed, with no deploy in
between.

Without a backend the same app falls back to Pyodide, **which cannot import
torch at all**. Since the July dialect conversion that is most of the bank, so
"no backend" reads on screen as "torch is not available in the browser sandbox".

## The database, and the failure that wasted a lot of time

`backend/.env` points `DATABASE_URL` at `localhost:54322/pdf_split_tool`. For a
long stretch nothing was listening there, and the app did not say so:

- `app/lifecycle.py` catches a failed schema bootstrap on purpose ("never block
  startup on this") and logs a WARNING, so `uvicorn` starts normally.
- `/health` answers **200** with the database completely gone — it only reads
  JSON off disk.
- Every endpoint that touches Postgres answers **500**.

From the browser that chain is invisible and mis-shaped. `/auth/signup` 500s →
no token → `getPracticeMode()` answers `"local"` → local mode has nowhere to run
torch → `TORCH_UNAVAILABLE`. The learner sees a sandbox message and concludes
the sandbox is the problem. **The sandbox is the symptom. The database is the
cause.**

Two things exist so that cannot happen again:

`scripts/dd_local_db.sh` puts a real Postgres where `.env` already says one is.

```bash
dd_local_db.sh up       # start + wait until it accepts connections (the runner calls this)
dd_local_db.sh status   # is it there, and does .env still agree with it?
dd_local_db.sh down     # stop, keep the data
dd_local_db.sh reset    # DESTROY the data and start clean (asks first)
dd_local_db.sh psql     # a shell on it
```

It reads every connection parameter out of `backend/.env` rather than repeating
them, and it refuses to act at all if `DATABASE_URL` names a host that is not
this machine. If the container is running on a port `.env` no longer uses, `up`
recreates it instead of reporting a green light over a backend talking to
nothing.

`delta_drills_local.sh` **refuses to print a URL until it has proved auth
works.** It posts a login for an account that does not exist and requires a
**401** — meaning the request reached Postgres, ran a query and found nobody. A
500 there is the database, and the script stops and says so. `/health` is not
used as a readiness check and should not be.

It then signs up a throwaway account and runs a tensor through
`/api/practice/run-code`, so the "torch 2.12.0+cpu on 16 cpus" line in the
banner is a measurement, not a claim.

The database container (`delta-drills-local-db`, volume `delta_drills_local_pg`)
is left running after Ctrl+C on purpose: it is cheap, it holds your local
progress, and the next start is instant.

### It is scratch data

Local accounts, attempts and mastery live in that volume plus
`backend/user_data/`. None of it is a copy of production, and `reset` destroys
all of it. Production is Neon + the Fly volume and is not reachable from here.

## Sign in with Google on localhost

There were **three** separate breaks stacked on top of each other here, and
each one hid the next. Two are fixed in this repo; the third is a Google-side
setting that no file here can assert on.

**1. The backend did not know the client id (fixed).** `backend/.env` had no
`GOOGLE_CLIENT_ID`, and `.env.example` never listed one — so nobody setting up
this repo would have added it. `settings.google_client_id` defaulted to `""`
and `/auth/google` answered **503 "Google sign-in is not configured on the
server"** for every sign-in attempt. The id is public (it ships to every
browser in `auth-config.js`), so it is now in both files. Probe it:

```bash
curl -s -X POST localhost:8000/auth/google -H 'Content-Type: application/json' \
  -d '{"credential":"junk"}' -w ' %{http_code}\n'
# 401 "Invalid Google token" = configured.  503 = not.
```

🔴 401 is the *healthy* answer there, the same way it is for `/auth/login`.

**2. The origins comment named a dead port (fixed).** `auth-config.js` told
readers to register `http://localhost:8770`, which nothing has served in a long
time — so the button had never worked on localhost, and the comment made that
look intentional. It now names what is actually registered.

**3. Google must have the origin registered (done — verified working).** The OAuth client
(`974393489971-…`, project `delta-drills`) had only the two Vercel origins
registered, so GSI answered every localhost load with `The given origin is not
allowed for the given client ID` and the button never rendered.
`http://localhost:5174` and `http://localhost:5173` were added as Authorized
JavaScript origins on 2026-09-01; the two Vercel origins were not touched
(removing one would break sign-in on production).

Ask Google directly rather than guessing — this is the only check that tells
you the truth, and `delta_drills_local.sh` now runs it at startup and warns:

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  -H 'Origin: http://localhost:5174' -H 'Referer: http://localhost:5174/' \
  "https://accounts.google.com/gsi/button?client_id=$CLIENT_ID&iframe_id=probe"
# 200 = Google will draw the button here. 403 = it will not.
```

That probe is a real oracle: it answers 200 for `https://delta-drills.vercel.app`
and 403 for `https://example.com`, so a 403 for localhost means localhost, not
a broken probe. (An earlier attempt using `/gsi/status` answered 403 for
*every* origin including the working production one — it proves nothing, do not
use it.)

🔴 There is **no API for this** — Google's OAuth client origins are console-only
(the IAP OAuth Admin API explicitly "cannot be used as a generic management API
for all OAuth clients", and is shut down as of March 2026). So this setting
lives in the Google Cloud console and **nowhere in this repo**, which means it
can drift without any test here failing. If the button stops rendering on
localhost, check the origins list first:
console.cloud.google.com → Google Auth Platform → Clients → `delta-drills`.

A bare `http://localhost` (no port) is registered as a fifth origin, so a
one-off local server on some other port still gets a button.

⏱️ **A saved origin is not a live origin.** Google's note on that page is *"5
minutes to a few hours"*; on 2026-09-01 it took **12 minutes**, and during that
window the button 403s exactly as if the origin had never been added. The
rollout is also **not atomic** — Google's edges disagreed with each other, so a
403 arriving *after* a 200 is normal mid-rollout and is not evidence the save
was lost. Re-probe before changing anything; the config was correct the whole
time it was failing.

🔴 **Only these two ports are registered.** `delta_drills_local.sh` accepts
`DELTA_DRILLS_FRONTEND_PORT`, and any other value puts you on an unregistered
origin where the Google button silently will not render again. Guest mode still
works there; sign-in does not. Use 5174 unless you have a reason not to, and
serve from `localhost` — `http://127.0.0.1:5174` is a *different origin* to
Google and is not registered, even though `app.js` treats it as local.

**The boot-time 403s are gone.** `/api/practice/kc-lattice` and
`/api/practice/diagnostic/status` were fired by deferred scripts
(`lesson-graph.js`, `why-graph.js`) before `practice/init.js` had awaited
`DDGuest.ensure()`, so they went out with **no `Authorization` header at all**
and FastAPI's bearer dependency answered 403 — not 401, which is what a *bad*
token gets.

That was never just console noise. `kc_lattice_read.js` memoizes the first
attempt in `_latticeReq`, so one 403 at boot pinned the knowledge graph to its
offline client-side fallback for the rest of the page load — the map and the
practice queue then disagreed about the same learner, which is the exact bug the
comment at the top of that file exists to prevent. `_fetch` there now waits on
`_sessionReady()` (the memoized `DDGuest.ensure()` promise `init.js` already
awaits, so no extra round trip) before issuing either request. Verified from the
network log: both now carry a bearer token and return 200.

## Guest mode on the deployed app

It works. Measured 2026-09-01 in a browser with cleared storage against
`delta-drills.vercel.app`: account minted, `practiceMode` `"backend"`, `import
torch` returning `2.13.0+cpu` from the fork runner, `Start new session` serving
a real graded drill. If you think it is broken, reproduce it there with site
data cleared before changing the banner — a localhost session proves nothing
about it, for all the reasons above.
