# CLI Reference — Supabase & Vercel

Quick reference for AI agents and developers working on Delta Drills.
Both CLIs are installed and authenticated on this machine.

---

## Vercel CLI

**Version:** 50.17.1
**Logged in as:** sethbgibson-6622
**Team/Scope:** `seth-gibsons-projects` (ID: `team_eGrwrqNQ97xD8TJZdXfhStdf`)
**Project:** `delta-drills`
**Production URL:** https://delta-drills.vercel.app
**Aliases:** https://textbook-to-notebooklm.vercel.app

### Important: Scope flag

Most Vercel commands require `--scope team_eGrwrqNQ97xD8TJZdXfhStdf` because the project belongs to a team, not a personal account. Without it, commands fail with `missing_scope`.

### Common commands

```bash
# List projects
vercel project ls --scope team_eGrwrqNQ97xD8TJZdXfhStdf

# Inspect current production deployment
vercel inspect delta-drills.vercel.app --scope team_eGrwrqNQ97xD8TJZdXfhStdf

# View deployment logs
vercel logs delta-drills.vercel.app --scope team_eGrwrqNQ97xD8TJZdXfhStdf

# List recent deployments
vercel ls delta-drills --scope team_eGrwrqNQ97xD8TJZdXfhStdf

# View environment variables
vercel env ls --scope team_eGrwrqNQ97xD8TJZdXfhStdf

# Pull environment variables locally
vercel env pull --scope team_eGrwrqNQ97xD8TJZdXfhStdf
```

### Deployment workflow

Vercel auto-deploys from the `deploy` branch via GitHub integration. You do NOT need `vercel deploy` manually. The flow is:

1. `deploy_delta_drills` (system command) pushes the `deploy` branch
2. GitHub webhook triggers Vercel
3. Vercel builds and deploys automatically

If you need to trigger a manual deployment:
```bash
# From the deploy worktree
cd /home/stellar-thread/Applications/Delta-Drills-Deployed
vercel --prod --scope team_eGrwrqNQ97xD8TJZdXfhStdf
```

### Linking (interactive only)

`vercel link` requires interactive terminal input and cannot be run by an AI agent in non-interactive mode. If the project becomes unlinked, the user must run:
```bash
cd /home/stellar-thread/Applications/Delta-Drills-Local
vercel link --scope seth-gibsons-projects
# Select "delta-drills" when prompted
```

---

## Supabase CLI

**Version:** 2.75.0
**Linked project ref:** `qaxtcaoydbpigomnfjpl`
**Project name:** textbook-to-notebooklm
**Region:** West US (Oregon)
**Dashboard:** https://supabase.com/dashboard/project/qaxtcaoydbpigomnfjpl
**API URL:** https://qaxtcaoydbpigomnfjpl.supabase.co

### Anon key (public, safe to commit)

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFheHRjYW95ZGJwaWdvbW5manBsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAxNzQ3MjQsImV4cCI6MjA4NTc1MDcyNH0.Mom-rTokqsvEbEshyvvfEjyL77AVa0LqJIg9FbpLvU4
```

### Common commands

```bash
# List all projects
supabase projects list

# Check migration status
supabase migration list

# Push new migrations to remote
supabase db push

# Pull remote schema changes
supabase db pull

# Dump remote schema
supabase db dump --schema public

# Lint local database
supabase db lint
```

### Querying data via REST API

The Supabase CLI does not have a direct SQL execution command for remote databases. Use curl with the REST API instead:

```bash
# Read all rows from a table (using anon key — respects RLS)
curl -s "https://qaxtcaoydbpigomnfjpl.supabase.co/rest/v1/practice_user_state?select=*" \
  -H "apikey: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFheHRjYW95ZGJwaWdvbW5manBsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAxNzQ3MjQsImV4cCI6MjA4NTc1MDcyNH0.Mom-rTokqsvEbEshyvvfEjyL77AVa0LqJIg9FbpLvU4" \
  -H "Authorization: Bearer <USER_JWT_OR_ANON_KEY>"

# Filter by email
curl -s "https://qaxtcaoydbpigomnfjpl.supabase.co/rest/v1/practice_user_state?user_email=eq.someone@example.com&select=*" \
  -H "apikey: <ANON_KEY>" \
  -H "Authorization: Bearer <ANON_KEY>"
```

Note: With the anon key and RLS enabled, you can only read rows matching `auth.email()`. To query all rows (admin), use the service_role key from the Supabase dashboard (never commit it).

### Database tables

| Table | Purpose |
|-------|---------|
| `practice_user_state` | Stores adaptive practice state per user as JSONB. Columns: `user_email` (PK), `state`, `created_at`, `updated_at`. RLS enabled. |

### Migrations

Migrations live in `supabase/migrations/`. To add a new one:

```bash
# Create a new migration file
touch supabase/migrations/00002_description.sql
# Edit it, then push
supabase db push
```

### Re-authentication

If the Supabase CLI token expires:
```bash
supabase login
supabase link --project-ref qaxtcaoydbpigomnfjpl
```

---

## Other Supabase projects on this account

| Project | Ref | Region | Notes |
|---------|-----|--------|-------|
| Delta Medicine | `vfpwnvqircuvwfxdpcxp` | East US | Separate project |
| textbook-to-notebooklm | `qaxtcaoydbpigomnfjpl` | West US | This project (Delta Drills) |

---

## File paths

| What | Path |
|------|------|
| Main worktree | `/home/stellar-thread/Applications/Delta-Drills-Local` (branch: `main`) |
| Deploy worktree | `/home/stellar-thread/Applications/Delta-Drills-Deployed` (branch: `deploy`) |
| Deploy command | `/usr/local/bin/deploy_delta_drills` (symlink to `scripts/deploy_delta_drills.sh`) |
| Supabase config | `supabase/config.toml` |
| Supabase migrations | `supabase/migrations/` |
| Vercel config | `.vercel/` (created after `vercel link`) |
