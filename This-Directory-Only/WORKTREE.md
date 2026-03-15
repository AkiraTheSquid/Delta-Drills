# Local Worktree

This directory is for local development and testing.

- Branch: main
- Local frontend: http://localhost:5173
- Local backend: http://localhost:8000
- This-directory-only files live under `This-Directory-Only/`
- Shared deployable files live under `Local_Deployed_Shared/`

To update the deploy worktree, run:

```
/usr/local/bin/deploy_delta_drills
```

Canonical implementation:
`/home/stellar-thread/Applications/Delta-Drills-Local/This-Directory-Only/scripts/deploy_delta_drills.sh`

The copy under the deploy worktree is only a forwarding wrapper and should not be edited directly.
