# Delta Drills

An interactive learning platform that adapts to each student's frontier, combining diagnostics, scaffolded lessons, and practice problems to build mastery.

## What This App Does

**Positioning:** Acts as Khan Academy / LeetCode for the ARENA curriculum, drilling its prerequisites. First MVP focuses on PyTorch fundamentals — a prerequisite block for students entering ARENA.

**Value add:**
- Diagnostic test reveals gaps in prerequisite knowledge
- Personalized learning paths: lessons → worked examples → problems
- Real-time adaptation: as students solve problems, the system builds a live model of their learning frontier and scaffolds content to the edge of their ability
- ARENA readiness: when prerequisites clear, students unlock ARENA problems

## What the MVP Delivers

### For Students
1. **Diagnostic** — pinpoint gaps in the prerequisite domain (e.g., PyTorch basics, linear algebra, calculus)
2. **Adaptive lessons** — only see topics needed for your gaps
3. **Worked examples** — patterns before practice
4. **Scaffolded problems** — difficulty and hints match your frontier
5. **Clear unlock path** — see when you're ready for ARENA

### For Authors
1. **Drag-and-drop lesson builder** — visual layout, no markup required
2. **Colab integration** — code runs live in notebooks
3. **Mastery tracking** — see student progress and inference quality
4. **Concept graph** — model prerequisite dependencies, optional export to ARENA

## Recent Changes

- **2026-08-22**: Added top-level README describing app vision and MVP scope.

## Key Folders

- `Local_Deployed_Shared/` — web frontend, lesson content, served assets
- `arena-procedural-drills/` — PyTorch problem bank and grader
- `docs/` — specs, handoffs, case studies
- `scripts/` — deploy and admin tools
- `ops/` — monitoring, background jobs

## Runtimes

- **Web:** Next.js + React ([serve locally](#local-development))
- **Backend:** Python + FastAPI (ITS engine)
- **Colab:** Jupyter notebooks compiled to web

---

## Getting Started

See `docs/` for architecture and setup guides. Local development runs on `:3000`:

```bash
cd Local_Deployed_Shared && npm run dev
```

## Deployment

Deploy to production (Fly.io + Neon):

```bash
bash scripts/deploy_delta_drills
```

---

For feature roadmap and open work, see `/home/stellar-thread/.claude/projects/-home-stellar-thread/memory/delta-drills.md`.
