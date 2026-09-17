# arena-book

## Purpose
- Source for the Jupyter Book that renders ARENA curriculum notebooks as a static, navigable website on Delta Drills' domain.
- Acts as the front door for the Courses tab — students click an ARENA section in `courses.js`, land on a Book page here, then click "Open in Colab" to run code.
- Also serves as the notebook target for ARENA course navigation. Practice-page source-link UI was removed; the practice page now focuses on imports shown above the editor.

## Owns
- Book configuration (`_config.yml`, `_toc.yml`, `intro.md`).
- Symlinks into `Local_Deployed_Shared/content/ARENA_4.0-main/` so notebooks render at GitHub-aligned paths (required for Colab launch URLs to resolve correctly against `callummcdougall/ARENA_3.0`).
- The build artifact (`_build/html/`) consumed by whatever Vercel project serves this Book.

## Does NOT own
- ARENA notebook content. Source of truth lives upstream at `callummcdougall/ARENA_3.0`. Local copy at `Local_Deployed_Shared/content/ARENA_4.0-main/`. Do not edit `.ipynb` files from inside this folder.
- Code execution. Phase 1 ships with click-out-to-Colab only. The deferred decision on inline execution lives in `~/.claude/projects/-home-stellar-thread/memory/delta-drills-jupyter-book-execution-decision.md`.
- Mastery tracking, grading, callbacks. Those live in Fly backend + Supabase, wired from the helper module a student imports inside Colab.

## Key Files
- `_config.yml`: Book metadata, theme, repository link, Colab launch button config.
- `_toc.yml`: Table of contents — defines navigation hierarchy + maps to symlinked notebook paths.
- `intro.md`: Landing page shown at Book root.
- `requirements.txt`: jupyter-book + plugins needed by the build environment.
- `chapter0_fundamentals/` … `chapter3_llm_evals/`: symlinks into the ARENA tree under `Local_Deployed_Shared/content/`.

## Data & External Dependencies
- Source notebooks: `Local_Deployed_Shared/content/ARENA_4.0-main/**/*.ipynb` (read-only from this folder's perspective).
- Build tool: `jupyter-book` (Python package).
- Upstream GitHub repo for Colab URLs: `callummcdougall/ARENA_3.0` (`main` branch).
- Consumer: `Local_Deployed_Shared/courses.js` will gain `url` fields on each section pointing into the deployed Book.

## How It Works (Flow)
1. `jupyter-book build .` reads `_config.yml` + `_toc.yml`, traverses the listed `.ipynb` files via symlinks, renders to HTML in `_build/html/`.
2. `_build/html/` is deployed as a static site (separate Vercel project from Delta Drills frontend).
3. Each rendered page carries an "Open in Colab" button. The button URL = `https://colab.research.google.com/github/callummcdougall/ARENA_3.0/blob/main/<path-from-book-root>.ipynb`. This only works because local symlink paths mirror the upstream GitHub layout exactly.
4. From Delta Drills, `courses.js` section cards link to `/arena/<chapter>/<section>` on the Book.

## Invariants & Constraints
- **Symlink paths must mirror ARENA's GitHub layout.** If you rename `chapter0_fundamentals` to anything else here, every Colab launch button breaks. Don't.
- **Notebooks are not executed at build time** (`execute_notebooks: "off"`). They require GPU/PyTorch we don't have in the build env. Cell outputs come from whatever was checkpointed into the `.ipynb` upstream.
- **Do not import or run any of the source notebooks from this folder.** This folder is purely a static-site build configuration.
- **Phase 1 is Colab-only.** Do NOT silently turn on Thebe / Binder / inline execution. That decision is deferred — see the memory file referenced under "Does NOT own."

## Extension Points
- Add an ARENA section to the Book: edit `_toc.yml`, add a `file:` entry pointing at the relative path of the new `.ipynb`. Run `jupyter-book build .` to verify.
- Change theme / Book chrome: `_config.yml` `html:` block + `sphinx.config:`.
- Add a per-page "Back to Delta Drills" link or progress indicator: edit `_config.yml` `html.extra_navbar` or write a small Sphinx extension.
- Surface a section URL in the Delta Drills app: append a `url` field to the corresponding section in `Local_Deployed_Shared/courses.js`.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Chapter 1 has incomplete `_exercises.ipynb` coverage** — `ACTIVE`
  - When it happens: building the Book or referencing chapter 1 sections.
  - Symptom: only `1.2`, `1.3.2`, and `1.5.3` exist as exercise notebooks under `chapter1_transformer_interp/exercises/`. The `courses.js` UI lists more sections (1.1, 1.3.1, 1.3.3, 1.3.4, 1.4.1, 1.4.2, 1.5.1, 1.5.2, 1.5.4) that have no corresponding notebook locally.
  - Root cause: ARENA's distribution model generates `*_exercises.ipynb` from `infrastructure/master_files/master_*.py`. Most chapter 1 generated outputs are not committed to the local snapshot.
  - Prevention/fix: when wiring `courses.js` URLs, only point at sections that resolve to a real notebook in `_toc.yml`. For missing ones, either (a) generate from master files via ARENA's conversion script, or (b) leave the section as a description-only card without a `url` field. Status `ACTIVE` until the user decides which.
  - Status: `ACTIVE`.

- **Symlink path drift** — `ACTIVE`
  - When it happens: someone reorganizes `Local_Deployed_Shared/content/` or renames `ARENA_4.0-main`.
  - Symptom: build fails with missing-file errors, or Colab buttons 404.
  - Root cause: this folder uses relative symlinks into the ARENA tree. Renaming the target breaks everything.
  - Prevention/fix: if the ARENA path moves, update symlinks here in lockstep. Run `ls -la` from this folder to verify symlinks resolve before each build.
  - Status: `ACTIVE`.

## Recent Changes
- 2026-05-13: Practice-page source notebook block removed; practice now surfaces imported helpers instead.
- 2026-04-29: Initial scaffold — `_config.yml`, `_toc.yml`, `intro.md`, `requirements.txt`, symlinks for the four ARENA chapters. Phase 1 design: Book on Vercel + Colab launch buttons. Inline-execution backend deferred (see project memory).
