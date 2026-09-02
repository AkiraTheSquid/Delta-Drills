# fonts

## Purpose
- The web fonts the ARENA notebook page is read in. Seth asked for LessWrong's
  reading surface copied exactly; this folder is the part of that answer we are
  allowed to ship.
- Self-hosted on purpose. Nothing on this page fetches a font at read time.

## Owns
- The ET Book faces (`.woff`) and their upstream licence.
- Being the thing `@font-face` in `../arena-notebook.css` points at.

## Does NOT own
- The `@font-face` rules themselves, the fallback chain, or any type size —
  all of that is `../arena-notebook.css`.
- Every other surface's type. The app's chrome is sans (`--font-sans` in
  `styles/variables.css`) and is untouched by this folder.
- The rail's font: it is sans on LessWrong too, so it inherits app chrome.

## Key Files
- `et-book-roman-line-figures.woff`: the headings face — `h1`/`h2`/`h3` and the
  section title. Weight 400, which is the whole point (ours used to be 700).
- `et-book-bold-line-figures.woff`: weight 700, for `<strong>` inside a heading.
- `et-book-display-italic-old-style-figures.woff`: italic 400, for `<em>`.
- `LICENSE`: ET Book's MIT licence. Keep it next to the files it covers.

## Data & External Dependencies
- Upstream: <https://github.com/edwardtufte/et-book> (MIT). Fetched once,
  vendored; there is no build step and no package manager entry for it.
- Consumed only by `../arena-notebook.css`, via relative `url("fonts/…")`.

## How It Works (Flow)
1. `arena-notebook.css` declares three `@font-face` rules for the family
   `ETBookRoman`, pointing at this folder.
2. `#page-arena-notebook` sets `--lw-header: "ETBookRoman", var(--lw-serif)`.
3. Headings resolve to ET Book; body text falls through `--lw-serif` to
   Palatino / URW Palladio / Georgia.

## Invariants & Constraints
- 🔴 **The family name must stay `ETBookRoman`.** That is LessWrong's own name
  for it (`themes/defaultPalette.ts`, `headerStack`), which is what makes a
  measurement taken off their page comparable to one taken off ours.
- 🔴 **Never add `warnock-pro`, and never link a Typekit stylesheet.** Their
  body face is commercial and is served from *LessWrong's* Adobe kit. It renders
  on their domain and on localhost, which is exactly what makes the mistake easy
  to make and hard to notice: it would work on your machine and 403 in
  production, and it would be spending someone else's licence either way.
- Keep `LICENSE` in this folder. MIT requires the notice to travel with the
  files, and a font folder with no licence beside it reads as unvetted.
- `.woff` only. A second format buys nothing — every browser this app supports
  has read WOFF for a decade — and doubles what has to stay in sync.

## Extension Points
- Another face for this page: drop the `.woff` here, add an `@font-face` in
  `../arena-notebook.css`, extend the check in `watch.py`.
- A different page wanting a serif: point it at `--lw-serif`; do not copy the
  `@font-face` rules into a second stylesheet.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Fetching the italic from the wrong directory** — `RESOLVED`
  - When it happens: vendoring the faces from upstream by hand.
  - Symptom: a 9 KB `.woff` that no browser will load, and italics silently
    fall back to a synthesised oblique Palatino. Nothing errors.
  - Root cause: upstream's folder is
    `et-book-display-italic-old-style-figures`, not `…-old-style`. GitHub Pages
    answers a missing path with a **200** HTML page, so `curl` looks fine and
    only the byte count gives it away.
  - Prevention/fix: `watch.py` asserts each file's WOFF magic (`wOFF`) and a
    plausible size, so a 404 page saved under a `.woff` name fails the watcher.

- **A ratio copied across two rem roots** — `ACTIVE`
  - When it happens: porting any further number off LessWrong.
  - Symptom: type that is subtly too tight or too loose, and a diff that says
    the two sides agree because both say `1.3`.
  - Root cause: their rem root is 13px and ours is 16px, so the same ratio is a
    different pixel value. Their `1.1rem` rail label is 14.3px, not 17.6px.
  - Prevention/fix: measure computed pixels, not declared ratios, and write
    pixels into `arena-notebook.css`. `tools/visual-diff` reads computed styles
    for this reason.

## Recent Changes
- 2026-09-02: Folder created. ET Book roman/bold/italic vendored from upstream
  (MIT) so the ARENA notebook page can use LessWrong's heading face without
  hotlinking their Typekit kit.
