/* ================================================================
   ONE MEMBER, ONE ROW, TWO COLUMNS.

   Seth, 2026-09-02: "each island that you created that's like the
   sticky notes is a row and it goes from left to right and each row
   like each profile has ... two columns: the far left column is what
   you already displayed, the middle column is what Delta Note did
   with the tiptap where it has the checkboxes with the three states."

     +--------------------------+----------------------------------+
     | avatar . name            | CHECKLIST                   2/6  |
     | 10 placement probes      | [x] finish 1.2                   |
     | PyTorch  ###...  54% +-17| [ ] read the einsum notes        |
     | Einops   ####..  67% +-16| [-] the optional exercise        |
     +--------------------------+----------------------------------+

   The left column is unchanged from the first pass — the Learner
   Home's own `.placement-area*` rows, drawn through
   `PlacementResults`. The right column is the day's three-state
   checklist, ported from Delta Note's sub-goals.

   ── 🔴 EXACTLY ONE COLUMN ON THIS PAGE IS AN EDITOR ──────────────
   Yours. Everybody else's is `DDChecklistDoc.renderDoc`, which walks
   the stored JSON and builds the same DOM without loading a single
   ProseMirror plugin. Twelve editors would be twelve node-view
   registries and eleven contenteditables nobody may type into.

   ── 🔴 THE EDITOR MUST BE TORN DOWN, NOT DROPPED ─────────────────
   `destroyAll()` is not tidiness. Its teardown FLUSHES the save
   debounce, so the words typed in the half-second before somebody
   clicks ▶ are written instead of lost — and it is the only thing
   that stops a ProseMirror view from outliving the DOM node it was
   mounted on, which is a leak per day clicked through.
   The page owner (`groups_view.js`) calls it before every repaint.
   ================================================================ */

const DDGroupsLane = (() => {
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  /* The one live editor, held here so the page can tear it down without
     knowing which row it is in. */
  let mine = null;

  const destroyAll = () => {
    if (mine) {
      const controller = mine;
      mine = null;
      controller.destroy();
    }
  };

  /* ---- the left column: the mastery bars ----------------------------- */

  /**
   * The rows under a member's name.
   *
   * `PlacementResults` owns the theta→readiness map; if it has not loaded
   * there is nothing honest to draw, so the card says so rather than
   * inventing a scale.
   *
   * @param {Array<{topic: string, theta: number, sd: number, probes: number}>} areas
   */
  const buildAreas = (areas) => {
    const wrap = el("div", "placement-areas dd-member-areas");
    const results = window.PlacementResults;
    if (!results || !Array.isArray(areas) || !areas.length) {
      wrap.appendChild(el("p", "dd-group-note", "No measurements yet."));
      return wrap;
    }

    wrap.appendChild(el("div", "placement-areas-head", "Weakest first"));

    /* Weakest first, ties broken toward the area with more evidence — the
       same order the Learner Home uses, so a member scanning their own row
       and somebody else's is reading the same list twice rather than two
       differently-sorted lists. */
    const rows = areas
      .slice()
      .sort(
        (a, b) =>
          results.readiness(a.theta) - results.readiness(b.theta) ||
          (b.probes || 0) - (a.probes || 0)
      );

    rows.forEach((area) => {
      const probes = Number(area.probes) || 0;
      const pct = Math.round(results.readiness(area.theta) * 100);
      const row = el("div", `placement-area${probes ? "" : " placement-area--unprobed"}`);
      row.appendChild(el("span", "placement-area-name", String(area.topic || "Other")));

      const bar = el("span", "placement-area-bar");
      const fill = document.createElement("i");
      fill.style.width = `${pct}%`;
      bar.appendChild(fill);
      row.appendChild(bar);

      row.appendChild(el("span", "placement-area-pct", `${pct}%`));
      const b = results.band(area.sd);
      row.appendChild(el("span", "placement-area-conf", b === null ? "" : `±${b}`));
      row.appendChild(
        el("span", "placement-area-probes", probes ? `${probes} probed` : "not probed")
      );
      wrap.appendChild(row);
    });
    return wrap;
  };

  const buildProfile = (member, isYou) => {
    const main = el("div", "dd-member-main");

    const head = el("div", "dd-member-head");
    head.appendChild(window.DDGroupsJoin.buildAvatar(member, "lg"));

    const who = el("div", "dd-member-who");
    const nameRow = el("div", "dd-member-name-row");
    nameRow.appendChild(el("span", "dd-member-name", member.display_name || "Learner"));
    /* Your own row is marked rather than moved: the roster is in join
       order, and a row that jumps to the front for one reader makes two
       people describing "the third one down" mean different things. */
    if (isYou) nameRow.appendChild(el("span", "dd-member-you", "you"));
    who.appendChild(nameRow);

    const probes = Number(member.probes) || 0;
    who.appendChild(
      el(
        "span",
        "dd-member-meta",
        probes
          ? `${probes} placement ${probes === 1 ? "probe" : "probes"}`
          : "hasn't taken the placement test"
      )
    );
    head.appendChild(who);
    main.appendChild(head);
    main.appendChild(buildAreas(member.areas));
    return main;
  };

  /* ---- the right column: the day's checklist -------------------------- */

  const countChip = (counts) =>
    counts && counts.total ? `${counts.checked}/${counts.total}` : "";

  /**
   * @param {{member: object, isYou: boolean, day: string, payload: string|null,
   *          onSave: (payload: string) => void}} deps
   *   `payload` is `null` while the day is still being read — which is a
   *   different thing to say than `""`, and the column says it differently.
   */
  const buildDay = ({ member, isYou, day, dayState, payload, onSave }) => {
    const column = el("div", "dd-member-day");

    const head = el("div", "dd-member-day-head");
    head.appendChild(el("span", "dd-member-day-title", "Checklist"));
    const chip = el("span", "dd-member-day-count", "");
    head.appendChild(chip);
    column.appendChild(head);

    const docs = window.DDChecklistDoc;

    /* 🔴 A DAY THAT WAS NEVER READ AND A DAY THAT FAILED TO READ ARE
       DIFFERENT SENTENCES. Both arrive here with no payload, and saying
       "Reading this day…" for the second one is a spinner that never
       stops — the page would look busy forever while nothing was in
       flight. The picker and the Retry button beside it are how a failed
       read is tried again. */
    if (dayState === "failed") {
      column.appendChild(
        el("p", "dd-group-note", "This day could not be read. Try again above.")
      );
      return column;
    }
    if (dayState !== "ready" || payload === null) {
      column.appendChild(el("p", "dd-group-note", "Reading this day…"));
      return column;
    }

    if (!isYou) {
      /* Somebody else's list: the same markup, no editor, no caret. */
      const view = el("div", "dd-checklist dd-checklist--read");
      if (!docs) {
        view.appendChild(el("p", "dd-group-note", "Checklists are unavailable."));
      } else if (!String(payload || "").trim()) {
        view.appendChild(el("p", "dd-checklist-blank", "Nothing written for this day."));
      } else {
        view.appendChild(docs.renderStored(payload));
        chip.textContent = countChip(docs.countsFromStored(payload));
      }
      column.appendChild(view);
      return column;
    }

    /* Your own: the live three-state editor. */
    const host = el("div", "dd-checklist dd-checklist--mine");
    column.appendChild(host);
    if (!window.DDChecklist || !docs) {
      host.appendChild(el("p", "dd-group-note", "The checklist editor is unavailable."));
      return column;
    }
    /* 🔴 A SAVE THAT FAILED HAS TO SAY SO ON THE PAGE. The editor writes
       through a half-second debounce, so the person who typed has already
       looked away by the time the request answers; a dropped connection or a
       list past the size cap would otherwise look exactly like a save, and
       the words would be gone on the next load with nothing having said a
       word about it. */
    const status = el("p", "dd-checklist-status", "");
    column.appendChild(status);
    const reportSave = (text) => {
      const answer = onSave ? onSave(text) : null;
      Promise.resolve(answer).then(
        (result) => {
          if (result && result.error) {
            status.textContent = `Not saved: ${result.error}`;
            status.classList.add("is-error");
          } else {
            status.textContent = "";
            status.classList.remove("is-error");
          }
        },
        (error) => {
          status.textContent = `Not saved: ${String(error?.message || error || "unknown error")}`;
          status.classList.add("is-error");
        }
      );
    };
    chip.textContent = countChip(docs.countsFromStored(payload));
    /* Belt and braces: the page tears the old editor down before it
       repaints, and this makes a second mount impossible even if a future
       caller forgets. Two live editors would both be saving the same row. */
    destroyAll();
    mine = window.DDChecklist.mount({
      element: host,
      payload: payload || "",
      onSave: reportSave,
      onCounts: (counts) => {
        chip.textContent = countChip(counts);
      },
    });
    /* The day this editor belongs to, so a save that resolves after a day
       change can be recognised as stale by whoever cares. */
    mine.day = day;
    return column;
  };

  /**
   * One member as a full-width row.
   *
   * @param {{member: object, isYou: boolean, day: string,
   *          dayState: "loading"|"ready"|"failed", payload: string|null,
   *          onSave: (payload: string) => Promise<{ok?: true, error?: string}>}} deps
   */
  const buildRow = (deps) => {
    const row = el("article", `dd-member${deps.isYou ? " is-you" : ""}`);
    row.appendChild(buildProfile(deps.member, deps.isYou));
    row.appendChild(buildDay(deps));
    return row;
  };

  return { buildRow, destroyAll };
})();

window.DDGroupsLane = DDGroupsLane;
