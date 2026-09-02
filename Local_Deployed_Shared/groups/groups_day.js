/* ================================================================
   THE DAY PICKER — which day's checklists the roster is showing.

   Seth, 2026-09-02: "at the top, for like which day it is, it should
   essentially have you choose the specific day for which you are
   displaying the goals for the tiptap for the accountability."

   So one control above the roster: ◀ a date ▶, plus a way back to
   today. Every member row's right-hand column shows THAT day.

   ── 🔴 EVERY DATE HERE IS LOCAL, AND NEVER toISOString() ─────────
   A day key is `YYYY-MM-DD` as the person's own calendar names it.
   `new Date().toISOString()` is UTC: west of Greenwich it names
   YESTERDAY for the whole evening, so a checklist typed at 8pm is
   written under one key and read back under another. It does not
   throw — the list simply comes back empty, on a page whose empty
   state looks exactly like that. Delta Note has the same rule written
   down in `js/shared/week_dates.js`, learned the same way.

   Arithmetic goes through the local Date constructor
   (`new Date(y, m - 1, d + delta)`), which is what makes ◀ and ▶
   correct across a DST boundary: adding 86 400 000 milliseconds to a
   timestamp lands on 23:00 the same day twice a year, and truncating
   that gives the day before.
   ================================================================ */

const DDGroupsDay = (() => {
  const pad = (n) => String(n).padStart(2, "0");

  /** A Date → `YYYY-MM-DD`, in the browser's own timezone. */
  const keyOf = (date) =>
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;

  const todayKey = () => keyOf(new Date());

  /** A key → a local Date at midnight, or null if it is not a day key. */
  const dateOf = (key) => {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(key || ""));
    if (!match) return null;
    const [, y, m, d] = match;
    const date = new Date(Number(y), Number(m) - 1, Number(d));
    /* A key like 2026-02-31 parses and then rolls into March. Rejecting it
       here keeps the picker from silently renaming the day somebody asked
       for. */
    return keyOf(date) === key ? date : null;
  };

  /** `key` moved `delta` days, staying on the local calendar. */
  const shiftKey = (key, delta) => {
    const date = dateOf(key) || new Date();
    return keyOf(new Date(date.getFullYear(), date.getMonth(), date.getDate() + delta));
  };

  /** What to call a day out loud: today and its neighbours by name, the
   *  rest by date. A roster is read mostly about today and yesterday. */
  const labelFor = (key) => {
    const date = dateOf(key);
    if (!date) return "";
    const today = todayKey();
    if (key === today) return "Today";
    if (key === shiftKey(today, -1)) return "Yesterday";
    if (key === shiftKey(today, 1)) return "Tomorrow";
    return date.toLocaleDateString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
    });
  };

  /**
   * The control.
   *
   * @param {{value: string, onChange: (key: string) => void}} deps
   * @returns {HTMLElement}
   */
  const buildPicker = ({ value, onChange }) => {
    const wrap = document.createElement("div");
    wrap.className = "dd-day-bar";

    const caption = document.createElement("span");
    caption.className = "dd-day-caption";
    caption.textContent = "Checklists for";
    wrap.appendChild(caption);

    const nav = document.createElement("div");
    nav.className = "dd-day-nav";

    const step = (delta, glyph, label) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ghost dd-day-step";
      button.textContent = glyph;
      button.setAttribute("aria-label", label);
      button.addEventListener("click", () => onChange(shiftKey(value, delta)));
      return button;
    };

    nav.appendChild(step(-1, "‹", "Previous day"));

    /* A real `<input type="date">`: it is the platform's own calendar, it
       is keyboard-accessible for free, and it already speaks
       `YYYY-MM-DD` — the same string the server is keyed by, so nothing
       between here and the database reformats a date. */
    const field = document.createElement("input");
    field.type = "date";
    field.className = "dd-day-input";
    field.value = value;
    field.setAttribute("aria-label", "Show checklists for this day");
    field.addEventListener("change", () => {
      /* An emptied field (the picker's own clear button) must not become
         "some other day" — put back the day that is on screen. */
      const next = dateOf(field.value) ? field.value : value;
      field.value = next;
      if (next !== value) onChange(next);
    });
    nav.appendChild(field);

    nav.appendChild(step(1, "›", "Next day"));
    wrap.appendChild(nav);

    const named = document.createElement("span");
    named.className = "dd-day-label";
    named.textContent = labelFor(value);
    wrap.appendChild(named);

    /* Only when it would do something. A "Today" button on today is a
       control that answers a click with nothing happening. */
    if (value !== todayKey()) {
      const back = document.createElement("button");
      back.type = "button";
      back.className = "ghost dd-day-today";
      back.textContent = "Today";
      back.addEventListener("click", () => onChange(todayKey()));
      wrap.appendChild(back);
    }

    return wrap;
  };

  return { keyOf, todayKey, dateOf, shiftKey, labelFor, buildPicker };
})();

window.DDGroupsDay = DDGroupsDay;
