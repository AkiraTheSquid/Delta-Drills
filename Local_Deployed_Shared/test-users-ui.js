/* ================================================================
   TEST-USERS-UI.JS — the two surfaces for a test-user demo

   The store, the identity swap and every decision about WHEN a switch is
   safe live in `test-users.js`. This file knows none of that. It renders
   two things off `window.DDTestUsers` and hands clicks straight back to
   it:

     1. The **Test users** section on the Account tab — create, act as,
        rename, reset, remove, and the acting-as state. INJECTED into the
        existing Account card rather than written into index.html, which
        is why nothing in the markup carries a `.dd-tu-*` class and why
        `styles/test-users.css` is the whole contract for how it looks.

     2. A **floating pill**, bottom-left. It is there while acting as
        somebody — that state has to be impossible to miss, or a practice
        session gets recorded against the wrong learner — and, once there
        IS a roster, on your own account too, so switching mid-demo does
        not mean walking back to the Account tab. Somebody who has never
        made a test user never sees it.

   Split out of test-users.js on 2026-08-23 when that file crossed into
   ORANGE. The seam is the obvious one and worth keeping: the store is
   testable with no DOM, and nothing in here may reach into localStorage
   or mint a token — if a rule about identity is needed here, it belongs
   next door with the rest of them.

   LOADED after test-users.js, whose `DDTestUsers` it reads at boot.
   ================================================================ */

(function installTestUsersUi(global) {
  const TU = global.DDTestUsers;
  /* The store is the hard dependency, and a missing one is a load-order
     mistake rather than something to paper over — say so once and mount
     nothing, instead of throwing inside a click handler later. */
  if (!TU) {
    console.warn("[test-users] test-users-ui.js loaded without test-users.js — nothing mounted");
    return;
  }

  const escapeHtml = (s) =>
    String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
    );

  const shortDate = (iso) => {
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "";
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  };

  // ── rendering ───────────────────────────────────────────────────
  let msgEl = null;
  const say = (text, kind) => {
    if (!msgEl) return;
    msgEl.textContent = text || "";
    msgEl.dataset.kind = kind || "";
  };

  const userRow = (u, activeId) => {
    const acting = u.id === activeId;
    const meta = [
      u.lastUsedAt ? `last used ${shortDate(u.lastUsedAt)}` : "never used",
      `made ${shortDate(u.createdAt)}`,
    ].join(" · ");
    return `
      <li class="dd-tu-row${acting ? " is-acting" : ""}" data-id="${escapeHtml(u.id)}">
        <div class="dd-tu-who">
          <span class="dd-tu-name">${escapeHtml(u.name)}</span>
          <span class="dd-tu-meta">${escapeHtml(meta)}</span>
        </div>
        <div class="dd-tu-row-actions">
          ${
            acting
              ? '<span class="dd-tu-badge">You are here</span>'
              : '<button type="button" class="ghost dd-tu-act" data-act="use">Act as</button>'
          }
          <button type="button" class="ghost dd-tu-mini" data-act="rename" title="Change the display name">Rename</button>
          <button type="button" class="ghost dd-tu-mini" data-act="reset" title="Start this test user over on a fresh, empty account"${acting ? " disabled" : ""}>Reset</button>
          <button type="button" class="ghost dd-tu-mini dd-tu-danger" data-act="remove" title="Forget this test user in this browser"${acting ? " disabled" : ""}>Remove</button>
        </div>
      </li>`;
  };

  let panel = null;
  const renderPanel = () => {
    if (!panel) return;
    const s = TU.session();
    const owner = TU.ownerEmail();
    const list = owner ? TU.list() : [];
    const listEl = panel.querySelector("#dd-tu-list");
    const stateEl = panel.querySelector("#dd-tu-state");
    const formEl = panel.querySelector("#dd-tu-new");

    panel.classList.toggle("dd-tu-locked", !owner);

    if (!owner) {
      stateEl.innerHTML =
        '<p class="dd-tu-note">Sign in with your own account first. Test users are stored against it, so signing in is what tells this browser whose demo accounts these are.</p>';
      listEl.innerHTML = "";
      formEl.hidden = true;
      return;
    }
    formEl.hidden = false;

    stateEl.innerHTML = s
      ? `<p class="dd-tu-acting"><span class="dd-tu-acting-text">Acting as <strong>${escapeHtml(
          s.user.name,
        )}</strong>. Your own account (${escapeHtml(
          s.owner.email,
        )}) is waiting.</span><button type="button" class="ghost dd-tu-exit" id="dd-tu-exit">Back to my account</button></p>`
      : `<p class="dd-tu-note">You are on your own account (${escapeHtml(
          owner,
        )}). Pick a test user below to hand the app to somebody else with an empty record — their placement result, mastery, XP and question history stay theirs, and yours is untouched.</p>`;

    listEl.innerHTML = list.length
      ? list.map((u) => userRow(u, s ? s.user.id : null)).join("")
      : '<li class="dd-tu-empty">No test users yet. Add one below — it becomes a real, empty account on the backend, so the placement test and the student model behave exactly as they would for a stranger.</li>';
  };

  const buildPanel = () => {
    const card = document.querySelector("#page-account .card");
    if (!card || document.getElementById("dd-test-users")) return;
    panel = document.createElement("section");
    panel.className = "dd-tu";
    panel.id = "dd-test-users";
    panel.innerHTML = `
      <div class="dd-tu-head">
        <span class="dd-tu-title">Test users</span>
        <span class="dd-tu-hint">
          Named, throwaway learners for showing the app to somebody. Each one is a real
          (empty) account on the backend, so the placement test, the lessons and the mastery
          model are the real ones — not a preview mode. Switching reloads the page; your own
          account is put back the moment you leave. The roster lives in this browser: clear
          site data and the names are gone.
        </span>
      </div>
      <div class="dd-tu-state" id="dd-tu-state"></div>
      <ul class="dd-tu-list" id="dd-tu-list"></ul>
      <form class="dd-tu-new" id="dd-tu-new">
        <input id="dd-tu-name" class="text-input" type="text" maxlength="${TU.MAX_NAME}"
               placeholder="Their name, e.g. Alice at the meetup" autocomplete="off" spellcheck="false" />
        <button class="primary" type="submit">Add test user</button>
      </form>
      <div class="dd-tu-msg hint" id="dd-tu-msg" role="status"></div>`;
    const anchor = card.querySelector(".account-actions");
    if (anchor) card.insertBefore(panel, anchor);
    else card.appendChild(panel);
    msgEl = panel.querySelector("#dd-tu-msg");

    panel.addEventListener("submit", async (e) => {
      if (e.target.id !== "dd-tu-new") return;
      e.preventDefault();
      const input = panel.querySelector("#dd-tu-name");
      const button = panel.querySelector("#dd-tu-new button");
      button.disabled = true;
      say("Creating an account for them…");
      const res = await TU.add(input.value);
      button.disabled = false;
      if (!res.ok) {
        say(res.reason, "bad");
        return;
      }
      input.value = "";
      say(
        `Added ${res.user.name}. Press “Act as” to hand them the app — a test user who has never been used opens on the welcome screen, exactly as a first-time visitor does.`,
        "ok",
      );
      renderPanel();
      renderPill();
    });

    panel.addEventListener("click", async (e) => {
      const exit = e.target.closest("#dd-tu-exit");
      if (exit) {
        TU.stopActing();
        return;
      }
      const button = e.target.closest("button[data-act]");
      if (!button) return;
      const row = button.closest(".dd-tu-row");
      const id = row && row.dataset.id;
      if (!id) return;
      const owner = TU.ownerEmail();
      const user = TU.list().find((u) => u.id === id);
      if (!user) return;

      if (button.dataset.act === "use") {
        button.disabled = true;
        say(`Signing in as ${user.name}…`);
        const res = await TU.actAs(user);
        if (!res.ok) {
          button.disabled = false;
          say(res.reason, "bad");
        }
        return; // success reloads
      }

      if (button.dataset.act === "rename") {
        startRename(row, user);
        return;
      }

      /* Reset and Remove both throw a record away, so each takes two
         clicks. Deliberately NOT window.confirm(): a native modal blocks
         the page (and every automated check of it) until it is dismissed,
         and this one is reachable from a phone mid-demo. */
      if (button.dataset.act === "reset" || button.dataset.act === "remove") {
        if (button.dataset.armed !== "1") {
          armOnce(button, button.dataset.act === "reset" ? "Reset — sure?" : "Remove — sure?");
          return;
        }
        if (button.dataset.act === "remove") {
          const gone = TU.remove(id);
          say(
            gone
              ? `Removed ${user.name} from this browser. The backend account itself stays behind — there is no endpoint to delete one.`
              : `${user.name} was not removed — go back to your own account first.`,
            gone ? "ok" : "bad",
          );
        } else {
          button.disabled = true;
          say(`Starting ${user.name} over…`);
          const res = await TU.reset(id);
          button.disabled = false;
          say(
            res.ok
              ? `${user.name} is on a fresh, empty account. The old one is abandoned, not deleted.`
              : res.reason,
            res.ok ? "ok" : "bad",
          );
        }
        renderPanel();
        renderPill();
      }
    });
  };

  /* A two-click guard that forgets itself, so a Remove armed and walked
     away from is not still armed when the page is next touched. */
  const armOnce = (button, label) => {
    const original = button.textContent;
    button.dataset.armed = "1";
    button.textContent = label;
    button.classList.add("is-armed");
    const disarm = () => {
      button.dataset.armed = "";
      button.textContent = original;
      button.classList.remove("is-armed");
    };
    setTimeout(disarm, 4000);
  };

  const startRename = (row, user) => {
    if (row.querySelector(".dd-tu-rename")) return;
    const who = row.querySelector(".dd-tu-who");
    const form = document.createElement("form");
    form.className = "dd-tu-rename";
    form.innerHTML = `
      <input class="text-input" type="text" maxlength="${TU.MAX_NAME}" value="${escapeHtml(user.name)}" />
      <button class="ghost" type="submit">Save</button>`;
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      TU.rename(user.id, form.querySelector("input").value);
      renderPanel();
      renderPill();
    });
    who.appendChild(form);
    form.querySelector("input").focus();
  };

  // ── the floating pill ───────────────────────────────────────────
  /* Deliberately fixed to a corner rather than added to the topbar: the
     topbar's height is load-bearing arithmetic (--dd-topbar-h in
     styles/layout.css, read by #page-practice and three other rules), and a
     demo affordance has no business changing it.

     It appears while acting as somebody — that state must be impossible to
     miss, or a session gets recorded against the wrong learner — and, once
     there IS a roster, while on your own account too, so switching does not
     require walking back to the Account tab mid-demo. Somebody who has
     never made a test user never sees it. */
  let pill = null;
  const renderPill = () => {
    const s = TU.session();
    const owner = TU.ownerEmail();
    const list = owner ? TU.list() : [];
    const wanted = TU.surfaceAllowed() && !!owner && (!!s || list.length > 0);

    if (!wanted) {
      if (pill) pill.hidden = true;
      return;
    }
    if (!pill) {
      pill = document.createElement("div");
      pill.className = "dd-tu-pill";
      pill.id = "dd-tu-pill";
      document.body.appendChild(pill);
      pill.addEventListener("click", async (e) => {
        const toggle = e.target.closest(".dd-tu-pill-btn");
        if (toggle) {
          pill.classList.toggle("is-open");
          return;
        }
        const back = e.target.closest(".dd-tu-pill-back");
        if (back) {
          TU.stopActing();
          return;
        }
        /* app.js binds [data-goto-tab] with a querySelectorAll at its own
           eval time, so a button injected later is never wired. Route it
           here instead of pretending the attribute works. */
        const manage = e.target.closest(".dd-tu-pill-manage");
        if (manage) {
          pill.classList.remove("is-open");
          if (typeof switchTab === "function") switchTab("account");
          document.getElementById("dd-test-users")?.scrollIntoView({ block: "center" });
          return;
        }
        const pick = e.target.closest(".dd-tu-pill-user");
        if (!pick) return;
        const user = TU.list().find((u) => u.id === pick.dataset.id);
        if (!user) return;
        pick.disabled = true;
        const res = await TU.actAs(user);
        if (!res.ok) {
          pick.disabled = false;
          pick.textContent = res.reason;
        }
      });
      /* Click-away closes it. Bubble phase, and the pill's own listener is
         on the pill, so a menu entry is handled before this ever runs. */
      document.addEventListener("click", (e) => {
        if (pill && !pill.contains(e.target)) pill.classList.remove("is-open");
      });
    }
    pill.hidden = false;
    pill.classList.toggle("is-acting", !!s);
    const others = list.filter((u) => !s || u.id !== s.user.id);
    pill.innerHTML = `
      <button type="button" class="dd-tu-pill-btn">
        <span class="dd-tu-pill-dot" aria-hidden="true"></span>
        <span class="dd-tu-pill-label">${
          s ? `Test user: <strong>${escapeHtml(s.user.name)}</strong>` : "Test users"
        }</span>
      </button>
      <div class="dd-tu-pill-menu">
        ${
          s
            ? `<button type="button" class="dd-tu-pill-back">← Back to my account</button><div class="dd-tu-pill-sep"></div>`
            : ""
        }
        ${
          others.length
            ? others
                .map(
                  (u) =>
                    `<button type="button" class="dd-tu-pill-user" data-id="${escapeHtml(
                      u.id,
                    )}">${escapeHtml(u.name)}</button>`,
                )
                .join("")
            : '<span class="dd-tu-pill-empty">No other test users</span>'
        }
        <div class="dd-tu-pill-sep"></div>
        <button type="button" class="dd-tu-pill-manage">Manage test users</button>
      </div>`;
  };

  // ── boot ────────────────────────────────────────────────────────
  const boot = () => {
    if (!TU.surfaceAllowed()) return;
    buildPanel();
    renderPanel();
    renderPill();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

  /* Signing in or out changes who the owner is, which changes the whole
     panel. guest-session.js and app.js both fire this. */
  global.addEventListener("delta:auth-state-changed", () => {
    renderPanel();
    renderPill();
  });

  /** Re-render both surfaces; exported for the console and for tests. */
  global.DDTestUsersUi = {
    refresh: () => {
      renderPanel();
      renderPill();
    },
  };
})(window);
