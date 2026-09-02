/* ================================================================
   GROUPS — GETTING INTO ONE

   Two surfaces, one file, because they are two states of the same
   question:

     • the DISCOVERY card, drawn instead of the roster while you are in
       no group — start one, pick one out of the directory, or paste the
       link somebody sent you;
     • the GROUP BAR, a strip above the roster once you are — who is in
       it, the link to hand the next person, and the way out.

   Ported from Delta Note's `accountability_discovery.js` + its avatar and
   directory modules, minus the ES-module boundaries this app's classic
   scripts do not have.

   ── 🔴 The consent is asked BEFORE the first read, not after ──────
   Joining a group publishes your area mastery to everyone in it. That is
   the whole feature, and it is also the one thing here a member cannot
   undo by clicking again — a peer may have read it already. So the
   dialog says exactly what will be shared, and the join does not happen
   until it is answered.

   THREE WAYS IN, ONE GATE. Create, invite link and public directory all
   run through `ask()`. It is one gate with three doors in front of it
   rather than three gates, because the third one added later is the one
   that forgets to ask.

   ── 🔴 No `confirm()`, anywhere ──────────────────────────────────
   The listing toggle arms itself and says what will happen, then does it
   on a second click. A native confirm blocks the page and this repo's
   browser checks cannot dismiss one, so it would make the surface
   untestable as well as ruder.
   ================================================================ */

const DDGroupsJoin = (() => {
  const store = () => window.DDGroupStore;

  /* ---- avatars ------------------------------------------------------
     A group drawn as the people in it. A count says "4 members"; a stack
     says "these four", which is the difference between a number and a
     room you might want to be in.

     🔴 The letters come from the SERVER (`initials`), never from slicing
     a name found elsewhere on the object. The public directory answers
     initials and no names at all, and a `displayName ?? initials`
     fallback is how that tighter boundary quietly stops being the one in
     effect — it would work perfectly on the roster, which carries names,
     and silently do nothing on the directory, which does not. */

  /* FNV-1a over the member id, which is per MEMBERSHIP: the same person
     is a different colour in two groups. The id is the only stable thing
     about a member this surface is allowed to know, and a colour derived
     from the name would change every time somebody edited theirs. */
  const avatarColor = (seed) => {
    let hash = 0x811c9dc5;
    const text = String(seed || "");
    for (let i = 0; i < text.length; i += 1) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return `hsl(${hash % 360} 58% 42%)`;
  };

  const buildAvatar = (member, size) => {
    const dot = document.createElement("span");
    dot.className = `dd-group-avatar${size ? ` is-${size}` : ""}`;
    const letters = String(member?.initials || "").trim() || "?";
    dot.textContent = letters;
    dot.style.background = avatarColor(member?.member_id || letters);
    /* The name when there is one, the letters when there is not. A
       directory avatar's label is its own letters — as much as the circle
       tells everyone else, and no more, which is the point. */
    const label = String(member?.display_name || "").trim() || letters;
    dot.title = label;
    dot.setAttribute("aria-label", label);
    return dot;
  };

  const buildAvatarStack = (members, max) => {
    const list = Array.isArray(members) ? members : [];
    const cap = max || 5;
    const stack = document.createElement("span");
    stack.className = "dd-group-avatars";
    /* One label for the row rather than a reading of every circle in it:
       the circles repeat a count that is already on screen. */
    stack.setAttribute("role", "img");
    stack.setAttribute("aria-label", list.length === 1 ? "1 member" : `${list.length} members`);
    list.slice(0, cap).forEach((m) => stack.appendChild(buildAvatar(m)));
    if (list.length > cap) {
      const more = document.createElement("span");
      more.className = "dd-group-avatar is-more";
      more.textContent = `+${list.length - cap}`;
      stack.appendChild(more);
    }
    return stack;
  };

  /* ---- the public directory ----------------------------------------- */

  const note = (host, message) => {
    host.replaceChildren();
    const p = document.createElement("p");
    p.className = "dd-group-note";
    p.textContent = message;
    host.appendChild(p);
  };

  /* 🔴 Which render is the current one, per host. Mounting starts a read
     and Refresh starts another; without this the slower of the two wins
     by finishing last, so a stale list — or an old failure over a fresh
     success — can replace what is already correct on screen. */
  const renders = new WeakMap();

  const buildDirectoryRow = (group, onPick) => {
    const row = document.createElement("div");
    row.className = "dd-public-group";
    row.appendChild(buildAvatarStack(group.members, 4));

    const label = document.createElement("span");
    label.className = "dd-public-name";
    /* textContent, not innerHTML: this string was typed by whoever started
       the group and is being drawn for strangers. */
    label.textContent = group.name || "Study group";
    row.appendChild(label);

    const count = document.createElement("span");
    count.className = "dd-public-count";
    count.textContent = group.member_count === 1 ? "1 member" : `${group.member_count} members`;
    row.appendChild(count);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost dd-public-join";
    if (group.is_member) {
      /* Shown rather than filtered out: a list that silently omits the
         group you are in reads as the group having vanished. */
      button.textContent = "You're in";
      button.disabled = true;
    } else if (group.member_count >= 12) {
      /* The cap the server enforces. Saying so here turns a refusal you
         would have met after clicking into one you can see before you do. */
      button.textContent = "Full";
      button.disabled = true;
    } else {
      button.textContent = "Join";
      button.addEventListener("click", () => onPick(group));
    }
    row.appendChild(button);
    return row;
  };

  const renderDirectory = async (host, onPick) => {
    const generation = (renders.get(host) || 0) + 1;
    renders.set(host, generation);
    note(host, "Looking for open groups…");
    const groups = await store().readPublicGroups();
    if (!host.isConnected || renders.get(host) !== generation) return;
    if (groups === null) {
      /* 🔴 A failed read is not an empty directory. */
      note(host, "Could not load the open groups just now. The invite-link box below still works.");
      return;
    }
    if (!groups.length) {
      note(host, 'No open groups yet. Start one and tick "anyone can find it" to be the first.');
      return;
    }
    host.replaceChildren();
    groups.forEach((g) => host.appendChild(buildDirectoryRow(g, onPick)));
  };

  /* ---- the discovery card ------------------------------------------- */

  const showError = (root, message) => {
    const slot = root.querySelector(".dd-group-error");
    if (!slot) return;
    slot.textContent = message || "";
    slot.hidden = !message;
  };

  /**
   * The card drawn instead of the roster while you are in no group.
   *
   * @param {HTMLElement} root
   * @param {{onJoined: () => void}} deps
   */
  const renderDiscovery = (root, deps) => {
    const invited = store().inviteFromLocation();
    const signedIn = store().hasAccount();
    root.innerHTML = `
      <section class="dd-group-discovery">
        <article class="dd-group-card">
          <h2>Practice beside somebody else</h2>
          <p class="dd-group-lede">A study group shows each member's readiness in every area of the curriculum, side by side — the same bars the Learner Home draws for you, for everyone in the group.</p>
          <p class="dd-group-error" role="alert" hidden></p>
          ${signedIn ? `
          ${invited ? `
          <div class="dd-group-invited">
            <p>You have been invited to a group.</p>
            <button class="primary dd-group-accept" type="button">Join this group</button>
          </div>` : ""}
          <div class="dd-group-block">
            <div class="dd-group-block-head">
              <label>Groups you can join</label>
              <button class="ghost dd-group-refresh" type="button">Refresh</button>
            </div>
            <div class="dd-public-list"></div>
          </div>
          <div class="dd-group-block">
            <label for="dd-group-name-input">Start a group</label>
            <div class="dd-group-row">
              <input id="dd-group-name-input" class="dd-group-input" type="text" maxlength="120" placeholder="Torch crew" />
              <button class="primary dd-group-create" type="button">Create</button>
            </div>
            <label class="dd-group-tick">
              <input type="checkbox" class="dd-group-public-choice" />
              Let anyone signed in find and join it
            </label>
          </div>
          <div class="dd-group-block">
            <label for="dd-group-join-input">Or paste an invite link</label>
            <div class="dd-group-row">
              <input id="dd-group-join-input" class="dd-group-input" type="text" placeholder="https://…?group=…" />
              <button class="ghost dd-group-join-go" type="button">Join</button>
            </div>
          </div>` : `<p class="dd-group-note">Groups belong to an account. Sign in to start or join one — a guest's progress lives only in this browser, so there is nothing for a group to read.</p>`}
        </article>
      </section>
      <dialog class="dd-group-consent">
        <h3>What the group will see</h3>
        <p>Every member of this group will be able to read your readiness in each area of the curriculum, how many placement probes measured it, the name you go by here, and the daily checklist you write on this page. They cannot see your email address or your attempts.</p>
        <p class="dd-group-consent-public" hidden>This group is listed, so anyone signed in can find it and join without an invite. The list shows its name, how many people are in it, and each member's initials — never names or email addresses.</p>
        <p>Anyone holding the group's invite link can join, so only share it with people you mean to share this with. Leaving the group removes you from it, and with you every readout of you the other members had.</p>
        <div class="dd-group-consent-actions">
          <button class="ghost dd-group-consent-no" type="button">Cancel</button>
          <button class="primary dd-group-consent-yes" type="button">Share my progress with this group</button>
        </div>
      </dialog>
    `;

    const dialog = root.querySelector(".dd-group-consent");
    let pending = null;

    /* ONE consent gate in front of all three ways in. */
    const ask = (run, listed) => {
      pending = run;
      showError(root, "");
      const extra = root.querySelector(".dd-group-consent-public");
      if (extra) extra.hidden = !listed;
      if (dialog?.showModal) dialog.showModal();
      else void runPending();
    };

    const runPending = async () => {
      const run = pending;
      pending = null;
      if (dialog?.close) dialog.close();
      if (!run) return;
      const result = await run();
      if (result.error || !result.group) {
        showError(root, result.error || "That did not work.");
        return;
      }
      store().clearInviteFromLocation();
      deps.onJoined(result.group);
    };

    root.querySelector(".dd-group-consent-no")?.addEventListener("click", () => {
      pending = null;
      dialog?.close?.();
    });
    root.querySelector(".dd-group-consent-yes")?.addEventListener("click", runPending);

    const name = () => store().defaultDisplayName();

    root.querySelector(".dd-group-accept")?.addEventListener("click", () => {
      ask(() => store().joinGroup(invited, name()), false);
    });

    root.querySelector(".dd-group-create")?.addEventListener("click", () => {
      const field = root.querySelector(".dd-group-input");
      const groupName = String(field?.value || "").trim() || "Study group";
      const listed = Boolean(root.querySelector(".dd-group-public-choice")?.checked);
      ask(() => store().createGroup(groupName, name(), listed ? "public" : "private"), listed);
    });

    root.querySelector(".dd-group-join-go")?.addEventListener("click", () => {
      const field = root.querySelector(".dd-group-join-input");
      const token = store().tokenFromPaste(field?.value || "");
      if (!token) {
        showError(root, "That does not look like an invite link.");
        return;
      }
      ask(() => store().joinGroup(token, name()), false);
    });

    /* The directory, for whoever can act on it. Drawing a guest a list of
       rooms they cannot enter — and spending the request to do it — would
       be worse than the sentence they get instead. */
    const list = root.querySelector(".dd-public-list");
    if (signedIn && list) {
      const pick = (group) => ask(() => store().joinPublicGroup(group.group_id, name()), true);
      void renderDirectory(list, pick);
      root.querySelector(".dd-group-refresh")?.addEventListener("click", () => {
        void renderDirectory(list, pick);
      });
    }
  };

  /* ---- the group bar ------------------------------------------------- */

  /**
   * The strip above the roster once you are in a group.
   *
   * @param {{group: any, onChanged: () => void}} deps
   * @returns {HTMLElement}
   */
  const buildGroupBar = ({ group, onChanged }) => {
    const listed = group.visibility === "public";
    const count = group.members?.length || 1;
    const bar = document.createElement("div");
    bar.className = "dd-group-bar";
    bar.innerHTML = `
      <span class="dd-group-title"></span>
      <span class="dd-group-count">${count === 1 ? "just you so far" : `${count} members`}</span>
      <span class="dd-group-visibility${listed ? " is-public" : ""}">${listed ? "Anyone can find it" : "Invite only"}</span>
      <span class="dd-group-actions">
        ${group.is_owner ? `<button class="ghost dd-group-visibility-btn" type="button" data-armed="no">${listed ? "Make it invite-only" : "Let anyone find it"}</button>` : ""}
        <button class="ghost dd-group-copy" type="button">Copy invite link</button>
        ${group.is_owner ? '<button class="ghost dd-group-rotate" type="button">New link</button>' : ""}
        <button class="ghost dd-group-leave" type="button">Leave</button>
      </span>
      <span class="dd-group-said" role="status"></span>
    `;
    bar.querySelector(".dd-group-title").textContent = group.name || "Study group";
    /* The people, before the words. The roster carries names here — you are
       in this group — so the circles get them as their labels. */
    bar.insertBefore(buildAvatarStack(group.members, 6), bar.firstChild);

    const said = bar.querySelector(".dd-group-said");
    const say = (message) => { if (said) said.textContent = message; };

    const toggle = bar.querySelector(".dd-group-visibility-btn");
    toggle?.addEventListener("click", async () => {
      const next = listed ? "private" : "public";
      if (toggle.dataset.armed !== "yes") {
        /* Two clicks, no modal. The first says what listing the group will
           mean for the people already in it — who did not choose it and are
           not being asked — and the second does it. */
        toggle.dataset.armed = "yes";
        toggle.textContent = next === "public" ? "Yes, list it" : "Yes, unlist it";
        say(next === "public"
          ? "Anyone signed in will see this group in the list, with its name, how many people are in it, and each member's initials — never names or email addresses. They can join without an invite."
          : "It will disappear from the list. Everyone already in it stays in it, and the invite link keeps working — replace that too if you want it shut.");
        return;
      }
      say("Saving…");
      const result = await store().setVisibility(group.group_id, next);
      if (result.error) { say(result.error); return; }
      say(next === "public" ? "Listed. Anyone signed in can find it now." : "Unlisted. It is invite-only again.");
      onChanged(result.group);
    });

    bar.querySelector(".dd-group-copy")?.addEventListener("click", async () => {
      const link = store().inviteLink(group.join_token);
      try {
        await navigator.clipboard.writeText(link);
        say("Invite link copied. Anyone holding it can join.");
      } catch (_) {
        /* Clipboard refused — an insecure origin, or a browser that wants a
           gesture it did not see. Showing the link is worse than copying it
           and better than a button that appears to do nothing. */
        say(link);
      }
    });

    bar.querySelector(".dd-group-rotate")?.addEventListener("click", async () => {
      say("Replacing the link…");
      const result = await store().rotateToken(group.group_id);
      if (result.error) { say(result.error); return; }
      say("Done. The old link no longer works. Everyone already in the group stayed in it.");
      onChanged(result.group);
    });

    bar.querySelector(".dd-group-leave")?.addEventListener("click", async () => {
      say("Leaving…");
      const result = await store().leaveGroup(group.group_id);
      if (result.error) { say(result.error); return; }
      onChanged(null);
    });

    return bar;
  };

  return { buildAvatar, buildAvatarStack, avatarColor, renderDiscovery, buildGroupBar };
})();

window.DDGroupsJoin = DDGroupsJoin;
