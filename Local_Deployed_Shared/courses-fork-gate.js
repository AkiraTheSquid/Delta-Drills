/* ================================================================
   COURSES-FORK-GATE.JS — first-click gate on the Courses tab's Colab links.

   The section links on the Courses tab point at the notebooks in Seth's fork
   `AkiraTheSquid/ARENA_3.0` (which carries practice notebooks upstream does
   not), read-only to everyone else — edits made there cannot be saved back. The fix is the same one the
   Account tab already documents: the student forks ARENA_3.0 once, tells us
   their GitHub username, and every Colab link on the page swaps the owner to
   their fork. Colab's `File → Save a copy in GitHub` then writes their work
   into their own repo.

   A GitHub username alone grants no write access, so the duplication is the
   fork click — this dialog just walks the student through it and records the
   username. No tokens, no OAuth, no backend.

   The username lives in `account_github_username`, the same key the Account
   tab reads and writes, so setting it here also fills in the Account field
   and repoints the Colab links on the Predicted and Practice tabs.
   ================================================================ */

(function initCoursesForkGate() {
  const USERNAME_KEY = "account_github_username";
  const PROMPTED_KEY = "delta_drills_courses_fork_prompted";
  // Fork SETH'S repo, not Callum's: it is the one with the supplementary
  // practice notebooks, and the one the app compiles its notebooks from.
  const FORK_URL = "https://github.com/AkiraTheSquid/ARENA_3.0/fork";
  // GitHub logins: alphanumeric plus internal single hyphens, max 39 chars.
  const USERNAME_RE = /^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$/;
  const CHECK_TIMEOUT_MS = 4000;

  const read = (key) => {
    try {
      return (localStorage.getItem(key) || "").trim();
    } catch (_) {
      return "";
    }
  };

  const saveUsername = (name) => {
    try {
      localStorage.setItem(USERNAME_KEY, name);
      localStorage.setItem(PROMPTED_KEY, "1");
    } catch (_) {
      /* private mode — the links still work for this session */
    }
    // Keep the Account tab's field in sync if it is already in the DOM.
    const field = document.getElementById("account-github-username");
    if (field) field.value = name;
    document.dispatchEvent(new CustomEvent("courses:github-owner-changed"));
  };

  const markPrompted = () => {
    try {
      localStorage.setItem(PROMPTED_KEY, "1");
    } catch (_) {
      /* ignore */
    }
  };

  // Ask GitHub whether <user>/ARENA_3.0 exists yet. Advisory only: a network
  // failure or rate limit must never block the student from opening a notebook.
  const forkExists = async (user) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CHECK_TIMEOUT_MS);
    try {
      const res = await fetch(`https://api.github.com/repos/${encodeURIComponent(user)}/ARENA_3.0`, {
        signal: controller.signal,
        headers: { Accept: "application/vnd.github+json" },
      });
      if (res.status === 404) return false;
      return true;
    } catch (_) {
      return true;
    } finally {
      clearTimeout(timer);
    }
  };

  const hrefFor = (notebookPath) =>
    typeof colabUpstreamHref === "function" ? colabUpstreamHref(notebookPath) : "";

  const openIn = (tab, href) => {
    if (tab && !tab.closed) tab.location.href = href;
    else window.open(href, "_blank", "noreferrer");
  };

  let activeGate = null;

  const closeGate = () => {
    if (!activeGate) return;
    document.removeEventListener("keydown", onGateKeydown);
    activeGate.remove();
    activeGate = null;
    document.body.classList.remove("modal-open");
  };

  const onGateKeydown = (e) => {
    if (e.key === "Escape") {
      e.preventDefault();
      closeGate();
    }
  };

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  };

  const open = (notebookPath) => {
    closeGate();

    const backdrop = el("div", "fork-gate-backdrop");
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) closeGate();
    });

    const modal = el("div", "fork-gate");
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-labelledby", "fork-gate-title");

    const title = el("h3", "fork-gate-title", "Save your work to your own GitHub?");
    title.id = "fork-gate-title";

    const blurb = el(
      "p",
      "fork-gate-blurb",
      "These exercises open Callum McDougall's original ARENA notebooks, which are read-only. " +
        "Fork them to your GitHub account and every link on this page will open your copy instead — " +
        "then Colab's File → Save a copy in GitHub keeps everything you write.",
    );

    const step1 = el("div", "fork-gate-step");
    step1.appendChild(el("span", "fork-gate-step-num", "1"));
    const step1Body = el("div", "fork-gate-step-body");
    step1Body.appendChild(el("p", "fork-gate-step-text", "Fork the ARENA repository (one time, opens GitHub)."));
    const forkLink = el("a", "fork-gate-fork-btn", "Fork AkiraTheSquid/ARENA_3.0 ↗");
    forkLink.href = FORK_URL;
    forkLink.target = "_blank";
    forkLink.rel = "noreferrer";
    step1Body.appendChild(forkLink);
    step1.appendChild(step1Body);

    const step2 = el("div", "fork-gate-step");
    step2.appendChild(el("span", "fork-gate-step-num", "2"));
    const step2Body = el("div", "fork-gate-step-body");
    step2Body.appendChild(el("p", "fork-gate-step-text", "Enter your GitHub username."));
    const form = document.createElement("form");
    form.className = "fork-gate-form";
    const input = document.createElement("input");
    input.type = "text";
    input.className = "fork-gate-input";
    input.placeholder = "your-github-username";
    input.autocomplete = "username";
    input.spellcheck = false;
    input.setAttribute("aria-label", "GitHub username");
    input.value = read(USERNAME_KEY);
    const submit = document.createElement("button");
    submit.type = "submit";
    submit.className = "fork-gate-submit";
    submit.textContent = "Save & open my copy";
    form.appendChild(input);
    form.appendChild(submit);
    step2Body.appendChild(form);
    const status = el("p", "fork-gate-status hidden");
    status.setAttribute("role", "status");
    step2Body.appendChild(status);
    step2.appendChild(step2Body);

    const skip = el("button", "fork-gate-skip", "Skip — open the read-only original");
    skip.type = "button";
    skip.addEventListener("click", () => {
      markPrompted();
      closeGate();
      window.open(hrefFor(notebookPath), "_blank", "noreferrer");
    });

    const footer = el("p", "fork-gate-footnote", "You can change this any time in the Account tab.");

    modal.appendChild(title);
    modal.appendChild(blurb);
    modal.appendChild(step1);
    modal.appendChild(step2);
    modal.appendChild(skip);
    modal.appendChild(footer);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);
    document.body.classList.add("modal-open");
    document.addEventListener("keydown", onGateKeydown);
    input.focus();
    activeGate = backdrop;

    // The fork check is async, and popup blockers reject window.open() once the
    // user gesture has been awaited — so claim the tab synchronously here and
    // point it at the notebook (or close it) once we know the answer.
    let overrideCheck = false;
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const name = input.value.trim().replace(/^@/, "").replace(/^https?:\/\/github\.com\//i, "");
      if (!USERNAME_RE.test(name)) {
        status.textContent = "That doesn't look like a GitHub username.";
        status.className = "fork-gate-status fork-gate-status-error";
        input.focus();
        return;
      }

      const tab = window.open("about:blank", "_blank");
      submit.disabled = true;
      status.textContent = "Checking your fork…";
      status.className = "fork-gate-status";

      const exists = overrideCheck || (await forkExists(name));
      if (!exists) {
        if (tab && !tab.closed) tab.close();
        submit.disabled = false;
        overrideCheck = true;
        submit.textContent = "Use it anyway";
        status.textContent = `Couldn't find ${name}/ARENA_3.0. Fork it in step 1 first, or press again to use it anyway.`;
        status.className = "fork-gate-status fork-gate-status-error";
        return;
      }

      saveUsername(name);
      closeGate();
      openIn(tab, hrefFor(notebookPath));
    });
  };

  window.CoursesForkGate = {
    // Prompt once: the student either sets a username or explicitly skips.
    needsPrompt: () => !read(USERNAME_KEY) && read(PROMPTED_KEY) !== "1",
    open,
  };
})();
