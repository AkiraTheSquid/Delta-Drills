/* ================================================================
   COURSES.JS — Courses tab: full list + live filter, with article
   detail view per course. Click a course → detail; Back → list.
   Per-course "Include for study?" Yes/No toggle persists in
   localStorage and mirrors between list-card and detail-hero.
   ================================================================ */

const ARENA_LOGO = "https://learn.arena.education/static/images/arena-logo.png";

const ARENA_DETAIL = {
  hero: {
    title: "ARENA Curriculum",
    subtitle: "AI safety, hands-on. From PyTorch fundamentals to original alignment research.",
    logo: ARENA_LOGO,
  },
  intro:
    "ARENA is a programme run by the London Initiative for Safe AI (LISA) that takes participants from coding fundamentals to the technical frontier of AI safety. The curriculum mixes coding exercises, paper replications, and an open-ended capstone project so you finish with both the skills and the artifacts to enter the field. All materials are open source and self-study friendly.",
  chapters: [
    {
      title: "Chapter 0 — Fundamentals",
      image: "https://images.squarespace-cdn.com/content/v1/67e146e032bcbc72c7a584bf/1742816993533-HAYMCMMHWW9WDNPN4A9J/funda.png?format=1500w",
      body:
        "Coding best practices, PyTorch fluency, and building your own CNNs and ResNets from scratch. The levelling chapter — everyone leaves on the same page so the rest of the programme can move.",
    },
    {
      title: "Chapter 1 — Transformer Interpretability",
      image: "https://images.squarespace-cdn.com/content/v1/67e146e032bcbc72c7a584bf/1742816993539-C7YP4RWTUQB9JIUAJX1W/mechinterp.png?format=1500w",
      body:
        "Build and train your own transformer, then take it apart. Covers mechanistic interpretability — circuits, attention heads, and the techniques pioneered by Anthropic's transformer-circuits work and Neel Nanda.",
    },
    {
      title: "Chapter 2 — Reinforcement Learning",
      image: "https://images.squarespace-cdn.com/content/v1/67e146e032bcbc72c7a584bf/1742816993544-BJGSRE009Z30UYTZYOPJ/rl.png?format=1500w",
      body:
        "RL fundamentals — agents, environments, accumulated reward — with experiments in OpenAI Gym. Then layer on Reinforcement Learning from Human Feedback (RLHF) and apply it to the transformer you trained earlier.",
    },
    {
      title: "Chapter 3 — LLM Evaluations",
      image: "https://images.squarespace-cdn.com/content/v1/67e146e032bcbc72c7a584bf/1742816993549-I095YBYX350KH88I5Z8V/evals.jpeg?format=1500w",
      body:
        "Build a multiple-choice benchmark from scratch and use it to evaluate current frontier models. Then move on to LM agents — how to construct them and how to measure their behaviour.",
    },
    {
      title: "Chapter 4 — Alignment Science",
      image: "https://images.squarespace-cdn.com/content/v1/67e146e032bcbc72c7a584bf/bbd568c4-e04d-4920-8edd-e5cfebb7bb96/science+of+misalignment.png?format=1500w",
      body:
        "A bucket for AI safety topics that don't fit cleanly into interpretability or evals — emergent misalignment, LLM psychology, the science of misalignment. The frontier of what alignment researchers at labs like Anthropic actually work on.",
    },
    {
      title: "Capstone Project",
      image: "https://images.squarespace-cdn.com/content/v1/67e146e032bcbc72c7a584bf/1742816993554-83FWIXW78N86GPKJS82N/DALL%C2%B7E%2B2022-09-28%2B12.07.07%2B-%2Bpainting%2Bof%2Ba%2Bhuman%2Busing%2Ba%2Bvery%2Blarge%2Bcomputer%2C%2Bin%2Bthe%2Bstyle%2Bof%2Bsci-fi%2Bartist%2BJim%2BBurns.png?format=1500w",
      body:
        "An open-ended project to close out the in-person programme. Pick a topic that hooked you during the course and spend a month building something real with the skills you picked up.",
    },
  ],
};

const COURSES_CATALOG = [
  {
    id: "arena",
    name: "ARENA Curriculum",
    logo: ARENA_LOGO,
    blurb: "Mechanistic interpretability, RL, evals, and alignment science — hands-on.",
    detail: ARENA_DETAIL,
  },
];

const COURSES_INCLUDE_STORAGE_KEY = "delta_drills_courses_include";

const loadIncludeState = () => {
  try {
    const raw = localStorage.getItem(COURSES_INCLUDE_STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
};

const saveIncludeState = (state) => {
  try {
    localStorage.setItem(COURSES_INCLUDE_STORAGE_KEY, JSON.stringify(state));
  } catch {
    /* quota or disabled storage — silently no-op */
  }
};

const getInclude = (courseId) => loadIncludeState()[courseId] || null;

const setInclude = (courseId, value) => {
  const state = loadIncludeState();
  if (value === null) delete state[courseId];
  else state[courseId] = value;
  saveIncludeState(state);
  syncIncludeControls(courseId, value);
};

const syncIncludeControls = (courseId, value) => {
  const groups = document.querySelectorAll(`[data-include-for="${courseId}"]`);
  groups.forEach((group) => {
    group.querySelectorAll("[data-include-value]").forEach((btn) => {
      const selected = btn.dataset.includeValue === value;
      btn.classList.toggle("is-selected", selected);
      btn.setAttribute("aria-checked", String(selected));
    });
  });
};

const buildIncludeControl = (courseId, scope) => {
  const wrap = document.createElement("div");
  wrap.className = `course-include course-include-${scope}`;
  wrap.dataset.includeFor = courseId;

  const label = document.createElement("span");
  label.className = "course-include-label";
  label.textContent = "Include course for study?";
  label.id = `course-include-label-${courseId}-${scope}`;

  const group = document.createElement("div");
  group.className = "course-include-options";
  group.setAttribute("role", "radiogroup");
  group.setAttribute("aria-labelledby", label.id);

  const current = getInclude(courseId);

  ["yes", "no"].forEach((value) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "course-include-option";
    btn.dataset.includeValue = value;
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", String(current === value));
    if (current === value) btn.classList.add("is-selected");

    const dot = document.createElement("span");
    dot.className = "course-include-dot";
    dot.setAttribute("aria-hidden", "true");

    const text = document.createElement("span");
    text.className = "course-include-text";
    text.textContent = value === "yes" ? "Yes" : "No";

    btn.appendChild(dot);
    btn.appendChild(text);
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      setInclude(courseId, value);
    });
    group.appendChild(btn);
  });

  wrap.appendChild(label);
  wrap.appendChild(group);
  return wrap;
};

(function initCoursesTab() {
  const input = document.getElementById("courses-search");
  const results = document.getElementById("courses-results");
  const listView = document.getElementById("courses-list-view");
  const detailView = document.getElementById("courses-detail-view");
  if (!input || !results || !listView || !detailView) return;

  const renderCard = (course) => {
    const card = document.createElement("article");
    card.className = "course-card";
    card.dataset.courseId = course.id;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `Open ${course.name}`);

    const img = document.createElement("img");
    img.className = "course-card-logo";
    img.src = course.logo;
    img.alt = `${course.name} logo`;
    img.loading = "lazy";
    img.referrerPolicy = "no-referrer";

    const body = document.createElement("div");
    body.className = "course-card-body";

    const name = document.createElement("div");
    name.className = "course-card-name";
    name.textContent = course.name;

    body.appendChild(name);
    if (course.blurb) {
      const blurb = document.createElement("div");
      blurb.className = "course-card-blurb";
      blurb.textContent = course.blurb;
      body.appendChild(blurb);
    }

    card.appendChild(img);
    card.appendChild(body);
    card.appendChild(buildIncludeControl(course.id, "list"));

    const open = (e) => {
      if (e.target.closest(".course-include")) return;
      showDetail(course);
    };
    card.addEventListener("click", open);
    card.addEventListener("keydown", (e) => {
      if (e.target.closest(".course-include")) return;
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        showDetail(course);
      }
    });
    return card;
  };

  const renderList = (query) => {
    const q = (query || "").trim().toLowerCase();
    const matches = q
      ? COURSES_CATALOG.filter((c) => c.name.toLowerCase().includes(q))
      : COURSES_CATALOG.slice();

    results.replaceChildren();
    if (!matches.length) {
      const empty = document.createElement("div");
      empty.className = "courses-empty";
      empty.textContent = `No courses match “${query}”.`;
      results.appendChild(empty);
      return;
    }
    matches.forEach((c) => results.appendChild(renderCard(c)));
  };

  const showList = () => {
    detailView.classList.add("hidden");
    detailView.replaceChildren();
    listView.classList.remove("hidden");
    if (typeof input.focus === "function") input.focus({ preventScroll: true });
  };

  const showDetail = (course) => {
    if (!course || !course.detail) return;
    detailView.replaceChildren(buildDetail(course));
    listView.classList.add("hidden");
    detailView.classList.remove("hidden");
    detailView.scrollTop = 0;
    window.scrollTo({ top: 0, behavior: "instant" });
  };

  const buildDetail = (course) => {
    const detail = course.detail;
    const article = document.createElement("article");
    article.className = "course-article";

    const back = document.createElement("button");
    back.type = "button";
    back.className = "course-back-btn";
    back.innerHTML = '<span aria-hidden="true">←</span> Back to courses';
    back.addEventListener("click", showList);
    article.appendChild(back);

    const hero = document.createElement("header");
    hero.className = "course-hero";
    const heroLogo = document.createElement("img");
    heroLogo.className = "course-hero-logo";
    heroLogo.src = detail.hero.logo;
    heroLogo.alt = `${detail.hero.title} logo`;
    heroLogo.referrerPolicy = "no-referrer";
    const heroText = document.createElement("div");
    heroText.className = "course-hero-text";
    const heroTitle = document.createElement("h1");
    heroTitle.className = "course-hero-title";
    heroTitle.textContent = detail.hero.title;
    const heroSub = document.createElement("p");
    heroSub.className = "course-hero-subtitle";
    heroSub.textContent = detail.hero.subtitle;
    heroText.appendChild(heroTitle);
    heroText.appendChild(heroSub);
    hero.appendChild(heroLogo);
    hero.appendChild(heroText);
    hero.appendChild(buildIncludeControl(course.id, "detail"));
    article.appendChild(hero);

    const intro = document.createElement("p");
    intro.className = "course-intro";
    intro.textContent = detail.intro;
    article.appendChild(intro);

    const chaptersWrap = document.createElement("section");
    chaptersWrap.className = "course-chapters";
    detail.chapters.forEach((ch, i) => {
      chaptersWrap.appendChild(buildChapter(ch, i));
    });
    article.appendChild(chaptersWrap);

    return article;
  };

  const buildChapter = (chapter, index) => {
    const row = document.createElement("section");
    row.className = `course-chapter ${index % 2 === 0 ? "course-chapter-left" : "course-chapter-right"}`;

    const img = document.createElement("img");
    img.className = "course-chapter-img";
    img.src = chapter.image;
    img.alt = `${chapter.title} illustration`;
    img.loading = "lazy";
    img.referrerPolicy = "no-referrer";

    const body = document.createElement("div");
    body.className = "course-chapter-body";
    const title = document.createElement("h2");
    title.className = "course-chapter-title";
    title.textContent = chapter.title;
    const text = document.createElement("p");
    text.className = "course-chapter-text";
    text.textContent = chapter.body;
    body.appendChild(title);
    body.appendChild(text);

    row.appendChild(img);
    row.appendChild(body);
    return row;
  };

  input.addEventListener("input", (e) => renderList(e.target.value));
  renderList("");
})();
