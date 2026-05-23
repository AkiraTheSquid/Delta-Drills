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
      color: "#DC2626",
      sections: [
        { number: "0.0", title: "Prerequisites", desc: "Essential PyTorch basics, einops/einsum libraries, and tensor manipulation fundamentals.", url: "/arena-book/chapter0_fundamentals/exercises/part0_prereqs/0.0_Prerequisites_exercises.html" },
        { number: "0.1", title: "Ray Tracing", desc: "Learn batched operations and linear algebra by rendering 3D meshes with raytracing.", url: "/arena-book/chapter0_fundamentals/exercises/part1_ray_tracing/0.1_Ray_Tracing_exercises.html" },
        { number: "0.2", title: "CNNs & ResNets", desc: "Build neural networks from scratch, from MNIST classifiers to ResNets for CIFAR-10.", url: "/arena-book/chapter0_fundamentals/exercises/part2_cnns/0.2_CNNs_%26_ResNets_exercises.html" },
        { number: "0.3", title: "Optimization", desc: "Implement SGD, RMSprop & Adam optimizers, and use Weights & Biases for experiment tracking.", url: "/arena-book/chapter0_fundamentals/exercises/part3_optimization/0.3_Optimization_exercises.html" },
        { number: "0.4", title: "Backpropagation", desc: "Build your own autograd system from scratch and train MLPs with custom backpropagation.", url: "/arena-book/chapter0_fundamentals/exercises/part4_backprop/0.4_Backprop_exercises.html" },
        { number: "0.5", title: "VAEs & GANs", desc: "Implement GANs and VAEs, foundational architectures for generative image models.", url: "/arena-book/chapter0_fundamentals/exercises/part5_vaes_and_gans/0.5_VAEs_%26_GANs_exercises.html" },
      ],
    },
    {
      title: "Chapter 1 — Transformer Interpretability",
      image: "https://images.squarespace-cdn.com/content/v1/67e146e032bcbc72c7a584bf/1742816993539-C7YP4RWTUQB9JIUAJX1W/mechinterp.png?format=1500w",
      body:
        "Build and train your own transformer, then take it apart. Covers mechanistic interpretability — circuits, attention heads, and the techniques pioneered by Anthropic's transformer-circuits work and Neel Nanda.",
      color: "#D97706",
      sections: [
        { number: "1.1", title: "Transformers from Scratch", desc: "Build a transformer from scratch and load pretrained GPT-2 weights.", url: "/arena-book/chapter1_transformer_interp/exercises/part1_transformer_from_scratch/1.1_Transformer_from_Scratch_exercises.html" },
        { number: "1.2", title: "Intro to Mech Interp", desc: "Learn TransformerLens to extract activations, apply hooks & find important attention heads.", url: "/arena-book/chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/1.2_Intro_to_Mech_Interp_exercises.html" },
        { number: "1.3.1", title: "Linear Probes", desc: "Train linear probes to detect deception in a model playing the game Coup.", url: "/arena-book/chapter1_transformer_interp/exercises/part31_linear_probes/1.3.1_Linear_Probes_exercises.html" },
        { number: "1.3.2", title: "Function Vectors & Model Steering", desc: "Steer model behaviour using activation interventions and the nnsight library.", url: "/arena-book/chapter1_transformer_interp/exercises/part32_function_vectors_and_model_steering/1.3.2_Function_Vectors_%26_Model_Steering_exercises.html" },
        { number: "1.3.3", title: "Interpretability with SAEs", desc: "Use SAEs to decompose LLM activation space, monitor cognition & steer behaviour.", url: "/arena-book/chapter1_transformer_interp/exercises/part33_interp_with_saes/1.3.3_Interpretability_with_SAEs_exercises.html" },
        { number: "1.3.4", title: "Activation Oracles", desc: "Implement activation oracles to reveal hidden knowledge and uncover forward-predictions.", url: "/arena-book/chapter1_transformer_interp/exercises/part34_activation_oracles/1.3.4_Activation_Oracles_exercises.html" },
        { number: "1.4.1", title: "Indirect Object Identification", desc: "Reverse-engineer the IOI circuit in GPT-2 small following 'Interpretability in the Wild'.", url: "/arena-book/chapter1_transformer_interp/exercises/part41_indirect_object_identification/1.4.1_Indirect_Object_Identification_exercises.html" },
        { number: "1.4.2", title: "SAE Circuits", desc: "Apply SAEs to circuit analysis, decomposing computations and tracing features through layers.", url: "/arena-book/chapter1_transformer_interp/exercises/part42_sae_circuits/1.4.2_SAE_Circuits_exercises.html" },
        { number: "1.5.1", title: "Balanced Bracket Classifier", desc: "Reverse-engineer the algorithm learned by a bracket-balancing transformer.", url: "/arena-book/chapter1_transformer_interp/exercises/part51_balanced_bracket_classifier/1.5.1_Balanced_Bracket_Classifier_exercises.html" },
        { number: "1.5.2", title: "Grokking & Modular Arithmetic", desc: "Discover Fourier circuits in modular arithmetic models and observe grokking in action.", url: "/arena-book/chapter1_transformer_interp/exercises/part52_grokking_and_modular_arithmetic/1.5.2_Grokking_%26_Modular_Arithmetic_exercises.html" },
        { number: "1.5.3", title: "OthelloGPT", desc: "Investigate emergent world representations in a GPT model trained on Othello games.", url: "/arena-book/chapter1_transformer_interp/exercises/part53_othellogpt/1.5.3_OthelloGPT_exercises.html" },
        { number: "1.5.4", title: "Superposition & SAEs", desc: "Replicate Anthropic's superposition paper and train SAEs to recover features.", url: "/arena-book/chapter1_transformer_interp/exercises/part54_toy_models_of_superposition_and_saes/1.5.4_Toy_Models_of_Superposition_%26_SAEs_exercises.html" },
        { number: "", title: "Monthly Algorithmic Problems", desc: "7 algorithmic challenges to test your interpretability skills in hackathon format." },
      ],
    },
    {
      title: "Chapter 2 — Reinforcement Learning",
      image: "https://images.squarespace-cdn.com/content/v1/67e146e032bcbc72c7a584bf/1742816993544-BJGSRE009Z30UYTZYOPJ/rl.png?format=1500w",
      body:
        "RL fundamentals — agents, environments, accumulated reward — with experiments in OpenAI Gym. Then layer on Reinforcement Learning from Human Feedback (RLHF) and apply it to the transformer you trained earlier.",
      color: "#059669",
      sections: [
        { number: "2.1", title: "Intro to RL", desc: "RL fundamentals: MDPs, policies, value functions, and multi-armed bandits.", url: "/arena-book/chapter2_rl/exercises/part1_intro_to_rl/2.1_Intro_to_RL_exercises.html" },
        { number: "2.2", title: "DQN & VPG", desc: "Implement DQN and Vanilla Policy Gradient for CartPole and beyond.", url: "/arena-book/chapter2_rl/exercises/part2_q_learning_and_policy_gradient/2.2_DQN_%26_VPG_exercises.html" },
        { number: "2.3", title: "PPO", desc: "Build a PPO agent from scratch and train it to master CartPole.", url: "/arena-book/chapter2_rl/exercises/part3_ppo/2.3_PPO_exercises.html" },
        { number: "2.4", title: "RLHF", desc: "Implement RLHF end-to-end, applying PPO to language model finetuning.", url: "/arena-book/chapter2_rl/exercises/part4_rlhf/2.4_RLHF_exercises.html" },
      ],
    },
    {
      title: "Chapter 3 — LLM Evaluations",
      image: "https://images.squarespace-cdn.com/content/v1/67e146e032bcbc72c7a584bf/1742816993549-I095YBYX350KH88I5Z8V/evals.jpeg?format=1500w",
      body:
        "Build a multiple-choice benchmark from scratch and use it to evaluate current frontier models. Then move on to LM agents — how to construct them and how to measure their behaviour.",
      color: "#2563EB",
      sections: [
        { number: "3.1", title: "Intro to Evals", desc: "Design threat models and specifications for evaluating model properties.", url: "/arena-book/chapter3_llm_evals/exercises/part1_intro_to_evals/3.1_Intro_to_Evals_exercises.html" },
        { number: "3.2", title: "Dataset Generation", desc: "Use LLMs to generate and refine high-quality evaluation datasets.", url: "/arena-book/chapter3_llm_evals/exercises/part2_dataset_generation/3.2_Dataset_Generation_exercises.html" },
        { number: "3.3", title: "Running Evals with Inspect", desc: "Run standardised LLM evaluations using UK AISI's Inspect library.", url: "/arena-book/chapter3_llm_evals/exercises/part3_running_evals_with_inspect/3.3_Running_Evals_with_Inspect_exercises.html" },
        { number: "3.4", title: "LLM Agents", desc: "Build LLM agents with scaffolding to play Wikipedia Racing and other tasks.", url: "/arena-book/chapter3_llm_evals/exercises/part4_llm_agents/3.4_LLM_Agents_exercises.html" },
      ],
    },
    {
      title: "Chapter 4 — Alignment Science",
      image: "https://images.squarespace-cdn.com/content/v1/67e146e032bcbc72c7a584bf/bbd568c4-e04d-4920-8edd-e5cfebb7bb96/science+of+misalignment.png?format=1500w",
      body:
        "A bucket for AI safety topics that don't fit cleanly into interpretability or evals — emergent misalignment, LLM psychology, the science of misalignment. The frontier of what alignment researchers at labs like Anthropic actually work on.",
      color: "#4F46E5",
      sections: [
        { number: "4.1", title: "Emergent Misalignment", desc: "Study emergent misalignment in finetuned models.", url: "/arena-book/chapter4_alignment_science/exercises/part1_emergent_misalignment/4.1_Emergent_Misalignment_exercises.html" },
        { number: "4.2", title: "Science of Misalignment", desc: "Two case studies in black-box investigation to understand and characterize seemingly misaligned behaviour.", url: "/arena-book/chapter4_alignment_science/exercises/part2_science_of_misalignment/4.2_Science_of_Misalignment_exercises.html" },
        { number: "4.3", title: "Interpreting Reasoning Models", desc: "Apply interpretability techniques to chain-of-thought reasoning models.", url: "/arena-book/chapter4_alignment_science/exercises/part3_interpreting_reasoning_models/4.3_Interpreting_Reasoning_Models_exercises.html" },
        { number: "4.4", title: "LLM Psychology & Persona Vectors", desc: "Explore persona vectors and psychological properties of language models.", url: "/arena-book/chapter4_alignment_science/exercises/part4_persona_vectors/4.4_LLM_Psychology_%26_Persona_Vectors_exercises.html" },
        { number: "4.5", title: "Investigator Agents", desc: "Use AI agents for investigating model behaviours (including petri & bloom).", url: "/arena-book/chapter4_alignment_science/exercises/part5_investigator_agents/4.5_Investigator_Agents_exercises.html" },
      ],
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

    if (Array.isArray(chapter.sections) && chapter.sections.length) {
      row.classList.add("course-chapter-clickable");
      row.tabIndex = 0;
      row.setAttribute("role", "button");
      row.setAttribute("aria-haspopup", "dialog");
      row.setAttribute("aria-label", `${chapter.title} — view sections`);
      if (chapter.color) row.style.setProperty("--chapter-color", chapter.color);
      row.addEventListener("click", () => openChapterModal(chapter));
      row.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openChapterModal(chapter);
        }
      });
    }
    return row;
  };

  let activeModal = null;

  const closeChapterModal = () => {
    if (!activeModal) return;
    document.removeEventListener("keydown", onModalKeydown);
    activeModal.remove();
    activeModal = null;
    document.body.classList.remove("modal-open");
  };

  const onModalKeydown = (e) => {
    if (e.key === "Escape") {
      e.preventDefault();
      closeChapterModal();
    }
  };

  const openChapterModal = (chapter) => {
    closeChapterModal();
    const backdrop = document.createElement("div");
    backdrop.className = "chapter-modal-backdrop";
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) closeChapterModal();
    });

    const modal = document.createElement("div");
    modal.className = "chapter-modal";
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-labelledby", "chapter-modal-title");
    if (chapter.color) modal.style.setProperty("--section-number-color", chapter.color);

    const header = document.createElement("div");
    header.className = "chapter-modal-header";
    const heading = document.createElement("h3");
    heading.id = "chapter-modal-title";
    heading.textContent = (chapter.title || "").replace(/^Chapter \d+ — /, "") || chapter.title;
    const close = document.createElement("button");
    close.type = "button";
    close.className = "chapter-modal-close";
    close.setAttribute("aria-label", "Close");
    close.textContent = "×";
    close.addEventListener("click", closeChapterModal);
    header.appendChild(heading);
    header.appendChild(close);

    const content = document.createElement("div");
    content.className = "chapter-modal-content";
    chapter.sections.forEach((s) => content.appendChild(buildSectionItem(s)));

    modal.appendChild(header);
    modal.appendChild(content);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);
    document.body.classList.add("modal-open");
    document.addEventListener("keydown", onModalKeydown);
    close.focus();
    activeModal = backdrop;
  };

  const buildSectionItem = (section) => {
    const item = document.createElement(section.url ? "a" : "div");
    item.className = section.url ? "section-item section-item-link" : "section-item";
    item.setAttribute("role", "listitem");
    if (section.url) {
      item.href = section.url;
      item.setAttribute(
        "aria-label",
        `Open ${section.number ? section.number + " " : ""}${section.title} in the curriculum book`,
      );
    }

    const num = document.createElement("span");
    num.className = "section-number";
    num.textContent = section.number || "";

    const info = document.createElement("div");
    info.className = "section-info";
    const t = document.createElement("div");
    t.className = "section-title";
    t.textContent = section.title;
    const d = document.createElement("div");
    d.className = "section-desc";
    d.textContent = section.desc;
    info.appendChild(t);
    info.appendChild(d);

    item.appendChild(num);
    item.appendChild(info);
    return item;
  };

  input.addEventListener("input", (e) => renderList(e.target.value));
  renderList("");
})();
