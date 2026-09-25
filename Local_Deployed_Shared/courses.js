/* ================================================================
   COURSES.JS — Courses tab: a catalog of courses, each enable-able.

   Back to a list (docs/spec-multi-course-catalog.md, 2026-09-25): a
   `#courses-list-view` card per `window.DeltaCourseRegistry.list()` entry
   (course-registry.js) with an enable toggle that reads/writes
   `GET/POST /api/practice/course-shares` — enabling a course sets its
   share above 0, which is what feeds it into practice (backend
   `app/course_mix.py`). "View course" swaps to `#courses-detail-view`,
   dispatched by the course's `detailKind`: "arena" renders the article
   below unchanged; "lesson-list" renders a small KC list that jumps into
   the Knowledge Graph tab. A back control returns to the list.

   ARENA's own render path — hero + source links + intro, then the
   alternating chapter rows, sections modal, in-app notebooks — is
   UNCHANGED from the one-course era described below.

   🔴 A SECTION ROW STAYS IN THE APP (Seth, 2026-09-01: "it won't actually
   take you to the Google Colab. It will instead stay inside of the app, and
   it will have an app version of those arena notebooks"). The row is a link
   to `?arena=<slug>` — a real href, so middle-click and "open in new tab"
   still work — and a plain left-click is intercepted and handed to
   `practice/arena-notebook.js`, which renders the upstream notebook on the
   notebook surface without a page load. The notebooks come from
   `scripts/compile_arena_notebooks.py`.

   COLAB IS STILL THERE, AS THE SECONDARY ROUTE. Each row carries a small
   "Colab ↗" link beside it, and that link is the one the fork convention from
   `stats/predicted-links.js` applies to: `colabUpstreamHref(path)` points at
   `<account_github_username>/ARENA_3.0` when the student has saved a GitHub
   username and falls back to `AkiraTheSquid/ARENA_3.0` otherwise (Seth's
   fork — the file the app compiles and the study group pulls), with
   `courses-fork-gate.js` intercepting the first click to offer to set it.
   Kept rather than deleted because a student with their own fork is working
   in it — an in-app notebook cannot save to their repo, and taking the door
   away would make the fork gate a feature with no way in.
   ================================================================ */

const ARENA_LOGO = "https://learn.arena.education/static/images/arena-logo.png";

// Section `url` fields are book pages served from this app. The matching
// upstream notebook is the same path with the prefix stripped and the
// extension swapped — see notebookPathForBookUrl() below.
const ARENA_BOOK_PREFIX = "/arena-book/";

const ARENA_DETAIL = {
  hero: {
    title: "ARENA Curriculum",
    subtitle: "AI safety, hands-on. From PyTorch fundamentals to original alignment research.",
    logo: ARENA_LOGO,
  },
  sources: [
    { label: "arena.education ↗", href: "https://www.arena.education/", title: "The ARENA programme homepage" },
    {
      label: "AkiraTheSquid/ARENA_3.0 ↗",
      href: "https://github.com/AkiraTheSquid/ARENA_3.0",
      title: "The original ARENA exercise repository on GitHub",
    },
    { label: "Curriculum book ↗", href: "https://learn.arena.education/", title: "The ARENA curriculum, published" },
  ],
  intro:
    "ARENA is a programme run by the London Initiative for Safe AI (LISA) that takes participants from coding fundamentals to the technical frontier of AI safety. The curriculum mixes coding exercises, paper replications, and an open-ended capstone project so you finish with both the skills and the artifacts to enter the field. All materials are open source and self-study friendly. Every section below opens Callum McDougall's notebook here in the app, against your own Python session — or in Colab, from the link beside it.",
  chapters: [
    {
      title: "Chapter 0 — Fundamentals",
      image: "https://images.squarespace-cdn.com/content/v1/67e146e032bcbc72c7a584bf/1742816993533-HAYMCMMHWW9WDNPN4A9J/funda.png?format=1500w",
      body:
        "Coding best practices, PyTorch fluency, and building your own CNNs and ResNets from scratch. The levelling chapter — everyone leaves on the same page so the rest of the programme can move.",
      color: "#DC2626",
      sections: [
        { number: "0.0", title: "Prerequisites", desc: "Essential PyTorch basics, einops/einsum libraries, and tensor manipulation fundamentals.", url: "/arena-book/chapter0_fundamentals/exercises/part0_prereqs/0.0_Prerequisites_exercises.html" },
        // 🔴 The `_with_supplementary_exercises` edition, which lives ONLY on
        // AkiraTheSquid/ARENA_3.0: the upstream notebook plus, under each
        // exercise, a collapsible practice section (prerequisite examples,
        // faded problems, challenges, siblings). Seth's study group pulls that
        // repo, the Colab link opens it, and compile_arena_notebooks.py reads
        // the fork first — one file on all three surfaces. No Jupyter Book page
        // exists for it; the url is only ever turned back into a notebook path.
        { number: "0.1", title: "Ray Tracing", desc: "Learn batched operations and linear algebra by rendering 3D meshes with raytracing. With supplementary practice under each exercise.", url: "/arena-book/chapter0_fundamentals/exercises/part1_ray_tracing/0.1_Ray_Tracing_exercises_with_supplementary_exercises.html" },
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
        // Upstream split 2.2 into two notebooks (ARENA 527f937, 2026-09-02); the
        // old part2_q_learning_and_policy_gradient notebook no longer exists.
        { number: "2.2.1", title: "DQN", desc: "Implement Deep Q-Networks and train an agent on CartPole.", url: "/arena-book/chapter2_rl/exercises/part21_dqn/2.2.1_DQN_exercises.html" },
        { number: "2.2.2", title: "VPG", desc: "Implement Vanilla Policy Gradient for CartPole and beyond.", url: "/arena-book/chapter2_rl/exercises/part22_vpg/2.2.2_VPG_exercises.html" },
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

// `/arena-book/<rest>.html` → `<rest>.ipynb`, the path inside ARENA_3.0.
// Book URLs are percent-encoded per segment (`%26` for `&`); decode here so
// colabUpstreamHref() can re-encode consistently for Colab's parser.
const notebookPathForBookUrl = (url) => {
  if (typeof url !== "string" || !url.startsWith(ARENA_BOOK_PREFIX)) return "";
  const rel = url.slice(ARENA_BOOK_PREFIX.length).replace(/\.html$/, ".ipynb");
  return rel
    .split("/")
    .map((seg) => {
      try {
        return decodeURIComponent(seg);
      } catch (_) {
        return seg;
      }
    })
    .join("/");
};

// Resolved fresh on every call so it reflects the current GitHub username —
// the fork gate can change the owner after the links have already rendered.
const arenaColabHrefFor = (notebookPath) => {
  if (!notebookPath) return "";
  return typeof colabUpstreamHref === "function" ? colabUpstreamHref(notebookPath) : "";
};

/* The id the in-app notebook is filed under: `0.0` → `0-0`, `1.3.1` → `1-3-1`.
   🔴 THIS MUST STAY IN STEP WITH `compile_arena_notebooks.py::_slug`, which
   mints the filename from the very same section number. They are two sides of
   one name, and the failure when they drift is a row that opens "that ARENA
   notebook has not been compiled here" — which reads as a missing file rather
   than as a renaming. `Local_Deployed_Shared/lessons/notebooks/watch.py`
   checks that every section courses.js links has a compiled notebook. */
const arenaSlugForSection = (section) => String(section.number || "").trim().replace(/\./g, "-");

(function initCoursesTab() {
  const detailView = document.getElementById("courses-detail-view");
  const listView = document.getElementById("courses-list-view");
  if (!detailView || !listView) return;

  const buildSources = (sources) => {
    const bar = document.createElement("nav");
    bar.className = "course-sources";
    bar.setAttribute("aria-label", "ARENA source links");
    (sources || []).forEach((s) => {
      const a = document.createElement("a");
      a.className = "course-source-link";
      a.href = s.href;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = s.label;
      if (s.title) a.title = s.title;
      bar.appendChild(a);
    });
    return bar;
  };

  const buildArticle = (detail) => {
    const article = document.createElement("article");
    article.className = "course-article";

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
    heroText.appendChild(buildSources(detail.sources));
    hero.appendChild(heroLogo);
    hero.appendChild(heroText);
    article.appendChild(hero);

    const intro = document.createElement("p");
    intro.className = "course-intro";
    intro.textContent = detail.intro;
    article.appendChild(intro);

    const chaptersWrap = document.createElement("section");
    chaptersWrap.className = "course-chapters";
    detail.chapters.slice(0, 1).forEach((ch, i) => chaptersWrap.appendChild(buildChapter({ ...ch, sections: ch.sections.filter(s => ["0.0", "0.1", "0.2"].includes(s.number)) }, i)));
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

  /* Which sections were actually compiled into this build — `null` until the
     index has been read, a Set afterwards.

     🔴 THE COMPILED NOTEBOOKS ARE NOT IN GIT. They are generated from
     `Local_Deployed_Shared/content/`, which is gitignored, and they are
     gitignored themselves, so a fresh clone — and any deploy — has this
     Courses tab, these rows, and none of the notebooks behind them. The rows
     must not become dead ends there: with no notebook to open, the Colab link
     is still a real route to the same material, so the row uses it.

     Primed when a chapter modal opens rather than at load, because a learner
     who never opens Courses should not pay for the index. */
  let compiledSlugs = null;
  const primeArenaIndex = () => {
    if (compiledSlugs || !window.ArenaNotebook) return;
    Promise.resolve(window.ArenaNotebook.sections())
      .then((list) => {
        compiledSlugs = new Set((list || []).map((entry) => entry.id));
      })
      .catch(() => {
        /* Leave it null. Unknown is not the same as empty: a click then tries
           the in-app view and gets its own, more specific message, which is
           the right answer for one failed fetch on an otherwise built tree. */
      });
  };

  const openChapterModal = (chapter) => {
    closeChapterModal();
    primeArenaIndex();
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

  /* A row is a WRAPPER holding two links: the section itself, which opens in
     the app, and a small "Colab ↗" beside it.

     🔴 The wrapper exists because an `<a>` may not contain another `<a>`. The
     row used to BE the anchor; putting the Colab link inside it would be
     invalid markup, and browsers recover from it by closing the outer anchor
     early — which silently makes the second half of the row unclickable. */
  const buildSectionItem = (section) => {
    const notebookPath = notebookPathForBookUrl(section.url);
    const slug = arenaSlugForSection(section);
    const row = document.createElement("div");
    row.className = "section-row";
    row.setAttribute("role", "listitem");

    // No notebook path at all (the "Monthly Algorithmic Problems" entry, which
    // has no upstream file) stays exactly what it was: a plain, inert row.
    const item = document.createElement(slug ? "a" : "div");
    item.className = slug ? "section-item section-item-link" : "section-item";
    if (slug) {
      item.dataset.arenaSlug = slug;
      /* A REAL href, not a `#`: middle-click, ctrl-click and "open in new tab"
         are how a reader queues up three sections at once, and a click handler
         that calls preventDefault on a `#` takes all of that away. The
         deep link is the one `arena-notebook.js` reads at load. */
      item.href = `?arena=${encodeURIComponent(slug)}`;
      item.setAttribute(
        "aria-label",
        `Open ${section.number ? section.number + " " : ""}${section.title} in the app`,
      );
      item.addEventListener("click", (e) => onSectionClick(e, slug));
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
    row.appendChild(item);

    if (notebookPath) {
      const colab = document.createElement("a");
      colab.className = "section-colab";
      colab.dataset.notebookPath = notebookPath;
      colab.href = arenaColabHrefFor(notebookPath);
      colab.target = "_blank";
      colab.rel = "noreferrer";
      colab.textContent = "Colab ↗";
      colab.title = "Open the original notebook in Google Colab";
      colab.setAttribute(
        "aria-label",
        `Open ${section.number ? section.number + " " : ""}${section.title} in Google Colab`,
      );
      colab.addEventListener("click", (e) => onColabLinkClick(e, notebookPath));
      row.appendChild(colab);
    }
    return row;
  };

  /* The section itself: rendered in the app, no page load.

     Modified clicks (new tab/window, middle-click) fall through to the href so
     browser-native open-in-background still works — and because that href is a
     real `?arena=` deep link, the new tab lands on the same notebook.

     If the notebook is not there — the compiler has never run in this checkout
     — `open()` returns false having already said so on screen, and the modal
     is closed either way so the learner is looking at the message rather than
     at the section list on top of it. */
  const onSectionClick = (e, slug) => {
    const modified = e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0;
    if (modified || !window.ArenaNotebook) return;

    /* Not compiled in this build, but the same section has a Colab link on the
       row: hand the click to it. The reader wanted this section, and being
       told the app cannot show it is worse than being shown it somewhere the
       app can. Only when the index says so — `null` means "not read yet", and
       guessing "missing" there would send a working build to Colab. */
    if (compiledSlugs && !compiledSlugs.has(slug)) {
      const colab = e.currentTarget.parentElement?.querySelector("a.section-colab");
      if (colab) {
        e.preventDefault();
        closeChapterModal();
        colab.click();
        return;
      }
    }

    e.preventDefault();
    closeChapterModal();
    window.ArenaNotebook.open(slug);
  };

  // First plain left-click on any Colab link goes through the fork gate, which
  // offers to point every link at the student's own ARENA_3.0 fork. Modified
  // clicks (new tab/window, middle-click) fall through to the rendered href so
  // browser-native open-in-background still works.
  const onColabLinkClick = (e, notebookPath) => {
    const gate = window.CoursesForkGate;
    const modified = e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0;
    if (modified || !gate || !gate.needsPrompt()) return;
    e.preventDefault();
    gate.open(notebookPath);
  };

  // The gate can change the GitHub owner after links are on the page. Re-resolve
  // every rendered href (including the ones inside an open modal) so a saved
  // username takes effect without a re-render.
  const refreshColabHrefs = () => {
    document.querySelectorAll("a.section-colab[data-notebook-path]").forEach((a) => {
      a.href = arenaColabHrefFor(a.dataset.notebookPath);
    });
  };
  document.addEventListener("courses:github-owner-changed", refreshColabHrefs);

  /* ================================================================
     CATALOG — the list view, one card per course-registry.js entry, and
     the detail dispatch that swaps a card's "View course" for the right
     renderer. Enabling a card is the ONLY control over a course's practice
     share now (Account tab's slider is gone) — the toggle round-trips
     `GET/POST /api/practice/course-shares`, which owns the number
     server-side (app/course_mix.py), same pattern arena-share.js used
     for the single-course version of this control.
     ================================================================ */

  const registry = () => (window.DeltaCourseRegistry ? window.DeltaCourseRegistry.list() : []);

  let shareState = null; // { courses: [{course, enabled, share, ...}], enable_share }
  // Toggles stay disabled until the initial GET lands — otherwise a toggle
  // during that window races its own load: the load can resolve AFTER the
  // toggle's own save and re-render the checkbox back to the pre-toggle state.
  let sharesReady = false;
  let loadError = null;
  // Bumped by every load and every save: a load that resolves after a newer
  // request (the learner's own toggle) is dropped instead of repainting over it.
  let generation = 0;
  const shareRow = (courseId) =>
    (shareState && shareState.courses && shareState.courses.find((r) => r.course === courseId)) || null;

  const loadCourseShares = async () => {
    const res = await apiFetch("/api/practice/course-shares");
    if (!res.ok) throw new Error("Could not load your course settings. Check your connection and sign-in.");
    return res.json();
  };
  const setCourseEnabled = async (courseId, enabled) => {
    const res = await apiFetch("/api/practice/course-shares", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ course: courseId, enabled }),
    });
    if (!res.ok) throw new Error("Could not save that — check your connection and sign-in.");
    return res.json();
  };

  const buildLessonListArticle = (course) => {
    const article = document.createElement("article");
    article.className = "course-article";

    const hero = document.createElement("header");
    hero.className = "course-hero";
    const heroText = document.createElement("div");
    heroText.className = "course-hero-text";
    const heroTitle = document.createElement("h1");
    heroTitle.className = "course-hero-title";
    heroTitle.textContent = course.label;
    const heroSub = document.createElement("p");
    heroSub.className = "course-hero-subtitle";
    heroSub.textContent = course.eyebrow || "";
    heroText.appendChild(heroTitle);
    heroText.appendChild(heroSub);
    hero.appendChild(heroText);
    article.appendChild(hero);

    const intro = document.createElement("p");
    intro.className = "course-intro";
    intro.textContent =
      course.intro ||
      "A small graph, on its own: a lesson and a couple of multiple-choice questions per concept. Open one to work it on the Knowledge Graph, scoped to just this course.";
    article.appendChild(intro);

    const list = document.createElement("ul");
    list.className = "course-lesson-list";
    Promise.resolve(course.milestoneKcs()).then((set) => {
      [...(set instanceof Set ? set : new Set(set || []))].forEach((kc) => {
        const row = document.createElement("li");
        const item = document.createElement("button");
        item.type = "button";
        item.className = "course-lesson-item";
        item.textContent = course.kcTitle ? course.kcTitle(kc) : kc;
        item.addEventListener("click", () => {
          window.deltaSetKgCourseFilter?.(course.id);
          switchTab("knowledge-graph");
          window.deltaFocusConceptGraphKc?.(kc);
        });
        row.appendChild(item);
        list.appendChild(row);
      });
    });
    article.appendChild(list);

    return article;
  };

  const DETAIL_BUILDERS = {
    arena: () => buildArticle(ARENA_DETAIL),
    "lesson-list": (course) => buildLessonListArticle(course),
  };

  const showList = () => {
    detailView.hidden = true;
    detailView.replaceChildren();
    listView.hidden = false;
  };

  const showDetail = (course) => {
    listView.hidden = true;
    detailView.hidden = false;
    const back = document.createElement("button");
    back.type = "button";
    back.className = "course-back-link";
    back.textContent = "← All courses";
    back.addEventListener("click", showList);
    const build = DETAIL_BUILDERS[course.detailKind] || DETAIL_BUILDERS.arena;
    detailView.replaceChildren(back, build(course));
  };

  const buildCourseCard = (course) => {
    const card = document.createElement("div");
    card.className = "course-catalog-card";

    const text = document.createElement("div");
    text.className = "course-catalog-card-text";
    const title = document.createElement("h3");
    title.className = "course-catalog-card-title";
    title.textContent = course.label;
    const eyebrow = document.createElement("p");
    eyebrow.className = "course-catalog-card-eyebrow";
    eyebrow.textContent = course.eyebrow || "";
    text.appendChild(title);
    text.appendChild(eyebrow);

    const actions = document.createElement("div");
    actions.className = "course-catalog-card-actions";

    const toggleLabel = document.createElement("label");
    toggleLabel.className = "course-catalog-toggle";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    // The course name keeps each card's toggle distinguishable to a screen
    // reader. The words are the course's own (course-registry.js): ARENA's
    // toggle only mixes its exercises in early — its concepts are the main
    // graph either way — while Delta Drills' is the only way into practice.
    const toggleWords = course.toggleLabel || "Add to practice";
    toggle.setAttribute("aria-label", `${course.label}: ${toggleWords}`);
    const row = shareRow(course.id);
    toggle.checked = !!(row && row.enabled);
    toggle.disabled = !sharesReady;
    toggle.addEventListener("change", async () => {
      ++generation;
      toggle.disabled = true;
      try {
        shareState = await setCourseEnabled(course.id, toggle.checked);
      } catch (err) {
        toggle.checked = !toggle.checked;
        toggle.disabled = false;
        window.alert?.(err.message);
        return;
      }
      toggle.disabled = false;
      // The save already landed — a refresh hiccup here must not roll the
      // checkbox back and claim the change failed when it didn't.
      try {
        await window.deltaRefreshKcLattice?.();
      } catch (_) {
        /* lattice refresh is best-effort; the saved setting still stands */
      }
      window.dispatchEvent(new CustomEvent("delta:adaptive-state-changed"));
    });
    const toggleText = document.createElement("span");
    toggleText.textContent = !sharesReady && loadError ? loadError : toggleWords;
    toggleLabel.appendChild(toggle);
    toggleLabel.appendChild(toggleText);

    const view = document.createElement("button");
    view.type = "button";
    view.className = "course-catalog-view-btn";
    view.textContent = "View course →";
    view.addEventListener("click", () => showDetail(course));

    actions.appendChild(toggleLabel);
    actions.appendChild(view);

    card.appendChild(text);
    card.appendChild(actions);
    return card;
  };

  const renderList = () => {
    // listView (#courses-list-view) IS the grid — its class is set in
    // index.html — so cards go straight in it, not into a second nested grid.
    listView.replaceChildren(...registry().map(buildCourseCard));
  };

  // Only a SUCCESSFUL load enables the toggles: a failed one keeps them
  // disabled with the error on the card, rather than showing every course as
  // off and editable. Waits for practice/init.js's mode announcement — before
  // it a fresh guest has no token yet and the GET would fail for that alone.
  const refreshShares = async () => {
    const mine = ++generation;
    try {
      const data = await loadCourseShares();
      if (mine !== generation) return;
      shareState = data;
      sharesReady = true;
      loadError = null;
    } catch (err) {
      if (mine !== generation) return;
      sharesReady = false;
      loadError = err.message;
    }
    renderList();
  };

  renderList();
  showList();
  window.addEventListener("delta:practice-mode-ready", refreshShares);
  if (window.DDPracticeModeReady) refreshShares();
})();
