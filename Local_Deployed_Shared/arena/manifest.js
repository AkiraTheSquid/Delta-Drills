/* ================================================================
   ARENA STAGE-1 PROBLEM REGISTRY

   Full ARENA 5.0 curriculum (chapters 0–4, 32 sections). Each entry
   feeds both the ARENA tab (rich detail panel) and the Predicted
   course scores stats sub-tab (table aggregated by chapter).

   To preserve the hand-crafted summaries and readiness data the
   registry started with, sections in `RICH_SECTIONS` keep their
   original specific values. The rest of the curriculum is filled in
   from chapter-level defaults via `buildProblem()`. Replace the
   placeholder readiness scores with user-specific values once Delta
   Drills can compute them.
   ================================================================ */

const ARENA_PLACEHOLDER_NOTE =
  "Placeholder readiness — replace with user-specific score from Delta Drills skill state in a later stage.";

const ARENA_LAUNCH_MODE = "Notebook file launch placeholder";

const ARENA_CHAPTER_DEFAULTS = {
  chapter0_fundamentals: {
    label: "Fundamentals",
    prerequisiteTags: ["NumPy", "PyTorch basics", "Linear algebra", "Python fluency"],
    skillWeights: [
      { skill: "NumPy", weight: 0.3 },
      { skill: "PyTorch", weight: 0.3 },
      { skill: "Linear algebra", weight: 0.2 },
      { skill: "Implementation", weight: 0.2 },
    ],
  },
  chapter1_transformer_interp: {
    label: "Transformer Interpretability",
    prerequisiteTags: ["Transformer architecture", "PyTorch", "TransformerLens", "Probing"],
    skillWeights: [
      { skill: "Transformers", weight: 0.35 },
      { skill: "Mech interp", weight: 0.3 },
      { skill: "PyTorch", weight: 0.2 },
      { skill: "Probing/analysis", weight: 0.15 },
    ],
  },
  chapter2_rl: {
    label: "Reinforcement Learning",
    prerequisiteTags: ["PyTorch", "Probability", "Optimization", "MDP intuition"],
    skillWeights: [
      { skill: "PyTorch", weight: 0.3 },
      { skill: "RL concepts", weight: 0.3 },
      { skill: "Probability", weight: 0.2 },
      { skill: "Optimization", weight: 0.2 },
    ],
  },
  chapter3_llm_evals: {
    label: "LLM Evaluations",
    prerequisiteTags: ["Python fluency", "Prompting", "Evaluation design", "Statistics"],
    skillWeights: [
      { skill: "LLM evals", weight: 0.35 },
      { skill: "Prompting", weight: 0.25 },
      { skill: "Dataset reasoning", weight: 0.2 },
      { skill: "Statistics", weight: 0.2 },
    ],
  },
  chapter4_alignment_science: {
    label: "Alignment Science",
    prerequisiteTags: ["LLM behavior", "Mech interp basics", "Prompting", "Eval design"],
    skillWeights: [
      { skill: "Alignment concepts", weight: 0.35 },
      { skill: "LLM analysis", weight: 0.25 },
      { skill: "Mech interp", weight: 0.2 },
      { skill: "Eval design", weight: 0.2 },
    ],
  },
};

const labelForScore = (score) => {
  if (score >= 75) return "Likely ready";
  if (score >= 60) return "Borderline ready";
  if (score >= 45) return "Partially ready";
  return "Needs prerequisite study";
};

// Hand-crafted overrides preserved from the original 4-entry manifest.
const RICH_SECTIONS = {
  "arena-0.0-prereqs": {
    summary:
      "Foundational array and tensor manipulation warm-up. This is a good first problem unit for testing the problem-registry flow because it is scoped tightly and maps cleanly onto Delta Drills prerequisite skills.",
    readinessScore: 78,
    readinessLabel: "Likely ready",
    readinessNote:
      "Stage 1 uses a placeholder readiness estimate. Later stages should replace this with a user-specific score computed from Delta Drills skill state.",
    prerequisiteTags: ["NumPy basics", "Array shapes", "Broadcasting intuition", "Einops familiarity"],
    skillWeights: [
      { skill: "NumPy", weight: 0.35 },
      { skill: "Broadcasting", weight: 0.3 },
      { skill: "Einops", weight: 0.2 },
      { skill: "Tensor shapes", weight: 0.15 },
    ],
  },
  "arena-0.1-ray-tracing": {
    summary:
      "A more geometric fundamentals unit with a precise notebook target. This gives you a second stage-1 problem page that is clearly different from prerequisites while still staying in the safer early-fundamentals area.",
    readinessScore: 64,
    readinessLabel: "Borderline ready",
    readinessNote:
      "Placeholder readiness only. Use this page to verify that Delta Drills can describe why a user might need prerequisite review before launching.",
    prerequisiteTags: ["Vector math", "NumPy indexing", "Broadcasting", "Linear algebra basics"],
    skillWeights: [
      { skill: "NumPy", weight: 0.25 },
      { skill: "Broadcasting", weight: 0.25 },
      { skill: "Linear algebra", weight: 0.3 },
      { skill: "Geometry reasoning", weight: 0.2 },
    ],
  },
  "arena-2.1-intro-to-rl": {
    summary:
      "An example of a heavier ARENA problem unit that still benefits from problem-level indexing inside Delta Drills. Stage 1 should let you inspect the exact notebook target and prerequisite profile before any deeper runtime integration exists.",
    readinessScore: 41,
    readinessLabel: "Needs prerequisite study",
    readinessNote:
      "This intentionally shows a lower placeholder readiness to exercise the dedicated problem-page UX for underprepared users.",
    prerequisiteTags: ["PyTorch basics", "Optimization", "Probability", "MDP intuition"],
    skillWeights: [
      { skill: "PyTorch", weight: 0.3 },
      { skill: "Optimization", weight: 0.2 },
      { skill: "Probability", weight: 0.25 },
      { skill: "RL concepts", weight: 0.25 },
    ],
  },
  "arena-3.1-intro-to-evals": {
    summary:
      "A stage-1 test case for later chapters. This page is mainly for verifying that Delta Drills can represent exact problem units, weighted skills, and launch metadata even when the eventual runtime path will likely need more infrastructure.",
    readinessScore: 53,
    readinessLabel: "Partially ready",
    readinessNote:
      "Use this as a test of chapter diversity in the registry rather than as proof of full browser execution support.",
    prerequisiteTags: ["Python fluency", "Prompting", "Evaluation design", "Basic statistics"],
    skillWeights: [
      { skill: "LLM evals", weight: 0.35 },
      { skill: "Prompting", weight: 0.2 },
      { skill: "Dataset reasoning", weight: 0.2 },
      { skill: "Statistics", weight: 0.25 },
    ],
  },
};

// Curriculum spec — every ARENA 5.0 section across chapters 0–4.
// `notebookPath` and `lessonPath` are deploy-relative paths under
// `content/ARENA_5.0-main/`, matching the on-disk layout linked into
// `arena-book/` via symlinks.
const ARENA_CURRICULUM = [
  // Chapter 0 — Fundamentals
  { id: "arena-0.0-prereqs", chapter: "chapter0_fundamentals", section: "0.0 Prerequisites", title: "Prerequisites",
    notebookPath: "content/ARENA_5.0-main/chapter0_fundamentals/exercises/part0_prereqs/0.0_Prerequisites_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter0_fundamentals/instructions/pages/00_[0.0]_Prerequisites.md",
    placeholderScore: 78 },
  { id: "arena-0.1-ray-tracing", chapter: "chapter0_fundamentals", section: "0.1 Ray Tracing", title: "Ray Tracing",
    notebookPath: "content/ARENA_5.0-main/chapter0_fundamentals/exercises/part1_ray_tracing/0.1_Ray_Tracing_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter0_fundamentals/instructions/pages/01_[0.1]_Ray_Tracing.md",
    placeholderScore: 64 },
  { id: "arena-0.2-cnns-and-resnets", chapter: "chapter0_fundamentals", section: "0.2 CNNs & ResNets", title: "CNNs & ResNets",
    notebookPath: "content/ARENA_5.0-main/chapter0_fundamentals/exercises/part2_cnns/0.2_CNNs_&_ResNets_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter0_fundamentals/instructions/pages/02_[0.2]_CNNs_&_ResNets.md",
    placeholderScore: 70 },
  { id: "arena-0.3-optimization", chapter: "chapter0_fundamentals", section: "0.3 Optimization", title: "Optimization",
    notebookPath: "content/ARENA_5.0-main/chapter0_fundamentals/exercises/part3_optimization/0.3_Optimization_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter0_fundamentals/instructions/pages/03_[0.3]_Optimization.md",
    placeholderScore: 60 },
  { id: "arena-0.4-backprop", chapter: "chapter0_fundamentals", section: "0.4 Backprop", title: "Backprop",
    notebookPath: "content/ARENA_5.0-main/chapter0_fundamentals/exercises/part4_backprop/0.4_Backprop_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter0_fundamentals/instructions/pages/04_[0.4]_Backprop.md",
    placeholderScore: 55 },
  { id: "arena-0.5-vaes-and-gans", chapter: "chapter0_fundamentals", section: "0.5 VAEs & GANs", title: "VAEs & GANs",
    notebookPath: "content/ARENA_5.0-main/chapter0_fundamentals/exercises/part5_vaes_and_gans/0.5_VAEs_&_GANs_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter0_fundamentals/instructions/pages/05_[0.5]_VAEs_&_GANs.md",
    placeholderScore: 50 },

  // Chapter 1 — Transformer Interp (ARENA 5.0 layout)
  { id: "arena-1.1-transformer-from-scratch", chapter: "chapter1_transformer_interp", section: "1.1 Transformer from Scratch", title: "Transformer from Scratch",
    notebookPath: "content/ARENA_5.0-main/chapter1_transformer_interp/exercises/part1_transformer_from_scratch/1.1_Transformer_from_Scratch_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter1_transformer_interp/instructions/pages/01_[1.1]_Transformer_from_Scratch.md",
    placeholderScore: 50 },
  { id: "arena-1.2-intro-to-mech-interp", chapter: "chapter1_transformer_interp", section: "1.2 Intro to Mech Interp", title: "Intro to Mech Interp",
    notebookPath: "content/ARENA_5.0-main/chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/1.2_Intro_to_Mech_Interp_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter1_transformer_interp/instructions/pages/02_[1.2]_Intro_to_Mech_Interp.md",
    placeholderScore: 45 },
  { id: "arena-1.3.1-linear-probes", chapter: "chapter1_transformer_interp", section: "1.3.1 Linear Probes", title: "Linear Probes",
    notebookPath: "content/ARENA_5.0-main/chapter1_transformer_interp/exercises/part31_linear_probes/1.3.1_Linear_Probes_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter1_transformer_interp/instructions/pages/11_[1.3.1]_Linear_Probes.md",
    placeholderScore: 42 },
  { id: "arena-1.3.2-function-vectors", chapter: "chapter1_transformer_interp", section: "1.3.2 Function Vectors & Model Steering", title: "Function Vectors & Model Steering",
    notebookPath: "content/ARENA_5.0-main/chapter1_transformer_interp/exercises/part32_function_vectors_and_model_steering/1.3.2_Function_Vectors_&_Model_Steering_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter1_transformer_interp/instructions/pages/12_[1.3.2]_Function_Vectors_&_Model_Steering.md",
    placeholderScore: 40 },
  { id: "arena-1.3.3-interp-with-saes", chapter: "chapter1_transformer_interp", section: "1.3.3 Interpretability with SAEs", title: "Interpretability with SAEs",
    notebookPath: "content/ARENA_5.0-main/chapter1_transformer_interp/exercises/part33_interp_with_saes/1.3.3_Interpretability_with_SAEs_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter1_transformer_interp/instructions/pages/13_[1.3.3]_Interpretability_with_SAEs.md",
    placeholderScore: 38 },
  { id: "arena-1.3.4-activation-oracles", chapter: "chapter1_transformer_interp", section: "1.3.4 Activation Oracles", title: "Activation Oracles",
    notebookPath: "content/ARENA_5.0-main/chapter1_transformer_interp/exercises/part34_activation_oracles/1.3.4_Activation_Oracles_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter1_transformer_interp/instructions/pages/14_[1.3.4]_Activation_Oracles.md",
    placeholderScore: 36 },
  { id: "arena-1.4.1-ioi", chapter: "chapter1_transformer_interp", section: "1.4.1 IOI", title: "Indirect Object Identification",
    notebookPath: "content/ARENA_5.0-main/chapter1_transformer_interp/exercises/part41_indirect_object_identification/1.4.1_Indirect_Object_Identification_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter1_transformer_interp/instructions/pages/21_[1.4.1]_Indirect_Object_Identification.md",
    placeholderScore: 35 },
  { id: "arena-1.4.2-sae-circuits", chapter: "chapter1_transformer_interp", section: "1.4.2 SAE Circuits", title: "SAE Circuits",
    notebookPath: "content/ARENA_5.0-main/chapter1_transformer_interp/exercises/part42_sae_circuits/1.4.2_SAE_Circuits_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter1_transformer_interp/instructions/pages/22_[1.4.2]_SAE_Circuits.md",
    placeholderScore: 33 },
  { id: "arena-1.5.1-balanced-brackets", chapter: "chapter1_transformer_interp", section: "1.5.1 Balanced Brackets", title: "Balanced Bracket Classifier",
    notebookPath: "content/ARENA_5.0-main/chapter1_transformer_interp/exercises/part51_balanced_bracket_classifier/1.5.1_Balanced_Bracket_Classifier_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter1_transformer_interp/instructions/pages/31_[1.5.1]_Balanced_Bracket_Classifier.md",
    placeholderScore: 38 },
  { id: "arena-1.5.2-grokking", chapter: "chapter1_transformer_interp", section: "1.5.2 Grokking", title: "Grokking & Modular Arithmetic",
    notebookPath: "content/ARENA_5.0-main/chapter1_transformer_interp/exercises/part52_grokking_and_modular_arithmetic/1.5.2_Grokking_&_Modular_Arithmetic_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter1_transformer_interp/instructions/pages/32_[1.5.2]_Grokking_&_Modular_Arithmetic.md",
    placeholderScore: 33 },
  { id: "arena-1.5.3-othellogpt", chapter: "chapter1_transformer_interp", section: "1.5.3 OthelloGPT", title: "OthelloGPT",
    notebookPath: "content/ARENA_5.0-main/chapter1_transformer_interp/exercises/part53_othellogpt/1.5.3_OthelloGPT_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter1_transformer_interp/instructions/pages/33_[1.5.3]_OthelloGPT.md",
    placeholderScore: 30 },
  { id: "arena-1.5.4-superposition", chapter: "chapter1_transformer_interp", section: "1.5.4 Superposition & SAEs", title: "Toy Models of Superposition & SAEs",
    notebookPath: "content/ARENA_5.0-main/chapter1_transformer_interp/exercises/part54_toy_models_of_superposition_and_saes/1.5.4_Toy_Models_of_Superposition_&_SAEs_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter1_transformer_interp/instructions/pages/34_[1.5.4]_Toy_Models_of_Superposition_&_SAEs.md",
    placeholderScore: 30 },

  // Chapter 2 — Reinforcement Learning
  { id: "arena-2.1-intro-to-rl", chapter: "chapter2_rl", section: "2.1 Intro to RL", title: "Intro to RL",
    notebookPath: "content/ARENA_5.0-main/chapter2_rl/exercises/part1_intro_to_rl/2.1_Intro_to_RL_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter2_rl/instructions/pages/01_[2.1]_Intro_to_RL.md",
    placeholderScore: 41 },
  { id: "arena-2.2.1-dqn", chapter: "chapter2_rl", section: "2.2.1 DQN", title: "Deep Q Networks",
    notebookPath: "content/ARENA_5.0-main/chapter2_rl/exercises/part21_dqn/2.2.1_Deep_Q_Networks_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter2_rl/instructions/pages/02_[2.2]_DQN_&_VPG.md",
    placeholderScore: 38 },
  { id: "arena-2.2.2-policy-gradient", chapter: "chapter2_rl", section: "2.2.2 Policy Gradient", title: "Policy Gradient",
    notebookPath: "content/ARENA_5.0-main/chapter2_rl/exercises/part22_vpg/2.2.2_Policy_Gradient_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter2_rl/instructions/pages/21_[2.2.2]_Policy_Gradient.md",
    placeholderScore: 35 },
  { id: "arena-2.3-ppo", chapter: "chapter2_rl", section: "2.3 PPO", title: "PPO",
    notebookPath: "content/ARENA_5.0-main/chapter2_rl/exercises/part3_ppo/2.3_PPO_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter2_rl/instructions/pages/03_[2.3]_PPO.md",
    placeholderScore: 32 },
  { id: "arena-2.4-rlhf", chapter: "chapter2_rl", section: "2.4 RLHF", title: "RLHF",
    notebookPath: "content/ARENA_5.0-main/chapter2_rl/exercises/part4_rlhf/2.4_RLHF_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter2_rl/instructions/pages/04_[2.4]_RLHF.md",
    placeholderScore: 30 },

  // Chapter 3 — LLM Evals
  { id: "arena-3.1-intro-to-evals", chapter: "chapter3_llm_evals", section: "3.1 Intro to Evals", title: "Intro to Evals",
    notebookPath: "content/ARENA_5.0-main/chapter3_llm_evals/exercises/part1_intro_to_evals/3.1_Intro_to_Evals_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter3_llm_evals/instructions/pages/01_[3.1]_Intro_to_Evals.md",
    placeholderScore: 53 },
  { id: "arena-3.2-dataset-generation", chapter: "chapter3_llm_evals", section: "3.2 Dataset Generation", title: "Dataset Generation",
    notebookPath: "content/ARENA_5.0-main/chapter3_llm_evals/exercises/part2_dataset_generation/3.2_Dataset_Generation_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter3_llm_evals/instructions/pages/02_[3.2]_Dataset_Generation.md",
    placeholderScore: 48 },
  { id: "arena-3.3-running-evals-with-inspect", chapter: "chapter3_llm_evals", section: "3.3 Running Evals with Inspect", title: "Running Evals with Inspect",
    notebookPath: "content/ARENA_5.0-main/chapter3_llm_evals/exercises/part3_running_evals_with_inspect/3.3_Running_Evals_with_Inspect_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter3_llm_evals/instructions/pages/03_[3.3]_Running_Evals_with_Inspect.md",
    placeholderScore: 45 },
  { id: "arena-3.4-llm-agents", chapter: "chapter3_llm_evals", section: "3.4 LLM Agents", title: "LLM Agents",
    notebookPath: "content/ARENA_5.0-main/chapter3_llm_evals/exercises/part4_llm_agents/3.4_LLM_Agents_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter3_llm_evals/instructions/pages/04_[3.4]_LLM_Agents.md",
    placeholderScore: 40 },

  // Chapter 4 — Alignment Science (NEW in ARENA 5.0)
  { id: "arena-4.1-emergent-misalignment", chapter: "chapter4_alignment_science", section: "4.1 Emergent Misalignment", title: "Emergent Misalignment",
    notebookPath: "content/ARENA_5.0-main/chapter4_alignment_science/exercises/part1_emergent_misalignment/4.1_Emergent_Misalignment_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter4_alignment_science/instructions/pages/01_[4.1]_Emergent_Misalignment.md",
    placeholderScore: 45 },
  { id: "arena-4.2-science-of-misalignment", chapter: "chapter4_alignment_science", section: "4.2 Science of Misalignment", title: "Science of Misalignment",
    notebookPath: "content/ARENA_5.0-main/chapter4_alignment_science/exercises/part2_science_of_misalignment/4.2_Science_of_Misalignment_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter4_alignment_science/instructions/pages/02_[4.2]_Science_of_Misalignment.md",
    placeholderScore: 42 },
  { id: "arena-4.3-interpreting-reasoning-models", chapter: "chapter4_alignment_science", section: "4.3 Interpreting Reasoning Models", title: "Interpreting Reasoning Models",
    notebookPath: "content/ARENA_5.0-main/chapter4_alignment_science/exercises/part3_interpreting_reasoning_models/4.3_Interpreting_Reasoning_Models_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter4_alignment_science/instructions/pages/03_[4.3]_Interpreting_Reasoning_Models.md",
    placeholderScore: 38 },
  { id: "arena-4.4-persona-vectors", chapter: "chapter4_alignment_science", section: "4.4 LLM Psychology & Persona Vectors", title: "LLM Psychology & Persona Vectors",
    notebookPath: "content/ARENA_5.0-main/chapter4_alignment_science/exercises/part4_persona_vectors/4.4_LLM_Psychology_&_Persona_Vectors_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter4_alignment_science/instructions/pages/04_[4.4]_LLM_Psychology_&_Persona_Vectors.md",
    placeholderScore: 35 },
  { id: "arena-4.5-investigator-agents", chapter: "chapter4_alignment_science", section: "4.5 Investigator Agents", title: "Investigator Agents",
    notebookPath: "content/ARENA_5.0-main/chapter4_alignment_science/exercises/part5_investigator_agents/4.5_Investigator_Agents_exercises.ipynb",
    lessonPath: "content/ARENA_5.0-main/chapter4_alignment_science/instructions/pages/05_[4.5]_Investigator_Agents.md",
    placeholderScore: 32 },
];

const buildArenaProblem = (entry) => {
  const def = ARENA_CHAPTER_DEFAULTS[entry.chapter];
  const rich = RICH_SECTIONS[entry.id] || {};
  const score = rich.readinessScore ?? entry.placeholderScore ?? 50;
  return {
    id: entry.id,
    chapterId: entry.chapter,
    chapterLabel: def.label,
    sectionLabel: entry.section,
    title: entry.title,
    summary:
      rich.summary ||
      `${def.label} section ${entry.section}: ${entry.title}. Placeholder summary — see ARENA materials for the full problem description.`,
    readinessScore: score,
    readinessLabel: rich.readinessLabel || labelForScore(score),
    readinessNote: rich.readinessNote || ARENA_PLACEHOLDER_NOTE,
    prerequisiteTags: rich.prerequisiteTags || def.prerequisiteTags,
    skillWeights: rich.skillWeights || def.skillWeights,
    lessonPath: entry.lessonPath,
    notebookPath: entry.notebookPath,
    backupNotebookPath: null,
    launchPath: entry.notebookPath,
    executionMode: ARENA_LAUNCH_MODE,
  };
};

window.ARENA_STAGE1_PROBLEMS = ARENA_CURRICULUM.map(buildArenaProblem);
