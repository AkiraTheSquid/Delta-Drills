# Plan for Integrating Spaced Repetition with the Existing Adaptive System

**Goal:** Augment the current adaptive-difficulty/prioritization system with spaced-repetition scheduling (FSRS) and knowledge-graph-based review, minimizing extra work. We’ll treat spaced repetition as a *parallel module* that schedules reviews of mastered subtopics, while the existing system continues selecting new problems. Key steps include mapping the current system, surveying open-source tools (SRS schedulers, KG builders, LLM prerequisite extractors), designing integration points, prototyping data flows, and planning validation.

## 1. Map the Existing Architecture and Data Interfaces  
Understand how the current system tracks subtopics and selects problems. Key components (from the formulas) are: 

- **Difficulty & Mastery Update:** After each question on subtopic *s*, a *baseline* score is updated via a recency factor α, combining the latest `score(n)` and prior baseline. This baseline roughly represents the learner’s mastery of *s*. The system also computes a running correctness *p(n)* and a difficulty multiplier that scales problem difficulty.  
- **Priority Calculation:** Each subtopic *s* has a user-set weight $w_s$ and a “learning rate” estimate $\hat r_s$ (an exponentially-weighted average of baseline deltas). The *priority* $g_s = w_s \cdot \hat r_s$ is used to pick the next topic.  
- **Problem Selection:** New problems are drawn at a target difficulty based on the topic’s current baseline and multiplier (Eqns from the user text).  

**Interfaces to note:** The system likely exposes APIs or events when a subtopic is mastered (maybe baseline exceeds a threshold) or when a question is answered (with its score). We need hooks at these points for the spaced-repetition module. Importantly, *the existing system drives “what to learn next” based on mastery; the SRS will drive “when to review past learning”.* We must preserve the existing difficulty-adjustment loop for new learning, and add a separate loop for review scheduling.

## 2. Survey Open-Source Tools and Data  

### Spaced-Repetition Schedulers (FSRS)
- **FSRS (Free Spaced Repetition Scheduler):**  The state-of-the-art open SRS. FSRS models each “card” (here, a subtopic) with three variables (difficulty, stability, retrievability) and predicts optimal review intervals. It is **scientifically backed** (machine-learned model) and has open-source implementations in Python and other languages【2†L196-L200】【25†L512-L519】. For example, the **py-fsrs** library provides FSRS in Python【15†L266-L270】. Benchmarks show FSRS vastly outperforms the old SM-2 schedule (FSRS-6 beats SM-2 on 99.6% of users by log-loss)【25†L512-L519】. *Benefit:* adopting FSRS will improve retention scheduling and reduce pointless reviews (it learns an optimal ease factor per item).  
- **FSRS4Anki / OSR:** The Open Spaced Repetition community provides tools and docs (FSRS4Anki wiki【16†L198-L205】) and an “awesome-FSRS” list【2†L220-L228】. We can use their libraries rather than re-implementing an SRS.

### Knowledge Graph Construction (Prerequisites)
- **KGGen (NeurIPS’25):** An open-source framework to extract a knowledge graph from unstructured text using LLMs【5†L313-L322】. It supports LLM backends (GPT, Gemini, Ollama etc.) and outputs entities/relations via DSPy. It can ingest course outlines or content to generate concepts and edges. (It also includes the MINE benchmark for graph quality【5†L329-L332】.)  
- **GraphRAG (Microsoft):** A RAG-based KGC pipeline that extracts entities and relations (including educational relations) via LLMs【29†L364-L372】. Though aimed at scientific text, its code is public. GraphRAG’s README emphasizes using graph “memory structures” from unstructured text to improve LLM outputs【29†L364-L372】.  
- **Neo4j LLM-Graph-Builder:** A mature open-source tool (4.5k stars) that uses LLMs and LangChain to build a Neo4j KG from documents (PDFs, web pages, etc.)【27†L298-L304】. It features a UI and API: upload content, choose LLM, and it extracts nodes/edges into Neo4j【27†L298-L304】.  
- **Graphusion (WWW’25):** A research codebase (24 stars) for zero-shot KG construction. It focuses on scientific domains but was shown to work in an “educational scenario” (TutorQA benchmark)【9†L1-L4】【8†L0-L9】. Graphusion explicitly includes a “Prerequisite_of” relation in its taxonomy【8†L13-L21】.  
- **LLM Prerequisite Research:** Recent papers validate LLMs for prerequisite extraction. A 2025 study found top LLMs (LLaMA4-Maverick, Claude-3, Qwen2, etc.) can predict prerequisite links with high F1 (~0.82–0.83) from topic names/descriptions【38†L660-L668】【38†L690-L697】. Another benchmark (ESCO-PrereqSkill) shows GPT-4 and similar models align closely with expert prerequisite graphs【35†L25-L33】【38†L660-L668】. While no off-the-shelf tool encapsulates this, it indicates **LLM prompting** can score “A is prerequisite of B” effectively.  

### Educational Datasets (for Validation)  
(Existing resources we could leverage for prototyping or evaluation.)  
- **Course Concept Pairs:** Datasets like *University Course Dataset* or *MOOC Prerequisite* contain labeled concept dependencies (often used in prerequisite prediction research)【source from prompt】.  
- **LectureBank/TutorialBank:** NLP/ML topic prerequisite chains【source】.  
- **Others:** Q-matrix datasets (KCs↔items), Kaggle/Cmu DataShop math (Junyi dataset), and Metacademy are publicly known prerequisite sources【prompt data】.  

### Student Modeling (Complementary)  
- Not directly asked, but we note mature frameworks exist (pyKT, GKT, etc.) that could inform how often to quiz known items. But since existing system uses a custom difficulty model, we focus on FSRS and simple graph heuristics.

## 3. Design Integration Points

**Overall Architecture:** We propose a *dual-loop system*. The original loop (adaptive difficulty & priority) continues for new content, and an *FSRS loop* runs in parallel to schedule reviews. Both loops feed a common “task queue.” Each cycle or session, the system selects tasks either to introduce new material or review old material based on priorities.

- **FSRS Card Model:** Treat each **subtopic** (or knowledge component) as an FSRS “card.” When the user first *learns* or shows mastery of subtopic *s* (e.g. baseline stabilizes), create an FSRS entry with initial difficulty. We can map the current baseline (0–100) to FSRS’s 1–10 scale (e.g. difficulty = 1 + 9·(1 – baseline/100)). Over time, FSRS will adjust each card’s difficulty and stability from review results.  
- **Event Hooks:** On each attempted question for subtopic *s*:  
  - Update the current baseline/difficulty (existing system).  
  - **If subtopic *s* is considered “mastered”** (e.g. user score consistently high), we add or update its FSRS record as “reviewed.” Give FSRS the result (correct/incorrect) and accuracy to update stability. For example, a correct solve (say grade>85) counts as a successful review (Easy/Good in FSRS terms); a struggle counts as a Hard or again. Use the actual score or binary correct to determine FSRS outcome.  
  - **Implicit Credit Propagation:** If we have a prerequisite/encompassing graph, we can optionally give *partial review credit* to prerequisites. For instance, when solving an advanced topic *B* correctly, we could feed FSRS a fractional “virtual review” for each predecessor *A*. Practically, we might call FSRS.update(A) with a reduced confidence (e.g. as a Hard if full review would be Good). This mimics Math Academy’s “implicit practice” concept【42†L124-L130】【41†L96-L105】. In absence of a ready graph, skip this or assume only direct prerequisites.  
- **Scheduling Loop:** Independently, FSRS will predict the next review date for each subtopic card. We maintain a timestamp of “due” reviews. At scheduling time (e.g. start of session or periodically), check which subtopics are due for review.  
- **Merging with Priority:** To combine with the existing priority system, we can treat *due review tasks* as urgent items. For example, any subtopic *s* due by FSRS could be given a very high interim weight (setting $g_s$ extremely large) so the system selects it immediately for review. Alternatively, we could maintain a separate “review queue” that is served before new learning tasks. The exact strategy can be tuned: e.g. always do all due reviews before new content, or interleave based on severity (e.g. how overdue).  
- **Parameter Mapping:** FSRS difficulty uses 1–10; we can initialize it from the subtopic’s current difficulty or baseline. As reviews happen, FSRS will learn a new difficulty/stability for that card. We should store FSRS’s state (difficulty, stability) alongside each subtopic in the database.  
- **API Flow:** Roughly: 
   1. **User solves a problem on subtopic s:** system computes new baseline, then calls `fsrs.on_review(s, result)` (where `result` encodes correctness). FSRS updates the card’s stability/difficulty. 
   2. **Session scheduling:** Check FSRS for any due subtopic (current date ≥ due date). For each due *s*, either insert a review task into the queue or boost its priority.
   3. **Select next task:** From union of “new topic candidates” (from existing method) and “due review tasks,” pick by priority.  
   4. **User does a review:** mark as reviewed in FSRS. If wrong, FSRS will often schedule the next review sooner (like normal SRS lapse).  

By running FSRS in parallel, we preserve the user’s adaptive difficulty pipeline untouched for introducing new material. FSRS simply *pops up* review tasks at the appropriate time.

## 4. Prototype Scoring and Data Flows

**Prerequisite/Encompass Graph Generation:** To propagate review credit or recommend review content, we need a dependency graph. We can bootstrap this via LLMs:

- **LLM Pairwise Prompts:** As a prototype, we can use an LLM (GPT-4, LLaMA, etc.) to score pairs (A,B) with the question “Is A a prerequisite of B?” or “What fraction of skill A is practiced by solving B?” using a structured JSON output. Yang et al. (2025) demonstrated bidirectional LLM prompting yields reliable prerequisite scores【38†L660-L668】【38†L690-L697】. We could implement a prompt pipeline: for each candidate pair of concepts in a domain, query the LLM and record a score between 0–1. This builds a weighted directed graph. Only high-scoring edges (e.g. >0.7) are kept as likely prerequisites.  
- **Use KG Frameworks:** Alternatively, use KGGen or GraphRAG to ingest course materials. For example, feed lecture notes or textbook chapters to **KGGen**; it will output entities and “subject-predicate-object” triples【5†L313-L322】. We would then filter for relationships of type PREREQUISITE or SUBSUMES (if template exists). Similarly, **GraphRAG** can process PDFs or web content to extract an initial graph. We may need to write custom instructions or templates for “find prerequisite relations between concepts” in these tools.  
- **Existing Datasets as Seed:** We can also seed the graph with known prerequisites from open datasets (e.g. linking “Algebra”→“Arithmetic” etc.), and then use LLM/RAG to expand or verify.  

**Data Flow Prototype:**  
1. **Content Ingestion:** Input course/subtopic descriptions into KG tools (KGGen or Neo4j-Builder). They output a raw KG of concepts/relations. For example, Graph builder (Neo4j) can be scripted via its API: upload a document, get Cypher output.  
2. **Graph Extraction:** Clean the KG to keep only education-relevant relations (Prerequisite, Uses, Is-a-Subskill). Possibly merge synonyms (entity resolution).  
3. **LLM Refinement:** For ambiguous cases, run targeted LLM queries to confirm whether A→B is a true prerequisite or an encompassing relation【42†L135-L143】.  
4. **FSRS and Implicit Reviews:** When a user answers a question on subtopic *B*, besides FSRS update for *B*, look up its neighbors in the KG. For each “encompassed” prerequisite *A* (i.e. A→B edges), apply an FSRS update to *A* with reduced effect. The exact factor could be tuned (Math Academy uses fractional weights). A simple heuristic: treat an advanced correct review as a “Hard” result on each encompassed skill A (giving partial credit) unless the student struggles with *B*, in which case propagate a small penalty upward.  
5. **Execution:** Use an FSRS library (e.g. **py-fsrs**). After each question, call something like `fsrs.review(card_id, outcome)` and retrieve `fsrs.next_interval(card_id)`.  The library will compute and store the next due date. Persist these in your DB.  

## 5. Validation and Rollout Strategy

Finally, plan experiments to validate benefit and fine-tune integration:

- **A/B Testing:** Deploy FSRS-enabled scheduling to a subset of learners or topics. Compare metrics against the baseline system: e.g. retention rates on older topics, number of explicit reviews given, student performance on surprise quizzes of past material, and user satisfaction.  
- **Offline Simulation:** Use historical response data (if available) to simulate what FSRS would have scheduled. Metrics: predicted retention vs actual recalls if we had waited longer, reduction in total reviews, etc.  
- **User Study:** Gather user feedback on forced reviews vs freedom. The literature notes too-rigid prerequisites can frustrate learners【42†L124-L130】, so ensure users can sometimes opt to defer a review or skip if they feel confident.  
- **Iterative Tuning:** The user can still provide feedback per problem (the α factor). We should monitor if FSRS’s intervals align with perceived difficulty. FSRS has an optimizer that can fit to user data if needed【19†L120-L127】, which could further customize the scheduling model.

## Recommended Open-Source Components

- **FSRS Scheduler:** Use the **py-fsrs** (Python) or other FSRS implementations【15†L266-L270】. This gives you a battle-tested scheduling engine and avoids reinventing the complex DSR model【19†L105-L114】【25†L512-L519】.  
- **Graph Construction:** Leverage **KGGen**【5†L313-L322】 or **GraphRAG**【29†L364-L372】 for generic KG extraction, and/or **Neo4j LLM Graph Builder**【27†L298-L304】 if you need a UI-driven setup. These frameworks handle LLM prompting, chunking, and output formatting. For prerequisite edges specifically, combine them with targeted LLM prompts (as in the prompt template in the question text) or the ESCO benchmark approach【38†L660-L668】.  
- **Knowledge Tracing (Optional):** If deeper student modeling is desired later, frameworks like **pyKT** or **GKT** could sit above the KG to track mastery over time. They can also inspire how to weight implicit practice from related items.  

**Building vs. Integrating:** We strongly recommend *integrating existing libraries* rather than coding from scratch. FSRS is open and well-documented【2†L196-L200】【25†L512-L519】. KG tools like KGGen/GraphRAG already implement the LLM pipelines needed. The main new code will glue these together with your database: setting up FSRS state updates, feeding content to the KG tool, and merging outputs into your selection logic. This hybrid approach leverages state-of-the-art open components for maximum gain with minimal custom effort.

