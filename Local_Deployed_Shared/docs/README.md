# docs

## Purpose
- Product and engineering reference docs for Delta Drills, especially ARENA integration, adaptive scheduling, and data-model decisions.

## Owns
- High-level architecture notes.
- Requirements and product-preference docs.
- Migration guidance for curriculum / scheduling changes.

## Does NOT own
- Runtime behavior or source-of-truth implementation logic.
- Backend API contracts that belong in code or schema modules.

## Key Files
- `arena_delta_product_preferences.txt`: product goals for ARENA + Delta Drills.
- `arena_delta_engineering_requirements.txt`: implementation-oriented requirements derived from the product goals.
- `data_architecture_delta_drills.txt`: current adaptive weighting, priority, and difficulty formulas.
- `ewma_now_dag_later_plan.txt`: what metadata and structure must be preserved now so EWMA-based course delivery can evolve into DAG-backed sequencing later.

## Invariants & Constraints
- Docs in this folder should describe canonical product/architecture intent, not temporary one-off hacks.
- When a scheduling or readiness decision becomes structurally important, document it here before it gets scattered across code comments.

## Recent Changes
- 2026-05-13: Added `ewma_now_dag_later_plan.txt` documenting the minimum metadata/schema needed to ship with EWMA now and add DAG sequencing later.
