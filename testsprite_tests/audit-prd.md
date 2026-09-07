# Delta Drills audit requirements

Delta Drills teaches ARENA prerequisite knowledge through placement, knowledge graphs, practice problems, lessons, and notebook exercises. This audit prioritizes functional behavior over cosmetic suggestions.

## Test environment

- Frontend: http://localhost:5173, serving the deployed worktree's Local_Deployed_Shared assets.
- Backend: current main worktree's FastAPI backend, proxied on the same origin. It uses a dedicated PostgreSQL test DB and isolated progress files. Production learner data is not used.
- Guest sessions are supported: allow guest-session provisioning to finish. No Google login is needed. Fresh browser contexts should receive fresh guest identities.
- All browser API calls must use the page origin via its configured api_base. Never switch to the production Fly API.
- Backend endpoints include /health, /auth/signup, /auth/login, /api/practice/diagnostic/status, /api/practice/diagnostic/plan, /api/practice/diagnostic/start, /api/practice/diagnostic/answer, and normal practice routes.

## Priority placement requirements

1. /diagnostic opens a visible placement page. A fresh learner can choose 1, 3, or 6 hours and start that exact plan. The test seeds practice estimates; it never awards mastery.
2. Six-hour placement audits all 44 enabled knowledge components (KCs), including ARENA-linked exercises, and presents coverage and uncertainty honestly. Twenty minutes per problem, six-hour total cap; shorter modes remain available.
3. An existing active legacy placement with zero answers and no plan must not trap the learner at "0 of at most 15" with no way to select the new six-hour plan. Preserve answered work; restarting an answered run must be deliberate.
4. Load next placement question displays an actual problem with its prompt and appropriate answer controls. Questions do not disappear during tab navigation, feedback, or reload.
5. "I don't know yet" records the currently served probe once, advances progress, and loads another usable problem. A reload should preserve an outstanding probe and its time budget.
6. Stopping or completing placement displays results on the placement page. Estimates, graph coverage, uncertain concepts, and edge contradictions must not be mislabeled as earned mastery.

## Broader functional coverage

- Welcome flow and automatic guest provisioning; no dead-end sign-in requirement for guest-supported learning features.
- Account-menu navigation to placement, knowledge graph, courses, and settings.
- Knowledge graph renders concepts and their prerequisite relationships; settings and disabled concepts affect diagnostic scope.
- Practice can display and execute a basic Python problem. Correct, incorrect, and skipped responses receive coherent feedback.
- ARENA course/exercise links and their mapped problems render. External Colab/Google pages are outside this local test environment; distinguish external-service limitations from app defects.
- Backend health reports 44 registered KCs with no unmapped KCs. Diagnostic plan/start/status contracts agree about hours, budget, coverage, and progress.
- Invalid or duplicate diagnostic answers must not create free evidence or consume time twice; non-current question IDs must be rejected.
- Mobile-width placement controls and menus remain reachable without blocking overlays or horizontal clipping.

Use at most 10 high-priority frontend tests initially. Include API assertions within those tests where needed. Do not run a literal six-hour test: verify the selected budget and use a small number of controlled interactions. Report precise reproduction steps, expected/actual results, browser console errors, and failing request details. Do not alter real accounts, send invitations, delete data, or deploy changes.
