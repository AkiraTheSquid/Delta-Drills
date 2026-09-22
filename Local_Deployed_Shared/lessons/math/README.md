# Math for ARENA 0.1

Open `index.html?math=rays` from the served app. The seven related coding
lesson pages link to this companion, including lessons reached from the
knowledge graph. Each companion topic links back to its existing KC and to
the ARENA 0.1 notebook. Content lives in `arena-0-1.json`.

This is a compact mathematical bridge to the **core** ray-tracing exercises.
It adds no KC nodes, prerequisite gates, scheduling weights, or mastery
credit. The existing PyTorch/ARENA graph ratio is preserved. A multiple-choice
answer is reasoning practice, not evidence that the learner can implement the
corresponding tensor operation. First-attempt scores last for the current
visit only. The coding ladder remains the implementation assessment.

| Math topic | Existing graph KC | ARENA transfer |
| --- | --- | --- |
| Points, displacement, affine parameters, camera samples | `raytracing.ray-parametrisation` | `make_rays_1d`, `make_rays_2d`, `intersect_ray_1d` |
| Column combinations, equation signs, residual vs membership | `raytracing.segment-intersection` | `intersect_ray_1d` |
| Determinant, singularity convention, validity mask, any over pairs | `raytracing.batched-segments` | `intersect_rays_1d`, batched triangle/mesh solves |
| Barycentric weights, edges, three-unknown system | `raytracing.triangle-intersection` | `triangle_ray_intersects`, `raytrace_triangle` |
| Norm, depth vs parameter, masked minimum | `raytracing.mesh-visibility` | `raytrace_mesh` |

## Scope and sources

Grounded in the user's local `arena-book` checkout:

- `chapter0_fundamentals/exercises/part0_prereqs/0.0_Prerequisites_exercises.ipynb`
  and `solutions.py`: scalar/vector/matrix arithmetic, norm and tensor prerequisites.
- `chapter0_fundamentals/exercises/part1_ray_tracing/0.1_Ray_Tracing_exercises.ipynb`
  and `solutions.py`: the seven core exercises above, their equations and conventions.
- Existing `lessons/pytorch/kp-{ray-parametrisation,segment-intersection,batched-segments,triangle-intersection,mesh-visibility}.md`
  and `lessons/arena_exercise_kcs.json`: live graph ownership and coding alternatives.

The source checkout is `/home/stellar-thread/Applications/Delta-Drills-Local/arena-book`.
Do not infer that the checkout was edited or its notebooks regenerated.

Information theory is intentionally deferred: entropy/KL/cross-entropy are
relevant to probability and loss lessons, not prerequisites of these core 0.1
exercises. Eigenvalues/SVD/general abstract algebra are likewise out of scope.

The notebook's optional video/lighting extensions are not a core gate. Their
additional math is rotation matrices (`RᵀR = I`, row-vectors transform with
`Rᵀ`), surface normals from a cross product, and a normalized dot product
for Lambertian lighting (`max(0, n̂·l̂)`). Reversing edge order reverses the
normal; light-direction convention must be stated. Zero-length normals cannot
be normalized. These deserve a separate optional lesson if bonus rendering
becomes the learning target; they are not silently counted as covered here.

## Authoring and checks

Each section teaches a general rule, shows a runnable concept cell, then one
worked example. Only after Continue does it show one MCQ at a time. No answer
is preselected; explanations appear only after Check answer. Each option has
misconception-specific feedback. Retry retains the original first-attempt score.

Plain Python cells keep mathematical steps visible without adding tensor API
prerequisites. Existing lesson rendering, editor, hidden checks, and execution
harness are reused. Every cell is independently runnable and prints a checked
result; hidden checks never become answer code. Coding lessons provide the
PyTorch transfer.

Run `This-Directory-Only/backend/.venv/bin/python3 scripts/validate_math_companion.py`
for content, graph mapping, source transfer, cell execution, and numerical
answer checks. After editing the linked KP files, also run the existing
lesson validation/compilation gates. No model critic is invoked (user override).
