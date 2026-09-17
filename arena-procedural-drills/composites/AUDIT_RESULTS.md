# Composite-drill audit results (2026-05-25 + patch pass 2026-05-26)

LLM-judge audit ran 5 parallel agents over all 180 composites. Each (drill, atom)
pair was checked: "would the test ASSERT-fail if this atom were corrupted?"

A follow-up patch pass added targeted assertions for every WEAK record. Final
state: **0 MISSING, 0 unpatched WEAK, 180/180 PASS** end-to-end exec.

## Initial audit (2026-05-25)

| part | PASS | WEAK | MISSING | total records |
|------|-----:|-----:|--------:|--------------:|
| part0+part1 | 89 | 33 | 0 | 122 |
| part2 (CNNs) | 59 | 1 | 0 | 60 |
| part3 (optim) | 57 | 4 | 0 | 61 |
| part4 (backprop) | 64 | 1 | 0 | 65 |
| part5 (VAE/GAN) | 56 | 4 | 0 | 60 |
| **Total** | **325** | **43** | **0** | **368** |

- **0 MISSING**: every claimed atom appears in solution code
- **43 WEAK**: test passes by accident (atom appears but corruption goes undetected)

## Patch pass (2026-05-26)

5 parallel patch agents took 8-9 drills each. All 39 affected drills patched; 180/180
PASS post-patch; mutation tests in agent A's sweep caught 8/8 corrupted-solution
attempts.

### Patch patterns applied

| pattern | scope | mechanism |
|---|---|---|
| **stride-0 spy** | 12 drills (part0/part1 einops-repeat) | monkey-patch `einops.repeat` to capture outputs, assert `0 in r.stride()` (broadcast view, not a copy). Combined with `inspect.getsource` forbidden-token list (.expand, broadcast_to, full-copy). |
| **source-grep `.unbind(`** | 7 drills (part1 unbind) | `assert '.unbind(' in inspect.getsource(cxN_*)` rules out `rays[:,0]` slicing shortcut. |
| **tuple-unpack regex** | 3 drills | `^\s*\w+\s*,\s*\w+\s*=` source-grep enforces destructure pattern. |
| **`data_ptr()` preservation** | 2 drills (part3 inplace-param-update) | `assert p.data.data_ptr() == ptr_before` rules out out-of-place reallocation. |
| **`no_grad`/`inference_mode` source-grep** | 2 drills (part3 inference-mode-step) | numerical-equality alone doesn't enforce; source-grep does. |
| **channel-mirror extraction** | 2 drills (part5 encoder-decoder-symmetric) | extract Conv2d in/out_channels, assert `dec_in == reversed(enc_out)`. |
| **ray-parametric eval** | 2 drills (part1 ray-parametric-form) | pick solved t, compute O+t*D, assert equals known intersection. |
| **barycentric inside-test** | 2 drills (part1 triangle-barycentric) | `(u>=0)&(v>=0)&(u+v<=1)` on inside + outside rays. |
| **unbroadcast peel case** | 1 drill (part4/027) | added `(5,3,4)→(5,1,4)` case forcing axis collapse. |
| **backward grad mutation** | 1 drill (part5/015) | snapshot params; assert mutated post-step. |
| **wandb step= kwarg semantics** | 1 drill (part5/021) | reject step-in-dict-payload, require `kwargs.get('step')`. |
| **wrong-atom metadata swap** | 4 drills | edit `metadata.delta_drills.atom_ids` + subtopics + primary_atom; catalog auto-regenerated. Swaps: vector-normalize-keepdim → broadcasting-rules (×2 in part0/016, /018); inference-mode-step → batchnorm-running-stats (part2/029); kept einops-rearrange via source-grep (part0/024). |

### Mid-stream fix: module-scope hoist (8 drills)

The first patch pass placed source-grep asserts at module-level — they ran
BEFORE `solution_body` exec'd (when only the stub raising NotImplementedError
was defined). All 8 affected drills (part1/012-018, 025) were patched by
hoisting the asserts inside `_test_cx<N>` body. Now they run AFTER solution
overrides the stub.

## Open audit follow-ups (none blocking)

- Re-run the audit periodically as new drills land (e.g. after future
  recommender-driven authoring waves).
- Consider auto-generating audit verdicts via a `mod`-style daemon rather
  than ad-hoc LLM judges.
