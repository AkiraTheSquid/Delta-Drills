# Composite-drill audit results (2026-05-25)

LLM-judge audit ran 5 parallel agents over all 180 composites. Each (drill, atom)
pair was checked: "would the test ASSERT-fail if this atom were corrupted?"

## Summary

| part | PASS | WEAK | MISSING | total records |
|------|-----:|-----:|--------:|--------------:|
| part0+part1 | 89 | 33 | 0 | 122 |
| part2 (CNNs) | 59 | 1 | 0 | 60 |
| part3 (optim) | 57 | 4 | 0 | 61 |
| part4 (backprop) | 64 | 1 | 0 | 65 |
| part5 (VAE/GAN) | 56 | 4 | 0 | 60 |
| **Total** | **325** | **43** | **0** | **368** |

- **0 MISSING**: every claimed atom appears in solution code
- **43 WEAK**: test does not actually catch atom corruption (atom passes by accident)

## Concentration

**33 of 43 WEAK in part0+part1.** Two systemic patterns:

1. **`einops-repeat` / `einops-repeat-broadcast` not stride-0-checked.** When
   the test only checks output values, a `.expand()` or full-copy implementation
   passes identically. Fix: add `assert out.stride()[axis] == 0` or
   `assert out.data_ptr() == src.data_ptr()`.
   - Hotspots: part0/{002,005,010,028,029,030}, part1/{008-012,025,026}.
   - **Counter-example** (the model): part1/006, 007, 026 — already enforce stride-0.

2. **`tensor-unbind` / `unbind-tuple-unpack` fungible with slicing.** Tests pass
   with `rays[:,0]` substituted for `t.unbind(rays, dim=1)`. Fix: add
   `assert '.unbind(' in inspect.getsource(cxN_*)` or similar source-grep.
   - Hotspots: part1/{013-018, 023}.

3. **Atom-mismatch fig leafs.** part0/016, 018 list `vector-normalize-keepdim`
   but the operation is sum-/softmax-normalize, not L2. Wrong atom claim, not
   weak test.

## Other parts (10 WEAK total)

Mostly minor:
- part2/029 — `inference-mode-step` (forward under no_grad, not optimizer step decorator)
- part3/005, 006 — `inplace-param-update` not data_ptr-checked (only id())
- part3/015, 016 — `inference-mode-step` numerical-only
- part4/027 — `unbroadcast-pattern` uses already-correct shape
- part5/007, 009 — `encoder-decoder-symmetric` only shape parity, not mirror
- part5/015 — `backward-on-scalar-loss` doesn't check `.grad` populated
- part5/021 — `wandb-log-step` step in dict not kwarg

## Fix strategy (deferred)

1. **part0+part1 stride-0 sweep**: ~12 drills, each needs 1-2 assertion lines.
2. **part0+part1 unbind source-grep**: ~7 drills, each needs an `inspect.getsource` assert.
3. **part0 wrong-atom replacement**: 016, 018 — swap `vector-normalize-keepdim` claim with a more accurate atom (e.g. `sum-and-broadcast-duality`).
4. **Other parts**: 10 targeted patches — each 1-3 lines.

Per-atom verdict JSON files preserved at /tmp/audit_part{01,2,3,4,5}.json
(not committed since /tmp is ephemeral; regenerate by re-running the 5 audit
agents on the same prompts).

## Next-session resume

Run the patch-pass: 5 parallel agents, each takes ~8 drills from the WEAK list
and adds the missing assertions. Verify with the same end-to-end exec harness
after each patch.
