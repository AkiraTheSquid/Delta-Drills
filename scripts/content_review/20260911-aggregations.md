# Aggregation replenishment — author self-review

Scope: `numpy.aggregations`, prompted by an exhausted-rung fallback to a
numeric-ranges question. The learner's production history showed recent
`partial` (Solo) attempts on q497, q498 and q987–989.

Added q1313–1332: 12 Solo and 8 Integrated. Pool sizes are now 18 Solo and
11 Integrated. Existing rung architecture and Faded items are unchanged.

## Content decisions

- Solo moves from basic whole-tensor arithmetic/counting into data-dependent
  thresholds, error measures and RMS. Inputs span vectors, matrices and 3D
  tensors; the operation remains global, without a newly introduced axis API.
- Integrated tasks combine reductions with elementwise operations, weighting,
  shape-independent counting, normalization or scalar boundary handling.
  Weighted averages, unequal-batch pooling, calibration, best-fit scale and
  offset, weighted variance and tolerance reports require additional decisions.
- Prompts state input/output contracts and define unfamiliar statistical
  quantities. They contain no API prescriptions, algorithm steps or worked
  solutions. Starters contain only the import and an empty function.
- Four cases per problem include ties, exact boundaries, singleton inputs,
  mixed signs, unequal sizes or rescaled weights as appropriate. Expected
  values are independently authored literals, not computed from the solution.
- Hidden `assert_code` checks enforce Python scalar types and, where promised,
  unchanged input tensors. Visible calls remain ordinary `solve(...)` calls.

## Graph corrections required by these drills

The aggregation lesson already demonstrated comparisons before the masking
lesson, but the syntax declaration belonged to masking. Moved ownership to
aggregations and explicitly explained strict/inclusive comparisons there.
Masking already depends on aggregations. No prerequisite exemption was added.

The atom graph contained `tensor-item-scalar` with no incoming edge or declared
root status. These drills now provide direct evidence for that existing atom.
Declared it an intentional tensor-literacy root, matching the atom graph's
separation from the Python course. The KC graph retains the full prerequisite
chain; no new Python-to-tensor atom gate is introduced.

## Validation

`scripts/test_aggregation_drills.py` runs every reference through the production
grader and rejects the starter, a constant copied from the first expected
value, an identity return, a tensor-return mistake and one authored near miss
for each problem. It also checks rung registration, prerequisite syntax,
prompt/starter leaks, hidden-assertion parity across all four graders, and
input mutation before fixture reset. Mechanical leak detection supplements
the prose review above; it does not certify arbitrary prose.

These exercises are preparation variants, not claims of complete ARENA
chapter coverage. This change does not retune promotion or scheduling.
