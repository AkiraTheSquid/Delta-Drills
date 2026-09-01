---
kc: numpy.reshape-flatten
---
## Findings
- 2026-09-01: 43-attempt `faded` stall at 14% accuracy in the 08-19→27 logs;
  the low-p run inside it scored 1/26 (mean predicted_p 0.237). Same root
  cause as [[numpy.ndarray-model]] (`a.T` untaught); kept as a separate note
  because the stall metric reports per concept and both fired independently.

## Checks
- `app/kc_stats.py` flags.
