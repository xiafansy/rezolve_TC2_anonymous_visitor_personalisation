# legacy/ — v1 & v2, kept for the audit trail

Superseded by the v3 unified engine at the repo root. Kept because the
project's story includes how each version's evaluation was falsified and
fixed; see the root README ("How the numbers were corrected").

| file | was | superseded by |
|---|---|---|
| `intent_inference.py` | v1 full-signal scorer (synthetic track) | `intent_engine.py` (prior + window + state evidence) |
| `intent_inference_real.py` | v2 two-stage scorer, browse-pattern rules | same rules live in `intent_engine.score_aggregates()` |
| `build_real_sessions.py` | v2 RetailRocket sessionizer — **whole-session features (leaky)** | root version censors at the first commercial event |
| `evaluate_real.py` / `evaluate_rees46_v2.py` | v2 validity checks on leaky features | root `evaluate_real.py` / `evaluate_rees46.py` (censored + prefix-3) |
| `evaluate.py` | v1 hindsight accuracy vs synthetic labels (97.5% — circular) | `evaluate_synthetic.py` (prefix accuracy, calibration, fitted gate) |
| `personalisation.py`, `demo.py`, `homepage_demo_data.json` | v1 layout mapping + console demo | root `personalisation.py` (stage-aware) + root `demo.py` (journey step-through) |
| `pure_random_dataset.py`, `reasonable_random_dataset.py`, `anonymous_*.csv` | v1 aggregate-row synthetic data | `make_event_dataset.py` (event-level, noisy, Low-intent-heavy) |

Known-wrong numbers that must not be quoted: v1 synthetic accuracy 97.5%
(label circularity); v2 "Evaluator 6.9%/9.7% purchase, hottest segment"
and the v2 REES46 table (13.5% Evaluator / 42.3% Decisive) — inflated by
post-cart feature leakage. Corrected figures are in the root README.
