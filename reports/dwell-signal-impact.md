# Dwell-time signal: measurement, changes, impact

*Response to Sauvik's request: add a 60–120s dwell band (PDP dwell) and test
its impact on the existing signals. Method: measure first on censored
train-window data (`measure_dwell.py`), derive weights from what the data
says, then re-run every held-out evaluation before/after.*

## 1 · What the data said

Two real retailers were measured; they **disagree about dwell direction and
agree about one thing — the idle cliff.**

**Per-page dwell (median inter-event gap, engaged sessions, buy-rate):**

| gap band | RetailRocket | REES46 |
|---|---|---|
| 15–60s | 1.76% | 5.68% |
| **60–120s** | **2.37%** | 5.42% |
| 120–300s | **3.02%** (peak) | 5.61% |
| 300–600s | 2.95% | 5.60% |
| **>600s** | **2.56% ↓** | **3.48% ↓↓** |

- RetailRocket: dwell is a real intent signal (rises ~2× into 120–300s), and
  it survives conditioning on the re-view rule (so it adds information,
  not shadow).
- REES46: flat 60–600s — dwell neither helps nor hurts there.
- **Both sites: credit must stop past ~600s.** That long between actions is a
  parked tab, not reading. The old engine granted unbounded Evaluator credit
  for any gap ≥ 60s/120s — a genuine bug this work fixes.

**Micro-visits (2 pre-commercial events, buy-rate by session span):**

| span | RetailRocket | REES46 |
|---|---|---|
| <15s | 0.87% | 8.58% |
| 60–120s | 1.06% | 6.10% |
| 120–300s | 1.30% | 5.65% |

Opposite directions: on a classic retailer, a lingering two-event visit is
warmer than a bounce; on the fast marketplace, the hottest micro-visits are
the *fastest* (decisive buyers racing to cart — whom the Decisive override
catches at the very next event anyway). Conclusion: **the 60–120s micro band
is a site-dependent signal** and gets a deliberately small weight.

## 2 · What changed in the engine

| change | weight | grounds |
|---|---|---|
| PDP dwell bands: gap 60–120s → Evaluator | +0.8 | "read the page" (RR: 2.37%) |
| gap 120–600s → Evaluator | +1.0 | RR peak band (3.02%) |
| **gap ≥ 600s → no dwell credit** (was: credit, unbounded) | 0 | idle cliff on both sites |
| micro-visit span 60–600s → Evaluator | +0.3 | Sauvik's band; small because site-dependent — enough to combine with an arrival prior and reach neutral-via-Unclear, never enough to out-vote behaviour |
| pace average: each gap capped at 600s | — | one overnight tab must not dominate `avg_gap` |
| offline mirror (`score_aggregates`): pace credit widened 120s→60s entry, idle-capped at 600s | +0.5 | RR band C (dwell beyond re-views) |

Considered and rejected: an *independent* pace→Evaluator credit for
no-revisit sessions (the measured lift exists on RR but this exact pattern —
engagement-shaped evidence without item engagement — is how the v2 dilution
bug happened; it stays an amplifier).

## 3 · Impact on existing signals (all held-out, censored)

**Synthetic decision-time accuracy — unchanged within noise, coverage up:**

| checkpoint | before | after |
|---|---|---|
| after 3 events (decided) | 83.0% | 82.7% |
| end of session (decided) | 86.3% | 85.9% |
| unclear @ k=2 | 1% | 0% |
| ECE | 0.120 | 0.121 |
| fitted gate | 0.43 | 0.41 |

**RetailRocket (final-month hold-out):** Evaluator purchase 2.82% → **2.86%**,
all other segments unchanged (Decisive 28.5%, Explorer 3.51%, Low-intent
0.24%). Prefix-3 agreement stays 99%.

**REES46 (Black-Friday hold-out):** stable — Evaluator 6.17% → 6.15% purchase
with coverage up 12.3% → 12.4% (the widened pace credit moved ~1.3k sessions
from Unclear into Evaluator without diluting it). Monotonicity still PASS
(6.15% > 2.23% > 1.89%), Decisive 40.2%, price-flavor lifts unchanged,
prefix-3 agreement 89%.

## 4 · Takeaways

1. Sauvik's 60–120s band is in, with the weight the data supports — modest as
   a positive signal, and site-dependent (strong on RetailRocket-like sites,
   flat on the REES46 marketplace).
2. The measurement surfaced a better universal rule than the one requested:
   **the 600s idle cap** — the only dwell finding both retailers agree on,
   and it fixed an unbounded-credit bug.
3. Existing signals are unharmed (tables above) — the promised impact test.
4. Weight-validation machinery note: every number here comes from censored
   features on held-out windows; next step on this thread is the ablation
   harness (each rule's marginal contribution), per Sauvik's "keep validating
   the weights".
