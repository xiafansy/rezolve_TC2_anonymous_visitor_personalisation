# Business case: what the intent model is for

*Response to Imran's "what does this bring the business?" — one page, numbers
from held-out, censored validation (see README for methodology).*

## The problem in money terms

An anonymous visitor gets the same homepage as everyone else, so acquisition
spend and page real-estate are allocated blind. The intent model's value is
**knowing which 20% of anonymous traffic is worth acting on, within 3 clicks,
and what to do for each segment** — without identity, history, or lookalikes.

## What the model concentrates

| dataset (held-out) | segment | traffic share | share of ALL purchases |
|---|---|---|---|
| REES46 (Nov) | Decisive | 7.7% | **55%** |
| REES46 (Nov) | Decisive + Evaluator | 20.1% | **69%** |
| RetailRocket (final month) | Decisive | 1.5% | **54%** |
| RetailRocket (final month) | Decisive + Evaluator | 3.3% | 60% |

Two-thirds of purchases sit in a fifth of sessions — and the model finds them
from browse-pattern signals alone, before checkout.

Against the obvious rivals (`reports/baseline-comparison.md`): the engine's
top-5% of RetailRocket traffic captures **68%** of purchases vs 16% for an
engagement-count heuristic and 5% for random targeting; on REES46 the
engagement heuristic is actually *worse than random* (fast decisive buyers
emit few events), which is exactly why a traffic heuristic can't substitute
for intent. Module-level: serving Evaluators a recently-viewed rail matches
their actual next behaviour **81%** of the time vs **6%** for a
popularity-for-everyone rail.

## What to do per segment, and the KPI it should move

| segment | action on the page | primary KPI (A/B) | guardrail |
|---|---|---|---|
| Decisive (buy rate 28–40%) | checkout support: sticky cart, one-tap pay, shipping reassurance; suppress discovery | checkout completion, time-to-purchase | refund/cancel rate |
| Evaluator (6%+, hottest browse segment) | comparison table, reviews up front, recently-viewed rail, clear CTA | add-to-cart rate | bounce |
| Explorer | cross-category trending, curated collections, save-for-later | pages/session, saves, return-visit conversion | exit rate |
| Low-intent (60–93% of traffic) | neutral-light: fast page, top categories, one promo | bounce no worse, page-weight budget | conversion no worse |
| Unclear (gate) | neutral balanced page | — (safety valve; monitor its share) | — |
| *price-conscious flavor* | value-first sorting, sale rail, budget picks | flagged-session conversion (offline: +0.3 to +4.5pp within every intent) | margin mix |

## What we are NOT claiming (and how to close the gap)

Offline validation proves the model **separates** conversion propensity; it
cannot prove the personalised page **causes** more conversion. Closing that
requires a standard A/B:

- **Arms:** control (current homepage) vs intent-personalised (this engine).
- **Unit:** anonymous session; assignment at first pageview; engine decides
  from click 3 (and instantly on the Decisive trigger).
- **Primary metric:** session conversion; secondary: cart rate, AOV;
  guardrails: bounce rate, page latency, Unclear share.
- **Sample size** (80% power, +5% relative on conversion): ~110k sessions/arm
  on a REES46-like base (5.6%), ~630k/arm on a RetailRocket-like base (1%).
  Days, not months, at these sites' traffic volumes.

## Why this and not a bigger model (today)

The rule engine ships with reasons attached to every decision, needs no
training data a cold-start site doesn't have, and its weights are now
measured, ablated and idle-capped on two retailers. Transformers / contextual
bandits are the roadmap once first-party traffic exists — the current
architecture (intent posterior = context, layouts = arms, conversion =
reward) is deliberately bandit-shaped so that upgrade swaps the weighting,
not the system.
