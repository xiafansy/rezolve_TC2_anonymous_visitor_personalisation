# TC2 · Anonymous-Visitor Personalisation

**Can we personalise the home page for an anonymous, first-time visitor — without
lookalike modelling?** This prototype infers *genuine, real-time intent* from the
signals a single session actually emits (referral, device, on-page behaviour,
browsing sequence) instead of asking "which known cohort does this stranger
resemble?".

## Approach

```
session signals ──> evidence-weighted intent scorer ──> homepage strategy ──> demo
 (observable        (transparent rules + confidence      (one layout per        (HTML)
  only)              gate -> NEUTRAL when unsure)         intent)
```

Design choices that define the project:

- **Rule-based scorer as the shipping logic, not ML.** A real cold-start has no
  labels to train on. A weighted-evidence scorer needs zero history, runs
  per-event in real time, and can *explain every verdict* — each inference
  carries its human-readable reasons.
- **A confidence gate that declines to personalise.** When evidence conflicts,
  the system says "Unclear" and serves a neutral page. Not personalising is a
  deliberate output, not a failure mode.
- **Honest evaluation.** The scorer never sees ground-truth labels or
  conversion outcomes; those are used only to validate it afterwards.

## Two tracks, two datasets

| | Track 1 · Synthetic | Track 2 · Real (RetailRocket) |
|---|---|---|
| data | `reasonable_random_dataset.py` → `anonymous_sessions.csv` (1,000 sessions, intent-labelled by construction) | Kaggle RetailRocket event log → `build_real_sessions.py` → 1.76M sessionized visits |
| what it proves | the full-signal pipeline end-to-end (referrer / device / search / sort / behaviour) | that intent archetypes exist in real traffic and are *predictive* |
| evaluation | accuracy vs known labels (`evaluate.py`) — with the caveat that the generator's rules are being partially inverted | temporal hold-out: thresholds fitted on months 1–3.5, validated on the final month (`evaluate_real.py`) |
| limit | circularity: synthetic labels came from rules we wrote | no referrer / device / search / price signals in the log |

The tracks are complementary: Track 1 demonstrates the architecture with every
signal the brief names; Track 2 replaces trust-me with evidence.

## What the real data taught us (findings)

1. **~75% of real sessions contain a single event; only ~10% have ≥3.**
   Behavioural inference can only ever serve the engaged tail — the neutral
   fallback plus acquisition context must carry the rest. (The engaged tail is
   worth it: it converts 9× the base rate.)
2. **Re-viewing the same item is the golden intent signal.** Sessions that
   re-view items ≥1.8× buy at ~10% vs ~4% below — and "focused on one category
   but never re-viewed an item" collapses to **1.0%** (a drive-by scan, not
   research). Our synthetic assumption "fast + focused = decisive buyer" was
   **wrong** on real traffic (those sessions buy at 1.2%).
3. **True decisive buyers are commercial-event-led**, not fast browsers: cart
   with ≤2 prior views → 75% purchase in-session. That's an event-triggered
   *mode switch* (Stage B), not a prediction.
4. **Time-of-day carries weak real signal** (evening ≈ 2× morning purchase
   rate) — usable as a soft prior, never as evidence of intent.

### Held-out validation (final month, never seen during tuning)

Stage-A intents are inferred from view-patterns only; conversion below is an
outcome the scorer never saw. Base purchase rate: **0.81%**.

| intent | coverage | cart rate | purchase rate |
|---|---|---|---|
| Decisive (Stage B) | 1.6% | 100%* | **22.8%** |
| Evaluator | 2.0% | 15.2% | **6.9%** |
| Explorer | 3.0% | 12.8% | **5.8%** |
| Unclear → neutral | 0.6% | 14.6% | 4.1% |
| Low-intent | 92.8% | 0.1% | 0.1% |

\* cart=100% by construction — Stage B *is* the add-to-cart trigger.
Monotonicity Evaluator > Explorer > Low-intent: **PASS**.

## Intent → homepage strategy

| intent | signature (real signals) | homepage serves |
|---|---|---|
| Decisive | cart with minimal prior browsing | checkout support: sticky cart, one-tap pay, shipping reassurance |
| Evaluator | re-views same items, single category, deliberate pace | comparison: specs side-by-side, reviews, recently-viewed rail, clear CTA |
| Explorer | ≥3 categories, high switch rate, no re-views | discovery: cross-category trending, curated collections |
| Low-intent / micro-visit | ≤2 events, gone in seconds | neutral-light: fast page, top categories, one broad promo |
| Unclear | conflicting evidence | neutral: balanced page; commit to nothing |
| Price-sensitive | *(Track 1 only — needs search/sort/price signals the real log lacks)* | sale rail, price-low sort default |

## Files

```
Track 1 (synthetic, full signal set)
  reasonable_random_dataset.py   generator: intent -> plausible signals
  intent_inference.py            v1 evidence scorer (referrer/device/search/behaviour)
  personalisation.py             intent -> homepage layout decision
  evaluate.py                    accuracy / confusion / per-class PR vs labels
  demo.py                        end-to-end walkthrough; exports homepage_demo_data.json
  personalisation_demo.html      visual demo (open in browser)

Track 2 (real, RetailRocket)
  build_real_sessions.py         event log -> 1.76M sessions with sequence features
  intent_inference_real.py       v2 two-stage scorer (browse-pattern + cart-trigger)
  evaluate_real.py               temporal hold-out validation (tables above)
```

### Data setup (Track 2)

Raw datasets are git-ignored (GitHub's 100MB/file limit; the sources below are
the canonical hosts). The repo carries the recipe, not the data.

**RetailRocket** — download from
[Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
and unzip into `archive/` (~900MB):
`events.csv`, `item_properties_part1/2.csv`, `category_tree.csv`. Then:

```
/usr/bin/python3 build_real_sessions.py    # writes real_sessions.csv (~134MB)
/usr/bin/python3 evaluate_real.py          # temporal hold-out validation
```

**REES46 multi-category store** — richer signals the RetailRocket log lacks:
real prices (unlocks the Price-sensitive archetype), readable category codes,
brands, and built-in session ids. Direct download from the publisher
([also on Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store))
into `archive_rees46/`:

```
mkdir -p archive_rees46 && cd archive_rees46
curl -LO https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz   # 1.7GB
curl -LO https://data.rees46.com/datasets/marketplace/2019-Nov.csv.gz   # 2.9GB
```

Columns: `event_time, event_type(view|cart|remove_from_cart|purchase),
product_id, category_id, category_code, brand, price, user_id, user_session`.

## Honest limitations

- Real-data features are session-level aggregates — an offline approximation of
  the incremental real-time computation (the realtime variant is Track 1's demo).
- Price-sensitivity is not observable in the RetailRocket log; it lives in the
  synthetic track only.
- Track 1 accuracy overstates what production would see (label circularity);
  Track 2's held-out lift is the number to trust.
- Hour-of-day timezone in the log is unknown; only relative differences used.
