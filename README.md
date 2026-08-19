# TC2 · Anonymous-Visitor Personalisation

**Can we personalise the home page for an anonymous, first-time visitor — without
lookalike modelling?** This prototype infers real-time intent from the signals a
single session emits (arrival context, on-page behaviour, browsing sequence) and
maps it to a homepage — with a calibrated option to *decline* and stay neutral.

> **v3.1** is a correctness pass over v3: four defects that made the code
> disagree with its own docstrings, a 102-test suite that would have caught
> them, and one-command reproduction. No rule, weight or threshold changed.
> See [CHANGELOG.md](CHANGELOG.md), Both real-data tables below are regenerated on the fixed engine.

## Architecture — one two-stage engine (`intent_engine.py`)

```
STAGE 1 · COLD START (event 0)           STAGE 2 · BEHAVIOURAL (every click)
referrer / device / landing / hour   +   newest actions (decayed window, last
        = soft PRIOR                     2-3 dominate) + session shape (state)
                     \                  /
                      posterior = softmax((prior + window + state) / T)
                                   |
              +--------------------+--------------------------+
              | conf >= gate : PERSONALISED intent layout      |
              | conf <  gate : NEUTRAL (declining is a choice) |
              | cart with <=2 prior views : DECISIVE override  |
              +-----------------------------------------------+
```

- A prior may *tilt* the neutral page (one accented block); it may not commit.
- Temperature and gate are **fitted** (NLL grid / target-precision search) on a
  fit split — never hand-picked, never tuned on reported test data.
- The Decisive override is a mode switch on a commercial trigger, not a
  prediction.
- Intents: Goal-driven · Evaluator · Explorer · Price-sensitive · Low-intent
  (+ `Unclear` from the gate, + `Decisive` override). Real logs without
  search/referrer signals can observe only Evaluator / Explorer / Low-intent,
  through the same engine (`score_aggregates()`).
- **Price sensitivity is a flavor, not a fifth real-data intent**: on real
  prices, cheap-leaning browsing (`price_rel_cat < 0.6`) lifts conversion
  *within every* intent, so it tilts merchandising while browse shape picks
  the layout.

## Three datasets, one honesty rule

The scorer never sees labels or outcomes; they are used only to validate.

| track | data | what it establishes |
|---|---|---|
| synthetic | `make_event_dataset.py` — 3k event-level sessions, 45% Low-intent, noisy archetypes | full-signal engine end-to-end; decision-time accuracy |
| real #1 | RetailRocket (Kaggle), 2.76M events → 1.76M sessions | behavioural intents separate real conversion; temporal hold-out |
| real #2 | REES46 (4.6GB, real prices), 110M events → 23M sessions | cross-retailer transfer with zero re-tuning; price flavor; Black-Friday robustness |

## How the numbers were corrected (read this before quoting any)

This project's evaluation was falsified and rebuilt three times — the audit
trail is the methodology story:

1. **v1 (circular):** 97.5% accuracy on synthetic data whose labels came from
   the same hand-written rules. Retired.
2. **v2 (leaky):** session-level features included behaviour *after* the first
   add-to-cart. Buyers re-open product pages during checkout, so re-viewing
   "predicted" buying partly because buying caused re-viewing. Quantified on
   RetailRocket's held-out month: Evaluator purchase 6.9% → **2.8%** once
   censored; the revisit-band gradient 3.4→24% collapsed to ≈2→3%; the claim
   "Evaluator out-converts Explorer" flipped to FAIL on this site.
3. **v3 (censored):** all behavioural features are **censored at the first
   commercial event** (`build_real_sessions.py`, `build_rees46_sessions.py`),
   the Decisive trigger uses `views_before_first_commercial`, and every table
   is decision-time honest. Superseded code and the exact wrong numbers live
   in [`legacy/`](legacy/README.md).
4. **v3.1 (this pass):** the censoring assertion that backed step 3 was
   `a <= b + c` on non-negative numbers — it could not fail. Rewritten to
   recount the log directly. Three silent defects fell out with it: cart-first
   sessions never reached the Decisive override; the "ungated" calibration
   pass was gated at 0.45 and fitting the gate against its own output; and at
   arrival the engine told callers to serve a committed intent page while the
   renderer drew a neutral one.

## Results

### Synthetic track — decision-time accuracy

Test split, fitted **T=1.0, gate=0.35** (`engine_params.json`, regenerate with
`python evaluate_synthetic.py`):

| after event | decided | acc (decided) | acc (all) | override |
|---|---|---|---|---|
| 0 (arrival) | 89% | 47% | 42% | 0% |
| 2 | 100% | 56% | 56% | 1.1% |
| 3 | 95% | **81%** | 75% | 1.9% |
| end | 97% | **84.5%** | 81% | 1.9% |

Confidence is calibrated (ECE 0.119; mean 84% when correct vs 69% when wrong).
Remaining Price-sensitive ↔ Evaluator confusion is genuine overlap — tuning it
away would recreate v1's circularity. Decisive override: 1.9% of sessions, 70%
purchase vs 11% base.

*(v3 reported gate 0.43 and 86%. The gate was being fitted against its own
gated output; 0.35 / 84.5% is what an honest ungated fit split gives.)*

### RetailRocket — held-out final month, censored features (base 0.81%)

| intent | coverage | purchase rate |
|---|---|---|
| Decisive (override) | 2.1% | **24.9%** |
| Evaluator | 1.8% | 2.8% |
| Explorer | 2.0% | 3.5% |
| Unclear → neutral | 1.8% | 1.5% |
| Low-intent | 92.3% | 0.15% |

What survives censoring here: the engaged-vs-micro split (~20×) and the
Decisive trigger. Within engaged browse patterns the differences are modest
and their ordering is **not stable on this site** — an honest negative result.
Prefix-3 agrees with full-session calls 99% of the time.

### REES46 — November (Black Friday) hold-out, censored, zero re-tuning (base 5.64%)

| intent | coverage | purchase rate | cheap-flavor lift |
|---|---|---|---|
| Decisive (override) | 7.9% | **40.3%** | +4.4pp |
| Evaluator | 12.3% | **6.2%** | +1.6pp |
| Explorer | 8.6% | 2.2% | +1.1pp |
| Unclear → neutral | 10.6% | 4.1% | +1.0pp |
| Low-intent | 60.6% | 1.8% | +0.4pp |

Monotonicity Evaluator > Explorer > Low-intent: **PASS** (also on prefix-3).
Censored revisit bands rise 3.0% → 10.0% (Oct) — re-viewing is a real but
site-dependent signal (~2-3× here, flat on RetailRocket; the leaky 8× was an
artifact). The price flavor lifts conversion within every intent on clean
features. Prefix-3 vs full-session agreement: 89%.

### ✔ Regenerated after the cart-first override fix

Both real-data tables above were re-run on the licensed archives after the
v3.1 override fix. Direction exactly as predicted: cart-first sessions moved
out of Low-intent into Decisive (RetailRocket Decisive coverage 1.5%→2.1%,
Low-intent floor 0.24%→0.15%; REES46 7.7%→7.9% and 1.9%→1.8%), with
Evaluator / Explorer / Unclear byte-identical. The synthetic table is
likewise from the fixed engine.

## What the real data taught us

1. **~75% of real sessions contain a single event; only ~10% have ≥3.**
   Behavioural inference can only ever serve the engaged tail — the neutral
   fallback plus acquisition context must carry the rest. The tail is worth
   it: it converts roughly 10× the base rate.
2. **Re-viewing the same item is the strongest browse-shape signal — but it is
   site-dependent.** ~2-3× on REES46, essentially flat on RetailRocket once
   censored. The leaky 8× gradient was an artifact of buyers re-opening pages
   during checkout. Publish it as a site-fitted signal, never a universal law.
3. **True decisive buyers are commercial-event-led, not fast browsers.** Cart
   with ≤2 prior views converts 28% (RetailRocket) to 40% (REES46) in-session.
   That is an event-triggered *mode switch*, not a prediction — which is
   exactly why it must fire on a real zero as well as a one or a two.
4. **"Fast + focused = decisive buyer" was wrong.** Our synthetic assumption
   inverted on real traffic: focused-on-one-category-but-never-re-viewed is a
   drive-by scan, and converts *below* base rate. The rule now scores it
   Low-intent.
5. **Time-of-day carries weak real signal** (evening ≈ 2× morning purchase
   rate) — usable as a soft prior, never as evidence of intent.

### What a stakeholder should take away

- Behavioural separation from **3 clicks**, no identity, no lookalikes:
  0.15–1.8% (low-intent floor) vs 2–6% (engaged intents) vs 25–40% (Decisive).
- The engine **knows when not to personalise** (fitted gate; Unclear → neutral).
- Every decision ships with its reasons (auditable evidence lists).
- Proving personalisation *lifts* conversion (vs merely separating it) needs
  an online A/B — outside this repo's reach.

## Intent → homepage (`personalisation.py`)

| serve mode | when | page |
|---|---|---|
| cold-accent | arrival, prior only | neutral base + ONE block tilted to the likely substantive intent |
| personalised | gate cleared | full intent layout (comparison / discovery / mission / value) |
| checkout | Decisive override | sticky cart, one-tap checkout, reassurance; suppress discovery |
| neutral / neutral-light | Unclear / Low-intent | balanced fast page, commit to nothing |
| *+ price-conscious flavor* | `price_rel_cat < 0.6` | value-first sorting, sale rail, budget picks on any layout |

## Files

```
intent_engine.py             two-stage engine + calibration (fit_temperature,
                             fit_gate, ECE) + score_aggregates() for real logs
personalisation.py           Inference -> stage-aware homepage (auditable)
make_event_dataset.py        event-level synthetic generator (seeded)
evaluate_synthetic.py        prefix accuracy, confusion, calibration, gate fit
                             -> writes engine_params.json
build_real_sessions.py       RetailRocket sessionizer, CENSORED + prefix-3
build_rees46_sessions.py     REES46 aggregator, CENSORED + prices + prefix-3
evaluate_real.py             RetailRocket temporal hold-out (censored + p3)
evaluate_rees46.py           REES46 cross-dataset hold-out + price flavor
demo.py -> demo.html         3 journeys, click-by-click homepage evolution
personalisation_demo.html    interactive full-signal mockup (open directly)

measure_dwell.py             dwell-band conversion measurement (weights come from here)
evaluate_ablation.py         per-rule knockout + weight sensitivity
evaluate_baselines.py        vs popularity / engagement-heuristic / random
evaluate_learned.py          LR + GBDT headroom check on identical features
ga4_prior_calibration.py     cold-start prior calibrated on real GA4 context

run_all.py                   everything that needs no download, one command
tests/                       102 unit tests (engine, layouts, censoring)
smoke_test_real_pipeline.py  end-to-end on fabricated data; asserts censoring
reports/                     meeting-ready findings (dwell impact, ablation,
                             baselines, business case, learned-weights headroom)
legacy/                      v1/v2 + the retired numbers (see its README)
```

## Run it

```bash
pip install -r requirements.txt
python run_all.py
```

That runs the unit tests, generates the synthetic events, fits temperature and
the gate, writes `demo.html`, and runs the real-pipeline smoke test on
fabricated data — no download at any point (~5s end to end).
`python run_all.py --quick` is the tests plus the smoke test. Individually:

```bash
python -m pytest -q                    # 102 unit tests
python smoke_test_real_pipeline.py     # real pipeline, fabricated data, asserts censoring
python make_event_dataset.py && python evaluate_synthetic.py
python demo.py                         # writes demo.html
```

### Real datasets (git-ignored; repo ships the recipe, not the data)

**RetailRocket** — [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
→ unzip into `archive/` (~900MB), then:

```bash
python build_real_sessions.py --archive archive --out real_sessions.csv
python evaluate_real.py --data real_sessions.csv
```

**REES46** — direct from the publisher
([also on Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)):

```bash
mkdir -p archive_rees46 && cd archive_rees46
curl -LO https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz   # 1.7GB
curl -LO https://data.rees46.com/datasets/marketplace/2019-Nov.csv.gz   # 2.9GB
cd .. && python build_rees46_sessions.py && python evaluate_rees46.py
```

## Honest limitations

- Synthetic accuracy still partially reflects generator design (mitigated by
  noise + Low-intent-heavy mix, not eliminated). Trust the censored real-data
  validity numbers.
- Real logs lack search/referrer/device, so Goal-driven and the full cold-start
  prior are exercised only on the synthetic track until first-party
  instrumentation exists.
- Offline aggregates approximate the incremental engine; prefix-3 tables are
  the closest offline stand-in for decision-time behaviour.
- November REES46 is Black-Friday traffic: gradients hold, absolute rates are
  promo-inflated; October is the calmer reference.
- The two real-data tables above predate the cart-first override fix — see
  *Numbers to regenerate*.
- Intent separation ≠ proven conversion lift; that requires an online A/B.
