# GA4 cold-start prior calibration (real context signals)

*Source: `bigquery-public-data.ga4_obfuscated_sample_ecommerce` (Google
Merchandise Store, Nov 2020–Jan 2021), aggregated in BigQuery by
`ga4_context_export.sql`, scored by `ga4_prior_calibration.py --cells`.
360,129 sessions, 4,848 purchases, base rate 1.35%.*

## Calibration table (suggested additive weight = 1.5 × (lift − 1), clipped)

| dim | level | sessions | buy% | lift | weight |
|---|---|---|---|---|---|
| referrer | referral | 63,524 | 1.66% | 1.23x | **+0.35** |
| referrer | other | 74,687 | 1.62% | 1.21x | +0.31 |
| referrer | direct | 83,459 | 1.29% | 0.96x | −0.06 |
| referrer | search | 122,841 | 1.10% | 0.81x | **−0.28** |
| referrer | ad | 15,618 | 0.98% | 0.73x | **−0.41** |
| device | mobile / desktop / tablet | 143k / 209k / 8k | 1.39 / 1.32 / 1.30% | ~1.0x | ±0.05 |
| continent | Americas / Asia / Europe / Oceania | 199k / 86k / 67k / 3.8k | 1.31–1.37% | ~1.0x | ±0.04 |
| continent | Africa | 3,715 | 1.16% | 0.86x | −0.21 |
| hour (UTC) | all four buckets | ~90k each | 1.31–1.37% | ~1.0x | ±0.04 |

## Reading

1. **Context is a weak signal — and that validates the architecture.** The
   strongest context weight on real traffic is ±0.4, versus 3–10× conversion
   multipliers from behavioural signals. The engine's design rule ("a prior
   tilts, behaviour decides", prior weights capped ≤1.5, cold gate high) is
   what this data recommends. Empirically most hand-set priors are, if
   anything, still too confident.
2. **Two directional surprises worth carrying:** organic-search arrivals
   convert *below* base (0.81x) and paid ads lowest (0.73x) on this store,
   while referral traffic is the warmest (1.23x). Caveats before
   generalising: this is Google's own merchandise store — its "referral"
   bucket includes internal cross-links (shop.googlemerchandisestore.com),
   i.e. partly returning/on-site traffic, and its search arrivals skew
   curiosity (brand tourists), not purchase missions.
3. **The flat hour-of-day table is timezone smearing, not absence of
   signal.** A global store aggregated on UTC hours flattens local daily
   rhythms — RetailRocket (single market) showed evening ≈ 2× morning. Local
   time is the right feature once first-party data exists.
4. **What this cannot calibrate: intent-direction weights.** GA4 gives
   context → *conversion propensity*, not context → *intent type* (a search
   arrival may still be mission-shaped even where it converts less). The
   engine's context→intent mapping therefore keeps its hand-set structure;
   what we adopt from this table is the *scale* (small), the *sign warnings*
   (don't treat ad/search arrivals as hot), and the method — the same SQL
   runs on any GA4 property, so a real client's first-party export calibrates
   the prior properly on day one.

## Repro

```bash
# paste ga4_context_export.sql into console.cloud.google.com/bigquery (sandbox),
# SAVE RESULTS -> CSV -> archive_ga4/ga4_context_cells.csv, then:
/usr/bin/python3 ga4_prior_calibration.py --cells archive_ga4/ga4_context_cells.csv
```

The 344-row cell table is committed (`archive_ga4/ga4_context_cells.csv`)
so this report reproduces without a BigQuery account.
