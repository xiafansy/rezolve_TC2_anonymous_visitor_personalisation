"""
Cross-dataset validation of the v2 intent inferer on REES46 (multi-category
store, Oct+Nov 2019) -- the second real retailer, with real prices.

What this run establishes:
  1. REPLICATION  -- do the behavioural bands found on RetailRocket hold on a
     different retailer? (revisit sweet-spot, breadth, drive-by scan)
  2. HOLD-OUT     -- thresholds were frozen before REES46 was ever loaded;
     October only sanity-checks distributions, November (Black Friday month!)
     is the evaluation window. A promo-distorted month is a deliberate
     robustness test.
  3. PRICE FLAVOR -- price_rel_cat < 0.6 (cheap-leaning within category) as an
     orthogonal modifier: does it add conversion lift WITHIN each intent?

November is evaluated on a fixed 2M-session random sample (seed 42) so the
reference Python scorer stays the single source of truth (no vectorised fork).
"""

import pandas as pd

from intent_inference_real import infer_realtime

SAMPLE_N = 2_000_000
SEED = 42

DTYPES = {c: "float32" for c in [
    "revisit_ratio", "top_category_share", "category_switch_rate",
    "median_gap_sec", "price_rel_cat", "n_unique_items", "n_categories"]}
DTYPES.update({"n_events": "int32", "n_views": "int32", "duration_sec": "int32"})

COLS = ["month", "n_events", "n_views", "revisit_ratio", "n_categories",
        "top_category_share", "category_switch_rate", "median_gap_sec",
        "duration_sec", "added_to_cart", "purchased", "price_rel_cat"]

FEATURES = ["n_events", "n_views", "revisit_ratio", "n_categories",
            "top_category_share", "category_switch_rate", "median_gap_sec",
            "duration_sec", "added_to_cart", "price_rel_cat"]


def revisit_replication(df):
    print("REPLICATION -- revisit bands, buy-rate by month "
          "(RetailRocket train pattern: 3.4 / 4.1 / 13.4 / 6.6)")
    for month, g in df.groupby("month"):
        eng = g[(g["n_events"] >= 3) & (g["n_views"] >= 2)]
        row = [month]
        for lo, hi in [(0, 1.2), (1.2, 1.8), (1.8, 2.5), (2.5, 99)]:
            m = eng[(eng["revisit_ratio"] >= lo) & (eng["revisit_ratio"] < hi)]
            row.append(f"{m['purchased'].mean():.1%}")
        print(f"  {row[0]}:  <1.2: {row[1]}   1.2-1.8: {row[2]}   "
              f"1.8-2.5: {row[3]}   >=2.5: {row[4]}")


def main():
    df = pd.read_csv("rees46_sessions.csv", usecols=COLS, dtype=DTYPES)
    print(f"sessions: Oct {len(df[df['month']=='2019-10']):,} | "
          f"Nov {len(df[df['month']=='2019-11']):,}\n")

    revisit_replication(df)

    # ---- November hold-out (Black Friday) ----------------------------------
    nov = df[df["month"] == "2019-11"]
    del df
    test = nov.sample(n=min(SAMPLE_N, len(nov)), random_state=SEED).copy()
    del nov

    res = [infer_realtime(r) for r in test[FEATURES].to_dict("records")]
    test["intent"] = [r.intent for r in res]
    test["stage"] = [r.stage for r in res]
    test["cheap"] = [r.price_conscious for r in res]

    print(f"\nNOVEMBER HOLD-OUT (2M-session sample; Black Friday month)")
    base = test["purchased"].mean()
    print(f"base purchase rate: {base:.2%}\n")

    tab = test.groupby("intent").agg(
        sessions=("purchased", "size"),
        coverage=("purchased", lambda s: len(s) / len(test)),
        cart_rate=("added_to_cart", "mean"),
        purchase_rate=("purchased", "mean"),
    )
    order = ["Decisive", "Evaluator", "Explorer", "Unclear", "Low-intent"]
    tab = tab.reindex([i for i in order if i in tab.index])
    tab["coverage"] = (tab["coverage"] * 100).round(1)
    tab["cart_rate"] = (tab["cart_rate"] * 100).round(1)
    tab["purchase_rate"] = (tab["purchase_rate"] * 100).round(2)
    print(tab.to_string())

    a = test[test["stage"] == "A"]
    ev, ex, lo = (a.loc[a["intent"] == i, "purchased"].mean()
                  for i in ["Evaluator", "Explorer", "Low-intent"])
    print(f"\nmonotonicity Evaluator > Explorer > Low-intent : "
          f"{'PASS' if ev > ex > lo else 'FAIL'} ({ev:.1%} > {ex:.1%} > {lo:.1%})")

    print("\nPRICE FLAVOR -- lift of cheap-leaning WITHIN each intent")
    for intent in ["Decisive", "Evaluator", "Explorer", "Low-intent"]:
        m = test[test["intent"] == intent]
        c, nc = m[m["cheap"]], m[~m["cheap"]]
        if len(c) < 1000:
            continue
        print(f"  {intent:<11} cheap: {c['purchased'].mean():6.2%} "
              f"(n={len(c):,})   not: {nc['purchased'].mean():6.2%} "
              f"(n={len(nc):,})")


if __name__ == "__main__":
    main()
