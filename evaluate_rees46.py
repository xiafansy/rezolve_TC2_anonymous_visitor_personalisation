"""
evaluate_rees46.py (v3) -- cross-dataset validation on REES46, through the
SAME unified engine as RetailRocket, on CENSORED (leakage-free) features.

Design
------
* Thresholds live in intent_engine.py and were set on RetailRocket -- REES46
  is evaluated with ZERO re-tuning (cross-retailer transfer test).
* October = sanity window; November (Black Friday) = reported hold-out,
  on a fixed 2M-session sample (seed 42).
* PRICE FLAVOR: censored `price_rel_cat < 0.6` marks cheap-leaning browsing;
  we report its conversion lift WITHIN each intent (it should tilt
  merchandising, not layout).
* PREFIX-3 table: the decision after the first 3 censored events.

Usage
-----
  python3 evaluate_rees46.py [--data rees46_sessions.csv]
"""

import argparse
import pandas as pd

from intent_engine import REAL_INTENTS, TwoStageEngine

SAMPLE_N = 2_000_000
SEED = 42

BASE_FEATS = ["n_events", "n_views", "revisit_ratio", "n_categories",
              "top_category_share", "category_switch_rate", "median_gap_sec",
              "duration_sec", "price_rel_cat"]

DTYPES = {c: "float32" for c in
          ["revisit_ratio", "top_category_share", "category_switch_rate",
           "median_gap_sec", "price_rel_cat", "n_categories", "n_unique_items",
           "revisit_ratio_p3", "top_category_share_p3", "category_switch_rate_p3",
           "median_gap_sec_p3", "price_rel_cat_p3", "n_categories_p3",
           "n_events", "n_views", "duration_sec",
           "n_events_p3", "n_views_p3", "duration_sec_p3"]}

COLS = (["month", "n_events_total", "added_to_cart", "purchased",
         "views_before_first_commercial"]
        + BASE_FEATS + [c + "_p3" for c in BASE_FEATS])


def row_features(row, suffix=""):
    f = {k: row.get(k + suffix) for k in BASE_FEATS}
    f["added_to_cart"] = row.get("added_to_cart")
    f["views_before_first_commercial"] = row.get("views_before_first_commercial")
    return f


def validity_table(df, intent_col):
    tab = df.groupby(intent_col).agg(
        sessions=("purchased", "size"),
        coverage=("purchased", lambda s: len(s) / len(df)),
        cart_rate=("added_to_cart", "mean"),
        purchase_rate=("purchased", "mean"),
    )
    order = ["Decisive", "Evaluator", "Explorer", "Unclear", "Low-intent"]
    tab = tab.reindex([i for i in order if i in tab.index])
    tab["coverage"] = (tab["coverage"] * 100).round(1)
    tab["cart_rate"] = (tab["cart_rate"] * 100).round(1)
    tab["purchase_rate"] = (tab["purchase_rate"] * 100).round(2)
    return tab


def monotonicity(df, col):
    p = df.groupby(col)["purchased"].mean()
    try:
        ok = p["Evaluator"] > p["Explorer"] > p["Low-intent"]
        return (f"monotonicity Evaluator > Explorer > Low-intent : "
                f"{'PASS' if ok else 'FAIL'} "
                f"({p['Evaluator']:.2%} > {p['Explorer']:.2%} > {p['Low-intent']:.2%})")
    except KeyError:
        return "monotonicity: some intents missing"


def revisit_bands(df, label):
    eng = df[(df["n_events"] >= 3) & (df["n_views"] >= 2)]
    cells = []
    for lo, hi in [(0, 1.2), (1.2, 1.8), (1.8, 2.5), (2.5, 99)]:
        m = eng[(eng["revisit_ratio"] >= lo) & (eng["revisit_ratio"] < hi)]
        cells.append(f"{m['purchased'].mean():.1%}" if len(m) else "--")
    print(f"  {label}:  <1.2: {cells[0]}   1.2-1.8: {cells[1]}   "
          f"1.8-2.5: {cells[2]}   >=2.5: {cells[3]}")


def main(data):
    df = pd.read_csv(data, usecols=lambda c: c in COLS, dtype=DTYPES)
    print(f"sessions: Oct {(df['month'] == '2019-10').sum():,} | "
          f"Nov {(df['month'] == '2019-11').sum():,}\n")

    print("REPLICATION -- CENSORED revisit bands, buy-rate by month")
    print("(RetailRocket censored hold-out: 2.1% / 2.0% / 3.0% / 2.8%)")
    for month, g in df.groupby("month"):
        revisit_bands(g, month)

    nov = df[df["month"] == "2019-11"]
    del df
    test = nov.sample(n=min(SAMPLE_N, len(nov)), random_state=SEED).copy()
    del nov

    eng = TwoStageEngine(intents=REAL_INTENTS)
    res = [eng.score_aggregates(row_features(r)) for r in test.to_dict("records")]
    test["intent"] = [x.intent for x in res]
    test["mode"] = [x.mode for x in res]
    test["cheap"] = [x.price_conscious for x in res]

    print(f"\nNOVEMBER HOLD-OUT (2M sample; Black Friday month; CENSORED features)")
    print(f"base purchase rate: {test['purchased'].mean():.2%}\n")
    print(validity_table(test, "intent").to_string())
    print("\n" + monotonicity(test[test["mode"] != "override"], "intent"))

    print("\nPRICE FLAVOR -- cheap-leaning lift WITHIN each intent (censored)")
    for intent in ["Decisive", "Evaluator", "Explorer", "Unclear", "Low-intent"]:
        m = test[test["intent"] == intent]
        c, nc = m[m["cheap"]], m[~m["cheap"]]
        if len(c) < 1000 or len(nc) < 1000:
            continue
        print(f"  {intent:<11} cheap: {c['purchased'].mean():6.2%} "
              f"(n={len(c):,})   not: {nc['purchased'].mean():6.2%} "
              f"(n={len(nc):,})")

    # ---- prefix-3 ------------------------------------------------------------
    t3 = test[test["n_events_p3"].notna()].copy()
    res3 = [eng.score_aggregates(row_features(r, "_p3"))
            for r in t3.to_dict("records")]
    t3["intent_p3"] = [x.intent for x in res3]
    t3["mode_p3"] = [x.mode for x in res3]

    print("\nPREFIX-3 -- decision after the first 3 censored events")
    print(validity_table(t3, "intent_p3").to_string())
    print("\n" + monotonicity(t3[t3["mode_p3"] != "override"], "intent_p3"))

    both = t3[(t3["intent_p3"] != "Unclear") & (t3["intent"] != "Unclear")]
    if len(both):
        print(f"\nprefix-3 vs full-session agreement: "
              f"{(both['intent_p3'] == both['intent']).mean():.0%} (n={len(both):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="rees46_sessions.csv")
    a = ap.parse_args()
    main(a.data)
