"""
evaluate_real.py (v3) -- temporal hold-out validation on RetailRocket,
run through the SAME unified engine as the synthetic track, on CENSORED
(leakage-free) features.

What changed vs v2
------------------
* Features are censored at the first commercial event (see
  build_real_sessions.py), so "Evaluator converts more" can no longer be an
  artifact of buyers re-opening the product page during checkout.
* Stage-B Decisive fires on `views_before_first_commercial <= 2` -- the
  honest version of "cart with minimal prior browsing".
* NEW: the same validity table computed on PREFIX-3 features -- does the
  intent called after just the first 3 clicks already separate conversion?
  That is the number a realtime homepage actually banks on.
* Same temporal split: fit on the first ~3.5 months, report the final month.

Usage
-----
  python3 evaluate_real.py --data real_sessions.csv [--split-ms 1439856000000]
"""

import argparse
import pandas as pd

from intent_engine import REAL_INTENTS, TwoStageEngine

DEFAULT_SPLIT_MS = 1439856000000  # 2015-08-18 UTC -> last month held out

BASE_FEATS = ["n_events", "n_views", "revisit_ratio", "n_categories",
              "top_category_share", "category_switch_rate", "median_gap_sec",
              "duration_sec"]


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


def monotonicity(df, intent_col):
    p = df.groupby(intent_col)["purchased"].mean()
    try:
        ok = p["Evaluator"] > p["Explorer"] > p["Low-intent"]
        return (f"monotonicity Evaluator > Explorer > Low-intent : "
                f"{'PASS' if ok else 'FAIL'} "
                f"({p['Evaluator']:.2%} > {p['Explorer']:.2%} > {p['Low-intent']:.2%})")
    except KeyError:
        return "monotonicity: some intents missing in this slice"


def fit_report(train):
    """Report train-window quantiles next to the thresholds the engine uses,
    so drift is visible. (Engine thresholds live in intent_engine.py.)"""
    eng = train[(train["n_events"] >= 3) & (train["n_views"] >= 2)]
    q = {
        "revisit_ratio p75 (engine uses 1.8)": round(eng["revisit_ratio"].quantile(.75), 2),
        "revisit_ratio p25 (engine uses 1.1)": round(eng["revisit_ratio"].quantile(.25), 2),
        "median_gap p60  (engine uses 120s)": round(eng["median_gap_sec"].quantile(.60), 1),
        "duration p25    (engine uses 90s)": round(eng["duration_sec"].quantile(.25), 1),
    }
    print("train-window quantiles vs engine thresholds:")
    for k, v in q.items():
        print(f"  {k}: {v}")


def main(data, split_ms):
    df = pd.read_csv(data)
    train = df[df["start_ms"] < split_ms]
    test = df[df["start_ms"] >= split_ms].copy()
    print(f"train sessions: {len(train):,}   test (final month): {len(test):,}\n")
    if len(train):
        fit_report(train)

    eng = TwoStageEngine(intents=REAL_INTENTS)

    # ---- full censored features ------------------------------------------------
    res = [eng.score_aggregates(row_features(r)) for r in test.to_dict("records")]
    test["intent"] = [x.intent for x in res]
    test["mode"] = [x.mode for x in res]

    print("\n" + "=" * 70)
    print("VALIDITY -- CENSORED full-session features (leakage-free)")
    print("Stage-A intents see NO commercial signals; conversion is unseen outcome")
    print("=" * 70)
    print(validity_table(test, "intent").to_string())
    base = test["purchased"].mean() * 100
    print(f"\nbase purchase rate: {base:.2f}%")
    print(monotonicity(test[test["mode"] != "override"], "intent"))

    # ---- prefix-3 features -------------------------------------------------------
    has_p3 = test["n_events_p3"].notna()
    t3 = test[has_p3].copy()
    res3 = [eng.score_aggregates(row_features(r, "_p3"))
            for r in t3.to_dict("records")]
    t3["intent_p3"] = [x.intent for x in res3]
    t3["mode_p3"] = [x.mode for x in res3]

    print("\n" + "=" * 70)
    print("VALIDITY -- PREFIX-3 features (decision after the first 3 clicks)")
    print("=" * 70)
    print(validity_table(t3, "intent_p3").to_string())
    print(f"\nbase purchase rate (prefix-eligible): {t3['purchased'].mean()*100:.2f}%")
    print(monotonicity(t3[t3["mode_p3"] != "override"], "intent_p3"))

    # ---- agreement: does the 3-click call survive the full session? --------------
    both = t3[(t3["intent_p3"] != "Unclear") & (t3["intent"] != "Unclear")]
    if len(both):
        agree = (both["intent_p3"] == both["intent"]).mean()
        print(f"\nprefix-3 vs full-session intent agreement: {agree:.0%} "
              f"(n={len(both):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="real_sessions.csv")
    ap.add_argument("--split-ms", type=int, default=DEFAULT_SPLIT_MS)
    a = ap.parse_args()
    main(a.data, a.split_ms)
