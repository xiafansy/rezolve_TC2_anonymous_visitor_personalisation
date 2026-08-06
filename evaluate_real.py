"""
Honest validation of the v2 (real-signal) intent inferer on RetailRocket.

There are NO intent labels in real traffic, so "accuracy" is meaningless here.
Instead we test what a production system would actually care about:

  1. PREDICTIVE VALIDITY -- do Stage-A intents, inferred from view-patterns
     alone, separate sessions by conversion outcomes they never saw?
     (Evaluator should out-convert Explorer should out-convert Low-intent.)
  2. COVERAGE -- how much of real traffic does each mode serve, and how often
     does the Unclear gate decline to personalise?
  3. TEMPORAL HONESTY -- thresholds are fitted on the first ~3.5 months;
     everything reported below comes from the held-out FINAL MONTH only.

Limitation (stated, not hidden): features are session-level aggregates, an
offline approximation of the incremental realtime computation.
"""

import pandas as pd

from intent_inference_real import (THRESHOLDS, infer_realtime)

SPLIT_MS = 1439856000000  # 2015-08-18 00:00 UTC -> last month is held out

FEATURES = ["n_events", "n_views", "revisit_ratio", "n_categories",
            "top_category_share", "category_switch_rate", "median_gap_sec",
            "duration_sec", "added_to_cart"]


def fit_thresholds(train):
    """Quantile-fit the behavioural cutoffs on TRAIN engaged sessions."""
    eng = train[(train["n_events"] >= 3) & (train["n_views"] >= 2)]
    fitted = {
        "revisit_hi": round(eng["revisit_ratio"].quantile(.75), 2),
        "revisit_lo": round(eng["revisit_ratio"].quantile(.25), 2),
        "gap_slow": round(eng["median_gap_sec"].quantile(.60), 1),
        "dur_micro": round(eng["duration_sec"].quantile(.25), 1),
    }
    print("fitted on train window:", fitted)
    print("in use (THRESHOLDS):   ",
          {k: THRESHOLDS[k] for k in fitted})
    return fitted


def main():
    df = pd.read_csv("real_sessions.csv")
    train = df[df["start_ms"] < SPLIT_MS]
    test = df[df["start_ms"] >= SPLIT_MS].copy()
    print(f"train sessions: {len(train):,}   test (final month): {len(test):,}\n")

    fit_thresholds(train)

    # ---- run inference on the held-out month ------------------------------
    rows = test[FEATURES].to_dict("records")
    res = [infer_realtime(r) for r in rows]
    test["intent"] = [r.intent for r in res]
    test["stage"] = [r.stage for r in res]
    test["confidence"] = [r.confidence for r in res]

    # ---- 1. coverage -------------------------------------------------------
    print("\nCOVERAGE (all held-out sessions)")
    cov = (test["intent"].value_counts(normalize=True) * 100).round(1)
    for k, v in cov.items():
        print(f"  {k:<11} {v:>5}%")

    # ---- 2. predictive validity -------------------------------------------
    print("\nPREDICTIVE VALIDITY (held-out month)")
    print("Stage-A intents are inferred from view-patterns only; cart/purchase")
    print("below are outcomes the scorer never saw.\n")
    tab = test.groupby("intent").agg(
        sessions=("purchased", "size"),
        cart_rate=("added_to_cart", "mean"),
        purchase_rate=("purchased", "mean"),
        mean_conf=("confidence", "mean"),
    )
    order = ["Decisive", "Evaluator", "Explorer", "Unclear", "Low-intent"]
    tab = tab.reindex([i for i in order if i in tab.index])
    tab["cart_rate"] = (tab["cart_rate"] * 100).round(1)
    tab["purchase_rate"] = (tab["purchase_rate"] * 100).round(2)
    tab["mean_conf"] = tab["mean_conf"].round(2)
    print(tab.to_string())

    base = test["purchased"].mean() * 100
    print(f"\n  base purchase rate (all test sessions): {base:.2f}%")

    a = test[test["stage"] == "A"]
    ev_p = a.loc[a["intent"] == "Evaluator", "purchased"].mean()
    ex_p = a.loc[a["intent"] == "Explorer", "purchased"].mean()
    lo_p = a.loc[a["intent"] == "Low-intent", "purchased"].mean()
    mono = ev_p > ex_p > lo_p
    print(f"  monotonicity Evaluator > Explorer > Low-intent : "
          f"{'PASS' if mono else 'FAIL'} "
          f"({ev_p:.1%} > {ex_p:.1%} > {lo_p:.1%})")

    # ---- 3. sample explanations -------------------------------------------
    print("\nSAMPLE EXPLANATIONS (held-out sessions)")
    for intent in ["Decisive", "Evaluator", "Explorer", "Unclear"]:
        pool = test[test["intent"] == intent]
        if not len(pool):
            continue
        row = pool.iloc[0]
        r = infer_realtime({f: row[f] for f in FEATURES})
        print(f"\n[{intent}] session of visitor {int(row['visitorid'])}, "
              f"purchased={bool(row['purchased'])}")
        print("    " + r.explain().replace("\n", "\n    "))


if __name__ == "__main__":
    main()
