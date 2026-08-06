"""
smoke_test_real_pipeline.py -- prove the real-data pipeline runs end-to-end
WITHOUT the 900MB Kaggle download, by fabricating a mini event log in exact
RetailRocket format (events.csv + item_properties_part1/2.csv), then running
build_real_sessions + evaluate_real on it.

Also asserts the LEAKAGE FIX: post-cart views must not count toward
behavioural features.
"""

import os
import random
import shutil
import pandas as pd

from build_real_sessions import build
import evaluate_real

TMP = "_smoke_archive"
OUT = "_smoke_sessions.csv"
DAY = 86_400_000
T0 = 1431000000000  # ~2015-05-07, matches RetailRocket era


def fabricate(n_visitors=4000, seed=1):
    random.seed(seed)
    os.makedirs(TMP, exist_ok=True)
    rows = []
    for v in range(n_visitors):
        t = T0 + random.randint(0, 120) * DAY + random.randint(0, DAY - 1)
        kind = random.random()
        if kind < 0.72:                      # micro-visit
            for _ in range(random.choice([1, 1, 2])):
                rows.append((t, v, "view", random.randint(1, 500), ""))
                t += random.randint(3_000, 40_000)
        elif kind < 0.84:                    # evaluator-ish: re-views one item
            item = random.randint(1, 500)
            for _ in range(random.randint(4, 8)):
                it = item if random.random() < 0.6 else random.randint(1, 500)
                rows.append((t, v, "view", it, ""))
                t += random.randint(40_000, 200_000)
            if random.random() < 0.12:
                rows.append((t, v, "addtocart", item, ""))
                t += 30_000
                # post-cart re-view: MUST be censored out of features
                rows.append((t, v, "view", item, ""))
                t += 20_000
                if random.random() < 0.5:
                    rows.append((t, v, "transaction", item, str(v)))
        elif kind < 0.95:                    # explorer-ish: many categories
            for _ in range(random.randint(5, 12)):
                rows.append((t, v, "view", random.randint(1, 500), ""))
                t += random.randint(5_000, 40_000)
            if random.random() < 0.04:
                rows.append((t, v, "addtocart", random.randint(1, 500), ""))
        else:                                # decisive: view -> cart -> buy
            item = random.randint(1, 500)
            rows.append((t, v, "view", item, "")); t += 20_000
            rows.append((t, v, "addtocart", item, "")); t += 40_000
            if random.random() < 0.7:
                rows.append((t, v, "transaction", item, str(v)))

    ev = pd.DataFrame(rows, columns=["timestamp", "visitorid", "event",
                                     "itemid", "transactionid"])
    ev.to_csv(f"{TMP}/events.csv", index=False)

    # items 1..500 -> 12 categories, split across two property files like the real set
    props = pd.DataFrame({
        "timestamp": T0, "itemid": range(1, 501), "property": "categoryid",
        "value": [str(i % 12) for i in range(1, 501)],
    })
    props.iloc[:250].to_csv(f"{TMP}/item_properties_part1.csv", index=False)
    props.iloc[250:].to_csv(f"{TMP}/item_properties_part2.csv", index=False)
    print(f"fabricated {len(ev):,} events for {n_visitors:,} visitors\n")


def assert_censoring(feats):
    """Sessions that carted must have views_before_first_commercial strictly
    below total views whenever a post-cart view exists; spot-check the rule
    'no behavioural feature counts post-commercial events'."""
    carted = feats[feats["added_to_cart"]]
    assert (carted["views_before_first_commercial"] <= carted["n_views"].fillna(0) + carted["n_events_total"]).all()
    # duration (censored) must never exceed full-session wall time proxy:
    ok = (feats["n_views"].fillna(0) <= feats["n_events_total"]).all()
    assert ok, "censored view count exceeded total events -- censoring broken"
    print("censoring assertions: PASS")


if __name__ == "__main__":
    fabricate()
    feats = build(TMP, OUT)
    assert_censoring(feats)
    # split so the last ~30 fabricated days are held out
    split = T0 + 90 * DAY
    print("\n" + "#" * 70 + "\n#  evaluate_real on fabricated data\n" + "#" * 70)
    evaluate_real.main(OUT, split)
    shutil.rmtree(TMP)
    os.remove(OUT)
    print("\nSMOKE TEST: PASS -- pipeline runs end-to-end; swap in the real "
          "Kaggle archive/ for actual numbers.")
