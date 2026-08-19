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
        elif kind < 0.98:                    # decisive: view -> cart -> buy
            item = random.randint(1, 500)
            rows.append((t, v, "view", item, "")); t += 20_000
            rows.append((t, v, "addtocart", item, "")); t += 40_000
            if random.random() < 0.7:
                rows.append((t, v, "transaction", item, str(v)))
        else:
            # Cart-FIRST: the visitor already knew the item (previous session,
            # a saved link, a re-order) and carts with ZERO prior views. Real
            # RetailRocket sessions look like this. Kept in the fixture because
            # it is exactly the case the old NaN-safe getter mis-read: a true
            # 0 collapsed to the 99 default and these sessions -- the highest
            # converting segment there is -- were served the Low-intent page.
            item = random.randint(1, 500)
            rows.append((t, v, "addtocart", item, "")); t += 30_000
            if random.random() < 0.8:
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
    """Verify the censoring INDEPENDENTLY, by recounting the fabricated log.

    The previous version compared `views_before_first_commercial` against
    `n_views + n_events_total`, which is true for any non-negative numbers --
    it could not fail, so it proved nothing. This one re-derives the expected
    pre-commercial view count straight from events.csv and demands equality.
    """
    ev = pd.read_csv(f"{TMP}/events.csv")
    ev = ev.sort_values(["visitorid", "timestamp"], kind="mergesort")
    # the fixture gives each visitor exactly one session, so visitorid is the key
    first_comm = (ev[ev["event"].isin(["addtocart", "transaction"])]
                  .groupby("visitorid")["timestamp"].min())
    ev = ev.join(first_comm.rename("fc"), on="visitorid")
    pre = ev[ev["fc"].isna() | (ev["timestamp"] < ev["fc"])]
    expected_views = (pre[pre["event"] == "view"].groupby("visitorid").size()
                      .reindex(ev["visitorid"].unique(), fill_value=0))
    expected_events = (pre.groupby("visitorid").size()
                       .reindex(ev["visitorid"].unique(), fill_value=0))

    got = feats.set_index("visitorid")
    exp_v = expected_views.reindex(got.index).fillna(0)
    exp_e = expected_events.reindex(got.index).fillna(0)

    assert (got["n_views"].fillna(0) == exp_v).all(),         "censored n_views disagrees with a direct recount of pre-commercial views"
    assert (got["n_events"].fillna(0) == exp_e).all(),         "censored n_events disagrees with a direct recount of pre-commercial events"
    assert (got["views_before_first_commercial"] == exp_v).all(),         "views_before_first_commercial is not the pre-commercial view count"

    carted = got[got["added_to_cart"]]
    assert (carted["n_events"].fillna(0) < carted["n_events_total"]).all(),         "a carted session kept all its events -- nothing was censored"
    assert (got["n_events_p3"].fillna(0) <= 3).all(), "prefix-3 exceeded 3 events"
    assert (got["n_events_p3"].fillna(0) <= got["n_events"].fillna(0)).all(),         "prefix-3 saw more events than the censored stream it is drawn from"

    n_cart_first = int((carted["views_before_first_commercial"] == 0).sum())
    assert n_cart_first > 0, "fixture no longer covers the cart-with-0-views case"
    print(f"censoring assertions: PASS "
          f"({len(carted):,} carted sessions, {n_cart_first:,} of them cart-first)")


def assert_decisive_override(feats):
    """The cart-first sessions MUST reach the Decisive override.

    Regression guard for the NaN-safe-getter bug: with `0 or 99`, every
    cart-with-zero-prior-views session scored Low-intent instead.
    """
    from intent_engine import REAL_INTENTS, TwoStageEngine
    eng = TwoStageEngine(intents=REAL_INTENTS)
    cart_first = feats[feats["added_to_cart"]
                       & (feats["views_before_first_commercial"] == 0)]
    intents = {eng.score_aggregates(r).intent
               for r in cart_first.to_dict("records")}
    assert intents == {"Decisive"},         f"cart-with-0-views must be Decisive, got {sorted(intents)}"
    print(f"decisive-override regression: PASS "
          f"({len(cart_first):,} cart-first sessions all routed to Checkout-support)")


if __name__ == "__main__":
    fabricate()
    feats = build(TMP, OUT)
    assert_censoring(feats)
    assert_decisive_override(feats)
    # split so the last ~30 fabricated days are held out
    split = T0 + 90 * DAY
    print("\n" + "#" * 70 + "\n#  evaluate_real on fabricated data\n" + "#" * 70)
    evaluate_real.main(OUT, split)
    shutil.rmtree(TMP)
    os.remove(OUT)
    print("\nSMOKE TEST: PASS -- pipeline runs end-to-end; swap in the real "
          "Kaggle archive/ for actual numbers.")
