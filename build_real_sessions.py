"""
Sessionize the RetailRocket event log (archive/) into per-session behavioural
features for anonymous-visitor intent inference.

Input
-----
archive/events.csv               timestamp(ms), visitorid, event, itemid, transactionid
archive/item_properties_part*.csv  used only to map itemid -> categoryid
archive/category_tree.csv        (not needed here; kept for later category rollups)

Sessionization
--------------
Standard 30-minute inactivity rule: a visitor's events belong to one session
until a gap > 30 min, which starts a new session.

Output
------
real_sessions.csv -- one row per session with ONLY observable signals:

  sequence / breadth
    n_events, n_views, n_unique_items, revisit_ratio (views per unique item),
    n_categories, top_category_share, category_switch_rate
  tempo
    duration_sec, median_gap_sec
  commercial
    added_to_cart, n_addtocart, purchased, first_cart_position
  context (to be TESTED for signal, not assumed)
    hour_of_day, day_of_week

No intent labels exist here -- this is real anonymous traffic. Validation is
done downstream via cluster structure and predictive validity, not accuracy
against ground truth.
"""

import numpy as np
import pandas as pd

SESSION_GAP_MS = 30 * 60 * 1000  # 30 min inactivity ends a session


# ---------------------------------------------------------------------------
# 1. Item -> category map (chunked scan of the big properties files)
# ---------------------------------------------------------------------------
def load_item_categories():
    parts = []
    for path in ["archive/item_properties_part1.csv",
                 "archive/item_properties_part2.csv"]:
        for chunk in pd.read_csv(path, chunksize=2_000_000):
            cat = chunk[chunk["property"] == "categoryid"]
            parts.append(cat[["timestamp", "itemid", "value"]])
    cats = pd.concat(parts, ignore_index=True)
    # Properties are time-varying snapshots; keep the latest per item.
    cats = cats.sort_values("timestamp").drop_duplicates("itemid", keep="last")
    mapping = cats.set_index("itemid")["value"].astype(int)
    print(f"item->category map: {len(mapping):,} items")
    return mapping


# ---------------------------------------------------------------------------
# 2. Sessionize events
# ---------------------------------------------------------------------------
def build_sessions():
    ev = pd.read_csv("archive/events.csv")
    ev = ev.sort_values(["visitorid", "timestamp"], kind="mergesort").reset_index(drop=True)

    # New session where visitor changes or gap exceeds threshold.
    new_visitor = ev["visitorid"].ne(ev["visitorid"].shift())
    gap = ev["timestamp"].diff()
    ev["session_id"] = (new_visitor | (gap > SESSION_GAP_MS)).cumsum()

    item_cat = load_item_categories()
    ev["category"] = ev["itemid"].map(item_cat)

    ts = pd.to_datetime(ev["timestamp"], unit="ms")
    ev["hour"] = ts.dt.hour
    ev["dow"] = ts.dt.dayofweek

    print(f"events: {len(ev):,}  sessions: {ev['session_id'].nunique():,}")
    return ev


def per_session_features(ev):
    is_view = ev["event"].eq("view")
    is_cart = ev["event"].eq("addtocart")

    # --- vectorised base aggregates ---------------------------------------
    g = ev.groupby("session_id")
    feats = pd.DataFrame({
        "visitorid": g["visitorid"].first(),
        "start_ms": g["timestamp"].first(),
        "end_ms": g["timestamp"].last(),
        "n_events": g.size(),
        "n_views": is_view.groupby(ev["session_id"]).sum(),
        "n_addtocart": is_cart.groupby(ev["session_id"]).sum(),
        "purchased": ev["event"].eq("transaction").groupby(ev["session_id"]).any(),
        "hour_of_day": g["hour"].first(),
        "day_of_week": g["dow"].first(),
    })
    feats["duration_sec"] = (feats["end_ms"] - feats["start_ms"]) / 1000.0
    feats["added_to_cart"] = feats["n_addtocart"] > 0

    # --- view-level breadth / focus ----------------------------------------
    views = ev[is_view]
    vg = views.groupby("session_id")
    feats["n_unique_items"] = vg["itemid"].nunique()
    feats["n_categories"] = vg["category"].nunique()

    # Share of views landing on the modal category (1.0 = fully focused).
    top_share = (views.groupby(["session_id", "category"]).size()
                 .groupby("session_id").max()
                 / vg.size())
    feats["top_category_share"] = top_share

    # Rate of consecutive views that jump category (0 = never, 1 = every step).
    switched = views["category"].ne(views["category"].shift()) & \
        views["session_id"].eq(views["session_id"].shift())
    feats["category_switch_rate"] = (
        switched.groupby(views["session_id"]).sum()
        / (vg.size() - 1).clip(lower=1)
    )

    feats["revisit_ratio"] = feats["n_views"] / feats["n_unique_items"]

    # --- tempo: median inter-event gap -------------------------------------
    gaps = ev["timestamp"].diff()
    same_session = ev["session_id"].eq(ev["session_id"].shift())
    feats["median_gap_sec"] = (
        gaps.where(same_session).groupby(ev["session_id"]).median() / 1000.0
    )

    # --- how early in the session the first add-to-cart happened -----------
    carts = ev[is_cart]
    first_cart = carts.groupby("session_id")["timestamp"].first()
    span = (feats["end_ms"] - feats["start_ms"]).replace(0, np.nan)
    feats["first_cart_position"] = (first_cart - feats["start_ms"]) / span

    feats = feats.drop(columns=["end_ms"]).reset_index()
    return feats


def main():
    ev = build_sessions()
    feats = per_session_features(ev)

    print("\nsession size distribution (n_events):")
    print(feats["n_events"].describe(percentiles=[.5, .75, .9, .99]).to_string())
    multi = feats[feats["n_events"] >= 3]
    print(f"\nsessions total: {len(feats):,}")
    print(f"sessions with >=3 events (inferable): {len(multi):,} "
          f"({len(multi)/len(feats):.1%})")
    print(f"cart rate: {feats['added_to_cart'].mean():.2%}   "
          f"purchase rate: {feats['purchased'].mean():.2%}")

    feats.to_csv("real_sessions.csv", index=False)
    print(f"\nsaved real_sessions.csv  ({len(feats):,} rows)")


if __name__ == "__main__":
    main()
