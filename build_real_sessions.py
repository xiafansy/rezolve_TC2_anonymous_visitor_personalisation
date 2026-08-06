"""
build_real_sessions.py (v3) -- sessionize the RetailRocket log, WITHOUT leakage.

THE LEAKAGE FIX (the most important change in v3)
-------------------------------------------------
v2 computed behavioural features over the ENTIRE session -- including views
that happened AFTER add-to-cart / purchase. Buyers re-open the product page
during checkout, so `revisit_ratio` and `duration` were inflated by the very
outcome we later "predicted". v3 censors: all behavioural features are
computed on events STRICTLY BEFORE the first commercial event. Outcome
columns (added_to_cart / purchased) still describe the full session.

Also new
--------
* prefix-3 features (`*_p3`): the same behavioural features computed on the
  first 3 censored events only -- what a realtime system knows at decision
  time ("last 2-3 clicks").
* `views_before_first_commercial`: drives the Stage-B Decisive override
  honestly (v2 used total n_views, which mixes pre and post).
* CLI args so the pipeline is testable: --archive DIR --out FILE.

Usage
-----
  python3 build_real_sessions.py --archive archive --out real_sessions.csv
"""

import argparse
import numpy as np
import pandas as pd

SESSION_GAP_MS = 30 * 60 * 1000  # 30 min inactivity ends a session
COMMERCIAL = ("addtocart", "transaction")


def load_item_categories(archive):
    parts = []
    for name in ["item_properties_part1.csv", "item_properties_part2.csv"]:
        for chunk in pd.read_csv(f"{archive}/{name}", chunksize=2_000_000):
            cat = chunk[chunk["property"] == "categoryid"]
            parts.append(cat[["timestamp", "itemid", "value"]])
    cats = pd.concat(parts, ignore_index=True)
    cats = cats.sort_values("timestamp").drop_duplicates("itemid", keep="last")
    mapping = cats.set_index("itemid")["value"].astype(int)
    print(f"item->category map: {len(mapping):,} items")
    return mapping


def sessionize(archive):
    ev = pd.read_csv(f"{archive}/events.csv")
    ev = ev.sort_values(["visitorid", "timestamp"], kind="mergesort").reset_index(drop=True)
    new_visitor = ev["visitorid"].ne(ev["visitorid"].shift())
    gap = ev["timestamp"].diff()
    ev["session_id"] = (new_visitor | (gap > SESSION_GAP_MS)).cumsum()
    ev["category"] = ev["itemid"].map(load_item_categories(archive))
    ts = pd.to_datetime(ev["timestamp"], unit="ms")
    ev["hour"] = ts.dt.hour
    ev["dow"] = ts.dt.dayofweek
    print(f"events: {len(ev):,}   sessions: {ev['session_id'].nunique():,}")
    return ev


def behavioural_features(ev, suffix=""):
    """Aggregate behavioural features for an (already filtered) event frame.
    Call with CENSORED events only -- this function trusts its input."""
    is_view = ev["event"].eq("view")
    g = ev.groupby("session_id")
    f = pd.DataFrame({
        f"n_events{suffix}": g.size(),
        f"n_views{suffix}": is_view.groupby(ev["session_id"]).sum(),
    })
    f[f"duration_sec{suffix}"] = (g["timestamp"].last() - g["timestamp"].first()) / 1000.0

    views = ev[is_view]
    vg = views.groupby("session_id")
    f[f"n_unique_items{suffix}"] = vg["itemid"].nunique()
    f[f"n_categories{suffix}"] = vg["category"].nunique()
    top = (views.groupby(["session_id", "category"]).size()
           .groupby("session_id").max() / vg.size())
    f[f"top_category_share{suffix}"] = top
    switched = views["category"].ne(views["category"].shift()) & \
        views["session_id"].eq(views["session_id"].shift())
    f[f"category_switch_rate{suffix}"] = (
        switched.groupby(views["session_id"]).sum() / (vg.size() - 1).clip(lower=1))
    f[f"revisit_ratio{suffix}"] = f[f"n_views{suffix}"] / f[f"n_unique_items{suffix}"]

    gaps = ev["timestamp"].diff()
    same = ev["session_id"].eq(ev["session_id"].shift())
    f[f"median_gap_sec{suffix}"] = (
        gaps.where(same).groupby(ev["session_id"]).median() / 1000.0)
    return f


def build(archive, out):
    ev = sessionize(archive)

    # ---- outcomes + context: computed on the FULL session -------------------
    g = ev.groupby("session_id")
    base = pd.DataFrame({
        "visitorid": g["visitorid"].first(),
        "start_ms": g["timestamp"].first(),
        "n_events_total": g.size(),
        "added_to_cart": ev["event"].eq("addtocart").groupby(ev["session_id"]).any(),
        "purchased": ev["event"].eq("transaction").groupby(ev["session_id"]).any(),
        "hour_of_day": g["hour"].first(),
        "day_of_week": g["dow"].first(),
    })

    # ---- THE CENSORING STEP --------------------------------------------------
    first_comm = (ev[ev["event"].isin(COMMERCIAL)]
                  .groupby("session_id")["timestamp"].min()
                  .rename("first_commercial_ms"))
    ev = ev.join(first_comm, on="session_id")
    pre = ev[ev["first_commercial_ms"].isna()
             | (ev["timestamp"] < ev["first_commercial_ms"])]
    print(f"censored events (pre-commercial): {len(pre):,} / {len(ev):,} "
          f"({len(pre)/len(ev):.1%})")

    feats = base.join(behavioural_features(pre), how="left")
    feats["views_before_first_commercial"] = feats["n_views"].fillna(0)

    # ---- prefix-3: what a realtime system knows after the first 3 clicks -----
    pre = pre.copy()
    pre["rank"] = pre.groupby("session_id").cumcount()
    feats = feats.join(behavioural_features(pre[pre["rank"] < 3], suffix="_p3"),
                       how="left")

    feats = feats.reset_index()
    feats.to_csv(out, index=False)

    print(f"\nsessions: {len(feats):,}")
    multi = feats[feats["n_events_total"] >= 3]
    print(f"sessions with >=3 events: {len(multi):,} ({len(multi)/len(feats):.1%})")
    print(f"cart rate: {feats['added_to_cart'].mean():.2%}   "
          f"purchase rate: {feats['purchased'].mean():.2%}")
    print(f"saved {out}")
    return feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive")
    ap.add_argument("--out", default="real_sessions.csv")
    a = ap.parse_args()
    build(a.archive, a.out)
