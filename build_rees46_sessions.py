"""
Aggregate the REES46 event stream (archive_rees46/) into per-session features,
CENSORED at the first commercial event (v3 leakage fix).

Why censoring (the v3 lesson, quantified on RetailRocket)
---------------------------------------------------------
Whole-session features let post-cart behaviour leak into "predictors": buyers
re-open product pages around checkout, inflating revisit_ratio for exactly the
sessions that convert (Evaluator purchase 6.9% leaky -> 2.8% censored; the
revisit-band gradient 3.4->24% collapses to 2.1->3.0%). All behavioural and
price features below are therefore computed on events STRICTLY BEFORE each
session's first cart/purchase. Outcome columns still describe the full session.

REES46 needs no sessionization (events carry `user_session`). What it adds
over RetailRocket: real prices -> the price-conscious flavor.

Outputs (rees46_sessions.csv), per session:
  outcomes (full session): added_to_cart, purchased, n_events_total
  censored behavioural:    n_events, n_views, n_unique_items, revisit_ratio,
                           n_categories, top_category_share,
                           category_switch_rate, duration_sec, median_gap_sec,
                           views_before_first_commercial
  censored price:          price_median_viewed, price_rel_cat
                           (vs month-level category median -- a catalog index,
                           not session behaviour, so computed on all views)
  prefix-3 (*_p3):         the same censored features on the first 3 events
"""

import numpy as np
import pandas as pd

FILES = {
    "2019-10": "archive_rees46/2019-Oct.csv.gz",
    "2019-11": "archive_rees46/2019-Nov.csv.gz",
}
OUT = "rees46_sessions.csv"
CHUNK = 4_000_000
EVENT_CODE = {"view": 0, "cart": 1, "remove_from_cart": 2, "purchase": 3}
COMMERCIAL = (1, 3)  # cart, purchase (remove implies a prior cart anyway)
USECOLS = ["event_time", "event_type", "product_id", "category_id",
           "price", "user_session"]


def load_month(path):
    parts = []
    for i, ch in enumerate(pd.read_csv(path, usecols=USECOLS, chunksize=CHUNK)):
        ch = ch.dropna(subset=["user_session"])
        ts = pd.to_datetime(ch["event_time"], format="%Y-%m-%d %H:%M:%S UTC",
                            utc=True)
        parts.append(pd.DataFrame({
            "ts": (ts.astype("int64") // 10**9).astype("int32"),
            "etype": ch["event_type"].map(EVENT_CODE).astype("int8"),
            "item": ch["product_id"].astype("int32"),
            "cat": ch["category_id"].astype("int64"),
            "price": ch["price"].astype("float32"),
            "sess": pd.util.hash_array(ch["user_session"].to_numpy(object)),
        }))
        print(f"  chunk {i+1}: {len(ch):,} rows", flush=True)
    ev = pd.concat(parts, ignore_index=True)
    del parts
    # Stable sort keeps the file's chronological order within equal timestamps
    # (REES46 has 1s resolution), so positional censoring is exact.
    ev = ev.sort_values(["sess", "ts"], kind="mergesort").reset_index(drop=True)
    print(f"  month total: {len(ev):,} events, "
          f"{ev['sess'].nunique():,} sessions", flush=True)
    return ev


def behavioural_features(ev, cat_median, suffix=""):
    """Feature block for an (already censored) event frame."""
    is_view = ev["etype"].eq(0)
    g = ev.groupby("sess")
    f = pd.DataFrame({
        f"n_events{suffix}": g.size().astype("int32"),
        f"n_views{suffix}": is_view.groupby(ev["sess"]).sum().astype("int32"),
        f"duration_sec{suffix}": (g["ts"].last() - g["ts"].first()).astype("int32"),
    })

    gaps = ev["ts"].diff()
    same = ev["sess"].eq(ev["sess"].shift())
    f[f"median_gap_sec{suffix}"] = gaps.where(same).groupby(ev["sess"]).median()

    v = ev[is_view]
    vg = v.groupby("sess")
    f[f"n_unique_items{suffix}"] = vg["item"].nunique()
    f[f"revisit_ratio{suffix}"] = (vg.size() / vg["item"].nunique()).astype("float32")
    f[f"n_categories{suffix}"] = vg["cat"].nunique()
    f[f"top_category_share{suffix}"] = (
        v.groupby(["sess", "cat"]).size().groupby("sess").max() / vg.size()
    ).astype("float32")
    vsame = v["sess"].eq(v["sess"].shift())
    switched = v["cat"].ne(v["cat"].shift()) & vsame
    f[f"category_switch_rate{suffix}"] = (
        switched.groupby(v["sess"]).sum() / (vg.size() - 1).clip(lower=1)
    ).astype("float32")

    f[f"price_median_viewed{suffix}"] = vg["price"].median().astype("float32")
    rel = (v["price"] / v["cat"].map(cat_median)).astype("float32")
    f[f"price_rel_cat{suffix}"] = rel.groupby(v["sess"]).median()
    return f


def month_features(ev, month):
    # ---- outcomes + context: FULL session -----------------------------------
    g = ev.groupby("sess")
    base = pd.DataFrame({
        "start_ts": g["ts"].first(),
        "n_events_total": g.size().astype("int32"),
        "added_to_cart": ev["etype"].eq(1).groupby(ev["sess"]).any(),
        "purchased": ev["etype"].eq(3).groupby(ev["sess"]).any(),
    })
    base["hour_of_day"] = (pd.to_datetime(base["start_ts"], unit="s", utc=True)
                           .dt.hour.astype("int8"))
    base["month"] = month

    # Catalog price index: month-level category medians (not session behaviour).
    cat_median = ev[ev["etype"].eq(0)].groupby("cat")["price"].median()

    # ---- THE CENSORING STEP: strictly before the first cart/purchase --------
    is_comm = ev["etype"].isin(COMMERCIAL)
    comm_seen = is_comm.groupby(ev["sess"]).cumsum()
    pre = ev[comm_seen.eq(0)]
    print(f"  censored events (pre-commercial): {len(pre):,} / {len(ev):,} "
          f"({len(pre)/len(ev):.1%})", flush=True)

    f = base.join(behavioural_features(pre, cat_median), how="left")
    f["views_before_first_commercial"] = f["n_views"].fillna(0).astype("int32")

    # ---- prefix-3: what a realtime system knows after 3 censored events -----
    pre = pre.copy()
    pre["rank"] = pre.groupby("sess").cumcount()
    f = f.join(behavioural_features(pre[pre["rank"] < 3], cat_median,
                                    suffix="_p3"), how="left")
    return f.reset_index()


def main():
    first = True
    for month, path in FILES.items():
        print(f"== {month}: {path}", flush=True)
        ev = load_month(path)
        f = month_features(ev, month)
        del ev
        eng = f[f["n_events_total"] >= 3]
        print(f"  sessions: {len(f):,} | >=3 events: {len(eng):,} "
              f"({len(eng)/len(f):.1%}) | cart {f['added_to_cart'].mean():.2%} "
              f"| buy {f['purchased'].mean():.2%}", flush=True)
        f.to_csv(OUT, index=False, mode="w" if first else "a", header=first)
        first = False
        del f
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
