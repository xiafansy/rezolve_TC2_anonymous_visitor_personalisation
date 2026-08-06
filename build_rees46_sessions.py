"""
Aggregate the REES46 event stream (archive_rees46/) into per-session features.

Unlike RetailRocket, REES46 needs no sessionization -- events carry a
`user_session` id. What it adds over RetailRocket:

  * real PRICES            -> price-behaviour features (Price-sensitive archetype)
  * remove_from_cart       -> a hesitation signal RetailRocket lacks
  * readable category codes + brands

Memory strategy: 42M (Oct) + 67M (Nov) rows won't fit comfortably as raw
strings, so each chunk is compressed to tight dtypes on arrival (uuid session
-> uint64 hash, event_type -> int8, timestamps -> int32 epoch seconds).

Output: rees46_sessions.csv, one row per session, observable signals only.
Price features (views only):
  price_median_viewed   median price of viewed items
  price_rel_cat         median of (view price / that category's month-median)
                        <1 = leaning cheap within category, >1 = premium
  price_first_to_last   last viewed price / first viewed price (drift down <1)
  price_stepdown_share  fraction of consecutive view steps where price dropped
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
    ev = ev.sort_values(["sess", "ts"], kind="mergesort").reset_index(drop=True)
    print(f"  month total: {len(ev):,} events, "
          f"{ev['sess'].nunique():,} sessions", flush=True)
    return ev


def month_features(ev, month):
    is_view = ev["etype"].eq(0)

    g = ev.groupby("sess")
    f = pd.DataFrame({
        "start_ts": g["ts"].first(),
        "n_events": g.size().astype("int32"),
        "n_views": is_view.groupby(ev["sess"]).sum().astype("int32"),
        "n_carts": ev["etype"].eq(1).groupby(ev["sess"]).sum().astype("int32"),
        "n_removes": ev["etype"].eq(2).groupby(ev["sess"]).sum().astype("int32"),
        "purchased": ev["etype"].eq(3).groupby(ev["sess"]).any(),
        "duration_sec": (g["ts"].last() - g["ts"].first()).astype("int32"),
    })
    f["added_to_cart"] = f["n_carts"] > 0

    # tempo: median inter-event gap within session
    gaps = ev["ts"].diff()
    same = ev["sess"].eq(ev["sess"].shift())
    f["median_gap_sec"] = gaps.where(same).groupby(ev["sess"]).median()

    # ---- view-pattern features (breadth / focus / revisit) -----------------
    v = ev[is_view]
    vg = v.groupby("sess")
    f["n_unique_items"] = vg["item"].nunique()
    f["revisit_ratio"] = (vg.size() / vg["item"].nunique()).astype("float32")
    f["n_categories"] = vg["cat"].nunique()
    f["top_category_share"] = (
        v.groupby(["sess", "cat"]).size().groupby("sess").max() / vg.size()
    ).astype("float32")
    vsame = v["sess"].eq(v["sess"].shift())
    switched = v["cat"].ne(v["cat"].shift()) & vsame
    f["category_switch_rate"] = (
        switched.groupby(v["sess"]).sum() / (vg.size() - 1).clip(lower=1)
    ).astype("float32")

    # ---- price features (views only) ----------------------------------------
    f["price_median_viewed"] = vg["price"].median().astype("float32")
    cat_median = v.groupby("cat")["price"].transform("median")
    rel = (v["price"] / cat_median).astype("float32")
    f["price_rel_cat"] = rel.groupby(v["sess"]).median()
    first_p = vg["price"].first()
    last_p = vg["price"].last()
    f["price_first_to_last"] = (last_p / first_p.replace(0, np.nan)).astype("float32")
    stepdown = (v["price"] < v["price"].shift()) & vsame
    f["price_stepdown_share"] = (
        stepdown.groupby(v["sess"]).sum() / (vg.size() - 1).clip(lower=1)
    ).astype("float32")

    f["hour_of_day"] = (pd.to_datetime(f["start_ts"], unit="s", utc=True)
                        .dt.hour.astype("int8"))
    f["month"] = month
    return f.reset_index()


def main():
    first = True
    for month, path in FILES.items():
        print(f"== {month}: {path}", flush=True)
        ev = load_month(path)
        f = month_features(ev, month)
        del ev
        eng = f[f["n_events"] >= 3]
        print(f"  sessions: {len(f):,} | >=3 events: {len(eng):,} "
              f"({len(eng)/len(f):.1%}) | cart {f['added_to_cart'].mean():.2%} "
              f"| buy {f['purchased'].mean():.2%}", flush=True)
        f.to_csv(OUT, index=False, mode="w" if first else "a", header=first)
        first = False
        del f
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
