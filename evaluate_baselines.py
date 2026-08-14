"""
evaluate_baselines.py -- Imran's question: how much better is intent-based
personalisation than (a) not personalising and (b) just showing popular stuff?

Offline logs can't measure true page-lift (no interventions), so we measure
the two honest proxies available:

1. PURCHASE CONCENTRATION (both datasets, held-out):
   rank sessions by the engine's intent tiers vs two rivals -- an
   engagement-count heuristic (n_events desc: "personalise for whoever
   clicks a lot") and random targeting -- and report what % of purchases
   sits in the top X% of traffic each policy would target.

2. NEXT-ACTION HIT-RATE (RetailRocket, event-level):
   at the 3rd censored event we commit to a module; does the REST of the
   session contain the behaviour that module serves?
     Evaluator  -> recently-viewed/comparison rail  -> hit: re-views a prefix item
     Explorer   -> cross-category discovery         -> hit: views a NEW category
     popularity -> top-100 rail (train-window)      -> hit: views a top-100 item
   Policy comparison: intent-matched modules (popularity as neutral fallback)
   vs popularity-for-everyone, on the same sessions.

Writes reports/baseline-comparison.md.
"""

import numpy as np
import pandas as pd

from intent_engine import REAL_INTENTS, TwoStageEngine

SPLIT_MS = 1439856000000
REES_SAMPLE = 500_000
SEED = 42
CUTS = [0.01, 0.05, 0.10, 0.20, 0.50]
TIER = {"Decisive": 5, "Evaluator": 4, "Explorer": 3, "Unclear": 2, "Low-intent": 1}

BASE_FEATS = ["n_events", "n_views", "revisit_ratio", "n_categories",
              "top_category_share", "category_switch_rate", "median_gap_sec",
              "duration_sec", "price_rel_cat"]


def infer(df):
    eng = TwoStageEngine(intents=REAL_INTENTS)
    recs = df.to_dict("records")
    tiers, confs = [], []
    for r in recs:
        f = {k: r.get(k) for k in BASE_FEATS}
        f["added_to_cart"] = r.get("added_to_cart")
        f["views_before_first_commercial"] = r.get("views_before_first_commercial")
        inf = eng.score_aggregates(f)
        tiers.append(TIER[inf.intent])
        confs.append(inf.confidence)
    df = df.copy()
    df["tier"], df["conf"] = tiers, confs
    return df


def capture_curve(df, order_cols, ascending=False):
    d = df.sort_values(order_cols, ascending=ascending, kind="mergesort")
    bought = d["purchased"].to_numpy(float)
    cum = np.cumsum(bought) / max(1.0, bought.sum())
    n = len(d)
    return {c: cum[min(n - 1, int(c * n) - 1)] * 100 for c in CUTS}


def concentration(name, df, lines, rng):
    df = infer(df)
    df["rand"] = rng.random(len(df))
    rows = [
        ("intent tiers (engine)", capture_curve(df, ["tier", "conf", "rand"])),
        ("engagement heuristic (n_events)", capture_curve(df, ["n_events", "rand"])),
        ("random targeting", capture_curve(df, ["rand"])),
    ]
    lines.append(f"\n### {name} — % of purchases captured in top-X% of traffic\n")
    lines.append("| policy | top 1% | 5% | 10% | 20% | 50% |")
    lines.append("|---|---|---|---|---|---|")
    for label, cv in rows:
        lines.append("| " + label + " | "
                     + " | ".join(f"{cv[c]:.0f}%" for c in CUTS) + " |")
    print(f"  concentration done: {name}", flush=True)


# ---------------------------------------------------------------------------
# Part 2: event-level next-action hit-rates (RetailRocket)
# ---------------------------------------------------------------------------
def next_action_hits(lines):
    ev = pd.read_csv("archive/events.csv")
    ev = ev.sort_values(["visitorid", "timestamp"], kind="mergesort").reset_index(drop=True)
    new_visitor = ev["visitorid"].ne(ev["visitorid"].shift())
    gap = ev["timestamp"].diff()
    ev["session_id"] = (new_visitor | (gap > 30 * 60 * 1000)).cumsum()

    cat_map = []
    for name in ["item_properties_part1.csv", "item_properties_part2.csv"]:
        for chunk in pd.read_csv(f"archive/{name}", chunksize=2_000_000):
            c = chunk[chunk["property"] == "categoryid"]
            cat_map.append(c[["timestamp", "itemid", "value"]])
    cat_map = (pd.concat(cat_map).sort_values("timestamp")
               .drop_duplicates("itemid", keep="last").set_index("itemid")["value"])
    ev["category"] = ev["itemid"].map(cat_map)

    # censor at first commercial event, rank within censored stream
    first_comm = (ev[ev["event"].isin(["addtocart", "transaction"])]
                  .groupby("session_id")["timestamp"].min())
    ev = ev.join(first_comm.rename("fc"), on="session_id")
    pre = ev[ev["fc"].isna() | (ev["timestamp"] < ev["fc"])].copy()
    pre["rank"] = pre.groupby("session_id").cumcount()

    # popularity rail: top-100 viewed items in the TRAIN window only
    train_views = ev[(ev["event"] == "view") & (ev["timestamp"] < SPLIT_MS)]
    top100 = set(train_views["itemid"].value_counts().head(100).index)

    # held-out sessions with a 3-event prefix and at least one later event
    feats = pd.read_csv("real_sessions.csv")
    test = feats[(feats["start_ms"] >= SPLIT_MS) & feats["n_events_p3"].notna()]
    test = infer_p3(test)

    pre = pre[pre["session_id"].isin(set(test["session_id"]))]
    prefix = pre[pre["rank"] < 3]
    rest = pre[pre["rank"] >= 3]
    has_rest = set(rest["session_id"])

    pv = prefix[prefix["event"] == "view"]
    rv = rest[rest["event"] == "view"]

    # hit: re-viewed a prefix item
    re_hit = set(rv.merge(pv[["session_id", "itemid"]].drop_duplicates(),
                          on=["session_id", "itemid"])["session_id"])
    # hit: viewed a category not in the prefix
    rc = rv.dropna(subset=["category"])[["session_id", "category"]]
    pc = pv.dropna(subset=["category"])[["session_id", "category"]].drop_duplicates()
    m = rc.merge(pc, on=["session_id", "category"], how="left", indicator=True)
    new_cat_hit = set(m.loc[m["_merge"] == "left_only", "session_id"])
    # hit: viewed a train-window top-100 item
    pop_hit = set(rv[rv["itemid"].isin(top100)]["session_id"])

    test = test[test["session_id"].isin(has_rest)].copy()
    test["hit_eval"] = test["session_id"].isin(re_hit)
    test["hit_expl"] = test["session_id"].isin(new_cat_hit)
    test["hit_pop"] = test["session_id"].isin(pop_hit)

    lines.append("\n### RetailRocket — does the served module contain the "
                 "visitor's actual next behaviour?\n")
    lines.append("Committed at the 3rd censored event; 'hit' = the rest of the "
                 "session contains what the module serves.\n")
    lines.append("| sessions (intent @ 3 clicks) | n | intent-matched module | popularity rail |")
    lines.append("|---|---|---|---|")
    ev_s = test[test["intent_p3"] == "Evaluator"]
    ex_s = test[test["intent_p3"] == "Explorer"]
    lines.append(f"| Evaluator → recently-viewed rail | {len(ev_s):,} | "
                 f"**{ev_s['hit_eval'].mean():.0%}** | {ev_s['hit_pop'].mean():.0%} |")
    lines.append(f"| Explorer → cross-category discovery | {len(ex_s):,} | "
                 f"**{ex_s['hit_expl'].mean():.0%}** | {ex_s['hit_pop'].mean():.0%} |")

    matched = np.where(test["intent_p3"] == "Evaluator", test["hit_eval"],
                       np.where(test["intent_p3"] == "Explorer", test["hit_expl"],
                                test["hit_pop"]))
    lines.append(f"\nPolicy over all {len(test):,} eligible sessions: "
                 f"intent-matched modules (popularity as neutral fallback) hit "
                 f"**{matched.mean():.0%}** vs popularity-for-everyone "
                 f"**{test['hit_pop'].mean():.0%}**.")
    print("  next-action hits done", flush=True)


def infer_p3(test):
    eng = TwoStageEngine(intents=REAL_INTENTS)
    out = []
    for r in test.to_dict("records"):
        f = {k: r.get(k + "_p3") for k in BASE_FEATS if k != "price_rel_cat"}
        f["price_rel_cat"] = None
        f["added_to_cart"] = r.get("added_to_cart")
        f["views_before_first_commercial"] = r.get("views_before_first_commercial")
        out.append(eng.score_aggregates(f).intent)
    test = test.copy()
    test["intent_p3"] = out
    return test


def main():
    rng = np.random.default_rng(SEED)
    lines = ["# Baselines: intent engine vs not-personalising vs popularity",
             "",
             "Generated by `evaluate_baselines.py` (held-out, censored).",
             "Offline logs cannot measure true page-lift; these are the two",
             "honest proxies: who a policy targets (concentration) and whether",
             "the served module matches next behaviour (hit-rate)."]

    print("RetailRocket concentration ...", flush=True)
    rr = pd.read_csv("real_sessions.csv")
    concentration("RetailRocket — final month",
                  rr[rr["start_ms"] >= SPLIT_MS], lines, rng)
    del rr

    print("REES46 concentration ...", flush=True)
    cols = (["month", "added_to_cart", "purchased",
             "views_before_first_commercial"] + BASE_FEATS)
    re = pd.read_csv("rees46_sessions.csv", usecols=lambda c: c in cols)
    nov = re[re["month"] == "2019-11"]
    concentration(f"REES46 — November ({REES_SAMPLE//1000}k sample, Black Friday)",
                  nov.sample(n=min(REES_SAMPLE, len(nov)), random_state=SEED),
                  lines, rng)
    del re, nov

    print("RetailRocket next-action hit-rates ...", flush=True)
    next_action_hits(lines)

    with open("reports/baseline-comparison.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote reports/baseline-comparison.md")


if __name__ == "__main__":
    main()
