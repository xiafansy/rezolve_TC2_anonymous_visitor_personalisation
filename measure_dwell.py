"""
measure_dwell.py -- derive dwell-time weights from data, not intuition.

Sauvik's ask: a 60-120s dwell band (he flags PDP dwell as a key intent
signal). Before touching the engine we measure, on CENSORED train-window
features only:

  A. MICRO-VISITS (censored n_events == 2): session duration = the one gap.
     Is a 60-120s two-event visit really as dead as a <15s bounce?
  B. ENGAGED sessions (>=3 events, >=2 views): median inter-event gap bands
     -- the offline proxy for per-PDP dwell. Where does conversion peak,
     and is there an IDLE plateau (left-the-tab) where credit should stop?
  C. Same gap bands CONDITIONED on low revisit (rr < 1.8) -- does dwell add
     signal beyond the re-view rule, or just shadow it?

Train windows only (RetailRocket: first ~3.5 months; REES46: October).
Held-out months stay untouched until the after/before impact run.
"""

import pandas as pd

SPLIT_MS = 1439856000000  # RetailRocket: last month starts here

DUR_BANDS = [(0, 15, "<15s"), (15, 60, "15-60s"), (60, 120, "60-120s"),
             (120, 300, "120-300s"), (300, 1e9, ">300s")]
GAP_BANDS = [(0, 5, "<5s"), (5, 15, "5-15s"), (15, 60, "15-60s"),
             (60, 120, "60-120s"), (120, 300, "120-300s"),
             (300, 600, "300-600s"), (600, 1e9, ">600s (idle?)")]


def bands(df, col, cuts, outcome="purchased"):
    rows = []
    for lo, hi, label in cuts:
        m = df[(df[col] >= lo) & (df[col] < hi)]
        if len(m) < 200:
            rows.append((label, len(m), None, None))
            continue
        rows.append((label, len(m), m["added_to_cart"].mean() * 100,
                     m[outcome].mean() * 100))
    print(f"  {'band':<14} {'n':>9}  {'cart%':>6}  {'buy%':>6}")
    for label, n, cart, buy in rows:
        c = f"{cart:6.2f}" if cart is not None else "   n/a"
        b = f"{buy:6.2f}" if buy is not None else "   n/a"
        print(f"  {label:<14} {n:>9,}  {c}  {b}")


def analyse(name, df):
    print(f"\n{'=' * 64}\n{name}  (train window, censored features)\n{'=' * 64}")

    micro2 = df[df["n_events"] == 2]
    print(f"\nA. micro-visits (censored n_events == 2, n={len(micro2):,})"
          f" -- duration bands")
    bands(micro2, "duration_sec", DUR_BANDS)
    one = df[df["n_events"] == 1]
    print(f"  [reference: 1-event sessions n={len(one):,} "
          f"cart {one['added_to_cart'].mean():.2%} buy {one['purchased'].mean():.2%}]")

    eng = df[(df["n_events"] >= 3) & (df["n_views"] >= 2)]
    print(f"\nB. engaged sessions (n={len(eng):,}) -- median inter-event gap bands")
    bands(eng, "median_gap_sec", GAP_BANDS)

    lowrev = eng[eng["revisit_ratio"] < 1.8]
    print(f"\nC. engaged & revisit<1.8 (n={len(lowrev):,}) -- gap bands "
          f"(dwell beyond the re-view rule?)")
    bands(lowrev, "median_gap_sec", GAP_BANDS)

    print(f"\nD. engaged sessions -- total duration bands (for the offline "
          f"'brief shallow visit' rule)")
    bands(eng, "duration_sec",
          [(0, 60, "<60s"), (60, 120, "60-120s"), (120, 300, "120-300s"),
           (300, 900, "300-900s"), (900, 1e9, ">900s")])


def main():
    rr = pd.read_csv("real_sessions.csv",
                     usecols=["start_ms", "n_events", "n_views", "duration_sec",
                              "median_gap_sec", "revisit_ratio",
                              "added_to_cart", "purchased"])
    analyse("RetailRocket", rr[rr["start_ms"] < SPLIT_MS])
    del rr

    re46 = pd.read_csv("rees46_sessions.csv",
                       usecols=["month", "n_events", "n_views", "duration_sec",
                                "median_gap_sec", "revisit_ratio",
                                "added_to_cart", "purchased"],
                       dtype={"n_events": "float32", "n_views": "float32",
                              "duration_sec": "float32",
                              "median_gap_sec": "float32",
                              "revisit_ratio": "float32"})
    analyse("REES46", re46[re46["month"] == "2019-10"])


if __name__ == "__main__":
    main()
