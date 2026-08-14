"""
ga4_prior_calibration.py -- calibrate the Stage-1 cold-start prior on real
context signals (GA4: traffic source, device, geo, hour).

Why: the engine's arrival prior (intent_engine.prior_evidence) is hand-set;
no real log we had carried referrer/device/geo. The GA4 obfuscated sample
(Google Merchandise Store) does. This script turns it into evidence:

  per context level (e.g. medium=organic, device=mobile, continent=Europe):
      sessions, purchase rate, lift vs base
      suggested additive prior weight = clip(1.5 * (lift - 1), -1.5, +1.5)
      (matching the engine's prior scale, where weights stay <= 1.5)

Cells below MIN_SESSIONS are reported as INSUFFICIENT and given no weight --
a 40-event preview produces a table of honest refusals, not fake numbers.

Data quality gate
-----------------
Spreadsheet round-trips mangle GA4 exports. The script detects and warns on:
  * event_timestamp in scientific notation (hour precision destroyed)
  * user_pseudo_id parsed as float (visitor identity destroyed)
Get the raw export: BigQuery public dataset
`bigquery-public-data.ga4_obfuscated_sample_ecommerce` (3 months, ~4.3M
events) or the full Kaggle download -- place CSVs in archive_ga4/.

Usage
-----
  /usr/bin/python3 ga4_prior_calibration.py [--glob 'archive_ga4/*.csv']

  # preferred: aggregate in BigQuery (ga4_context_export.sql), download the
  # few-hundred-row cell table, then:
  /usr/bin/python3 ga4_prior_calibration.py --cells archive_ga4/ga4_context_cells.csv
"""

import argparse
import glob as globmod

import numpy as np
import pandas as pd

MIN_SESSIONS = 500
SESSION_GAP_US = 30 * 60 * 1_000_000  # GA4 timestamps are microseconds

SOCIAL = ("instagram", "facebook", "twitter", "t.co", "pinterest", "tiktok",
          "youtube", "linkedin", "reddit")

COLS = ["event_timestamp", "event_name", "user_pseudo_id",
        "traffic_source.medium", "traffic_source.source",
        "device.category", "geo.continent", "geo.country"]


def referrer_bucket(medium, source):
    m, s = str(medium).lower(), str(source).lower()
    if m in ("(none)", "(direct)", "nan") or s == "(direct)":
        return "direct"
    if m == "organic":
        return "search"
    if m in ("cpc", "ppc", "paidsearch", "display", "cpm"):
        return "ad"
    if m == "email" or "email" in s:
        return "email"
    if m == "referral" and any(x in s for x in SOCIAL):
        return "social"
    if m == "referral":
        return "referral"
    return "other"


def quality_gate(df):
    warns = []
    ts = df["event_timestamp"]
    if ts.dtype == object and ts.astype(str).str.contains("E+", regex=False).any():
        warns.append("event_timestamp in scientific notation -> hour precision destroyed")
    else:
        # microsecond timestamps ending in many zeros = precision loss
        t = pd.to_numeric(ts, errors="coerce").dropna()
        if len(t) and (t % 1_000_000_000 == 0).mean() > 0.9:
            warns.append("event_timestamp precision truncated (spreadsheet round-trip?)")
    uid = df["user_pseudo_id"].astype(str)
    if uid.str.match(r"^\d+\.\d+$").mean() > 0.5:
        warns.append("user_pseudo_id looks float-mangled -> visitor identity unreliable")
    return warns


def load(pattern):
    paths = sorted(globmod.glob(pattern)) or sorted(set(globmod.glob("archive/*ga4*")))
    if not paths:
        raise SystemExit(f"no GA4 csvs found (looked for {pattern} and archive/*ga4*)")
    frames = []
    for p in paths:
        df = pd.read_csv(p, usecols=lambda c: c in COLS, low_memory=False)
        frames.append(df)
        print(f"loaded {p}: {len(df):,} rows")
    df = pd.concat(frames, ignore_index=True)

    for w in quality_gate(df):
        print(f"  !! DATA QUALITY: {w}")

    # long-format exports repeat the event row per param -> dedupe to events
    df["ts"] = pd.to_numeric(df["event_timestamp"], errors="coerce")
    ev = (df.dropna(subset=["ts", "user_pseudo_id"])
            .drop_duplicates(["user_pseudo_id", "ts", "event_name"]))
    print(f"events after dedupe: {len(ev):,}  visitors: {ev['user_pseudo_id'].nunique():,}")
    return ev


def sessionize(ev):
    ev = ev.sort_values(["user_pseudo_id", "ts"], kind="mergesort")
    new = (ev["user_pseudo_id"].ne(ev["user_pseudo_id"].shift())
           | (ev["ts"].diff() > SESSION_GAP_US))
    ev = ev.assign(session_id=new.cumsum())
    g = ev.groupby("session_id")
    first = g[["traffic_source.medium", "traffic_source.source",
               "device.category", "geo.continent", "ts"]].first()
    s = pd.DataFrame({
        "referrer": [referrer_bucket(m, src) for m, src in zip(
            first["traffic_source.medium"], first["traffic_source.source"])],
        "device": first["device.category"].str.lower(),
        "continent": first["geo.continent"],
        "hour": pd.to_datetime(first["ts"], unit="us", errors="coerce").dt.hour,
        "purchased": ev["event_name"].eq("purchase").groupby(ev["session_id"]).any(),
        "carted": ev["event_name"].eq("add_to_cart").groupby(ev["session_id"]).any(),
    }, index=first.index)
    print(f"sessions: {len(s):,}  purchase rate: {s['purchased'].mean():.2%}")
    return s


def calibrate(s, dim):
    base = s["purchased"].mean()
    print(f"\n-- prior calibration by {dim} (base purchase {base:.2%}) --")
    print(f"{'level':<14} {'sessions':>9} {'buy%':>7} {'lift':>6}  suggested weight")
    for level, grp in s.groupby(dim, dropna=False, observed=True):
        n = len(grp)
        if n < MIN_SESSIONS:
            print(f"{str(level):<14} {n:>9,} {'--':>7} {'--':>6}  INSUFFICIENT (<{MIN_SESSIONS})")
            continue
        rate = grp["purchased"].mean()
        lift = rate / base if base else float("nan")
        w = float(np.clip(1.5 * (lift - 1), -1.5, 1.5))
        print(f"{str(level):<14} {n:>9,} {rate:>6.2%} {lift:>5.2f}x  {w:+.2f}")


def calibrate_cells(path):
    """BigQuery-aggregated mode: read the cell table produced by
    ga4_context_export.sql (referrer, device, continent, hour_bucket,
    sessions, purchases, carts) and print the same calibration tables."""
    c = pd.read_csv(path)
    total_sessions = c["sessions"].sum()
    total_purchases = c["purchases"].sum()
    base = total_purchases / total_sessions
    print(f"cells: {len(c):,}  sessions: {total_sessions:,}  "
          f"purchases: {total_purchases:,}  base rate: {base:.2%}")
    for dim in ["referrer", "device", "continent", "hour_bucket"]:
        g = c.groupby(dim, dropna=False)[["sessions", "purchases"]].sum()
        print(f"\n-- prior calibration by {dim} (base purchase {base:.2%}) --")
        print(f"{'level':<14} {'sessions':>9} {'buy%':>7} {'lift':>6}  suggested weight")
        for level, row in g.sort_values("sessions", ascending=False).iterrows():
            n = int(row["sessions"])
            if n < MIN_SESSIONS:
                print(f"{str(level):<14} {n:>9,} {'--':>7} {'--':>6}  "
                      f"INSUFFICIENT (<{MIN_SESSIONS})")
                continue
            rate = row["purchases"] / n
            lift = rate / base
            w = float(np.clip(1.5 * (lift - 1), -1.5, 1.5))
            print(f"{str(level):<14} {n:>9,} {rate:>6.2%} {lift:>5.2f}x  {w:+.2f}")
    print("\nAdopt a weight only where the table shows one; everything else "
          "stays at the hand-set default until real volume exists.")


def main(pattern):
    ev = load(pattern)
    s = sessionize(ev)
    if s["purchased"].sum() < 30:
        print("\n!! fewer than 30 purchases in the file -- this is a preview "
              "extract, not the dataset; weights below will be INSUFFICIENT.")
    s["hour_bucket"] = pd.cut(s["hour"], [0, 6, 12, 18, 24], right=False,
                              labels=["night", "morning", "afternoon", "evening"])
    for dim in ["referrer", "device", "continent", "hour_bucket"]:
        calibrate(s, dim)
    print("\nAdopt a weight only where the table shows one; everything else "
          "stays at the hand-set default until real volume exists.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="archive_ga4/*.csv")
    ap.add_argument("--cells", help="BigQuery-aggregated cell CSV "
                                    "(from ga4_context_export.sql)")
    a = ap.parse_args()
    if a.cells:
        calibrate_cells(a.cells)
    else:
        main(a.glob)
