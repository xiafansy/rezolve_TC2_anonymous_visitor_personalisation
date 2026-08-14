"""
evaluate_ablation.py -- per-rule knockout + weight sensitivity on held-out data.

Answers Sauvik's "keep validating the weights": for every scoring rule in the
offline engine path, what actually breaks when it's removed, and how sensitive
are the conclusions to its exact weight (x0.5 / x1.5)?

Metrics per run (held-out, censored):
  cov%        coverage of each committed intent
  eval_buy%   Evaluator purchase rate  (dilution detector)
  expl_buy%   Explorer purchase rate
  top2_share  % of ALL purchases captured by Decisive + Evaluator sessions
  mono        Evaluator > Explorer > Low-intent purchase ordering

Datasets: RetailRocket final month (full) and REES46 November (500k sample,
seed 42 -- Black Friday). Writes reports/weight-ablation.md.
"""

import pandas as pd

from intent_engine import REAL_INTENTS, TwoStageEngine

SPLIT_MS = 1439856000000
REES_SAMPLE = 500_000
SEED = 42

RULES = ["micro", "micro_short", "micro_linger", "revisit_core",
         "revisit_focus", "revisit_pace", "no_revisit", "broad_cats",
         "cat_switch", "scan_no_engage", "brief_shallow", "few_signals",
         "decisive_override", "cheap_flavor"]
SENSITIVITY_RULES = ["micro", "revisit_core", "revisit_focus", "broad_cats",
                     "cat_switch", "scan_no_engage", "brief_shallow"]

BASE_FEATS = ["n_events", "n_views", "revisit_ratio", "n_categories",
              "top_category_share", "category_switch_rate", "median_gap_sec",
              "duration_sec", "price_rel_cat"]


def load_retailrocket():
    df = pd.read_csv("real_sessions.csv")
    test = df[df["start_ms"] >= SPLIT_MS]
    return test


def load_rees46():
    cols = (["month", "added_to_cart", "purchased",
             "views_before_first_commercial"] + BASE_FEATS)
    df = pd.read_csv("rees46_sessions.csv", usecols=lambda c: c in cols)
    nov = df[df["month"] == "2019-11"]
    return nov.sample(n=min(REES_SAMPLE, len(nov)), random_state=SEED)


def to_records(test):
    rows = []
    recs = test.to_dict("records")
    for r in recs:
        f = {k: r.get(k) for k in BASE_FEATS}
        f["added_to_cart"] = r.get("added_to_cart")
        f["views_before_first_commercial"] = r.get("views_before_first_commercial")
        rows.append((f, bool(r["purchased"]), bool(r["added_to_cart"])))
    return rows


def run(rows, disabled=(), scale=None):
    eng = TwoStageEngine(intents=REAL_INTENTS, disabled_rules=disabled,
                         weight_scale=scale)
    n = len(rows)
    stats = {i: [0, 0] for i in
             ["Decisive", "Evaluator", "Explorer", "Unclear", "Low-intent"]}
    total_buys = 0
    for f, buy, _cart in rows:
        inf = eng.score_aggregates(f)
        s = stats[inf.intent]
        s[0] += 1
        s[1] += buy
        total_buys += buy
    out = {}
    for i, (cnt, buys) in stats.items():
        out[f"cov_{i}"] = cnt / n * 100
        out[f"buy_{i}"] = (buys / cnt * 100) if cnt else float("nan")
    top2 = stats["Decisive"][1] + stats["Evaluator"][1]
    out["top2_share"] = top2 / max(1, total_buys) * 100
    out["mono"] = (out["buy_Evaluator"] > out["buy_Explorer"] > out["buy_Low-intent"])
    return out


def fmt_row(name, m, base):
    d = lambda k: m[k] - base[k]
    return (f"| {name} | {m['cov_Evaluator']:.1f} ({d('cov_Evaluator'):+.1f}) "
            f"| {m['buy_Evaluator']:.2f} ({d('buy_Evaluator'):+.2f}) "
            f"| {m['buy_Explorer']:.2f} ({d('buy_Explorer'):+.2f}) "
            f"| {m['top2_share']:.1f} ({d('top2_share'):+.1f}) "
            f"| {'PASS' if m['mono'] else 'FAIL'} |")


def ablate(name, rows, lines):
    base = run(rows)
    lines.append(f"\n### {name}\n")
    lines.append(f"Baseline: Evaluator cov {base['cov_Evaluator']:.1f}% "
                 f"buy {base['buy_Evaluator']:.2f}% · Explorer buy "
                 f"{base['buy_Explorer']:.2f}% · Decisive+Evaluator capture "
                 f"{base['top2_share']:.1f}% of purchases · mono "
                 f"{'PASS' if base['mono'] else 'FAIL'}\n")
    lines.append("| removed rule | Eval cov% (Δ) | Eval buy% (Δ) | "
                 "Expl buy% (Δ) | top2 purch share% (Δ) | mono |")
    lines.append("|---|---|---|---|---|---|")
    for rid in RULES:
        m = run(rows, disabled={rid})
        lines.append(fmt_row(f"−{rid}", m, base))
        print(f"  knocked out {rid}", flush=True)

    lines.append("\n**Weight sensitivity (x0.5 / x1.5):**\n")
    lines.append("| rule | Eval buy% @x0.5 | @x1.0 | @x1.5 | top2 share @x0.5 | @x1.0 | @x1.5 |")
    lines.append("|---|---|---|---|---|---|---|")
    for rid in SENSITIVITY_RULES:
        lo = run(rows, scale={rid: 0.5})
        hi = run(rows, scale={rid: 1.5})
        lines.append(f"| {rid} | {lo['buy_Evaluator']:.2f} | "
                     f"{base['buy_Evaluator']:.2f} | {hi['buy_Evaluator']:.2f} "
                     f"| {lo['top2_share']:.1f} | {base['top2_share']:.1f} "
                     f"| {hi['top2_share']:.1f} |")
        print(f"  sensitivity {rid}", flush=True)
    return base


def main():
    lines = ["# Weight ablation & sensitivity (held-out, censored)",
             "",
             "Generated by `evaluate_ablation.py`. Each row = one rule knocked",
             "out (or its weight scaled); deltas vs the full engine baseline.",
             "`top2 purch share` = % of all purchases sitting in sessions the",
             "engine labels Decisive or Evaluator — the concentration the",
             "business case banks on."]

    print("RetailRocket final month ...", flush=True)
    rr = to_records(load_retailrocket())
    ablate(f"RetailRocket — final month (n={len(rr):,})", rr, lines)
    del rr

    print("REES46 November sample ...", flush=True)
    re = to_records(load_rees46())
    ablate(f"REES46 — November sample (n={len(re):,}, Black Friday)", re, lines)

    with open("reports/weight-ablation.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote reports/weight-ablation.md")


if __name__ == "__main__":
    main()
