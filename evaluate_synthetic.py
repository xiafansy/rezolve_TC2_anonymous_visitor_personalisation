"""
evaluate_synthetic.py -- honest evaluation of the v3 two-stage engine.

What is measured (and why it's different from v1's single number)
-----------------------------------------------------------------
1. PREFIX ACCURACY -- the question production actually asks is "what should
   the homepage be after event k?", so we score the engine at k = 0 (arrival,
   Stage-1 prior only), 1, 2, 3, 5, and end-of-session. v1's 97.5% was a
   whole-session-hindsight number; these are decision-time numbers.
2. CALIBRATION -- temperature is FITTED on a fit split (NLL grid search) and
   ECE is reported on the test split. Confidence now means something.
3. GATE -- the Unclear threshold is FITTED (smallest threshold reaching a
   target precision on decided sessions), then coverage/precision are
   reported on test. "Decline to personalise" becomes a tuned decision,
   not a magic 0.42.
4. OVERRIDE -- Decisive (cart-with-<=2-views) hit rate and outcome.

Split: sessions are randomly assigned 40% fit / 60% test (seeded). The fit
split is used ONLY for temperature + gate; all reported tables are test-only.
"""

import hashlib
import json
import pandas as pd

from intent_engine import (ALL_INTENTS, TwoStageEngine, fit_temperature,
                           fit_gate, expected_calibration_error)

SEED = 7
FIT_FRACTION = 0.40
TARGET_PRECISION = 0.85
CHECKPOINTS = [0, 1, 2, 3, 5, "end"]
PARAMS_OUT = "engine_params.json"   # fitted T + gate; demo.py reads this

CTX_COLS = ["referrer", "device", "landing", "hour"]
EVENT_COLS = ["type", "ts", "item", "category", "query", "sort_key", "filter_key"]


def load():
    s = pd.read_csv("synthetic_sessions.csv")
    e = pd.read_csv("synthetic_events.csv", keep_default_na=False)
    events_by_sid = {sid: g[EVENT_COLS].to_dict("records")
                     for sid, g in e.sort_values(["session_id", "idx"]).groupby("session_id")}
    return s, events_by_sid


def run_session(engine, ctx, events, checkpoints):
    """Replay one session; capture Inference at each checkpoint. Returns
    {checkpoint: Inference} plus the final raw scores (for calibration)."""
    out = {}
    inf = engine.start_session(ctx)
    if 0 in checkpoints:
        out[0] = inf
    for k, ev in enumerate(events, 1):
        inf = engine.observe(ev)
        if k in checkpoints:
            out[k] = inf
    out["end"] = inf
    # carry last known inference forward for checkpoints beyond session length
    for c in checkpoints:
        if c not in out and c != "end":
            out[c] = inf
    return out, engine._combined_scores() if engine.override is None else None


def assign_split(session_id):
    """Deterministic per-session split.

    Was `random.random()` walked down the frame, which silently reshuffles
    every session's split the moment the generator's size or row order
    changes. Hashing the id keeps a session on the same side of the fence
    forever, so fit and test can never trade places between regenerations.
    """
    h = hashlib.blake2b(f"{SEED}:{session_id}".encode(), digest_size=8).digest()
    return "fit" if int.from_bytes(h, "big") / 2**64 < FIT_FRACTION else "test"


def main():
    sessions, events_by_sid = load()
    sessions["split"] = [assign_split(sid) for sid in sessions["session_id"]]
    fit_df = sessions[sessions.split == "fit"]
    test_df = sessions[sessions.split == "test"]
    print(f"fit sessions: {len(fit_df):,}   test sessions: {len(test_df):,}\n")

    # ---------------- pass 1: FIT temperature + gate on fit split -----------
    # Calibrate on scores captured MID-session (k=3) as well as at the end,
    # because production decisions happen throughout, not only in hindsight.
    score_rows, labels = [], []
    for _, row in fit_df.iterrows():
        ctx = {c: row[c] for c in CTX_COLS}
        eng = TwoStageEngine()
        eng.start_session(ctx)
        events = events_by_sid.get(row.session_id, [])
        for k, evt in enumerate(events, 1):
            eng.observe(evt)
            if k == 3 and eng.override is None:
                score_rows.append(eng._combined_scores()); labels.append(row.intent)
        if eng.override is None:  # overrides bypass softmax; skip for calib
            score_rows.append(eng._combined_scores()); labels.append(row.intent)

    T = fit_temperature(score_rows, labels)
    print(f"fitted temperature: {T}")

    # Genuinely ungated now: the engine used to coerce gate=0.0 back to its
    # 0.45 default (falsy-`or` bug), so every "ungated" confidence below 0.45
    # arrived pre-labelled Unclear and was scored as WRONG -- the gate was
    # being fitted against its own output.
    eng = TwoStageEngine(temperature=T, gate=0.0, cold_gate=0.0)
    assert eng.gate == 0.0 and eng.cold_gate == 0.0, "ungated pass is gated"
    confs, corrects = [], []
    for _, row in fit_df.iterrows():
        ctx = {c: row[c] for c in CTX_COLS}
        res, _ = run_session(eng, ctx, events_by_sid.get(row.session_id, []), ["end"])
        inf = res["end"]
        if inf.mode == "override":
            continue
        confs.append(inf.confidence)
        corrects.append(1 if inf.intent == row.intent else 0)
    gate, cov_fit, prec_fit = fit_gate(confs, corrects, TARGET_PRECISION)
    print(f"fitted gate: {gate:.2f}  (fit-split coverage {cov_fit:.0%}, "
          f"precision {prec_fit:.0%}, target {TARGET_PRECISION:.0%})\n")

    # ---------------- pass 2: EVALUATE on test split -------------------------
    eng = TwoStageEngine(temperature=T, gate=gate, cold_gate=max(gate, 0.55))
    records = []
    for _, row in test_df.iterrows():
        ctx = {c: row[c] for c in CTX_COLS}
        res, _ = run_session(eng, ctx, events_by_sid.get(row.session_id, []), CHECKPOINTS)
        for cp in CHECKPOINTS:
            inf = res[cp]
            records.append(dict(
                session_id=row.session_id, checkpoint=str(cp), true=row.intent,
                pred=inf.intent, conf=inf.confidence, mode=inf.mode,
                served=inf.served, purchased=row.purchased,
            ))
    r = pd.DataFrame(records)

    # ---- prefix accuracy table ----------------------------------------------
    print("=" * 68)
    print("PREFIX ACCURACY (test split) -- what the homepage would decide")
    print("=" * 68)
    print(f"{'after':>8} | {'decided':>8} | {'acc(decided)':>12} | "
          f"{'acc(all)':>9} | {'unclear':>8} | {'override':>8}")
    for cp in [str(c) for c in CHECKPOINTS]:
        g = r[r.checkpoint == cp]
        decided = g[(g.pred != "Unclear")]
        beh = decided[decided["mode"] != "override"]
        acc_dec = (beh.pred == beh.true).mean() if len(beh) else float("nan")
        # acc(all): Unclear counts as wrong unless the true label would have
        # been served acceptably by Neutral -- strictest read: count as wrong.
        acc_all = (g.pred == g.true).mean()
        print(f"{cp:>8} | {len(decided)/len(g):>7.0%} | {acc_dec:>12.1%} | "
              f"{acc_all:>9.1%} | {(g.pred=='Unclear').mean():>8.0%} | "
              f"{(g['mode']=='override').mean():>8.1%}")

    # ---- end-of-session detail ----------------------------------------------
    end = r[r.checkpoint == "end"]
    beh_end = end[end["mode"] != "override"]
    dec_end = beh_end[beh_end.pred != "Unclear"]

    print("\nCONFUSION (test, end-of-session, decided & non-override)")
    cm = pd.crosstab(dec_end.true, dec_end.pred).reindex(
        index=ALL_INTENTS, columns=ALL_INTENTS, fill_value=0)
    print(cm.to_string())

    print("\nPER-CLASS (decided)")
    rows = []
    for lbl in ALL_INTENTS:
        tp = cm.loc[lbl, lbl]
        fp = cm[lbl].sum() - tp
        fn = cm.loc[lbl].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0
        rows.append(dict(intent=lbl, precision=round(p, 3), recall=round(rc, 3),
                         f1=round(f1, 3), support=int(cm.loc[lbl].sum())))
    print(pd.DataFrame(rows).set_index("intent").to_string())

    # ---- calibration ----------------------------------------------------------
    ece = expected_calibration_error(
        beh_end.conf.tolist(), (beh_end.pred == beh_end.true).astype(int).tolist())
    print(f"\nCALIBRATION (test, end): ECE = {ece:.3f} at fitted T={T} "
          f"(gate-decided precision {(dec_end.pred == dec_end.true).mean():.1%})")

    print(f"\nmean confidence when correct: "
          f"{beh_end[beh_end.pred == beh_end.true].conf.mean():.0%}   "
          f"when wrong: {beh_end[(beh_end.pred != beh_end.true) & (beh_end.pred != 'Unclear')].conf.mean():.0%}")

    # ---- override outcomes -----------------------------------------------------
    ov = end[end["mode"] == "override"]
    if len(ov):
        print(f"\nDECISIVE OVERRIDE: fired on {len(ov)/len(end):.1%} of sessions; "
              f"purchase rate among them {ov.purchased.mean():.0%} "
              f"(base {end.purchased.mean():.0%})")

    # ---- what gets SERVED at k=2 (the user's 'last 2-3 clicks' moment) --------
    k2 = r[r.checkpoint == "2"]
    print("\nSERVED PAGE MIX after 2 events (test):")
    print((k2.served.value_counts(normalize=True) * 100).round(1).to_string())

    # Publish the fitted params so demo.py (and any other consumer) uses the
    # numbers this run actually produced. They used to be hand-copied into
    # demo.py, and had already drifted (demo said 0.43, the fit says 0.41).
    params = dict(temperature=T, gate=round(gate, 4),
                  cold_gate=round(max(gate, 0.55), 4),
                  target_precision=TARGET_PRECISION, seed=SEED,
                  fit_sessions=int(len(fit_df)), test_sessions=int(len(test_df)),
                  ece=round(float(ece), 4))
    with open(PARAMS_OUT, "w") as fh:
        json.dump(params, fh, indent=2)
    print(f"\nfitted params -> {PARAMS_OUT}: "
          f"TwoStageEngine(temperature={T}, gate={gate:.2f})")


if __name__ == "__main__":
    main()
