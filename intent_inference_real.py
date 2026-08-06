"""
Intent inference v2 -- adapted to REAL anonymous-session signals (RetailRocket).

What changed vs the synthetic-signal inferer (intent_inference.py)
------------------------------------------------------------------
The real event log has no referrer / device / search / sort signals, but it has
something better: true browsing sequences. And the real data *corrected* two of
our synthetic assumptions:

  * "Fast + focused = decisive buyer"  ->  WRONG on real traffic. Short focused
    bursts convert worst (1.2%). True decisive buyers are commercial-event-led
    (view once -> cart -> buy, 75% purchase rate).
  * "Researchers are slow to convert"  ->  WRONG. Deep evaluators (re-viewing
    the same item, single category) are the hottest browsing segment: 9.7%
    purchase and 41% of all engaged-session purchases.

Architecture: two stages, mirroring what runs in real time.
  Stage A  browse-pattern intent from view-only signals (no commercial events,
           so downstream conversion is an honest validation target).
  Stage B  event-triggered override: an add-to-cart with minimal prior
           browsing flips the session to Decisive checkout-support mode.

Intents (v2): Evaluator | Explorer | Low-intent | (gate) Unclear | (override) Decisive
Price-sensitive is NOT inferable from this dataset (no price/search signals) --
kept in the synthetic pipeline as a demonstration of how it slots in.

Honesty rules
-------------
* Thresholds are fit as quantiles on a TRAINING window (first ~3.5 months);
  evaluation happens on the final month only (evaluate_real.py).
* Stage-A evidence uses only pre-commercial view-pattern features.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Behavioural thresholds.
#
# Derived from TRAIN-window quantiles by evaluate_real.py (--fit); the values
# below are those fitted numbers, kept explicit so the scorer is fully legible.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "revisit_hi": 1.8,       # ~p75 of revisit_ratio     -> re-viewing items
    "revisit_lo": 1.1,       # ~p25                       -> hardly any re-views
    "scan_rev_max": 1.4,     # below this, category focus = drive-by scan
    "cats_broad": 3,         # >=3 categories touched     -> browsing wide
    "switch_hi": 0.5,        # >=50% of view steps change category
    "focus_hi": 0.9,         # >=90% of views in one category
    "views_deep": 4,         # enough views to call a pattern
    "gap_slow": 120.0,       # median inter-event gap (s) -> deliberate pace
    "dur_micro": 90.0,       # sessions shorter than this barely happened
    "events_micro": 2,       # <=2 events = micro-visit
}

INTENTS = ["Evaluator", "Explorer", "Low-intent"]
MIN_CONFIDENCE = 0.45


@dataclass
class Evidence:
    intent: str
    weight: float
    reason: str


def gather_evidence(s, t=THRESHOLDS):
    """Stage-A evidence from view-pattern signals ONLY (no cart/purchase)."""
    ev = []

    def add(intent, w, reason):
        ev.append(Evidence(intent, w, reason))

    n_views = s.get("n_views", 0) or 0
    revisit = s.get("revisit_ratio", 1.0) or 1.0
    n_cats = s.get("n_categories", 0) or 0
    focus = s.get("top_category_share", 0.0) or 0.0
    switch = s.get("category_switch_rate", 0.0) or 0.0
    gap = s.get("median_gap_sec", 0.0) or 0.0
    dur = s.get("duration_sec", 0.0) or 0.0
    n_events = s.get("n_events", 0) or 0

    # --- micro-visit: barely any signal at all ------------------------------
    if n_events <= t["events_micro"] or n_views < 2:
        add("Low-intent", 2.5, f"micro-visit ({n_events} events) -- almost no signal left behind")
        if dur < t["dur_micro"]:
            add("Low-intent", 1.0, f"gone in {dur:.0f}s")
        return ev  # nothing else worth weighing

    # --- Evaluator: re-viewing the same item(s) is the REQUIRED core signal.
    # Train-window bands: revisit>=1.8 buys at 10%+ vs 4% below; focus or slow
    # pace WITHOUT re-views buys at only 1-5%, so those are amplifiers, never
    # sufficient evidence on their own (the v2.0 dilution bug).
    if revisit >= t["revisit_hi"]:
        add("Evaluator", 3.0, f"re-viewed items {revisit:.1f}x on average (comparison behaviour)")
        if focus >= 0.75:
            add("Evaluator", 1.5, f"{focus:.0%} of views inside one category while re-viewing")
        if gap >= t["gap_slow"]:
            add("Evaluator", 0.5, f"deliberate pace (median {gap:.0f}s between actions)")
    elif revisit <= t["revisit_lo"]:
        add("Explorer", 1.0, "almost never returned to an item")

    # --- Explorer: wide, switchy browsing -----------------------------------
    if n_cats >= t["cats_broad"]:
        add("Explorer", 3.0, f"touched {n_cats} categories (browsing wide)")
    if switch >= t["switch_hi"]:
        add("Explorer", 2.0, f"changed category on {switch:.0%} of view steps")

    # --- category scan without item engagement = drive-by, not evaluation ---
    # (focus>=0.9, revisit<1.4, views>=4 buys at 1.0% on train -- low-intent)
    if focus >= t["focus_hi"] and revisit < t["scan_rev_max"] and n_views >= t["views_deep"]:
        add("Low-intent", 1.5, "scanned one category without engaging any item twice")

    # --- short shallow visits lean low-intent -------------------------------
    if dur < t["dur_micro"] and revisit < t["revisit_hi"]:
        add("Low-intent", 2.0, f"brief shallow visit ({dur:.0f}s, no re-views)")
    if n_views <= 3 and revisit < t["revisit_hi"] and n_cats <= 2:
        add("Low-intent", 1.0, "too few distinct signals to suggest a mission")

    return ev


@dataclass
class Inference:
    intent: str                 # Evaluator | Explorer | Low-intent | Unclear | Decisive
    confidence: float
    scores: dict
    reasons: list = field(default_factory=list)
    stage: str = "A"            # "A" browse-pattern | "B" commercial override

    def explain(self):
        rank = ", ".join(f"{k} {v:.1f}" for k, v in
                         sorted(self.scores.items(), key=lambda kv: -kv[1]))
        why = "; ".join(self.reasons[:3])
        return (f"{self.intent} (stage {self.stage}, confidence {self.confidence:.0%})\n"
                f"    scores: {rank}\n    because: {why}")


def infer_realtime(s, min_confidence=MIN_CONFIDENCE, t=THRESHOLDS):
    """
    Full two-stage inference for one session dict.

    Stage B fires when commercial action arrives with minimal prior browsing --
    the visitor knew what they came for; switch to checkout support.
    Otherwise Stage A scores browse patterns, with an Unclear gate.
    """
    n_views = s.get("n_views", 0) or 0
    if s.get("added_to_cart") and n_views <= 2:
        return Inference(
            intent="Decisive", confidence=0.95,
            scores={"Decisive": 1.0},
            reasons=["added to cart with almost no browsing first "
                     "(arrived already decided)"],
            stage="B",
        )

    scores = {i: 0.0 for i in INTENTS}
    reasons_by = {i: [] for i in INTENTS}
    for e in gather_evidence(s, t):
        scores[e.intent] += e.weight
        reasons_by[e.intent].append((e.weight, e.reason))

    total = sum(scores.values())
    winner = max(scores, key=scores.get)
    confidence = scores[winner] / total if total > 0 else 0.0
    reasons = [r for _, r in sorted(reasons_by[winner], reverse=True)]

    if confidence < min_confidence:
        return Inference("Unclear", confidence, scores,
                         ["signals point in multiple directions -- decline to commit"],
                         stage="A")
    return Inference(winner, confidence, scores, reasons, stage="A")


# ---------------------------------------------------------------------------
# Homepage strategies for the real-signal intents
# ---------------------------------------------------------------------------
REAL_STRATEGIES = {
    "Decisive": "Checkout-support mode: sticky cart summary, one-tap checkout, "
                "shipping/returns reassurance. Get out of the way.",
    "Evaluator": "Comparison mode: specs side-by-side, reviews up front, "
                 "recently-viewed rail, and a CLEAR add-to-cart -- this is the "
                 "hottest browsing segment (9.7% purchase).",
    "Explorer": "Discovery mode: cross-category trending, curated collections, "
                "visible cart. Broad browsers still buy (7.9%).",
    "Low-intent": "Neutral-light: fast page, top categories, one broad promo. "
                  "Don't over-invest; capture a bookmark/newsletter if offered.",
    "Unclear": "Neutral: balanced search + trending + categories. "
               "Commit to nothing until signals separate.",
}


if __name__ == "__main__":
    demo = {
        "n_events": 7, "n_views": 6, "revisit_ratio": 3.0,
        "n_categories": 1, "top_category_share": 1.0,
        "category_switch_rate": 0.0, "median_gap_sec": 140.0,
        "duration_sec": 900.0, "added_to_cart": False,
    }
    r = infer_realtime(demo)
    print(r.explain())
    print("    strategy:", REAL_STRATEGIES[r.intent])
