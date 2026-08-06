"""
Intent inference layer for anonymous, first-time visitors.

Goal
----
Infer a visitor's *genuine real-time intent* from the limited signals available
in a single, un-authenticated session -- WITHOUT lookalike modelling and WITHOUT
any historical / labelled data about this visitor.

Why rule-based (not ML) as the primary logic?
----------------------------------------------
In the real cold-start scenario there are no intent labels to train on, so a
transparent weighted-evidence scorer is what would actually ship. It needs zero
training data, runs per-session in real time, and -- crucially -- it can explain
*why* it reached a verdict, which is what powers an honest personalisation story.

The scorer reads ONLY observable session signals. It never sees the ground-truth
`Intent` column; that column is used solely for offline evaluation (evaluate.py).

Signals deliberately NOT used
-----------------------------
- Time_of_Day : pure noise in the current data (no relationship to intent), so a
                faithful inferer must not lean on it.
- Browsing sequence : not captured in the dataset yet (a known gap / next step).
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# The four intents we distinguish
# ---------------------------------------------------------------------------
INTENTS = ["Goal-driven", "Explorer", "Research", "Price-sensitive"]


# ---------------------------------------------------------------------------
# Bucketing for continuous signals
#
# Sensible analyst buckets (low / medium / high), chosen from domain reasoning
# about shopper behaviour rather than reverse-engineered generator bounds.
# ---------------------------------------------------------------------------
def _scroll_bucket(v):
    if v < 45:
        return "low"
    if v <= 75:
        return "medium"
    return "high"


def _views_bucket(v):
    if v <= 6:
        return "low"
    if v <= 12:
        return "medium"
    return "high"


def _duration_bucket(v):
    if v < 300:
        return "short"
    if v <= 600:
        return "medium"
    return "long"


def _as_bool(v):
    """CSV round-trips booleans as the strings 'True' / 'False'."""
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


# ---------------------------------------------------------------------------
# Evidence rules
#
# Each rule contributes weighted evidence toward one intent. Weights encode how
# discriminative a signal is (a distinctive sort mode is worth more than a
# weakly-correlated referrer). The result is an interpretable additive score.
# ---------------------------------------------------------------------------
@dataclass
class Evidence:
    intent: str
    weight: float
    reason: str


def gather_evidence(s):
    """Return a list of Evidence items fired by session dict `s`."""
    ev = []

    def add(intent, weight, reason):
        ev.append(Evidence(intent, weight, reason))

    # --- Sort type: the single most discriminative on-page signal ----------
    sort_type = str(s.get("Sort_Type", "")).strip()
    if sort_type == "Trending":
        add("Explorer", 3.0, "sorting by Trending (browsing, not seeking)")
    elif sort_type in ("Relevance", "Newest"):
        add("Goal-driven", 2.5, f"sorting by {sort_type} (looking for a specific match)")
    elif sort_type == "Highest Rated":
        add("Research", 3.0, "sorting by Highest Rated (quality comparison)")
    elif sort_type == "Price Low-High":
        # Sorting by price alone is weak evidence of bargain-hunting -- researchers
        # sort by price too. Let Category=Sale (below) carry the Price-sensitive call.
        add("Research", 1.5, "sorting by Price Low-High (comparing options)")
        add("Price-sensitive", 0.5, "sorting by Price Low-High")

    # --- Category ----------------------------------------------------------
    category = str(s.get("Category", "")).strip()
    if category == "Sale":
        add("Price-sensitive", 3.0, "browsing the Sale category")
    elif category in ("Running", "Electronics", "Beauty"):
        add("Goal-driven", 0.5, f"in a purchase-oriented category ({category})")

    # --- Search behaviour --------------------------------------------------
    search_used = _as_bool(s.get("Search_Used", False))
    if search_used:
        add("Goal-driven", 1.0, "used search (arrived with something in mind)")
    else:
        add("Explorer", 1.5, "did not search (open-ended browsing)")

    if str(s.get("Search_Query", "")).strip().lower() == "discount":
        add("Price-sensitive", 1.5, "searched for 'discount'")

    # --- Filtering ---------------------------------------------------------
    if _as_bool(s.get("Filter_Used", False)):
        add("Research", 1.0, "applied filters (narrowing deliberately)")
        add("Price-sensitive", 1.0, "applied filters (narrowing deliberately)")
    else:
        add("Explorer", 1.0, "no filters (casual browsing)")
        add("Goal-driven", 0.5, "no filters (went straight to it)")

    # --- Device ------------------------------------------------------------
    if str(s.get("Device", "")).strip() == "Desktop":
        add("Research", 0.8, "on desktop (comfortable for deep comparison)")

    # --- Referrer ----------------------------------------------------------
    referrer = str(s.get("Referrer", "")).strip()
    if referrer == "Instagram":
        add("Explorer", 1.5, "arrived from Instagram (discovery channel)")
    elif referrer == "Facebook":
        add("Explorer", 1.0, "arrived from Facebook (discovery channel)")
    elif referrer == "Email":
        add("Price-sensitive", 1.0, "arrived from Email (promo-driven)")
        add("Research", 0.5, "arrived from Email")
    elif referrer == "Google":
        add("Goal-driven", 1.0, "arrived from Google (active search)")
        add("Research", 0.3, "arrived from Google")

    # --- Depth of engagement ----------------------------------------------
    scroll = _scroll_bucket(s.get("Scroll_Depth", 0))
    if scroll == "high":
        add("Explorer", 1.5, "very high scroll depth (grazing the whole page)")
    elif scroll == "low":
        add("Goal-driven", 1.5, "low scroll depth (found it fast)")
    else:
        add("Research", 0.5, "moderate scroll depth")
        add("Price-sensitive", 0.5, "moderate scroll depth")

    views = _views_bucket(s.get("Product_Views", 0))
    if views == "high":
        add("Explorer", 1.5, "viewed many products (broad browsing)")
    elif views == "low":
        add("Goal-driven", 1.5, "viewed few products (focused)")
    else:
        add("Research", 0.5, "viewed a moderate number of products")
        add("Price-sensitive", 0.5, "viewed a moderate number of products")

    duration = _duration_bucket(s.get("Session_Duration_sec", 0))
    if duration == "long":
        add("Explorer", 1.0, "long session (leisurely browsing)")
        add("Research", 1.0, "long session (careful evaluation)")
    elif duration == "short":
        add("Goal-driven", 1.5, "short session (in and out)")
    else:
        add("Price-sensitive", 0.5, "medium-length session")

    # --- Commercial signals ------------------------------------------------
    if _as_bool(s.get("Purchase", False)):
        add("Goal-driven", 1.0, "completed a purchase")
        add("Price-sensitive", 0.5, "completed a purchase")
    if _as_bool(s.get("Add_to_Cart", False)):
        add("Goal-driven", 0.5, "added to cart")

    return ev


# ---------------------------------------------------------------------------
# Inference result
# ---------------------------------------------------------------------------
@dataclass
class Inference:
    intent: str                       # predicted intent (argmax)
    confidence: float                 # share of total evidence going to winner
    scores: dict                      # raw score per intent
    reasons: list = field(default_factory=list)  # top human-readable reasons

    def explain(self):
        pct = ", ".join(
            f"{i} {self.scores[i]:.1f}" for i in
            sorted(self.scores, key=self.scores.get, reverse=True)
        )
        head = f"{self.intent} (confidence {self.confidence:.0%})"
        why = "; ".join(self.reasons[:3])
        return f"{head}\n    scores: {pct}\n    because: {why}"


def infer(session):
    """Infer intent for a single session dict. Uses observable signals only."""
    scores = {i: 0.0 for i in INTENTS}
    reasons_by_intent = {i: [] for i in INTENTS}

    for e in gather_evidence(session):
        scores[e.intent] += e.weight
        reasons_by_intent[e.intent].append((e.weight, e.reason))

    total = sum(scores.values())
    winner = max(scores, key=scores.get)
    confidence = scores[winner] / total if total > 0 else 0.0

    # Top reasons that pushed toward the winning intent, strongest first.
    top_reasons = [
        r for _, r in sorted(reasons_by_intent[winner], reverse=True)
    ]

    return Inference(
        intent=winner,
        confidence=confidence,
        scores=scores,
        reasons=top_reasons,
    )


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = {
        "Referrer": "Google",
        "Device": "Mobile",
        "Category": "Running",
        "Search_Used": True,
        "Search_Query": "running shoes",
        "Scroll_Depth": 30,
        "Product_Views": 3,
        "Filter_Used": False,
        "Sort_Type": "Relevance",
        "Session_Duration_sec": 120,
        "Add_to_Cart": True,
        "Purchase": True,
    }
    print(infer(demo).explain())
