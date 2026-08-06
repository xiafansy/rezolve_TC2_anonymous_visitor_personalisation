"""
personalisation.py (v3) -- turn an Inference into a homepage, stage-aware.

Serving modes
-------------
  cold-accent   arrival (Stage 1 only): NEUTRAL base layout, with ONE accent
                block tilted by the prior's argmax. A prior may tilt; it may
                not commit.
  personalised  behavioural confidence cleared the gate: full intent layout.
  checkout      Decisive override: get out of the visitor's way.
  neutral       Unclear: balanced page, commit to nothing.

Every block carries a rationale -> the whole chain (signal -> intent ->
layout) stays auditable.
"""

from dataclasses import dataclass, field

from intent_engine import Inference


@dataclass
class Homepage:
    mode: str                 # cold-accent | personalised | checkout | neutral
    intent: str
    confidence: float
    hero: str
    blocks: list              # [(title, rationale)]
    suppress: list = field(default_factory=list)
    reasons: list = field(default_factory=list)


LAYOUTS = {
    "Goal-driven": dict(
        hero="Search-first hero, pre-scoped to the visitor's category",
        blocks=[
            ("Best sellers in their category", "decisive buyers convert on proven items"),
            ("Free shipping / fast checkout banner", "remove last-mile hesitation"),
            ("Recently viewed rail", "one tap back to the exact product"),
        ],
        suppress=["editorial lookbooks", "cross-category discovery"]),
    "Evaluator": dict(
        hero="Comparison hero: the re-viewed items side-by-side",
        blocks=[
            ("Spec / price comparison table", "they are actively comparing -- do the work for them"),
            ("Reviews up front", "quality evidence closes evaluators"),
            ("Recently viewed rail + clear add-to-cart", "hottest browsing segment; make committing easy"),
        ],
        suppress=["broad promos", "unrelated trending"]),
    "Explorer": dict(
        hero="Full-bleed 'Trending now' visual, no search box competing",
        blocks=[
            ("Cross-category trending", "sustain the browse"),
            ("Curated collections", "give the wandering a spine"),
            ("Newsletter / save-for-later", "capture the visit even without a sale"),
        ],
        suppress=["aggressive checkout prompts"]),
    "Price-sensitive": dict(
        hero="Sale hero with countdown + biggest % off",
        blocks=[
            ("Deals rail sorted price-low", "mirror their own sort behaviour"),
            ("Price-drop alerts signup", "deal hunters return for drops"),
            ("Bundle offers", "raise basket without raising price resistance"),
        ],
        suppress=["full-price new arrivals"]),
    "Low-intent": dict(
        hero="Fast, light hero: top categories only",
        blocks=[
            ("Top categories grid", "orient in one glance"),
            ("One broad promo", "single low-pressure hook"),
        ],
        suppress=["heavy media", "modals", "anything slow"]),
}

NEUTRAL = dict(
    hero="Balanced hero: search + trending side by side",
    blocks=[
        ("Search bar", "serve the mission if there is one"),
        ("Trending now", "serve the browse if there is one"),
        ("Top categories", "orient everyone"),
    ],
    suppress=[])

CHECKOUT = dict(
    hero="Sticky cart summary with one-tap checkout",
    blocks=[
        ("Shipping / returns reassurance", "the one doubt that kills a decided sale"),
        ("'Complete your order' CTA", "they decided; get out of the way"),
    ],
    suppress=["all discovery content", "promos that add doubt"])


def render(inf: Inference) -> Homepage:
    if inf.mode == "override":
        L = CHECKOUT
        return Homepage("checkout", "Decisive", inf.confidence, L["hero"],
                        L["blocks"], L["suppress"], inf.reasons)

    if inf.mode == "cold":
        # Accent = the most likely SUBSTANTIVE intent. Low-intent winning the
        # cold prior just means "probably a drive-by"; the interesting tilt is
        # "if they turn out to be someone, who?" -- the runner-up.
        ranked = sorted(inf.probs, key=inf.probs.get, reverse=True)
        accent = next((i for i in ranked if i != "Low-intent"), ranked[0])
        L = dict(NEUTRAL)
        blocks = list(L["blocks"])
        if inf.probs.get(accent, 0) >= 0.10 and accent in LAYOUTS:
            blocks = [LAYOUTS[accent]["blocks"][0]] + blocks  # one tilted block
        return Homepage("cold-accent", accent, inf.confidence, L["hero"],
                        blocks, L["suppress"],
                        [f"prior tilt toward {accent} -- no behaviour observed yet"]
                        + inf.reasons)

    if inf.intent == "Unclear":
        L = NEUTRAL
        return Homepage("neutral", "Unclear", inf.confidence, L["hero"],
                        L["blocks"], L["suppress"], inf.reasons)

    L = LAYOUTS[inf.intent]
    mode = "neutral-light" if inf.intent == "Low-intent" else "personalised"
    return Homepage(mode, inf.intent, inf.confidence, L["hero"],
                    L["blocks"], L["suppress"], inf.reasons)
