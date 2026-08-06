"""
Homepage personalisation layer.

Turns an inferred intent (+ confidence) into a concrete home-page layout: an
ordered list of content blocks, a hero treatment, blocks to suppress, and the
UX rationale behind each choice.

Design principles
-----------------
1. Intent drives layout, not identity. We never use who the visitor "is like"
   (no lookalike) -- only what this session's signals imply they want *now*.
2. Confidence gates commitment. When the inferer isn't sure (confidence below a
   threshold), we serve a balanced NEUTRAL home page rather than confidently
   showing the wrong thing. Getting personalisation wrong is worse than not
   personalising at all.
3. Every block carries a rationale, so the experience is auditable end-to-end:
   session signal -> intent -> layout decision.
"""

from dataclasses import dataclass

from intent_inference import infer

# Below this winner-confidence we decline to personalise and serve NEUTRAL.
# Rationale: with 4 intents, a confident call sits well above the 0.25 floor;
# offline, wrong predictions cluster around ~0.43 confidence, so a threshold in
# the low-0.40s screens out most mistakes while keeping most correct calls.
MIN_CONFIDENCE = 0.42


@dataclass
class Block:
    title: str
    rationale: str


@dataclass
class Homepage:
    intent: str            # the intent this layout serves ("Neutral" if fallback)
    confidence: float
    personalised: bool     # False when we fell back to Neutral
    headline: str
    primary_goal: str
    hero: str
    blocks: list           # list[Block], in display order
    suppress: list         # things we deliberately de-emphasise
    tone: str
    reasons: list          # why we inferred this intent (from the inferer)


# ---------------------------------------------------------------------------
# Per-intent home-page strategies.
#
# {category} is filled from the session so the layout points at what the visitor
# is actually engaging with. Explorer/Neutral stay category-light on purpose.
# ---------------------------------------------------------------------------
STRATEGIES = {
    "Goal-driven": {
        "headline": "Find it fast",
        "primary_goal": "Strip friction between arrival and checkout",
        "hero": "Search-first hero: a large search box pre-scoped to {category}, "
                "with the visitor's likely query as placeholder text",
        "tone": "Efficient, minimal, conversion-focused",
        "blocks": [
            ("Best sellers in {category}",
             "A decisive buyer converts fastest on proven, popular products"),
            ("Free shipping & fast-checkout banner",
             "Reassure on cost and speed to remove last-mile hesitation"),
            ("Recently viewed / quick re-add",
             "One tap back to the exact product they came for"),
            ("Back-in-stock & top-rated in {category}",
             "Narrow, high-signal shortlist -- no browsing detour"),
        ],
        "suppress": ["Editorial lookbooks", "Broad cross-category discovery carousels"],
    },
    "Explorer": {
        "headline": "Discover what's trending",
        "primary_goal": "Sustain browsing depth and inspire; capture an email",
        "hero": "Full-bleed lifestyle imagery: 'Trending now' with a bold visual, "
                "no search box competing for attention",
        "tone": "Visual, editorial, low-pressure",
        "blocks": [
            ("Trending & new arrivals carousel",
             "Fresh, changing content rewards the open-ended browser"),
            ("Curated collections / lookbook",
             "Themed inspiration keeps a grazing session moving"),
            ("Shop by category grid",
             "Visual entry points for someone with no fixed destination"),
            ("Newsletter / early-access sign-up",
             "Low intent to buy now -- convert the visit into a contactable lead"),
        ],
        "suppress": ["Prominent search bar", "Hard-sell checkout CTAs"],
    },
    "Research": {
        "headline": "Compare with confidence",
        "primary_goal": "Support careful evaluation with proof and detail",
        "hero": "Comparison-oriented hero for {category}: 'Top rated', with "
                "filters (rating, price, features) surfaced immediately",
        "tone": "Informative, trustworthy, detail-rich",
        "blocks": [
            ("Highest-rated in {category}",
             "Social proof is the deciding factor for an evaluator"),
            ("Side-by-side comparison tool",
             "Lets a researcher weigh specs without leaving the page"),
            ("Buying guides & expert reviews",
             "Deep content matches a long, deliberate session"),
            ("Verified reviews & ratings spotlight",
             "Aggregated trust signals to de-risk the decision"),
        ],
        "suppress": ["Urgency countdown timers", "Thin, image-only product tiles"],
    },
    "Price-sensitive": {
        "headline": "Today's best deals",
        "primary_goal": "Surface savings and create gentle urgency",
        "hero": "Deals hero: 'Up to X% off', clearance banner, promo-code strip "
                "front and centre",
        "tone": "Value-led, energetic, urgency-aware",
        "blocks": [
            ("Sale grid sorted by biggest discount",
             "A bargain-hunter scans by savings, not by product"),
            ("Price-drop & clearance highlights",
             "Reinforce that now is a good time to buy"),
            ("Coupon / promo-code panel",
             "Matches the promo-driven way they arrived"),
            ("Limited-time deals countdown",
             "Gentle urgency nudges a price-led shopper to commit"),
        ],
        "suppress": ["Full-price editorial", "Premium/luxury positioning"],
    },
    # Served when confidence is too low to commit to any single intent.
    "Neutral": {
        "headline": "Welcome",
        "primary_goal": "Cover all bases when intent is unclear -- do no harm",
        "hero": "Balanced hero: search box alongside a 'Trending now' visual, "
                "neither dominating",
        "tone": "Balanced, broadly useful",
        "blocks": [
            ("Search + top categories",
             "Serve goal-driven visitors without crowding out browsers"),
            ("Trending products",
             "Safe, broadly appealing entry point"),
            ("Featured deals strip",
             "A modest nod to value without going all-in on discounts"),
            ("New arrivals",
             "Fresh content for the undecided"),
        ],
        "suppress": ["Aggressive single-intent commitment"],
    },
}


def _fill(text, category):
    return text.replace("{category}", str(category) if category else "your category")


def build_homepage(session, min_confidence=MIN_CONFIDENCE):
    """Full pipeline for one session: signals -> intent -> home-page layout."""
    result = infer(session)
    personalised = result.confidence >= min_confidence
    key = result.intent if personalised else "Neutral"
    strat = STRATEGIES[key]
    category = session.get("Category", "")

    blocks = [Block(_fill(t, category), _fill(r, category))
              for (t, r) in strat["blocks"]]

    return Homepage(
        intent=key,
        confidence=result.confidence,
        personalised=personalised,
        headline=strat["headline"],
        primary_goal=strat["primary_goal"],
        hero=_fill(strat["hero"], category),
        blocks=blocks,
        suppress=strat["suppress"],
        tone=strat["tone"],
        reasons=result.reasons[:3],
    )


def render_text(hp, session=None):
    """Human-readable render of a Homepage for the CLI demo."""
    lines = []
    tag = "PERSONALISED" if hp.personalised else "NEUTRAL (low confidence)"
    lines.append(f"HOME PAGE -> {hp.intent}  [{tag}]  confidence {hp.confidence:.0%}")
    if hp.reasons:
        lines.append(f"  inferred because: {'; '.join(hp.reasons)}")
    lines.append(f"  goal : {hp.primary_goal}")
    lines.append(f"  tone : {hp.tone}")
    lines.append(f"  HERO : {hp.hero}")
    lines.append("  BLOCKS:")
    for i, b in enumerate(hp.blocks, 1):
        lines.append(f"    {i}. {b.title}")
        lines.append(f"       -> {b.rationale}")
    lines.append(f"  SUPPRESS: {', '.join(hp.suppress)}")
    return "\n".join(lines)


if __name__ == "__main__":
    demo = {
        "Referrer": "Instagram", "Device": "Mobile", "Category": "Fashion",
        "Search_Used": False, "Search_Query": "", "Scroll_Depth": 92,
        "Product_Views": 16, "Filter_Used": False, "Sort_Type": "Trending",
        "Session_Duration_sec": 700, "Add_to_Cart": False, "Purchase": False,
    }
    print(render_text(build_homepage(demo), demo))
