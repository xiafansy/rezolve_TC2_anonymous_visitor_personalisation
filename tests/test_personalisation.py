"""The signal -> intent -> layout chain has to stay auditable and complete.

`render()` indexes LAYOUTS[inf.intent] directly, so any intent added to the
taxonomy without a matching layout is a KeyError in production, on a live
homepage. These tests make that a red build instead.
"""

import pytest

from intent_engine import ALL_INTENTS, Inference, TwoStageEngine
from personalisation import CHECKOUT, LAYOUTS, NEUTRAL, render


def _inf(intent, mode="behavioural", conf=0.9, probs=None, served=None):
    probs = probs or {i: (0.9 if i == intent else 0.025) for i in ALL_INTENTS}
    return Inference(intent=intent, confidence=conf, probs=probs, scores={},
                     reasons=["because"], mode=mode, served=served or intent)


# ---------------------------------------------------------------------------
# completeness
# ---------------------------------------------------------------------------
def test_every_intent_in_the_taxonomy_has_a_layout():
    missing = [i for i in ALL_INTENTS if i not in LAYOUTS]
    assert not missing, f"no homepage defined for {missing} -- render() would raise"


@pytest.mark.parametrize("intent", ALL_INTENTS)
def test_every_layout_is_renderable_and_explains_itself(intent):
    page = render(_inf(intent))
    assert page.hero
    assert page.blocks, f"{intent} renders an empty page"
    for block in page.blocks:
        title, rationale = block
        assert title and rationale, f"{intent} has a block with no rationale"


@pytest.mark.parametrize("layout", list(LAYOUTS.values()) + [NEUTRAL, CHECKOUT])
def test_no_layout_block_is_missing_its_rationale(layout):
    assert all(len(b) == 2 and all(b) for b in layout["blocks"])


# ---------------------------------------------------------------------------
# serving modes
# ---------------------------------------------------------------------------
def test_override_serves_checkout_and_suppresses_discovery():
    page = render(_inf("Decisive", mode="override", conf=0.95))
    assert page.mode == "checkout"
    assert page.intent == "Decisive"
    assert any("discovery" in s for s in page.suppress)


def test_unclear_serves_a_neutral_page_that_commits_to_nothing():
    page = render(_inf("Unclear", conf=0.3))
    assert page.mode == "neutral"
    assert page.blocks == NEUTRAL["blocks"]


def test_low_intent_serves_the_light_page_not_a_personalised_one():
    assert render(_inf("Low-intent")).mode == "neutral-light"


def test_cleared_gate_serves_the_full_intent_layout():
    page = render(_inf("Evaluator"))
    assert page.mode == "personalised"
    assert page.blocks == LAYOUTS["Evaluator"]["blocks"]


# ---------------------------------------------------------------------------
# the cold-start accent: a prior may tilt, it may not commit
# ---------------------------------------------------------------------------
def test_cold_accent_keeps_the_neutral_base_and_adds_at_most_one_block():
    probs = {"Low-intent": 0.5, "Explorer": 0.3, "Evaluator": 0.1,
             "Goal-driven": 0.06, "Price-sensitive": 0.04}
    page = render(_inf("Unclear", mode="cold", conf=0.5, probs=probs))
    assert page.mode == "cold-accent"
    assert page.intent == "Explorer", "accent should be the top SUBSTANTIVE intent"
    assert len(page.blocks) == len(NEUTRAL["blocks"]) + 1
    assert page.blocks[0] == LAYOUTS["Explorer"]["blocks"][0]
    assert all(b in page.blocks for b in NEUTRAL["blocks"]), "neutral base was replaced"


def test_cold_accent_never_accents_low_intent():
    probs = {"Low-intent": 0.9, "Explorer": 0.04, "Evaluator": 0.03,
             "Goal-driven": 0.02, "Price-sensitive": 0.01}
    page = render(_inf("Unclear", mode="cold", conf=0.9, probs=probs))
    assert page.intent != "Low-intent"


def test_a_weak_prior_does_not_tilt_the_page_at_all():
    probs = {"Low-intent": 0.94, "Explorer": 0.03, "Evaluator": 0.015,
             "Goal-driven": 0.01, "Price-sensitive": 0.005}
    page = render(_inf("Unclear", mode="cold", conf=0.94, probs=probs))
    assert page.blocks == NEUTRAL["blocks"], "a <10% prior should earn no accent"


def test_cold_accent_states_that_no_behaviour_was_observed():
    page = render(_inf("Unclear", mode="cold", conf=0.5))
    assert any("no behaviour observed yet" in r for r in page.reasons)


# ---------------------------------------------------------------------------
# end to end: engine -> page
# ---------------------------------------------------------------------------
def test_arrival_never_produces_a_personalised_page():
    eng = TwoStageEngine()
    for ref in ["google", "instagram", "email", "direct", "ad", "facebook"]:
        page = render(eng.start_session(dict(referrer=ref, landing="sale", hour=21)))
        assert page.mode == "cold-accent"


def test_a_cart_first_visitor_gets_the_checkout_page():
    eng = TwoStageEngine()
    eng.start_session(dict(referrer="direct", landing="product"))
    page = render(eng.observe({"type": "addtocart", "ts": 0, "item": "A",
                               "category": "running"}))
    assert page.mode == "checkout"
