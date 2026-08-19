"""Unit tests for the two-stage intent engine.

The repo shipped one end-to-end smoke test and no unit tests, so every rule
weight, the decay, the gate and the calibration helpers were only ever
exercised through 3k-session aggregate tables -- where a broken rule shows up
as a slightly different percentage, not as a failure. These pin the behaviour
the README claims, plus regression guards for the two defects those aggregate
tables hid.
"""

import math

import pytest

from intent_engine import (ALL_INTENTS, REAL_INTENTS, SessionState,
                           TwoStageEngine, _num, _softmax,
                           expected_calibration_error, fit_gate,
                           fit_temperature, prior_evidence, window_evidence)


# ---------------------------------------------------------------------------
# construction / configuration
# ---------------------------------------------------------------------------
def test_explicit_zero_gate_is_respected():
    """gate=0.0 means "never decline", not "use the default".

    Regression: `self.gate = gate or DEFAULTS["gate"]` turned 0.0 into 0.45,
    so evaluate_synthetic's ungated calibration pass was gated and the fitted
    gate was being measured against its own output.
    """
    eng = TwoStageEngine(gate=0.0, cold_gate=0.0)
    assert eng.gate == 0.0
    assert eng.cold_gate == 0.0


def test_explicit_zero_decay_is_respected():
    assert TwoStageEngine(decay=0.0).decay == 0.0


def test_defaults_still_apply_when_omitted():
    eng = TwoStageEngine()
    assert (eng.temperature, eng.gate, eng.cold_gate, eng.decay) == (2.0, 0.45, 0.55, 0.80)


@pytest.mark.parametrize("kwargs", [dict(temperature=0.0), dict(temperature=-1),
                                    dict(decay=1.5), dict(decay=-0.1)])
def test_invalid_hyperparameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TwoStageEngine(**kwargs)


def test_restricted_intent_set_never_emits_unobservable_intents():
    """The real-data track cannot observe Goal-driven / Price-sensitive."""
    eng = TwoStageEngine(intents=REAL_INTENTS)
    inf = eng.score_aggregates(dict(n_events=8, n_views=8, revisit_ratio=2.4,
                                    n_categories=1, top_category_share=1.0,
                                    median_gap_sec=90, duration_sec=700,
                                    added_to_cart=False,
                                    views_before_first_commercial=99))
    assert set(inf.probs) <= set(REAL_INTENTS)


# ---------------------------------------------------------------------------
# NaN-safe feature reads
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, default, expected", [
    (0, 99, 0.0),                 # the bug: a real zero must survive
    (0.0, 1.0, 0.0),
    (None, 7.0, 7.0),
    (float("nan"), 7.0, 7.0),
    ("not a number", 3.0, 3.0),
    (True, 0.0, 1.0),
    (False, 5.0, 0.0),
    (2.5, 0.0, 2.5),
])
def test_num_preserves_zero_and_defaults_on_missing(raw, default, expected):
    assert _num(raw, default) == expected


# ---------------------------------------------------------------------------
# Stage 1 -- the cold-start prior
# ---------------------------------------------------------------------------
def test_prior_may_tilt_but_never_commits():
    """No arrival context on its own should clear the cold gate."""
    eng = TwoStageEngine()
    for ctx in [dict(referrer="google", device="desktop", landing="product", hour=21),
                dict(referrer="instagram", device="mobile", landing="home", hour=21),
                dict(referrer="email", device="desktop", landing="sale", hour=19)]:
        inf = eng.start_session(ctx)
        assert inf.mode == "cold"
        assert inf.intent == "Unclear", f"{ctx} committed on arrival: {inf.intent}"
        assert inf.served.startswith("Neutral")


def test_prior_tilt_points_the_right_way():
    eng = TwoStageEngine()
    sale = eng.start_session(dict(referrer="email", landing="sale", hour=19))
    insta = eng.start_session(dict(referrer="instagram", landing="home", hour=21))
    assert sale.probs["Price-sensitive"] > sale.probs["Explorer"]
    assert insta.probs["Explorer"] > insta.probs["Price-sensitive"]


def test_prior_always_carries_the_low_intent_base_rate():
    assert "Low-intent" in [e.intent for e in prior_evidence({})]


def test_unknown_context_values_are_ignored_not_crashed():
    inf = TwoStageEngine().start_session(dict(referrer="carrier-pigeon",
                                              device="fridge",
                                              landing="somewhere", hour=None))
    assert math.isclose(sum(inf.probs.values()), 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Stage 2 -- incremental behaviour
# ---------------------------------------------------------------------------
def _views(*specs, start=0, step=90):
    """(item, category) pairs -> view events spaced `step` seconds apart."""
    return [{"type": "view", "ts": start + i * step, "item": it, "category": c}
            for i, (it, c) in enumerate(specs)]


def test_revisiting_an_item_reads_as_evaluator():
    eng = TwoStageEngine()
    eng.start_session({})
    inf = None
    for e in _views(("A", "home"), ("B", "home"), ("A", "home"),
                    ("A", "home"), ("B", "home"), ("A", "home")):
        inf = eng.observe(e)
    assert inf.intent == "Evaluator", inf.explain()


def test_breadth_without_revisiting_reads_as_explorer():
    eng = TwoStageEngine()
    eng.start_session({})
    inf = None
    for e in _views(("a", "fashion"), ("b", "home"), ("c", "electronics"),
                    ("d", "beauty"), ("e", "running"), ("f", "fashion"), step=12):
        inf = eng.observe(e)
    assert inf.intent == "Explorer", inf.explain()


def test_deal_search_is_sticky_and_survives_the_decay_window():
    """A price-seeking action is a property of the session, not of the moment."""
    eng = TwoStageEngine()
    eng.start_session({})
    eng.observe({"type": "search", "ts": 0, "query": "clearance deals"})
    inf = None
    for e in _views(("s1", "sale"), ("s2", "sale"), ("s1", "sale"),
                    ("s3", "sale"), ("s1", "sale"), start=10, step=40):
        inf = eng.observe(e)
    assert inf.intent == "Price-sensitive", inf.explain()


def test_window_evidence_decays():
    eng = TwoStageEngine(decay=0.5)
    eng.start_session({})
    eng.observe({"type": "sort", "ts": 0, "sort_key": "trending"})
    first = eng.window["Explorer"]
    assert first > 0
    eng.observe({"type": "view", "ts": 5, "item": "z", "category": "home"})
    assert eng.window["Explorer"] == pytest.approx(first * 0.5)


def test_a_single_click_stays_humble():
    eng = TwoStageEngine()
    eng.start_session(dict(referrer="google", landing="product"))
    inf = eng.observe({"type": "view", "ts": 0, "item": "A", "category": "running"})
    assert inf.intent in ("Unclear", "Low-intent"), inf.explain()


def test_idle_gap_is_capped_so_a_parked_tab_cannot_dominate_pace():
    eng = TwoStageEngine()
    eng.start_session({})
    eng.observe({"type": "view", "ts": 0, "item": "A", "category": "home"})
    eng.observe({"type": "view", "ts": 40_000, "item": "B", "category": "home"})
    assert eng.state.avg_gap_sec == pytest.approx(600.0)


def test_long_dwell_past_the_idle_cap_earns_no_evaluator_credit():
    st = SessionState()
    st.views_per_item = {"A": 1}
    prev = {"type": "view", "ts": 0, "item": "A", "category": "home"}
    reading = window_evidence({"type": "view", "ts": 200, "item": "B",
                               "category": "home"}, st, prev)
    parked = window_evidence({"type": "view", "ts": 9_000, "item": "B",
                              "category": "home"}, st, prev)
    assert any(e.intent == "Evaluator" and "dwelled" in e.reason for e in reading)
    assert not any("dwelled" in e.reason for e in parked)


# ---------------------------------------------------------------------------
# the Decisive override
# ---------------------------------------------------------------------------
def test_override_fires_on_cart_with_minimal_browsing():
    eng = TwoStageEngine()
    eng.start_session(dict(referrer="google", landing="product"))
    eng.observe({"type": "view", "ts": 0, "item": "A", "category": "running"})
    inf = eng.observe({"type": "addtocart", "ts": 40, "item": "A", "category": "running"})
    assert (inf.intent, inf.mode, inf.served) == ("Decisive", "override", "Checkout-support")


def test_override_does_not_fire_after_a_long_browse():
    eng = TwoStageEngine()
    eng.start_session({})
    for e in _views(*[(f"i{i}", "home") for i in range(6)]):
        eng.observe(e)
    inf = eng.observe({"type": "addtocart", "ts": 999, "item": "i0", "category": "home"})
    assert inf.mode != "override"


def test_override_is_sticky_once_fired():
    eng = TwoStageEngine()
    eng.start_session({})
    eng.observe({"type": "addtocart", "ts": 0, "item": "A", "category": "home"})
    inf = None
    for e in _views(*[(f"j{i}", "beauty") for i in range(5)], start=10):
        inf = eng.observe(e)
    assert inf.intent == "Decisive"


def test_override_fires_with_zero_prior_views():
    """Cart-first sessions -- the single highest-converting segment there is.

    Regression: the offline getter ended in `or default`, so a true 0 in
    `views_before_first_commercial` collapsed to the 99 default, failed the
    `<= 2` test, and these visitors were served the Low-intent page.
    """
    eng = TwoStageEngine(intents=REAL_INTENTS)
    inf = eng.score_aggregates(
        dict(n_events=1, n_views=0, revisit_ratio=0.0, n_categories=0,
             top_category_share=0.0, category_switch_rate=0.0,
             median_gap_sec=0.0, duration_sec=0.0, added_to_cart=True,
             views_before_first_commercial=0))
    assert inf.intent == "Decisive", inf.explain()
    assert inf.served == "Checkout-support"


@pytest.mark.parametrize("views_before", [0, 1, 2])
def test_override_boundary_is_inclusive(views_before):
    eng = TwoStageEngine(intents=REAL_INTENTS)
    inf = eng.score_aggregates(dict(n_events=views_before + 1, n_views=views_before,
                                    added_to_cart=True,
                                    views_before_first_commercial=views_before))
    assert inf.intent == "Decisive"


def test_override_boundary_excludes_three_prior_views():
    eng = TwoStageEngine(intents=REAL_INTENTS)
    inf = eng.score_aggregates(dict(n_events=4, n_views=3, added_to_cart=True,
                                    views_before_first_commercial=3))
    assert inf.intent != "Decisive"


# ---------------------------------------------------------------------------
# the offline (aggregate) path
# ---------------------------------------------------------------------------
def _agg(**kw):
    row = dict(n_events=1, n_views=0, revisit_ratio=0.0, n_categories=0,
               top_category_share=0.0, category_switch_rate=0.0,
               median_gap_sec=0.0, duration_sec=0.0, added_to_cart=False,
               views_before_first_commercial=99)
    row.update(kw)
    return row


def test_micro_visit_is_low_intent():
    eng = TwoStageEngine(intents=REAL_INTENTS)
    assert eng.score_aggregates(
        _agg(n_events=2, n_views=2, duration_sec=20)).intent == "Low-intent"


def test_aggregate_revisit_is_evaluator_and_breadth_is_explorer():
    eng = TwoStageEngine(intents=REAL_INTENTS)
    ev = eng.score_aggregates(_agg(n_events=9, n_views=9, revisit_ratio=2.6,
                                   n_categories=1, top_category_share=1.0,
                                   median_gap_sec=110, duration_sec=900))
    ex = eng.score_aggregates(_agg(n_events=10, n_views=10, revisit_ratio=1.0,
                                   n_categories=5, top_category_share=0.3,
                                   category_switch_rate=0.8,
                                   median_gap_sec=15, duration_sec=400))
    assert ev.intent == "Evaluator", ev.explain()
    assert ex.intent == "Explorer", ex.explain()


def test_missing_features_do_not_crash_the_offline_path():
    inf = TwoStageEngine(intents=REAL_INTENTS).score_aggregates({})
    assert inf.intent in set(REAL_INTENTS) | {"Unclear"}
    assert math.isclose(sum(inf.probs.values()), 1.0, rel_tol=1e-9)


def test_price_flavor_is_orthogonal_to_the_layout_choice():
    """Cheap-leaning browsing tilts merchandising; it must not change intent."""
    eng = TwoStageEngine(intents=REAL_INTENTS)
    base = _agg(n_events=9, n_views=9, revisit_ratio=2.6, n_categories=1,
                top_category_share=1.0, median_gap_sec=110, duration_sec=900)
    plain = eng.score_aggregates(base)
    cheap = eng.score_aggregates({**base, "price_rel_cat": 0.4})
    dear = eng.score_aggregates({**base, "price_rel_cat": 1.3})
    assert cheap.intent == dear.intent == plain.intent
    assert cheap.price_conscious and not dear.price_conscious


def test_disabled_rule_removes_its_contribution():
    row = _agg(n_events=9, n_views=9, revisit_ratio=2.6, n_categories=1,
               top_category_share=1.0, median_gap_sec=110, duration_sec=900)
    on = TwoStageEngine(intents=REAL_INTENTS).score_aggregates(row)
    off = TwoStageEngine(intents=REAL_INTENTS,
                         disabled_rules={"revisit_core"}).score_aggregates(row)
    assert on.scores["Evaluator"] > off.scores["Evaluator"]


def test_weight_scale_moves_the_score_proportionally():
    row = _agg(n_events=9, n_views=9, revisit_ratio=2.6, n_categories=1,
               top_category_share=1.0, median_gap_sec=110, duration_sec=900)
    base = TwoStageEngine(intents=REAL_INTENTS).score_aggregates(row)
    half = TwoStageEngine(intents=REAL_INTENTS,
                          weight_scale={"revisit_core": 0.5}).score_aggregates(row)
    assert half.scores["Evaluator"] == pytest.approx(base.scores["Evaluator"] - 1.5)


def test_disabling_the_override_routes_the_session_back_to_scoring():
    eng = TwoStageEngine(intents=REAL_INTENTS, disabled_rules={"decisive_override"})
    inf = eng.score_aggregates(_agg(n_events=1, added_to_cart=True,
                                    views_before_first_commercial=0))
    assert inf.intent != "Decisive"


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def test_gate_declines_when_evidence_conflicts():
    strict = TwoStageEngine(gate=0.99, cold_gate=0.99)
    strict.start_session({})
    inf = None
    for e in _views(("A", "home"), ("A", "home"), ("B", "beauty")):
        inf = strict.observe(e)
    assert inf.intent == "Unclear"
    assert inf.served == "Neutral"


def test_zero_gate_never_declines():
    assert TwoStageEngine(gate=0.0, cold_gate=0.0).start_session({}).intent != "Unclear"


# ---------------------------------------------------------------------------
# calibration helpers
# ---------------------------------------------------------------------------
def test_softmax_is_a_distribution_and_temperature_flattens_it():
    scores = {"a": 3.0, "b": 1.0, "c": 0.0}
    sharp, flat = _softmax(scores, 0.5), _softmax(scores, 5.0)
    assert math.isclose(sum(sharp.values()), 1.0)
    assert math.isclose(sum(flat.values()), 1.0)
    assert sharp["a"] > flat["a"]


def test_softmax_is_shift_invariant():
    a = _softmax({"x": 1.0, "y": 2.0}, 1.0)
    b = _softmax({"x": 101.0, "y": 102.0}, 1.0)
    assert a["x"] == pytest.approx(b["x"])


def test_fit_temperature_prefers_a_sharp_t_when_scores_are_reliable():
    assert fit_temperature([{"a": 4.0, "b": 0.0}] * 100, ["a"] * 100) <= 1.0


def test_fit_temperature_prefers_a_flat_t_when_scores_are_noise():
    rows = [{"a": 4.0, "b": 0.0}] * 100
    assert fit_temperature(rows, ["a"] * 50 + ["b"] * 50) >= 2.0


def test_fit_gate_reaches_the_target_precision():
    confs = [i / 100 for i in range(30, 100)]
    correct = [1 if c > 0.6 else 0 for c in confs]
    t, cov, prec = fit_gate(confs, correct, target_precision=0.9)
    assert prec >= 0.9
    assert 0 < cov <= 1


def test_fit_gate_returns_the_smallest_qualifying_threshold():
    t, _, prec = fit_gate([0.30, 0.50, 0.80, 0.90], [0, 1, 1, 1],
                          target_precision=1.0)
    assert t == pytest.approx(0.31, abs=0.02)
    assert prec == 1.0


def test_fit_gate_handles_an_unreachable_target_without_crashing():
    t, cov, prec = fit_gate([0.4, 0.5, 0.6], [0, 0, 0], target_precision=0.9)
    assert prec < 0.9
    assert 0.0 <= t <= 1.0


def test_fit_gate_on_empty_input():
    assert fit_gate([], [], target_precision=0.85)[1] == 0.0


def test_ece_is_zero_for_a_perfectly_calibrated_forecaster():
    assert expected_calibration_error([0.95] * 100, [1] * 95 + [0] * 5) < 0.02


def test_ece_catches_overconfidence():
    assert expected_calibration_error([0.95] * 100, [1] * 50 + [0] * 50) > 0.4


def test_ece_of_nothing_is_zero():
    assert expected_calibration_error([], []) == 0.0


# ---------------------------------------------------------------------------
# session bookkeeping
# ---------------------------------------------------------------------------
def test_start_session_clears_the_previous_visitor():
    eng = TwoStageEngine()
    eng.start_session({})
    eng.observe({"type": "addtocart", "ts": 0, "item": "A", "category": "home"})
    assert eng.current().intent == "Decisive"
    eng.start_session({})
    assert eng.override is None
    assert eng.state.n_events == 0
    assert eng.current().intent != "Decisive"


def test_state_counters_track_the_stream():
    eng = TwoStageEngine()
    eng.start_session({})
    for e in _views(("A", "home"), ("A", "home"), ("B", "beauty"), step=30):
        eng.observe(e)
    st = eng.state
    assert (st.n_events, st.n_views, st.n_unique_items, st.n_categories) == (3, 3, 2, 2)
    assert st.revisit_ratio == pytest.approx(1.5)
    assert st.top_category_share == pytest.approx(2 / 3)
    assert st.duration_sec == pytest.approx(60.0)


def test_every_decision_carries_its_reasons():
    eng = TwoStageEngine()
    eng.start_session(dict(referrer="instagram"))
    inf = None
    for e in _views(("a", "fashion"), ("b", "home"), ("c", "electronics"),
                    ("d", "beauty"), ("e", "running"), step=12):
        inf = eng.observe(e)
    assert inf.reasons, "an auditable engine must never decide without reasons"
    assert "->" in inf.explain()


def test_probabilities_cover_exactly_the_configured_intents():
    inf = TwoStageEngine().start_session({})
    assert set(inf.probs) == set(ALL_INTENTS)
    assert math.isclose(sum(inf.probs.values()), 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# arrival: the engine and the renderer must agree about what "cold" means
# ---------------------------------------------------------------------------
def test_cold_mode_never_serves_a_committed_page_even_when_confident():
    """`served` used to become the winning intent once the cold gate cleared.

    At the fitted T=1.0 that happens routinely, so the engine was telling a
    caller to serve the full Low-intent page off arrival context alone --
    while personalisation.render() drew a neutral cold-accent page regardless.
    Two modules, two different answers to "what is arrival?".
    """
    eng = TwoStageEngine(temperature=1.0, cold_gate=0.0)   # forced to commit
    inf = eng.start_session(dict(referrer="instagram", device="mobile",
                                 landing="home", hour=21))
    assert inf.mode == "cold"
    assert inf.confidence >= eng.cold_gate, "test needs a cleared cold gate"
    assert inf.served.startswith("Neutral (accent:")


def test_cold_accent_label_matches_the_renderer_choice():
    from personalisation import render
    eng = TwoStageEngine(temperature=1.0)
    for ctx in [dict(referrer="instagram", landing="home", hour=21),
                dict(referrer="email", landing="sale", hour=19),
                dict(referrer="google", landing="product", hour=13),
                dict(referrer="direct", landing="category", hour=9),
                {}]:
        inf = eng.start_session(ctx)
        assert inf.served == f"Neutral (accent: {render(inf).intent})", ctx


def test_cold_accent_is_never_low_intent():
    eng = TwoStageEngine(temperature=1.0)
    assert "Low-intent" not in eng.start_session({}).served


def test_behavioural_mode_still_serves_the_intent_it_commits_to():
    eng = TwoStageEngine(temperature=1.0)
    eng.start_session({})
    inf = None
    for e in _views(("A", "home"), ("B", "home"), ("A", "home"),
                    ("A", "home"), ("B", "home"), ("A", "home")):
        inf = eng.observe(e)
    assert inf.mode == "behavioural"
    assert inf.served == inf.intent == "Evaluator"
