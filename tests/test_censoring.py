"""The leakage fix is this project's central methodological claim.

v2 aggregated behavioural features over the whole session, including views
that happened *after* add-to-cart -- buyers re-open the product page during
checkout, so `revisit_ratio` "predicted" buying partly because buying caused
re-viewing. v3 censors at the first commercial event. That claim was backed by
one end-to-end smoke test whose assertion (`a <= b + c` on non-negative
numbers) could not fail.

These tests build a tiny log with hand-computed expected values, so the
censoring, the sessionization gap, prefix-3 and `views_before_first_commercial`
each fail loudly and individually.
"""

import pandas as pd
import pytest

from build_real_sessions import SESSION_GAP_MS, build, sessionize

MIN = 60_000
T0 = 1_431_000_000_000


def _write_archive(tmp_path, rows):
    """rows: (timestamp, visitorid, event, itemid) -> a RetailRocket-shaped archive."""
    ev = pd.DataFrame([(t, v, e, i, "") for t, v, e, i in rows],
                      columns=["timestamp", "visitorid", "event", "itemid",
                               "transactionid"])
    ev.to_csv(tmp_path / "events.csv", index=False)
    # items 1..9 -> categories 1,2,3 repeating: item i -> category ((i-1) % 3) + 1
    props = pd.DataFrame({"timestamp": T0, "itemid": range(1, 10),
                          "property": "categoryid",
                          "value": [str((i - 1) % 3 + 1) for i in range(1, 10)]})
    props.iloc[:5].to_csv(tmp_path / "item_properties_part1.csv", index=False)
    props.iloc[5:].to_csv(tmp_path / "item_properties_part2.csv", index=False)
    return str(tmp_path)


# visitor 1 -- the leakage case: 2 distinct views, then cart, then the SAME
#              item re-opened twice during checkout. Uncensored that reads as
#              revisit_ratio 4/2 = 2.0 ("Evaluator"); censored it is 2/2 = 1.0.
# visitor 2 -- a genuine evaluator: 5 views of 2 items, no commercial event.
# visitor 3 -- cart-first: adds to cart having viewed nothing this session.
# visitor 4 -- explorer: 6 views across 3 categories, no commercial event.
LOG = [
    (T0 + 0 * MIN, 1, "view", 1),
    (T0 + 1 * MIN, 1, "view", 2),
    (T0 + 2 * MIN, 1, "addtocart", 1),
    (T0 + 3 * MIN, 1, "view", 1),
    (T0 + 4 * MIN, 1, "view", 1),
    (T0 + 5 * MIN, 1, "transaction", 1),

    (T0 + 0 * MIN, 2, "view", 4),
    (T0 + 2 * MIN, 2, "view", 5),
    (T0 + 4 * MIN, 2, "view", 4),
    (T0 + 6 * MIN, 2, "view", 5),
    (T0 + 8 * MIN, 2, "view", 4),

    (T0 + 0 * MIN, 3, "addtocart", 7),
    (T0 + 1 * MIN, 3, "transaction", 7),

    (T0 + 0 * MIN, 4, "view", 1),
    (T0 + 1 * MIN, 4, "view", 2),
    (T0 + 2 * MIN, 4, "view", 3),
    (T0 + 3 * MIN, 4, "view", 4),
    (T0 + 4 * MIN, 4, "view", 5),
    (T0 + 5 * MIN, 4, "view", 6),
]


@pytest.fixture(scope="module")
def feats(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rr")
    archive = _write_archive(tmp, LOG)
    return build(archive, str(tmp / "sessions.csv")).set_index("visitorid")


# ---------------------------------------------------------------------------
# the censoring itself
# ---------------------------------------------------------------------------
def test_post_cart_views_are_excluded_from_behavioural_features(feats):
    v1 = feats.loc[1]
    assert v1["n_views"] == 2, "the two post-cart re-views leaked back in"
    assert v1["n_events"] == 2
    assert v1["n_unique_items"] == 2
    assert v1["revisit_ratio"] == pytest.approx(1.0)


def test_the_uncensored_reading_really_would_have_been_different(feats):
    """Guard against a censoring step that is a no-op on this fixture."""
    v1 = feats.loc[1]
    assert v1["n_events_total"] == 6
    assert v1["n_events"] < v1["n_events_total"]
    # uncensored revisit_ratio would be 4 views / 2 unique items = 2.0
    assert v1["revisit_ratio"] < 2.0


def test_duration_stops_at_the_first_commercial_event(feats):
    """Full session spans 5 minutes; the censored window is the first minute."""
    assert feats.loc[1]["duration_sec"] == pytest.approx(60.0)


def test_outcomes_still_describe_the_whole_session(feats):
    v1 = feats.loc[1]
    assert bool(v1["added_to_cart"]) and bool(v1["purchased"])
    assert not bool(feats.loc[2]["added_to_cart"])


def test_a_session_with_no_commercial_event_is_untouched(feats):
    v2 = feats.loc[2]
    assert v2["n_views"] == 5
    assert v2["n_events"] == v2["n_events_total"] == 5
    assert v2["revisit_ratio"] == pytest.approx(2.5)
    assert v2["duration_sec"] == pytest.approx(480.0)


# ---------------------------------------------------------------------------
# views_before_first_commercial -- what the Decisive override fires on
# ---------------------------------------------------------------------------
def test_views_before_first_commercial_counts_only_pre_cart_views(feats):
    assert feats.loc[1]["views_before_first_commercial"] == 2


def test_cart_first_session_reports_zero_prior_views(feats):
    """Not NaN, not 99 -- zero. The override boundary depends on it."""
    assert feats.loc[3]["views_before_first_commercial"] == 0
    assert bool(feats.loc[3]["added_to_cart"])


def test_cart_first_session_reaches_the_decisive_override(feats):
    from intent_engine import REAL_INTENTS, TwoStageEngine
    row = feats.reset_index()
    row = row[row["visitorid"] == 3].to_dict("records")[0]
    inf = TwoStageEngine(intents=REAL_INTENTS).score_aggregates(row)
    assert inf.intent == "Decisive", inf.explain()


# ---------------------------------------------------------------------------
# breadth / focus features
# ---------------------------------------------------------------------------
def test_category_features_are_computed_on_censored_views(feats):
    v4 = feats.loc[4]
    assert v4["n_categories"] == 3
    assert v4["top_category_share"] == pytest.approx(2 / 6)
    assert v4["category_switch_rate"] == pytest.approx(1.0)


def test_focused_session_reports_a_single_category(feats):
    # visitor 2 views items 4 and 5 -> categories 1 and 2
    assert feats.loc[2]["n_categories"] == 2


# ---------------------------------------------------------------------------
# prefix-3: what a live homepage knows after three clicks
# ---------------------------------------------------------------------------
def test_prefix3_sees_at_most_three_events(feats):
    assert (feats["n_events_p3"].dropna() <= 3).all()
    assert feats.loc[4]["n_events_p3"] == 3


def test_prefix3_is_drawn_from_the_censored_stream(feats):
    """Visitor 1 only has 2 pre-commercial events, so p3 cannot exceed them."""
    assert feats.loc[1]["n_events_p3"] == 2


def test_prefix3_features_reflect_only_the_first_three_events(feats):
    v4 = feats.loc[4]
    assert v4["n_categories_p3"] == 3          # items 1,2,3 -> categories 1,2,3
    assert v4["duration_sec_p3"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# sessionization
# ---------------------------------------------------------------------------
def test_a_gap_longer_than_the_threshold_starts_a_new_session(tmp_path):
    rows = [(T0, 9, "view", 1),
            (T0 + 60_000, 9, "view", 2),
            (T0 + 60_000 + SESSION_GAP_MS + 1, 9, "view", 3)]
    ev = sessionize(_write_archive(tmp_path, rows))
    assert ev["session_id"].nunique() == 2


def test_activity_inside_the_threshold_stays_one_session(tmp_path):
    rows = [(T0, 9, "view", 1),
            (T0 + SESSION_GAP_MS - 1, 9, "view", 2)]
    ev = sessionize(_write_archive(tmp_path, rows))
    assert ev["session_id"].nunique() == 1


def test_two_visitors_never_share_a_session(tmp_path):
    rows = [(T0, 1, "view", 1), (T0 + 1000, 2, "view", 2)]
    ev = sessionize(_write_archive(tmp_path, rows))
    assert ev["session_id"].nunique() == 2
