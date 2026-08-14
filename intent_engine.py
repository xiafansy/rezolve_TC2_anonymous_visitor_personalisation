"""
intent_engine.py -- v3 unified two-stage intent engine.

Architecture (as requested)
---------------------------
  STAGE 1 · COLD START (event 0, arrival)
      The visitor has done nothing yet. The only observables are acquisition
      context: referrer, device, landing page type, hour of day. These form a
      SOFT PRIOR over intents -- enough to tilt a neutral homepage ("accent"),
      never enough to fully personalise on their own.

  STAGE 2 · BEHAVIOURAL (events 1..k, incremental)
      Every click/action updates the score. Two kinds of evidence:
        * windowed evidence  -- fired by the newest event, decayed each step so
          the LAST 2-3 ACTIONS dominate (gamma^3 ~ 0.5);
        * state evidence     -- recomputed each step from running session shape
          (revisit ratio, category breadth, micro-visit), not decayed.
      posterior = softmax( (prior + decayed_window + state) / temperature )

  OVERRIDE · DECISIVE (commercial trigger)
      add-to-cart with <=2 prior views flips the session to checkout-support
      mode. This is a mode switch, not a prediction (RetailRocket: 75%+
      in-session purchase).

  GATE · UNCLEAR
      Calibrated confidence below threshold -> decline to personalise, serve
      NEUTRAL. Temperature and gate are FITTED on a held-out split
      (fit_temperature / fit_gate), not hand-picked.

What this fixes vs v1/v2
------------------------
  * v1/v2 scored a finished session; v3 is incremental -- it can answer
    "what should the homepage be NOW?" after every single event.
  * confidence was raw score-share; v3 confidence is a temperature-calibrated
    softmax probability.
  * arrival context (referrer/device/landing) and behaviour were separate
    tracks; v3 fuses them as prior x likelihood.
  * recency: v2 aggregates weighted the 1st and 30th click equally; v3 decays
    old evidence so the most recent 2-3 actions carry the signal.

The engine also exposes `score_aggregates()` so the real-data track
(RetailRocket, which only has offline session aggregates) runs through the
SAME intent taxonomy and gate -- with the leakage fix applied upstream
(features censored at the first commercial event, see build_real_sessions.py).
"""

from dataclasses import dataclass, field
import math

# ---------------------------------------------------------------------------
# Intent taxonomy (unified across both tracks)
# ---------------------------------------------------------------------------
ALL_INTENTS = ["Goal-driven", "Evaluator", "Explorer", "Price-sensitive", "Low-intent"]

# Real-data track (RetailRocket) has no search / price / referrer signals, so
# Goal-driven and Price-sensitive are structurally unobservable there.
REAL_INTENTS = ["Evaluator", "Explorer", "Low-intent"]

DISCOUNT_TERMS = ("discount", "cheap", "deal", "deals", "clearance", "sale", "coupon")


@dataclass
class Evidence:
    intent: str
    weight: float
    reason: str
    stage: str  # "prior" | "window" | "state"


# ---------------------------------------------------------------------------
# STAGE 1 -- cold-start prior from acquisition context
# ---------------------------------------------------------------------------
def prior_evidence(ctx):
    """
    ctx: dict with any of
      referrer: google|direct|instagram|facebook|email|ad|other
      device:   mobile|desktop|tablet
      landing:  home|category|product|sale
      hour:     0-23
    Weights are deliberately SMALL (<=1.5): a prior tilts, behaviour decides.
    """
    ev = []
    add = lambda i, w, r: ev.append(Evidence(i, w, r, "prior"))

    ref = str(ctx.get("referrer", "")).strip().lower()
    if ref == "google":
        add("Goal-driven", 1.2, "arrived from search (came looking for something)")
        add("Evaluator", 0.4, "search arrivals often compare")
    elif ref == "instagram":
        add("Explorer", 1.2, "arrived from Instagram (discovery channel)")
    elif ref == "facebook":
        add("Explorer", 0.8, "arrived from Facebook (discovery channel)")
    elif ref == "email":
        add("Price-sensitive", 1.0, "arrived from email (promo-driven channel)")
        add("Evaluator", 0.3, "email arrivals often return to a considered item")
    elif ref == "ad":
        add("Price-sensitive", 0.6, "arrived from a paid ad")
        add("Explorer", 0.4, "ad clicks skew impulse/browse")
    elif ref == "direct":
        add("Goal-driven", 0.6, "typed the address (has a mission)")

    landing = str(ctx.get("landing", "")).strip().lower()
    if landing == "product":
        add("Goal-driven", 1.0, "deep-linked straight to a product page")
    elif landing == "sale":
        add("Price-sensitive", 1.5, "landed on the sale page")
    elif landing == "category":
        add("Goal-driven", 0.3, "landed on a category (narrowed already)")
        add("Evaluator", 0.3, "category landings precede comparison")

    dev = str(ctx.get("device", "")).strip().lower()
    if dev == "desktop":
        add("Evaluator", 0.4, "on desktop (comfortable for comparison)")

    hour = ctx.get("hour", None)
    if hour is not None and 18 <= int(hour) <= 23:
        # RetailRocket finding: evening ~2x morning purchase rate. Soft prior only.
        add("Goal-driven", 0.15, "evening session (higher buy propensity)")
        add("Evaluator", 0.15, "evening session (higher buy propensity)")

    # Base-rate honesty: most anonymous traffic is a drive-by. The engine
    # starts humble; behaviour has to EARN a personalised page.
    add("Low-intent", 1.0, "base rate: most first visits are micro-visits")
    return ev


# ---------------------------------------------------------------------------
# STAGE 2 -- incremental behavioural scorer
# ---------------------------------------------------------------------------
class SessionState:
    """Running counters, cheap enough to keep per live session."""

    def __init__(self):
        self.n_events = 0
        self.n_views = 0
        self.views_per_item = {}
        self.categories = {}
        self.last_ts = None
        self.first_ts = None
        self.searched = False          # any search
        self.nondeal_search = False    # searched a product term (not a deal term)
        self.carted = False
        self.price_signal_count = 0    # deal search / price sort / price filter
        self.sale_views = 0            # views inside the Sale section
        self.gap_sum = 0.0             # for average inter-event pace
        self.gap_n = 0

    # -- derived ------------------------------------------------------------
    @property
    def n_unique_items(self):
        return len(self.views_per_item)

    @property
    def revisit_ratio(self):
        return self.n_views / self.n_unique_items if self.views_per_item else 0.0

    @property
    def n_categories(self):
        return len(self.categories)

    @property
    def top_category_share(self):
        if not self.categories:
            return 0.0
        return max(self.categories.values()) / max(1, self.n_views)

    @property
    def duration_sec(self):
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        return max(0.0, self.last_ts - self.first_ts)

    @property
    def avg_gap_sec(self):
        return self.gap_sum / self.gap_n if self.gap_n else 0.0


def window_evidence(event, state, prev_event):
    """Evidence fired by THIS event (later decayed => recent actions dominate)."""
    ev = []
    add = lambda i, w, r: ev.append(Evidence(i, w, r, "window"))
    etype = event.get("type")

    if etype == "view":
        item = event.get("item")
        cat = event.get("category")
        if item is not None and state.views_per_item.get(item, 0) >= 1:
            add("Evaluator", 2.5, "re-viewed an item just seen (comparison behaviour)")
        if prev_event is not None and prev_event.get("type") == "view":
            if cat is not None and prev_event.get("category") not in (None, cat):
                add("Explorer", 1.0, "hopped category between consecutive views")
            gap = (event.get("ts", 0) or 0) - (prev_event.get("ts", 0) or 0)
            # PDP dwell bands, measured on censored train windows
            # (measure_dwell.py): buy-rate peaks at 120-300s and falls off a
            # cliff past ~600s -- that long is a parked tab, not reading.
            if 60 <= gap < 120:
                add("Evaluator", 0.8, f"dwelled {gap:.0f}s on the previous item (read the page)")
            elif 120 <= gap < 600:
                add("Evaluator", 1.0, f"dwelled {gap:.0f}s on the previous item (deliberate consideration)")
            elif 0 < gap < 5:
                add("Explorer", 0.4, "rapid-fire flicking between items")
            # gap >= 600s: idle cap -- no dwell credit for a left-open tab.
        if str(cat).strip().lower() == "sale":
            add("Price-sensitive", 1.0, "viewing items in the Sale section")

    elif etype == "search":
        q = str(event.get("query", "")).lower()
        if any(t in q for t in DISCOUNT_TERMS):
            add("Price-sensitive", 2.0, f"searched for a deal term ('{q}')")
        else:
            add("Goal-driven", 1.8, f"searched '{q}' (arrived with something in mind)")

    elif etype == "sort":
        key = str(event.get("sort_key", "")).lower()
        if key == "price_asc":
            add("Price-sensitive", 1.6, "sorted by price low-to-high")
        elif key in ("rating", "reviews"):
            add("Evaluator", 1.6, "sorted by rating (quality comparison)")
        elif key == "trending":
            add("Explorer", 1.5, "sorted by trending (browsing, not seeking)")
        elif key in ("relevance", "newest"):
            add("Goal-driven", 1.0, f"sorted by {key} (seeking a match)")

    elif etype == "filter":
        key = str(event.get("filter_key", "")).lower()
        add("Evaluator", 0.8, "applied a filter (narrowing deliberately)")
        if key == "price":
            add("Price-sensitive", 1.2, "filtered on price")

    elif etype == "addtocart":
        # Non-override carts still say "converging on a choice".
        add("Goal-driven", 1.0, "added to cart")

    return ev


def state_evidence(state):
    """Whole-session shape, recomputed fresh each step (not decayed).

    Sticky traits live here on purpose: a deal search or a price-sort is a
    property of the SESSION, not of the moment -- it must not decay away
    like windowed evidence does (the v3.0 Price-sensitive recall bug).
    """
    ev = []
    add = lambda i, w, r: ev.append(Evidence(i, w, r, "state"))

    if state.n_events == 0:
        add("Low-intent", 1.6, "no behaviour observed yet -- base rate says humble")
        return ev
    if state.n_events <= 2:
        add("Low-intent", 1.6, f"only {state.n_events} event(s) so far -- barely any signal")
        if state.duration_sec < 60:
            add("Low-intent", 0.6, "under a minute on site")
        elif state.duration_sec < 600:
            # Sauvik's 60-120s dwell band, extended to the measured idle cap.
            # Site-dependent (RetailRocket: micro buy-rate rises with dwell;
            # REES46: falls), so the tilt is deliberately small -- enough to
            # combine with a prior and reach Unclear/neutral, never enough
            # to out-vote behavioural evidence.
            add("Evaluator", 0.3, f"lingered {state.duration_sec:.0f}s rather than bouncing")
        return ev

    rr = state.revisit_ratio
    price_hunting = state.price_signal_count >= 1

    # --- sticky price-seeking trait -----------------------------------------
    if price_hunting:
        add("Price-sensitive", 1.3, "took a price-seeking action (deal search / price sort / price filter)")
        if state.price_signal_count >= 2:
            add("Price-sensitive", 1.2, f"{state.price_signal_count} separate price-seeking actions")
    if state.n_views >= 3 and state.sale_views / max(1, state.n_views) >= 0.5:
        add("Price-sensitive", 1.0, "most views inside the Sale section")

    # --- sticky search trait --------------------------------------------------
    if state.nondeal_search:
        add("Goal-driven", 1.2, "came in through a product search")
        if state.top_category_share >= 0.8:
            add("Goal-driven", 0.8, "search + stayed inside one category")
        if state.avg_gap_sec and state.avg_gap_sec < 40:
            add("Goal-driven", 0.5, "moving fast (knows what they want)")

    # --- re-viewing: the golden comparison signal -----------------------------
    # Credit routing: a deal-hunter re-views too -- when price signals are
    # present, the revisit pattern supports Price-sensitive, not Evaluator.
    if rr >= 1.8:
        if price_hunting:
            add("Price-sensitive", 1.5, f"re-viewing deal candidates {rr:.1f}x")
        else:
            w = 2.0 if state.n_views >= 6 else 1.0
            add("Evaluator", w, f"re-viewing items {rr:.1f}x on average (golden signal)")
    if (not price_hunting and state.avg_gap_sec >= 60
            and state.n_views >= 5 and rr >= 1.4):
        add("Evaluator", 0.8, f"deliberate pace ({state.avg_gap_sec:.0f}s between actions)")

    # --- breadth --------------------------------------------------------------
    if state.n_categories >= 3:
        add("Explorer", 1.8, f"touched {state.n_categories} categories (browsing wide)")
    # Real-data finding: focused scan WITHOUT re-views is a drive-by (1.0% buy).
    if state.n_views >= 4 and state.top_category_share >= 0.9 and rr < 1.4 \
            and not state.nondeal_search:
        add("Low-intent", 1.5, "scanned one category without engaging any item twice")
    return ev


# ---------------------------------------------------------------------------
# Inference result
# ---------------------------------------------------------------------------
@dataclass
class Inference:
    intent: str            # winning intent | "Decisive" | "Unclear"
    confidence: float      # calibrated max-probability
    probs: dict            # calibrated distribution over intents
    scores: dict           # raw additive scores (auditable)
    reasons: list = field(default_factory=list)
    mode: str = "behavioural"   # "cold" | "behavioural" | "override"
    served: str = "Neutral"     # what the homepage layer should render
    # Orthogonal price flavor (REES46 track): cheap-leaning browsing lifts
    # conversion within EVERY intent, so it tilts merchandising, not layout.
    price_conscious: bool = False

    def explain(self):
        rank = ", ".join(f"{k} {self.probs[k]:.0%}"
                         for k in sorted(self.probs, key=self.probs.get, reverse=True))
        why = "; ".join(self.reasons[:3])
        return (f"{self.intent} (mode={self.mode}, conf {self.confidence:.0%}) "
                f"-> serve {self.served}\n    probs: {rank}\n    because: {why}")


def _softmax(scores, temperature):
    mx = max(scores.values())
    exps = {k: math.exp((v - mx) / max(1e-6, temperature)) for k, v in scores.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class TwoStageEngine:
    """
    One instance per live session.

        eng = TwoStageEngine()
        eng.start_session({"referrer": "instagram", "device": "mobile", ...})
        print(eng.current())          # <- cold-start decision (Stage 1 only)
        eng.observe(event); eng.current()   # <- refreshed after every action
    """

    DEFAULTS = dict(temperature=2.0, gate=0.45, cold_gate=0.55, decay=0.80)

    def __init__(self, intents=None, temperature=None, gate=None,
                 cold_gate=None, decay=None, disabled_rules=None,
                 weight_scale=None):
        self.intents = list(intents or ALL_INTENTS)
        self.temperature = temperature or self.DEFAULTS["temperature"]
        self.gate = gate or self.DEFAULTS["gate"]
        self.cold_gate = cold_gate or self.DEFAULTS["cold_gate"]
        self.decay = decay or self.DEFAULTS["decay"]
        # Ablation hooks (offline score_aggregates path): rule ids to silence
        # entirely, and per-rule weight multipliers for sensitivity analysis.
        self.disabled_rules = set(disabled_rules or ())
        self.weight_scale = dict(weight_scale or {})
        self._reset()

    def _reset(self):
        self.state = SessionState()
        self.prior = {i: 0.0 for i in self.intents}
        self.window = {i: 0.0 for i in self.intents}
        self.reasons = {i: [] for i in self.intents}
        self.prev_event = None
        self.override = None

    # -- session lifecycle ----------------------------------------------------
    def start_session(self, ctx=None):
        self._reset()
        for e in prior_evidence(ctx or {}):
            if e.intent in self.prior:
                self.prior[e.intent] += e.weight
                self.reasons[e.intent].append((e.weight, e.reason))
        return self.current()

    def observe(self, event):
        """Feed one event dict: {type, ts, item, category, query, sort_key, ...}"""
        st = self.state
        ts = event.get("ts")
        if ts is not None:
            if st.first_ts is None:
                st.first_ts = ts
            st.last_ts = ts

        # ---- OVERRIDE check BEFORE counting this event ----------------------
        if event.get("type") in ("addtocart", "purchase") and st.n_views <= 2 \
                and self.override is None:
            self.override = Inference(
                intent="Decisive", confidence=0.95,
                probs={"Decisive": 1.0}, scores={"Decisive": 1.0},
                reasons=["commercial action with almost no browsing first "
                         "(arrived already decided)"],
                mode="override", served="Checkout-support",
            )

        # ---- decay old window evidence: recent 2-3 actions dominate ---------
        for i in self.intents:
            self.window[i] *= self.decay

        # ---- new evidence ----------------------------------------------------
        for e in window_evidence(event, st, self.prev_event):
            if e.intent in self.window:
                self.window[e.intent] += e.weight
                self.reasons[e.intent].append((e.weight, e.reason))

        # ---- update counters -------------------------------------------------
        if self.prev_event is not None and ts is not None:
            prev_ts = self.prev_event.get("ts")
            if prev_ts is not None and ts >= prev_ts:
                # Idle cap: one overnight tab-park must not dominate the pace.
                st.gap_sum += min(ts - prev_ts, 600.0)
                st.gap_n += 1
        st.n_events += 1
        etype = event.get("type")
        if etype == "view":
            st.n_views += 1
            item, cat = event.get("item"), event.get("category")
            if item is not None:
                st.views_per_item[item] = st.views_per_item.get(item, 0) + 1
            if cat is not None:
                st.categories[cat] = st.categories.get(cat, 0) + 1
            if str(cat).strip().lower() == "sale":
                st.sale_views += 1
        elif etype == "search":
            st.searched = True
            q = str(event.get("query", "")).lower()
            if any(t in q for t in DISCOUNT_TERMS):
                st.price_signal_count += 1
            else:
                st.nondeal_search = True
        elif etype == "sort" and str(event.get("sort_key", "")).lower() == "price_asc":
            st.price_signal_count += 1
        elif etype == "filter" and str(event.get("filter_key", "")).lower() == "price":
            st.price_signal_count += 1
        elif etype == "addtocart":
            st.carted = True

        self.prev_event = event
        return self.current()

    # -- scoring ---------------------------------------------------------------
    def _combined_scores(self):
        combined = {i: self.prior[i] + self.window[i] for i in self.intents}
        for e in state_evidence(self.state):
            if e.intent in combined:
                combined[e.intent] += e.weight
        return combined

    def current(self):
        if self.override is not None:
            return self.override

        scores = self._combined_scores()
        probs = _softmax(scores, self.temperature)
        winner = max(probs, key=probs.get)
        conf = probs[winner]

        # collect state reasons fresh (they aren't stored)
        state_rs = [(e.weight, e.reason) for e in state_evidence(self.state)
                    if e.intent == winner]
        top = [r for _, r in sorted(self.reasons[winner] + state_rs, reverse=True)]

        cold = self.state.n_events == 0
        mode = "cold" if cold else "behavioural"
        gate = self.cold_gate if cold else self.gate

        if conf < gate:
            served = f"Neutral (accent: {winner})" if cold else "Neutral"
            return Inference("Unclear", conf, probs, scores,
                             ["signals point in multiple directions -- decline to commit"]
                             + top[:1], mode, served)
        return Inference(winner, conf, probs, scores, top, mode, served=winner)

    # -- offline adapter for the real-data track --------------------------------
    def score_aggregates(self, f):
        """
        Score a CENSORED aggregate feature row (RetailRocket track).
        f keys: n_events, n_views, revisit_ratio, n_categories,
                top_category_share, median_gap_sec, duration_sec,
                added_to_cart, views_before_first_commercial
                (+ optional price_rel_cat -> price-conscious flavor, REES46)
        """
        g = lambda k, d=0.0: (f.get(k) if f.get(k) == f.get(k) else d) or d  # NaN-safe

        # Cheap-leaning within category (median viewed price < 60% of the
        # category median). CENSORED upstream like every other feature.
        cheap = ("cheap_flavor" not in self.disabled_rules
                 and 0 < g("price_rel_cat", 0.0) < 0.60)
        cheap_reason = "browsing the cheap end of categories (value tilt)"

        if ("decisive_override" not in self.disabled_rules
                and g("added_to_cart")
                and g("views_before_first_commercial", 99) <= 2):
            return Inference("Decisive", 0.95, {"Decisive": 1.0}, {"Decisive": 1.0},
                             ["cart with almost no prior browsing"]
                             + ([cheap_reason] if cheap else []),
                             mode="override", served="Checkout-support",
                             price_conscious=cheap)

        scores = {i: 0.0 for i in self.intents}
        reasons = {i: [] for i in self.intents}

        def add(rid, i, w, r):
            if rid in self.disabled_rules or i not in scores:
                return
            w *= self.weight_scale.get(rid, 1.0)
            scores[i] += w
            reasons[i].append((w, r))

        n_events, n_views = g("n_events"), g("n_views")
        rr, n_cats = g("revisit_ratio", 1.0), g("n_categories")
        focus, gap, dur = g("top_category_share"), g("median_gap_sec"), g("duration_sec")

        if n_events <= 2 or n_views < 2:
            add("micro", "Low-intent", 2.5, f"micro-visit ({n_events:.0f} events)")
            if dur < 90:
                add("micro_short", "Low-intent", 1.0, f"gone in {dur:.0f}s")
            elif dur < 600:
                add("micro_linger", "Evaluator", 0.3,
                    f"lingered {dur:.0f}s rather than bouncing")
        else:
            if rr >= 1.8:
                add("revisit_core", "Evaluator", 3.0,
                    f"re-viewed items {rr:.1f}x (comparison)")
                if focus >= 0.75:
                    add("revisit_focus", "Evaluator", 1.5,
                        f"{focus:.0%} of views in one category")
                # Dwell credit widened to the measured band (60s reading
                # threshold) and idle-capped at 600s (parked tab).
                if 60 <= gap < 600:
                    add("revisit_pace", "Evaluator", 0.5, "deliberate pace")
            elif rr <= 1.1:
                add("no_revisit", "Explorer", 1.0, "almost never returned to an item")
            if n_cats >= 3:
                add("broad_cats", "Explorer", 3.0, f"touched {n_cats:.0f} categories")
            if g("category_switch_rate") >= 0.5:
                add("cat_switch", "Explorer", 2.0, "switched category on half of view steps")
            if focus >= 0.9 and rr < 1.4 and n_views >= 4:
                add("scan_no_engage", "Low-intent", 1.5, "category scan without item engagement")
            if dur < 90 and rr < 1.8:
                add("brief_shallow", "Low-intent", 2.0, "brief shallow visit")
            if n_views <= 3 and rr < 1.8 and n_cats <= 2:
                add("few_signals", "Low-intent", 1.0, "too few distinct signals")

        probs = _softmax(scores, self.temperature)
        winner = max(probs, key=probs.get)
        conf = probs[winner]
        top = [r for _, r in sorted(reasons[winner], reverse=True)]
        if cheap:
            top.append(cheap_reason)
        if conf < self.gate:
            return Inference("Unclear", conf, probs, scores,
                             ["conflicting evidence"], "behavioural", "Neutral",
                             price_conscious=cheap)
        return Inference(winner, conf, probs, scores, top, "behavioural", winner,
                         price_conscious=cheap)


# ---------------------------------------------------------------------------
# Calibration -- fitted on a held-out split, never hand-picked
# ---------------------------------------------------------------------------
def fit_temperature(score_rows, labels, grid=None):
    """
    Minimise NLL of the true label under softmax(scores/T).
    score_rows: list of dicts intent->raw score;  labels: list of true intents.
    """
    grid = grid or [0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    best_t, best_nll = None, float("inf")
    for t in grid:
        nll = 0.0
        for scores, y in zip(score_rows, labels):
            p = _softmax(scores, t).get(y, 1e-9)
            nll -= math.log(max(p, 1e-9))
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t


def fit_gate(confidences, correct, target_precision=0.85):
    """
    Smallest threshold whose decided-set precision >= target.
    Returns (threshold, coverage, precision) on the fit split.
    """
    pairs = sorted(zip(confidences, correct))
    best = (0.99, 0.0, 1.0)
    for t in [x / 100 for x in range(25, 96, 2)]:
        dec = [(c, ok) for c, ok in pairs if c >= t]
        if not dec:
            continue
        prec = sum(ok for _, ok in dec) / len(dec)
        cov = len(dec) / len(pairs)
        if prec >= target_precision:
            return t, cov, prec
        best = (t, cov, prec)
    return best


def expected_calibration_error(confidences, correct, bins=10):
    """Standard ECE: |accuracy - confidence| averaged over confidence bins."""
    tot, ece = len(confidences), 0.0
    if tot == 0:
        return 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        ece += (len(idx) / tot) * abs(acc - conf)
    return ece


if __name__ == "__main__":
    eng = TwoStageEngine()
    print("== arrival (Stage 1 only) ==")
    print(eng.start_session({"referrer": "instagram", "device": "mobile",
                             "landing": "home", "hour": 21}).explain())
    journey = [
        {"type": "view", "ts": 0, "item": "A", "category": "fashion"},
        {"type": "view", "ts": 30, "item": "B", "category": "home"},
        {"type": "view", "ts": 55, "item": "C", "category": "electronics"},
        {"type": "view", "ts": 130, "item": "C", "category": "electronics"},
        {"type": "sort", "ts": 150, "sort_key": "rating"},
        {"type": "view", "ts": 260, "item": "C", "category": "electronics"},
    ]
    for k, e in enumerate(journey, 1):
        print(f"\n== after event {k}: {e['type']} ==")
        print(eng.observe(e).explain())
