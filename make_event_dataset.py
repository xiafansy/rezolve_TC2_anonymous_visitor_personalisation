"""
make_event_dataset.py -- EVENT-LEVEL synthetic sessions (v3).

Why a new generator
-------------------
The v1 generator emitted one aggregate row per session, so incremental
inference ("what do we know after click 2?") was untestable. v3 emits the
actual event stream: arrival context + a timestamped sequence of
view / search / sort / filter / addtocart / purchase events.

Realism carried over from the RetailRocket findings:
  * Low-intent dominates traffic (default 45% of sessions, 1-3 events);
  * Evaluators re-view the same item(s) -- the golden signal;
  * archetypes are NOISY: 15% of sessions get off-archetype events, and
    context (referrer etc.) is only correlated with intent, never determined.

Outputs
-------
  synthetic_sessions.csv  session_id, intent(label), referrer, device, landing,
                          hour, n_events, added_to_cart, purchased
  synthetic_events.csv    session_id, idx, ts, type, item, category, query,
                          sort_key, filter_key
"""

import random
import pandas as pd

SEED = 42
N_SESSIONS = 3000

MIX = {  # traffic mix (Low-intent heavy, per real-data reality)
    "Low-intent": 0.45, "Explorer": 0.15, "Evaluator": 0.15,
    "Goal-driven": 0.15, "Price-sensitive": 0.10,
}

CATS = ["running", "fashion", "electronics", "beauty", "home", "sale"]
QUERIES = {
    "running": ["running shoes", "trail runners"], "fashion": ["summer dress", "black jacket"],
    "electronics": ["wireless headphones", "gaming mouse"], "beauty": ["skincare set", "perfume"],
    "home": ["desk lamp", "coffee table"], "sale": ["discount", "clearance deals"],
}
DEAL_QUERIES = ["discount", "cheap deals", "clearance", "sale items"]


def _pick(weighted):  # {value: weight} -> value
    vals, w = zip(*weighted.items())
    return random.choices(vals, weights=w)[0]


def _ctx(intent):
    """Acquisition context, CORRELATED with intent but overlapping heavily."""
    if intent == "Goal-driven":
        ref = _pick({"google": 0.55, "direct": 0.25, "email": 0.08, "instagram": 0.07, "ad": 0.05})
        landing = _pick({"product": 0.45, "category": 0.3, "home": 0.25})
        device = _pick({"mobile": 0.5, "desktop": 0.4, "tablet": 0.1})
    elif intent == "Evaluator":
        ref = _pick({"google": 0.4, "direct": 0.25, "email": 0.15, "facebook": 0.1, "instagram": 0.1})
        landing = _pick({"category": 0.4, "product": 0.3, "home": 0.3})
        device = _pick({"desktop": 0.55, "mobile": 0.35, "tablet": 0.1})
    elif intent == "Explorer":
        ref = _pick({"instagram": 0.4, "facebook": 0.2, "direct": 0.2, "google": 0.15, "ad": 0.05})
        landing = _pick({"home": 0.6, "category": 0.3, "sale": 0.1})
        device = _pick({"mobile": 0.6, "tablet": 0.2, "desktop": 0.2})
    elif intent == "Price-sensitive":
        ref = _pick({"email": 0.35, "ad": 0.2, "google": 0.2, "direct": 0.15, "facebook": 0.1})
        landing = _pick({"sale": 0.45, "home": 0.3, "category": 0.25})
        device = _pick({"mobile": 0.55, "desktop": 0.35, "tablet": 0.1})
    else:  # Low-intent: basically the marginal distribution
        ref = _pick({"instagram": 0.25, "google": 0.2, "direct": 0.2, "facebook": 0.15,
                     "ad": 0.1, "email": 0.1})
        landing = _pick({"home": 0.5, "category": 0.25, "product": 0.15, "sale": 0.1})
        device = _pick({"mobile": 0.65, "desktop": 0.2, "tablet": 0.15})
    hour = _pick({9: 1, 12: 1.4, 15: 1.4, 18: 1.6, 21: 1.8, 23: 0.8})
    return dict(referrer=ref, device=device, landing=landing, hour=hour)


def _item(cat, i):
    return f"{cat[:3]}-{i:03d}"


def _events(intent, ctx):
    """Generate the event stream for one session. Returns (events, carted, bought)."""
    ev, ts = [], 0.0

    def emit(**kw):
        nonlocal ts
        ev.append(dict(ts=round(ts, 1), **kw))

    def step(lo, hi):
        nonlocal ts
        ts += random.uniform(lo, hi)

    carted = bought = False

    if intent == "Goal-driven":
        cat = random.choice([c for c in CATS if c != "sale"])
        # 15% decisive sub-mode: deep link -> 1-2 views -> cart. This is the
        # real-data "commercial-event-led" buyer the Stage-B override serves.
        if random.random() < 0.15:
            target = _item(cat, random.randint(0, 3))
            emit(type="view", item=target, category=cat); step(10, 40)
            if random.random() < 0.5:
                emit(type="view", item=target, category=cat); step(8, 30)
            carted = True
            emit(type="addtocart", item=target, category=cat); step(15, 60)
            if random.random() < 0.75:
                bought = True; emit(type="purchase", item=target, category=cat)
            return ev, carted, bought
        if random.random() < 0.75:
            emit(type="search", query=random.choice(QUERIES[cat])); step(4, 15)
        if random.random() < 0.4:
            emit(type="sort", sort_key=random.choice(["relevance", "newest"])); step(3, 10)
        target = _item(cat, random.randint(0, 3))
        for _ in range(random.randint(2, 4)):
            emit(type="view", item=_item(cat, random.randint(0, 9)), category=cat); step(8, 45)
        emit(type="view", item=target, category=cat); step(15, 60)
        if random.random() < 0.35:  # quick confirm re-view
            emit(type="view", item=target, category=cat); step(10, 40)
        if random.random() < 0.7:
            carted = True; emit(type="addtocart", item=target, category=cat); step(20, 90)
            if random.random() < 0.6:
                bought = True; emit(type="purchase", item=target, category=cat)

    elif intent == "Evaluator":
        cat = random.choice([c for c in CATS if c != "sale"])
        pool = [_item(cat, i) for i in range(random.randint(2, 4))]
        if random.random() < 0.5:
            emit(type="search", query=random.choice(QUERIES[cat])); step(5, 20)
        if random.random() < 0.5:
            emit(type="sort", sort_key="rating"); step(4, 12)
        if random.random() < 0.4:
            emit(type="filter", filter_key=random.choice(["brand", "size"])); step(4, 12)
        for _ in range(random.randint(6, 12)):
            it = random.choice(pool)
            if random.random() < 0.15:  # occasional stray view
                oc = random.choice(CATS)
                emit(type="view", item=_item(oc, random.randint(0, 9)), category=oc)
            else:
                emit(type="view", item=it, category=cat)
            step(40, 180)  # deliberate pace
        if random.random() < 0.35:
            carted = True; emit(type="addtocart", item=random.choice(pool), category=cat)
            step(30, 120)
            if random.random() < 0.3:
                bought = True; emit(type="purchase", item=pool[0], category=cat)

    elif intent == "Explorer":
        cats = random.sample(CATS, k=random.randint(3, 5))
        if random.random() < 0.55:
            emit(type="sort", sort_key="trending"); step(3, 10)
        for _ in range(random.randint(7, 16)):
            c = random.choice(cats)
            emit(type="view", item=_item(c, random.randint(0, 19)), category=c)
            step(4, 35)  # quick flicking
        if random.random() < 0.12:
            c = random.choice(cats)
            carted = True; emit(type="addtocart", item=_item(c, 1), category=c)
            if random.random() < 0.3:
                step(20, 60); bought = True; emit(type="purchase", item=_item(c, 1), category=c)

    elif intent == "Price-sensitive":
        if ctx["landing"] != "sale" and random.random() < 0.6:
            emit(type="search", query=random.choice(DEAL_QUERIES)); step(4, 15)
        if random.random() < 0.7:
            emit(type="sort", sort_key="price_asc"); step(3, 10)
        if random.random() < 0.5:
            emit(type="filter", filter_key="price"); step(4, 12)
        pool_cat = _pick({"sale": 0.6, random.choice(CATS[:-1]): 0.4})
        pool = [_item(pool_cat, i) for i in range(random.randint(3, 6))]
        for _ in range(random.randint(4, 9)):
            emit(type="view", item=random.choice(pool), category=pool_cat); step(15, 90)
        if random.random() < 0.5:
            carted = True; emit(type="addtocart", item=pool[0], category=pool_cat)
            step(20, 80)
            if random.random() < 0.45:
                bought = True; emit(type="purchase", item=pool[0], category=pool_cat)

    else:  # Low-intent micro-visit
        n = _pick({1: 0.5, 2: 0.3, 3: 0.2})
        c = random.choice(CATS)
        for _ in range(n):
            emit(type="view", item=_item(c, random.randint(0, 19)),
                 category=c if random.random() < 0.7 else random.choice(CATS))
            step(3, 25)

    # 15% noise: inject 1-2 off-archetype events mid-stream
    if random.random() < 0.15 and len(ev) >= 3:
        pos = random.randint(1, len(ev) - 1)
        c = random.choice(CATS)
        noise = dict(type="view", ts=ev[pos]["ts"] + 1,
                     item=_item(c, random.randint(0, 19)), category=c)
        ev.insert(pos + 1, noise)

    return ev, carted, bought


def main():
    random.seed(SEED)
    sess_rows, ev_rows = [], []
    for sid in range(1, N_SESSIONS + 1):
        intent = _pick(MIX)
        ctx = _ctx(intent)
        events, carted, bought = _events(intent, ctx)
        sess_rows.append(dict(session_id=sid, intent=intent, **ctx,
                              n_events=len(events), added_to_cart=carted,
                              purchased=bought))
        for i, e in enumerate(events):
            ev_rows.append(dict(session_id=sid, idx=i, ts=e["ts"], type=e["type"],
                                item=e.get("item", ""), category=e.get("category", ""),
                                query=e.get("query", ""), sort_key=e.get("sort_key", ""),
                                filter_key=e.get("filter_key", "")))

    s = pd.DataFrame(sess_rows)
    e = pd.DataFrame(ev_rows)
    s.to_csv("synthetic_sessions.csv", index=False)
    e.to_csv("synthetic_events.csv", index=False)
    print(f"sessions: {len(s):,}   events: {len(e):,}")
    print("\nintent mix:")
    print((s['intent'].value_counts(normalize=True) * 100).round(1).to_string())
    print(f"\ncart rate {s['added_to_cart'].mean():.1%}   "
          f"purchase rate {s['purchased'].mean():.1%}")


if __name__ == "__main__":
    main()
