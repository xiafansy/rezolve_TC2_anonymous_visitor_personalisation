"""
End-to-end pipeline walkthrough on REAL sessions from anonymous_sessions.csv.

For each intent it picks a representative session and prints the full chain
(signals -> inferred intent -> personalised home page), then shows a genuinely
ambiguous session triggering the NEUTRAL fallback.

Also exports homepage_demo_data.json -- a diverse sample used by the visual demo
so the page renders the real engine's output rather than duplicating the logic.
"""

import json

import pandas as pd

from intent_inference import infer
from personalisation import build_homepage, render_text, MIN_CONFIDENCE

SIGNAL_COLS = [
    "Referrer", "Device", "Category", "Search_Used", "Search_Query",
    "Scroll_Depth", "Product_Views", "Filter_Used", "Sort_Type",
    "Session_Duration_sec", "Add_to_Cart", "Purchase",
]


def to_session(row):
    s = {c: row[c] for c in SIGNAL_COLS}
    s["Search_Query"] = "" if pd.isna(s["Search_Query"]) else s["Search_Query"]
    return s


def main():
    df = pd.read_csv("anonymous_sessions.csv")

    # Attach prediction + confidence to every row.
    results = [infer(to_session(r)) for _, r in df.iterrows()]
    df["Pred"] = [r.intent for r in results]
    df["Conf"] = [r.confidence for r in results]

    print("#" * 68)
    print("#  ANONYMOUS-VISITOR PERSONALISATION -- END-TO-END PIPELINE")
    print("#" * 68)

    # One confident, correct exemplar per intent.
    for intent in ["Goal-driven", "Explorer", "Research", "Price-sensitive"]:
        pool = df[(df["Intent"] == intent) & (df["Pred"] == intent)
                  & (df["Conf"] >= MIN_CONFIDENCE)].sort_values("Conf", ascending=False)
        row = pool.iloc[0]
        s = to_session(row)
        print("\n" + "-" * 68)
        print(f"SESSION #{int(row['Session_ID'])}  (true intent: {intent})")
        print("  signals: " + ", ".join(
            f"{k}={s[k]}" for k in
            ["Referrer", "Device", "Category", "Search_Used", "Sort_Type",
             "Scroll_Depth", "Product_Views", "Session_Duration_sec"]))
        print(render_text(build_homepage(s), s))

    # A low-confidence session -> NEUTRAL fallback.
    amb = df[df["Conf"] < MIN_CONFIDENCE].sort_values("Conf")
    if len(amb):
        row = amb.iloc[0]
        s = to_session(row)
        print("\n" + "-" * 68)
        print(f"SESSION #{int(row['Session_ID'])}  (true intent: {row['Intent']}) "
              f"-- AMBIGUOUS")
        print("  signals: " + ", ".join(
            f"{k}={s[k]}" for k in
            ["Referrer", "Device", "Category", "Search_Used", "Sort_Type",
             "Scroll_Depth", "Product_Views", "Session_Duration_sec"]))
        print(render_text(build_homepage(s), s))

    # ---- Export a diverse sample for the visual demo ----------------------
    export = []
    seen_intent = {}
    for _, row in df.iterrows():
        s = to_session(row)
        hp = build_homepage(s)
        # Keep the demo set small but varied: a few per predicted bucket.
        bucket = hp.intent
        if seen_intent.get(bucket, 0) >= 3:
            continue
        seen_intent[bucket] = seen_intent.get(bucket, 0) + 1
        export.append({
            "session_id": int(row["Session_ID"]),
            "true_intent": row["Intent"],
            "signals": {k: (bool(s[k]) if isinstance(s[k], (bool,))
                            else (int(s[k]) if isinstance(s[k], (int,)) else s[k]))
                        for k in SIGNAL_COLS},
            "inferred_intent": hp.intent,
            "confidence": round(hp.confidence, 3),
            "personalised": hp.personalised,
            "reasons": hp.reasons,
            "homepage": {
                "headline": hp.headline,
                "primary_goal": hp.primary_goal,
                "tone": hp.tone,
                "hero": hp.hero,
                "blocks": [{"title": b.title, "rationale": b.rationale}
                           for b in hp.blocks],
                "suppress": hp.suppress,
            },
        })

    with open("homepage_demo_data.json", "w") as f:
        json.dump(export, f, indent=2, default=str)
    print("\n" + "#" * 68)
    print(f"# Exported {len(export)} sample sessions -> homepage_demo_data.json")
    print("#" * 68)


if __name__ == "__main__":
    main()
