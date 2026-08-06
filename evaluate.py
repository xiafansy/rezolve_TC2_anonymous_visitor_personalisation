"""
Offline evaluation of the rule-based intent inferer.

Runs infer() over every session in anonymous_sessions.csv, then compares the
prediction against the ground-truth `Intent` label (which the inferer never
sees) to report accuracy, a confusion matrix, and per-class precision / recall
/ F1. Also prints a few worked examples of the explainable output.

Metrics are computed with plain pandas/numpy -- no sklearn dependency.
"""

import pandas as pd

from intent_inference import INTENTS, infer

DATA = "anonymous_sessions.csv"

# Signals the inferer is allowed to see. `Intent` is intentionally excluded.
SIGNAL_COLS = [
    "Referrer", "Device", "Category", "Search_Used", "Search_Query",
    "Scroll_Depth", "Product_Views", "Filter_Used", "Sort_Type",
    "Session_Duration_sec", "Add_to_Cart", "Purchase",
]


def confusion_matrix(y_true, y_pred, labels):
    m = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for t, p in zip(y_true, y_pred):
        m.loc[t, p] += 1
    return m


def per_class_metrics(cm):
    """Precision / recall / F1 / support from a confusion matrix (rows=true)."""
    rows = []
    for lbl in cm.index:
        tp = cm.loc[lbl, lbl]
        fp = cm[lbl].sum() - tp          # predicted lbl but wasn't
        fn = cm.loc[lbl].sum() - tp      # was lbl but predicted otherwise
        support = cm.loc[lbl].sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        rows.append({
            "Intent": lbl, "Precision": precision, "Recall": recall,
            "F1": f1, "Support": support,
        })
    return pd.DataFrame(rows).set_index("Intent")


def main():
    df = pd.read_csv(DATA)
    df["Search_Query"] = df["Search_Query"].fillna("")

    preds, confs = [], []
    for _, row in df.iterrows():
        session = {c: row[c] for c in SIGNAL_COLS}
        result = infer(session)
        preds.append(result.intent)
        confs.append(result.confidence)

    df["Predicted"] = preds
    df["Confidence"] = confs
    correct = (df["Predicted"] == df["Intent"])

    # ---- Headline accuracy ------------------------------------------------
    acc = correct.mean()
    print("=" * 64)
    print(f"OVERALL ACCURACY : {acc:.1%}  ({correct.sum()}/{len(df)})")
    print(f"Mean confidence  : {df['Confidence'].mean():.1%}")
    print("=" * 64)

    # ---- Confusion matrix -------------------------------------------------
    cm = confusion_matrix(df["Intent"], df["Predicted"], INTENTS)
    print("\nCONFUSION MATRIX  (rows = true, cols = predicted)\n")
    print(cm.to_string())

    # ---- Per-class metrics ------------------------------------------------
    metrics = per_class_metrics(cm)
    print("\nPER-CLASS METRICS\n")
    print(metrics.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nMacro-F1 : {metrics['F1'].mean():.3f}")

    # ---- Confidence when right vs wrong -----------------------------------
    print("\nMean confidence when CORRECT : "
          f"{df.loc[correct, 'Confidence'].mean():.1%}")
    print("Mean confidence when WRONG   : "
          f"{df.loc[~correct, 'Confidence'].mean():.1%}")

    # ---- A few explained examples ----------------------------------------
    print("\n" + "=" * 64)
    print("SAMPLE EXPLANATIONS")
    print("=" * 64)
    for _, row in df.head(4).iterrows():
        session = {c: row[c] for c in SIGNAL_COLS}
        result = infer(session)
        mark = "OK " if result.intent == row["Intent"] else "MISS"
        print(f"\n[{mark}] true={row['Intent']}")
        print("    " + result.explain().replace("\n", "\n    "))


if __name__ == "__main__":
    main()
