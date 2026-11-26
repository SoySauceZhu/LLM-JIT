"""Shared five-class metrics, including terminal failures."""

from collections import Counter

from mortgage_prompt import LABELS


def classification_scores(matrix):
    n = sum(sum(row.values()) for row in matrix.values())
    per_class = {}
    for label in LABELS:
        support = sum(matrix[label].values())
        predicted = sum(row[label] for row in matrix.values())
        tp = matrix[label][label]
        per_class[label] = {
            "support": support, "recall": tp / support if support else None,
            "fpr": (predicted - tp) / (n - support) if support < n else None,
            "f1": 2 * tp / (support + predicted) if support + predicted else 0.0,
        }
    return {
        "n": n,
        "valid_rate": 1 - sum(row["FAIL"] for row in matrix.values()) / n if n else None,
        "accuracy": sum(matrix[label][label] for label in LABELS) / n if n else None,
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(LABELS) if n else None,
        "per_class": per_class,
    }


def score(rows, field, *, paired=False):
    matrix = {ref: dict.fromkeys((*LABELS, "FAIL"), 0) for ref in LABELS}
    agree = 0
    for row in rows:
        label = row[field]
        matrix[row["reference_label"]][label or "FAIL"] += 1
        agree += bool(label and label == row.get("teacher_label"))
    return {
        **classification_scores(matrix), "correct": sum(matrix[k][k] for k in LABELS),
        "teacher_matches": agree if paired else None,
        "teacher_agreement_all_cases": agree / len(rows) if paired and rows else None,
        "predicted_counts": dict(Counter(row[field] or "FAIL" for row in rows)),
        "confusion_rows_reference_columns_prediction": matrix,
    }


def teacher_agreement_by_class(rows, field):
    """Group matches by the Teacher's valid label, including Student failures in support."""
    support = Counter(row["teacher_label"] for row in rows)
    matches = Counter(row["teacher_label"] for row in rows if row[field] == row["teacher_label"])
    return {label: {"support": support[label], "matches": matches[label],
                    "agreement": matches[label] / support[label] if support[label] else None}
            for label in LABELS}
