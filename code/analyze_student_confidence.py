"""Reproduce the offline confidence audit; no model calls or reference edits."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/confidence/gemma3-270m-it-v3-lora-curve"
DEST = ROOT / "result/validation/gemma3-270m-it-v3-confidence"
LABELS = "ABCDE"
BINS = [0, .4, .6, .8, .9, .95, 1.000001]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(row, key="five_way_label"):
    s, t, y = (row[k] for k in (key, "teacher_label", "reference_label"))
    if not t:
        return "teacher_invalid"
    if s == t == y:
        return "all_equal"
    if s == t:
        return "student_teacher_not_reference"
    if t == y:
        return "teacher_reference_not_student"
    if s == y:
        return "student_reference_not_teacher"
    return "all_different"


def stats(rows, key="five_way_label"):
    n = len(rows)
    correct = sum(r[key] == r["reference_label"] for r in rows)
    matches = sum(bool(r["teacher_label"]) and r[key] == r["teacher_label"] for r in rows)
    return dict(n=n, correct=correct, teacher_matches=matches,
                accuracy=correct / n if n else None,
                agreement=matches / n if n else None,
                mean_confidence=mean(r["confidence"] for r in rows) if n else None)


def audit(rows):
    bins = []
    for lo, hi in zip(BINS, BINS[1:]):
        subset = [r for r in rows if lo <= r["confidence"] < hi]
        bins.append(dict(lower=lo, upper=min(hi, 1), **stats(subset)))
    groups = {}
    for name in sorted(set(map(group, rows))):
        subset = [r for r in rows if group(r) == name]
        groups[name] = dict(**stats(subset), high95=sum(r["confidence"] >= .95 for r in subset),
                            low80=sum(r["confidence"] < .8 for r in subset))
    thresholds = []
    for threshold in [0, .6, .8, .9, .95, .99, .995]:
        selected = [r for r in rows if r["confidence"] >= threshold]
        # Operational routing requires an actual accepted output and enough label mass.
        eligible = [r for r in selected if r["strict_label"] and r["label_mass"] >= .5]
        ids = {r["complaint_id"] for r in eligible}
        hybrid = [dict(r, routed_label=r["strict_label"] if r["complaint_id"] in ids else r["teacher_label"])
                  for r in rows]
        thresholds.append(dict(threshold=threshold, five_way=stats(selected),
                               strict_on_same_subset=stats(selected, "strict_label"),
                               eligible=stats(eligible, "strict_label"), coverage=len(eligible) / len(rows),
                               hybrid=stats(hybrid, "routed_label")))
    invalid = [r for r in rows if not r["strict_label"]]
    per_class = {}
    for label in LABELS:
        teacher_rows = [r for r in rows if r["teacher_label"] == label]
        ref_rows = [r for r in rows if r["reference_label"] == label]
        per_class[label] = dict(teacher=stats(teacher_rows), reference=stats(ref_rows),
                               teacher_reference_correct=sum(r["teacher_label"] == label for r in ref_rows))
    ece = {target: sum(b["n"] / len(rows) * abs(b["mean_confidence"] - b[target])
                       for b in bins if b["n"]) for target in ("accuracy", "agreement")}
    return dict(five_way=stats(rows), strict=stats(rows, "strict_label"), groups=groups,
                bins=bins, thresholds=thresholds, binned_ece=ece, per_class=per_class,
                teacher_confusions=dict(Counter(f"{r['teacher_label'] or 'INVALID'}->{r['five_way_label']}"
                                                 for r in rows if r["teacher_label"] != r["five_way_label"])),
                reference_confusions=dict(Counter(f"{r['reference_label']}->{r['five_way_label']}"
                                                   for r in rows if r["reference_label"] != r["five_way_label"])),
                invalid=dict(**stats(invalid),
                             ids=[r["complaint_id"] for r in invalid],
                             confidence_range=[min(r["confidence"] for r in invalid), max(r["confidence"] for r in invalid)],
                             label_mass_range=[min(r["label_mass"] for r in invalid), max(r["label_mass"] for r in invalid)]),
                valid_label_mass_median=median(r["label_mass"] for r in rows if r["strict_label"]))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    checkpoints, all_rows = {}, {}
    validation = {r["complaint_id"]: r for r in read_csv(ROOT / "data/mortgage_complaint.csv")
                  if r["split"] == "validation"}
    assert len(validation) == 1013
    teacher = {r["complaint_id"]: r for r in read_csv(
        ROOT / "output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all/predictions.csv")
               if r["split"] == "validation"}
    for path in sorted(SOURCE.glob("*/summary.json")):
        summary = json.loads(path.read_text())["checkpoints"][0]
        csv_path = path.parent / summary["confidence_file"]
        archived_path = ROOT / summary["validation_run"] / "predictions.csv"
        assert digest(archived_path) == summary["predictions_sha256"]
        rows, archived = read_csv(csv_path), read_csv(archived_path)
        assert len(rows) == len(archived) == len(validation)
        assert len({r["complaint_id"] for r in rows}) == len(validation)
        for row, previous in zip(rows, archived, strict=True):
            ident = row["complaint_id"]
            assert ident == previous["complaint_id"]
            assert row["reference_label"] == previous["reference_label"] == validation[ident]["reference_label"]
            assert row["teacher_label"] == previous["teacher_label"] == teacher[ident]["teacher_label"]
            assert row["strict_label"] == previous["student_label"]
            assert row["five_way_label"] == row["archived_first_token_choice"] == previous["first_token_choice"]
            for key in ["confidence", "label_mass", *[f"p_{c}" for c in LABELS]]:
                row[key] = float(row[key])
            assert abs(sum(row[f"p_{c}"] for c in LABELS) - 1) < 1e-5
            assert row["five_way_label"] == max(LABELS, key=lambda c: row[f"p_{c}"])
            assert abs(row["confidence"] - max(row[f"p_{c}"] for c in LABELS)) < 1e-6
        result = audit(rows)
        assert abs(result["five_way"]["accuracy"] - summary["five_way_cfpb_accuracy"]) < 1e-12
        assert abs(result["strict"]["accuracy"] - summary["archived_strict_cfpb_accuracy"]) < 1e-12
        metrics_path = ROOT / summary["validation_run"] / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        assert result["five_way"]["teacher_matches"] == metrics["first_token_choice"]["teacher_matches"]
        assert result["strict"]["teacher_matches"] == metrics["strict_generation"]["teacher_matches"]
        checkpoints[path.parent.name] = result
        all_rows[path.parent.name] = {r["complaint_id"]: r for r in rows}
        for source in [path, csv_path, archived_path, metrics_path]:
            source_hashes[str(source.relative_to(ROOT))] = digest(source)
    assert len(checkpoints) == 12
    latest = list(all_rows["n9000"].values())
    train_path = ROOT / "data/lora/teacher_deepseek-v4.1-flash-1017_v3/history_n9000.csv"
    train = read_csv(train_path)
    assert not set(validation).intersection(r["complaint_id"] for r in train)
    teacher_accuracy = sum(r["teacher_label"] == r["reference_label"] for r in latest) / len(latest)
    output = dict(definition="A-E conditional first-token softmax; confidence is not calibrated correctness",
                  sources_sha256=source_hashes, checkpoints=checkpoints,
                  teacher_accuracy=teacher_accuracy,
                  joint_accuracy_agreement_upper_bound=1 + teacher_accuracy,
                  train_teacher_counts=dict(Counter(r["teacher_label"] for r in train)))
    for source in [train_path, ROOT / "data/mortgage_complaint.csv",
                   ROOT / "output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all/predictions.csv"]:
        source_hashes[str(source.relative_to(ROOT))] = digest(source)
    review = []
    for row in latest:
        if row["five_way_label"] == row["reference_label"] == row["teacher_label"] and row["strict_label"]:
            continue
        ident = row["complaint_id"]
        late = [all_rows[k][ident] for k in ["n7000", "n8000", "n9000"]]
        review.append(dict(complaint_id=ident, group=group(row), reference_label=row["reference_label"],
                           teacher_label=row["teacher_label"], student_label=row["five_way_label"],
                           strict_label=row["strict_label"], confidence=row["confidence"], label_mass=row["label_mass"],
                           late3_agree_teacher=all(r["five_way_label"] == row["teacher_label"] for r in late),
                           late3_min_confidence=min(r["confidence"] for r in late),
                           raw_issue=validation[ident]["raw_issue"], text=validation[ident]["text"]))
    review.sort(key=lambda r: (r["group"], -r["confidence"]))
    with (DEST / "review_cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    output["review_count"] = len(review)
    (DEST / "analysis.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    plot(checkpoints)
    print(f"Verified {len(checkpoints)} checkpoints × {len(latest)} validation rows; artifacts: {DEST}")


def plot(checkpoints):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    sizes = [int(k[1:]) for k in checkpoints]
    for key, label, color in [("mean_confidence", "Mean confidence", "#d97706"),
                               ("agreement", "Teacher agreement", "#2563eb"),
                               ("accuracy", "Reference accuracy", "#059669")]:
        ax.plot(sizes, [r["five_way"][key] * 100 for r in checkpoints.values()], "o-", label=label, color=color)
    ax.axhline(90, color="gray", linestyle="--")
    ax.set(xlabel="Training examples", ylabel="Percent", title="Confidence rises faster than reference accuracy")
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    bins = checkpoints["n9000"]["bins"]
    for key, label, color in [("agreement", "Teacher agreement", "#2563eb"), ("accuracy", "Reference accuracy", "#059669")]:
        ax.plot([b["mean_confidence"] for b in bins], [b[key] for b in bins], "o-", label=label, color=color)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for b in bins:
        ax.annotate(f"n={b['n']}", (b["mean_confidence"], b["accuracy"]), xytext=(0, -13), textcoords="offset points", fontsize=8, ha="center")
    ax.set(xlabel="Mean confidence in bin", ylabel="Observed rate", xlim=(.3, 1.02), ylim=(-.08, 1.02), title="N9000: conditional confidence vs observed rates")
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    names = ["student_teacher_not_reference", "teacher_reference_not_student", "all_different"]
    labels = ["Student = teacher\nDifferent reference", "Teacher = reference\nDifferent student", "All three differ"]
    groups = checkpoints["n9000"]["groups"]
    low = [groups[k]["n"] - groups[k]["high95"] for k in names]
    high = [groups[k]["high95"] for k in names]
    ax.bar(labels, low, color="#94a3b8", label="Confidence < 95%")
    ax.bar(labels, high, bottom=low, color="#d97706", label="Confidence >= 95%")
    for i, k in enumerate(names):
        ax.text(i, groups[k]["n"] + 2, str(groups[k]["n"]), ha="center")
    ax.set(ylabel="Reference mismatches", title="N9000: 244 reference mismatches (one invalid teacher separate)")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    thresholds = checkpoints["n9000"]["thresholds"]
    for key, label, color in [("agreement", "Teacher agreement", "#2563eb"), ("accuracy", "Reference accuracy", "#059669")]:
        ax.plot([t["coverage"] * 100 for t in thresholds], [t["eligible"][key] * 100 for t in thresholds], "o-", label=label, color=color)
    for t in thresholds[1:]:
        ax.annotate(f"p>={t['threshold']}", (t["coverage"] * 100, t["eligible"]["agreement"] * 100), xytext=(0, 5), textcoords="offset points", fontsize=8, ha="center")
    ax.axhline(90, color="gray", linestyle="--")
    ax.set(xlabel="Coverage (% of all 1,013 cases)", ylabel="Selected-set rate (%)", title="Selection trades coverage for agreement / accuracy", ylim=(70, 103))
    ax.legend(fontsize=9, loc="lower left")
    fig.suptitle("Gemma 3 270M confidence audit | October validation, n=1,013", fontsize=16)
    fig.savefig(DEST / "confidence_analysis.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
