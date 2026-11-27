"""Offline Confidence-Gated Teacher Fallback (CGTF) on archived validation data."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from evaluation import score
from mortgage_prompt import LABELS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "output/confidence/gemma3-270m-it-v3-lora-curve"
DEFAULT_OUTPUT = ROOT / "result/validation/gemma3-270m-it-v3-cgtf"
EXPORT_THRESHOLDS = (0, .6, .8, .9, .95, .99, .995, 1)
MODES = {"strict": "strict_label", "five_way": "five_way_label"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def route(rows, threshold, mode="strict"):
    """Use only confidence to choose the source; ties stay with the Student."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1]")
    field = MODES[mode]
    result = []
    for row in rows:
        confidence = row["confidence"]
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be finite and in [0, 1]")
        escalated = confidence < threshold
        student = row[field]
        final = row["teacher_label"] if escalated else student
        result.append({**row, "mode": mode, "threshold": threshold,
                       "student_label": student, "escalated_to_teacher": escalated,
                       "selected_source": "teacher" if escalated else "student",
                       "final_label": final, "final_status": "ok" if final else "invalid",
                       "label_changed": final != student})
    return result


def metrics_for(rows, threshold, mode="strict"):
    routed = route(rows, threshold, mode)
    retained = [r for r in routed if not r["escalated_to_teacher"]]
    escalated = [r for r in routed if r["escalated_to_teacher"]]
    result = {"mode": mode, "threshold": threshold, "n_total": len(rows),
              "student_requests": len(rows), "teacher_requests": len(escalated),
              "teacher_request_rate": len(escalated) / len(rows) if rows else None,
              "student_retained": len(retained),
              "student_coverage": len(retained) / len(rows) if rows else None,
              "archived_teacher_attempts_on_escalated": sum(r.get("teacher_attempts", 0) for r in escalated),
              "escalated_teacher_invalid": sum(not r["teacher_label"] for r in escalated),
              "retained_student_invalid": sum(not r["student_label"] for r in retained),
              "label_changed": sum(r["label_changed"] for r in routed),
              "escalated_label_unchanged": sum(not r["label_changed"] for r in escalated),
              "reference_gains": sum(r["student_label"] != r["reference_label"] == r["final_label"] for r in escalated),
              "reference_losses": sum(r["student_label"] == r["reference_label"] != r["final_label"] for r in escalated)}
    for prefix, subset, field in [("hybrid", routed, "final_label"),
                                  ("retained", retained, "student_label"),
                                  ("rejected_student", escalated, "student_label")]:
        scores = score(subset, field, paired=True)
        for key in ("accuracy", "macro_f1", "teacher_agreement_all_cases", "correct", "teacher_matches", "valid_rate"):
            result[f"{prefix}_{key}"] = scores[key]
        if prefix == "retained":
            for label in LABELS:
                result[f"retained_reference_{label}"] = scores["per_class"][label]["support"]
                result[f"retained_predicted_{label}"] = scores["predicted_counts"].get(label, 0)
    require(result["teacher_requests"] + result["student_retained"] == len(rows), "Route count mismatch")
    require(result["hybrid_correct"] == score(rows, MODES[mode])["correct"] + result["reference_gains"] - result["reference_losses"],
            "Reference gains/losses do not reconcile")
    require(result["hybrid_teacher_matches"] == result["retained_teacher_matches"] + len(escalated) - result["escalated_teacher_invalid"],
            "Teacher agreement decomposition does not reconcile")
    return result


def ledger_rows(rows, threshold, mode="strict"):
    fields = ("complaint_id", "split", "input_sha256", "mode", "threshold", "reference_label",
              "student_label", "strict_label", "five_way_label", "confidence", "label_mass", "teacher_label",
              "teacher_attempts", "escalated_to_teacher", "selected_source", "final_label", "final_status", "label_changed")
    return [{key: r[key] for key in fields} for r in route(rows, threshold, mode)]


def load_checkpoint(summary_path, teacher, validation):
    summary = json.loads(summary_path.read_text())
    require(summary["split"] == "validation" and len(summary["checkpoints"]) == 1, "Expected one validation checkpoint")
    checkpoint = summary["checkpoints"][0]
    confidence_path = summary_path.parent / checkpoint["confidence_file"]
    archived_path = ROOT / checkpoint["validation_run"] / "predictions.csv"
    metrics_path = archived_path.with_name("metrics.json")
    config_path = archived_path.with_name("config.json")
    require(sha256(archived_path) == checkpoint["predictions_sha256"], "Archived predictions hash changed")
    rows = read_csv(confidence_path)
    archive = read_csv(archived_path)
    require(len(rows) == len(archive) == len(validation) == 1013, "Expected all 1,013 validation cases")
    require(len({r["complaint_id"] for r in rows}) == len(rows), "Duplicate confidence IDs")
    for row, previous in zip(rows, archive, strict=True):
        ident = row["complaint_id"]
        require(ident == previous["complaint_id"] and ident in validation and ident in teacher, "ID mismatch")
        t = teacher[ident]
        require(previous["split"] == t["split"] == "validation", "Split mismatch")
        require(row["reference_label"] == previous["reference_label"] == t["reference_label"] == validation[ident]["reference_label"],
                "Reference mismatch")
        require(row["teacher_label"] == previous["teacher_label"] == t["teacher_label"], "Teacher mismatch")
        require(previous["input_sha256"] == t["input_sha256"], "Input hash mismatch")
        require(row["strict_label"] == previous["student_label"], "Strict label mismatch")
        require(row["five_way_label"] == row["archived_first_token_choice"] == previous["first_token_choice"], "Five-way label mismatch")
        require(row["reference_label"] in LABELS and row["five_way_label"] in LABELS, "Invalid target/choice")
        require(row["teacher_label"] in (*LABELS, "") and row["strict_label"] in (*LABELS, ""), "Invalid label")
        for key in ("confidence", "label_mass", *(f"p_{c}" for c in LABELS)):
            row[key] = float(row[key])
            require(math.isfinite(row[key]) and 0 <= row[key] <= 1, "Invalid probability")
        require(abs(sum(row[f"p_{c}"] for c in LABELS) - 1) < 1e-5, "Probability sum mismatch")
        require(abs(max(row[f"p_{c}"] for c in LABELS) - row["confidence"]) < 1e-6, "Confidence mismatch")
        require(row["five_way_label"] == max(LABELS, key=lambda c: row[f"p_{c}"]), "Argmax mismatch")
        row.update(split="validation", input_sha256=previous["input_sha256"], teacher_attempts=int(t["attempts"]))
    metrics = json.loads(metrics_path.read_text())
    require(metrics["complete"] and metrics["pairing_status"] == "complete", "Incomplete archive")
    for mode, metric_key in (("strict", "strict_generation"), ("five_way", "first_token_choice")):
        baseline = score(rows, MODES[mode], paired=True)
        for key in ("accuracy", "macro_f1", "teacher_matches", "correct"):
            require(abs(baseline[key] - metrics[metric_key][key]) < 1e-10, "Baseline score mismatch")
    sources = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p)
               for p in (summary_path, confidence_path, archived_path, metrics_path, config_path)}
    return rows, sources


def plot_curves(metrics, output, focus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    selected = [r for r in metrics if r["checkpoint"] == focus and r["mode"] == "strict"]
    diagnostic = [r for r in metrics if r["checkpoint"] == focus and r["mode"] == "five_way"]
    x = [r["threshold"] for r in selected]
    specs = [("accuracy", "Reference accuracy", "#059669"),
             ("teacher_agreement_all_cases", "Teacher agreement", "#2563eb"),
             ("macro_f1", "Macro-F1 (five classes)", "#d97706")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (metric, label, color) in zip(axes[0], specs):
        ax.plot(x, [r[f"hybrid_{metric}"] for r in selected], color=color, label="Hybrid, original strict output")
        ax.plot(x, [r[f"hybrid_{metric}"] for r in diagnostic], color=color, linestyle=":", label="Hybrid, five-way diagnostic")
        ax.axhline(selected[0][f"hybrid_{metric}"], color="#64748b", linestyle="--", label="Student baseline")
        ax.axhline(selected[-1][f"hybrid_{metric}"], color="#991b1b", linestyle="-.", label="Teacher baseline (all routed here)")
        ax.set(title=label, xlabel="Confidence threshold", ylabel="Score", xlim=(0, 1))
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    for metric, label, color in specs:
        ax.plot(x, [r[f"retained_{metric}"] for r in selected], color=color, label=label)
    ax.set(title="Student on retained samples only", xlabel="Confidence threshold", ylabel="Conditional score", xlim=(0, 1))
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(x, [r["teacher_requests"] for r in selected], color="#7c3aed", label="Requests escalated to Teacher")
    ax.plot(x, [r["student_retained"] for r in selected], color="#0f766e", label="Requests retained by Student")
    ax.set(title="Routing volume (N = 1,013)", xlabel="Confidence threshold", ylabel="Requests", xlim=(0, 1))
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    for metric, label, color in specs:
        ax.plot([r["teacher_request_rate"] for r in selected], [r[f"hybrid_{metric}"] for r in selected], color=color, label=label)
    ax.set(title="Hybrid quality vs Teacher request fraction", xlabel="Teacher request fraction", ylabel="Score", xlim=(0, 1))
    ax.legend(fontsize=8)
    fig.suptitle(f"Confidence-Gated Teacher Fallback | Gemma {focus} | October validation", fontsize=16)
    fig.savefig(output / "threshold_curves.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    checkpoints = sorted({r["checkpoint"] for r in metrics})
    for i, checkpoint in enumerate(checkpoints):
        rows = [r for r in metrics if r["checkpoint"] == checkpoint and r["mode"] == "strict"]
        color = plt.cm.viridis(i / max(1, len(checkpoints) - 1))
        for ax, (metric, label, _) in zip(axes, specs):
            ax.plot([r["teacher_request_rate"] for r in rows], [r[f"hybrid_{metric}"] for r in rows],
                    color=color, label=checkpoint)
            ax.set(title=label, xlabel="Teacher request fraction", ylabel="Hybrid score", xlim=(0, 1))
    axes[-1].legend(fontsize=8, ncol=2)
    fig.suptitle("All 12 checkpoints | Pure confidence gate, original strict outputs", fontsize=15)
    fig.savefig(output / "checkpoint_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--focus", default="n9000")
    parser.add_argument("--export-threshold", action="append", type=float,
                        help="Repeat to export additional per-case ledgers and include thresholds in the sweep")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    export_thresholds = sorted(set((*EXPORT_THRESHOLDS, *(args.export_threshold or []))))
    for threshold in export_thresholds:
        require(math.isfinite(threshold) and 0 <= threshold <= 1, "Threshold outside [0, 1]")
    thresholds = sorted(set([i / 100 for i in range(101)] + export_thresholds))
    summaries = sorted(args.source.resolve().glob("*/summary.json"))
    require(bool(summaries), "No confidence checkpoints found")
    teacher_path = ROOT / "output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all/predictions.csv"
    data_path = ROOT / "data/mortgage_complaint.csv"
    teacher = {r["complaint_id"]: r for r in read_csv(teacher_path) if r["split"] == "validation"}
    validation = {r["complaint_id"]: r for r in read_csv(data_path) if r["split"] == "validation"}
    loaded = {}
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in (teacher_path, data_path, Path(__file__).resolve(), ROOT / "code/evaluation.py")}
    for path in summaries:
        rows, hashes = load_checkpoint(path, teacher, validation)
        loaded[path.parent.name] = rows
        sources.update(hashes)
    require(args.focus in loaded, "Focus checkpoint not present")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = []
    baselines = {}
    for checkpoint, rows in loaded.items():
        baselines[checkpoint] = {mode: score(rows, field, paired=True) for mode, field in MODES.items()}
        baselines[checkpoint]["teacher"] = score(rows, "teacher_label", paired=True)
        for mode in MODES:
            previous = None
            for threshold in thresholds:
                row = {"checkpoint": checkpoint, **metrics_for(rows, threshold, mode)}
                if previous is not None:
                    require(row["teacher_requests"] >= previous["teacher_requests"], "Nonmonotonic routing")
                    require(row["hybrid_teacher_matches"] >= previous["hybrid_teacher_matches"], "Nonmonotonic agreement")
                previous = row
                metrics.append(row)
            # Save reviewable per-request records for each named threshold, both output definitions.
            for threshold in export_thresholds:
                name = f"threshold_{format(threshold, '.12g').replace('.', 'p')}"
                write_csv(args.output / checkpoint / mode / f"{name}.csv", ledger_rows(rows, threshold, mode))
        print(f"Verified and scored {checkpoint}: {len(thresholds)} thresholds, two output definitions", flush=True)
    write_csv(args.output / "threshold_metrics.csv", metrics)
    summary = {"method": "Confidence-Gated Teacher Fallback (CGTF)",
               "record_name": "CGTF Validation Ledger", "split": "validation", "n": 1013,
               "primary_mode": "strict", "diagnostic_mode": "five_way", "focus_checkpoint": args.focus,
               "rule": "confidence < threshold -> archived Teacher; otherwise original Student label",
               "invalid_policy": "No extra invalid-output or label-mass gate; invalid selected label remains failure",
               "agreement_reference": "Frozen DeepSeek Teacher; invalid labels never count as matches",
               "thresholds": thresholds, "export_thresholds": export_thresholds,
               "checkpoint_count": len(loaded), "actual_new_teacher_api_calls": 0,
               "cost_note": "Teacher requests are simulated escalated cases; archived attempts include historical retries; no prices assigned",
               "sources_sha256": sources, "baselines": baselines,
               "metrics_file": "threshold_metrics.csv"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if not args.skip_plots:
        # In these archived results no confidence equals 1, so threshold 1 is the all-Teacher endpoint.
        require(all(r["confidence"] < 1 for r in loaded[args.focus]), "Plot endpoint is not all Teacher")
        plot_curves(metrics, args.output, args.focus)
    print(f"CGTF artifacts: {args.output}")


if __name__ == "__main__":
    main()
