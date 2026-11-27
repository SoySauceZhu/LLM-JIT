"""Summarize the LR learning curve and replay confidence-gated Teacher fallback."""

import csv
import gzip
import io
import json

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score

from confidence_gated_fallback import EXPORT_THRESHOLDS, metrics_for, require, route, write_csv
from mortgage_prompt import LABELS

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


DISPLAY_THRESHOLDS = (0, .30, .40, .50, .60, .70, .80, .90, .95, .99, .995, 1)
THRESHOLDS = sorted({*DISPLAY_THRESHOLDS, *EXPORT_THRESHOLDS, *(i / 100 for i in range(101))})
BUDGETS = (.10, .25, .50)
QUALITY = ("accuracy", "teacher_agreement_all_cases", "macro_f1")
TITLES = ("CFPB accuracy", "DeepSeek agreement", "CFPB macro-F1")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def confidence_summary(rows):
    probabilities = np.array([[row[f"p_{label}"] for label in LABELS] for row in rows])
    targets = np.array([LABELS.index(row["reference_label"]) for row in rows])
    outcome = probabilities.argmax(axis=1) == targets
    confidence = probabilities.max(axis=1)
    bins = []
    for i in range(10):
        mask = np.minimum((confidence * 10).astype(int), 9) == i
        bins.append({"confidence_min": i / 10, "confidence_max": (i + 1) / 10,
                     "n": int(mask.sum()), "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
                     "accuracy": float(outcome[mask].mean()) if mask.any() else None})
    return {"mean_confidence": float(confidence.mean()),
            "correct_mean_confidence": float(confidence[outcome].mean()) if outcome.any() else None,
            "incorrect_mean_confidence": float(confidence[~outcome].mean()) if (~outcome).any() else None,
            "ece_10_bins": sum(b["n"] / len(rows) * abs(b["mean_confidence"] - b["accuracy"]) for b in bins if b["n"]),
            "multiclass_brier": float(np.square(probabilities - np.eye(len(LABELS))[targets]).sum(axis=1).mean()),
            "confidence_correctness_auroc": float(roc_auc_score(outcome, confidence)) if len(set(outcome)) == 2 else None,
            "bins": bins}


def best(rows, metric):
    return max(rows, key=lambda row: (row[metric], -row["teacher_requests"], -row["threshold"]))


def simulate(output, student_rows, student, teacher):
    gate_metrics, operating_points = [], []
    ledger_fields = ("n_train", "complaint_id", "threshold", "confidence", "reference_label", "student_label",
                     "teacher_label", "teacher_attempts", "escalated_to_teacher", "selected_source", "final_label",
                     "final_status", "label_changed")
    with gzip.GzipFile(output / "gate_predictions.csv.gz", "wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=ledger_fields, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            for n, rows in student_rows.items():
                sweep = [{"n_train": n, **metrics_for(rows, threshold, mode="five_way")} for threshold in THRESHOLDS]
                require(all(sweep[i]["teacher_requests"] <= sweep[i + 1]["teacher_requests"]
                            for i in range(len(sweep) - 1)), f"Nonmonotonic routing at n{n}")
                for endpoint, baseline, calls in ((sweep[0], student[n], 0), (sweep[-1], teacher, len(rows))):
                    require(endpoint["teacher_requests"] == calls and
                            all(endpoint[f"hybrid_{metric}"] == baseline[metric] for metric in (*QUALITY, "valid_rate")),
                            f"Gate endpoint differs from baseline at n{n}")
                gate_metrics.extend(sweep)
                export = set(DISPLAY_THRESHOLDS)
                for budget in (*BUDGETS, 1.0):
                    feasible = [row for row in sweep if row["teacher_request_rate"] <= budget]
                    for objective in ("accuracy", "macro_f1"):
                        chosen = best(feasible, f"hybrid_{objective}")
                        export.add(chosen["threshold"])
                        operating_points.append({"n_train": n, "teacher_budget": budget, "objective": objective,
                                                 **{key: chosen[key] for key in ("threshold", "teacher_requests",
                                                   "teacher_request_rate", "hybrid_accuracy", "hybrid_macro_f1",
                                                   "hybrid_teacher_agreement_all_cases", "reference_gains",
                                                   "reference_losses")}})
                for threshold in sorted(export):
                    writer.writerows(route(rows, threshold, mode="five_way"))
    return gate_metrics, operating_points


def draw_plots(output, student, teacher, gate, focus):
    plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    color = "#116a82"
    teacher_color = "#c5652c"
    sizes = list(student)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), layout="constrained")
    for ax, metric, title in zip(axes, QUALITY, TITLES, strict=True):
        ax.plot(sizes, [student[n][metric] for n in sizes], "o-", color=color, ms=4, label="BGE + LR")
        ax.axhline(teacher[metric], color=teacher_color, linestyle="--", label="DeepSeek Teacher")
        ax.set(title=title, xlabel="History Teacher labels", xticks=[0, 3000, 6000, 9000], ylim=(0, 1.04))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.22)
    axes[0].legend(loc="lower right", frameon=False)
    fig.savefig(output / "learning_curve.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), layout="constrained")
    examples = list(dict.fromkeys((0, 500, focus, sizes[-1])))
    palette = ("#857a98", "#51a288", color, "#314677")
    for ax, metric, title in zip(axes, QUALITY, TITLES, strict=True):
        for n, series_color in zip(examples, palette):
            points = [row for row in gate if row["n_train"] == n]
            ax.plot([p["teacher_request_rate"] for p in points], [p[f"hybrid_{metric}"] for p in points],
                    color=series_color, label=f"n{n}")
        ax.axhline(teacher[metric], color=teacher_color, linestyle="--", label="DeepSeek Teacher")
        ax.set(title=title, xlabel="Teacher request rate", xlim=(0, 1), ylim=(0, 1.04))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.22)
    axes[0].legend(loc="lower right", frameon=False, ncols=2)
    fig.savefig(output / "gate_tradeoff.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 4.4), layout="constrained")
    ax.plot((0, 1), (0, 1), "--", color="#777777", linewidth=1, label="Perfect calibration")
    for n, series_color in zip(examples, palette):
        bins = [b for b in student[n]["confidence"]["bins"] if b["n"]]
        ax.plot([b["mean_confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", ms=4,
                color=series_color, label=f"n{n}")
    ax.set(title="Confidence vs. observed CFPB accuracy", xlabel="Mean Student confidence",
           ylabel="CFPB accuracy in bin", xlim=(0, 1.03), ylim=(0, 1.03))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=.22)
    fig.savefig(output / "confidence_calibration.png")
    plt.close(fig)


def write_results(output, config, student_rows, student, teacher):
    for n, rows in student_rows.items():
        student[n]["confidence"] = confidence_summary(rows)
    validation_metrics = [{"model": "BGE + LR", "n_train": n, "n_validation": scores["n"],
                           **{k: scores[k] for k in (*QUALITY, "valid_rate", "fit_seconds", "fit_iterations")},
                           **{k: v for k, v in scores["confidence"].items() if k != "bins"},
                           **{f"recall_{c}": scores["per_class"][c]["recall"] for c in LABELS},
                           **{f"f1_{c}": scores["per_class"][c]["f1"] for c in LABELS}}
                          for n, scores in student.items()]
    validation_metrics.append(dict.fromkeys(validation_metrics[0]) | {
        "model": config["teacher"], "n_validation": teacher["n"],
        **{k: teacher[k] for k in (*QUALITY, "valid_rate")},
        **{f"recall_{c}": teacher["per_class"][c]["recall"] for c in LABELS},
        **{f"f1_{c}": teacher["per_class"][c]["f1"] for c in LABELS}})
    write_csv(output / "validation_metrics.csv", validation_metrics)
    gate, operating = simulate(output, student_rows, student, teacher)
    write_csv(output / "gate_threshold_metrics.csv", gate)
    write_csv(output / "gate_budget_summary.csv", operating)
    save_json(output / "analysis.json", {"n_validation": teacher["n"], "checkpoints": student, "teacher": teacher,
              "majority_C": sum(r["reference_label"] == "C" for r in student_rows[0]) / teacher["n"],
              "gate": {"rule": "confidence < threshold -> frozen Teacher; equality retains Student",
                       "thresholds": THRESHOLDS, "ledger_thresholds": DISPLAY_THRESHOLDS,
                       "actual_new_teacher_api_calls": 0}})
    focus = max(student, key=lambda n: student[n]["macro_f1"])
    draw_plots(output, student, teacher, gate, focus)
    write_note(output, config, student, teacher, gate, operating, focus)


def write_note(output, config, student, teacher, gate, operating, focus):
    best_acc = max(student, key=lambda n: student[n]["accuracy"])
    best_agree = max(student, key=lambda n: student[n]["teacher_agreement_all_cases"])
    last = max(student)
    best_f1 = student[focus]
    base = student[0]
    gpu = config["embedding"]["gpu"] or config["embedding"]["device"]
    quality_table = ["| 节点 | 准确率 | Teacher 一致率 | Macro-F1 | B 召回 | E 召回 |",
                     "|---:|---:|---:|---:|---:|---:|"]
    for name, scores in [*((f"n{n}", s) for n, s in student.items()), ("DeepSeek Teacher", teacher)]:
        quality_table.append(f"| {name} | {scores['accuracy']:.2%} | {scores['teacher_agreement_all_cases']:.2%} | "
                             f"{scores['macro_f1']:.4f} | {scores['per_class']['B']['recall']:.2%} | "
                             f"{scores['per_class']['E']['recall']:.2%} |")
    gate_table = ["| 目标 | Teacher 预算 | 阈值 | Teacher 请求 | Hybrid 准确率 | 一致率 | Macro-F1 | 改正 / 改错 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in operating:
        if row["n_train"] == focus:
            goal = "准确率" if row["objective"] == "accuracy" else "Macro-F1"
            gate_table.append(f"| {goal} | {row['teacher_budget']:.0%} | {row['threshold']:g} | "
                              f"{row['teacher_requests']} ({row['teacher_request_rate']:.2%}) | "
                              f"{row['hybrid_accuracy']:.2%} | {row['hybrid_teacher_agreement_all_cases']:.2%} | "
                              f"{row['hybrid_macro_f1']:.4f} | {row['reference_gains']} / {row['reference_losses']} |")
    gate_best = best(gate, "hybrid_macro_f1")
    gate_best_acc = best(gate, "hybrid_accuracy")
    confidence = best_f1["confidence"]
    high_confidence = confidence["bins"][-1]
    f1_delta = best_f1["macro_f1"] - teacher["macro_f1"]
    n_last = student[last]
    doc = f"""# BGE Small + LR 的 October validation 实验

在 **{gpu}** 上完成 13 个节点，均使用相同的 **{teacher['n']:,} 条** 2024 年 10 月验证样本。Student 的最高准确率为 **{student[best_acc]['accuracy']:.2%}（n{best_acc}）**，最高 Teacher 一致率为 **{student[best_agree]['teacher_agreement_all_cases']:.2%}（n{best_agree}）**，最高 macro-F1 为 **{best_f1['macro_f1']:.4f}（n{focus}）**。DeepSeek Teacher 的准确率为 **{teacher['accuracy']:.2%}**，macro-F1 为 **{teacher['macro_f1']:.4f}**。

## Embedding 与 LR 设计

- 冻结 `BAAI/bge-small-en-v1.5`，固定 revision `{config['embedding']['revision']}`。它面向英语文本，输出 384 维向量，能以较低的编码开销作为轻量 Student 的基线。用投诉原文编码，将超过 510 个内容 token 的叙述按 510 个连续 token 分块。对每块的 `[CLS]` 向量做 L2 归一化，按块的内容 token 数加权，再做文档级 L2 归一化。编码只用投诉原文，输入不含标签、日期或 v3 few-shot 示例。[BGE 模型卡](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/{config['embedding']['revision']})
- n0 使用 v3 的五段类别描述做零样本原型。类别权重为 `10 × 类别描述单位向量`，五个偏置均为 `0`。通过同一线性层与 softmax 输出标签和置信度。缩放系数在运行前固定。
- n100 至 n9000 各自独立拟合五分类 softmax LR。使用 L2 正则、`C=10`、L-BFGS、可学习截距和等样本权重。训练目标为冻结的 DeepSeek v3 标签。每个节点逐行核对同名 LoRA 训练前缀，reference 标签仅用于评估。验证期只计算预测与指标，11 至 12 月 test 未评分。CUDA 编码用 FP16，CLS 归一化和聚合用 FP32，LR 拟合在 CPU 上完成。
- 编码覆盖 9,000 条 history、1,013 条 validation 和 5 段类别描述，共 {config['embedding']['chunks']:,} 块，耗时 {config['embedding']['seconds']:.1f} 秒。validation 有 {config['validation_multichunk_cases']} 条使用多块，最长 {config['validation_max_content_tokens']:,} 个内容 token。13 份 JSON 可直接用 `softmax(x @ coef.T + intercept)` 恢复概率。

## Student 学习曲线

![BGE + LR 与 Teacher 的准确率、一致率和 macro-F1](learning_curve.png)

准确率与 macro-F1 按 CFPB 消费者自选标签计算。一致率以是否预测同一个合法 DeepSeek 字母为准。三个指标均基于整批验证样本。Teacher 有 {teacher['n'] - teacher['teacher_matches']} 条无效输出，计为错误及不一致。Macro-F1 对 A–E 五类的 F1 等权平均。

{chr(10).join(quality_table)}

- 从零样本到 n{focus}，准确率变化 **{(best_f1['accuracy'] - base['accuracy']) * 100:+.2f} 个百分点**，macro-F1 变化 **{best_f1['macro_f1'] - base['macro_f1']:+.4f}**。n{focus} 的 macro-F1 相对 Teacher 为 **{f1_delta:+.4f}**。n{last} 的准确率为 **{n_last['accuracy']:.2%}**，macro-F1 为 **{n_last['macro_f1']:.4f}**。
- n0 在 {teacher['n']} 条投诉中预测了 {base['predicted_counts'].get('B', 0)} 次 B，而实际 B 类为 {teacher['per_class']['B']['support']} 条。n100 则预测了 {student[100]['predicted_counts'].get('C', 0)} 次 C，B 和 E 的召回分别为 {student[100]['per_class']['B']['recall']:.2%} 和 {student[100]['per_class']['E']['recall']:.2%}。n100 的 Teacher 标签里，C 占 {student[100]['training_label_counts']['C']} 条。
- n{focus} 的 B 类召回为 **{best_f1['per_class']['B']['recall']:.2%}**，E 类召回为 **{best_f1['per_class']['E']['recall']:.2%}**。验证集中 B 有 {teacher['per_class']['B']['support']} 条，E 有 {teacher['per_class']['E']['support']} 条。n{focus} 在 B/E 分别输出 {best_f1['predicted_counts'].get('B', 0)} / {best_f1['predicted_counts'].get('E', 0)} 次。分节点、分标签混淆矩阵与 Teacher 一致率见 `analysis.json`。

## 置信度与 Teacher 回退

置信度为 Student 五类 softmax 概率的最大值。`confidence < τ` 时回退到同一投诉的 DeepSeek 冻结预测，等于阈值时保留 Student。模拟覆盖 0 至 1 每隔 0.01 的阈值以及 0.995。τ=0 完全保留 Student，τ=1 在本实验中全部选择 Teacher。Teacher 无效输出保留为失败。模拟实际新增 Teacher API 调用为 0。

![置信度门控下的 Teacher 请求率与混合分类质量](gate_tradeoff.png)

![置信度分箱与实际正确率](confidence_calibration.png)

n{focus} 的正确与错误样本平均置信度分别为 **{confidence['correct_mean_confidence']:.3f}** 和 **{confidence['incorrect_mean_confidence']:.3f}**。按 10 个等宽分箱计算的 ECE 为 **{confidence['ece_10_bins']:.3f}**，以置信度区分回答正确与否的 AUROC 为 **{confidence['confidence_correctness_auroc']:.3f}**。最高置信度区间 `[0.9, 1]` 的 {high_confidence['n']} 条样本平均置信度为 **{high_confidence['mean_confidence']:.2%}**，实际准确率为 **{high_confidence['accuracy']:.2%}**，说明输出概率偏高。

下表固定 Student 为 n{focus}，分别按准确率和 macro-F1 在各 Teacher 请求预算内挑选阈值。得分相同时优先选更少 Teacher 请求，再选较低阈值。“改正 / 改错”以同一条投诉的 Student 原输出为基准。

{chr(10).join(gate_table)}

- 最高 hybrid macro-F1 为 **{gate_best['hybrid_macro_f1']:.4f}**（n{gate_best['n_train']}，τ={gate_best['threshold']:g}，{gate_best['teacher_requests']} 次 Teacher 请求，占 {gate_best['teacher_request_rate']:.2%}）。该点准确率为 **{gate_best['hybrid_accuracy']:.2%}**，一致率为 **{gate_best['hybrid_teacher_agreement_all_cases']:.2%}**，较纯 Teacher 的 macro-F1 高 **{gate_best['hybrid_macro_f1'] - teacher['macro_f1']:+.4f}**。
- 最高 hybrid 准确率为 **{gate_best_acc['hybrid_accuracy']:.2%}**（n{gate_best_acc['n_train']}，τ={gate_best_acc['threshold']:g}，Teacher 请求率 {gate_best_acc['teacher_request_rate']:.2%}）。比纯 Teacher 多答对 **{gate_best_acc['hybrid_correct'] - teacher['correct']} 条**。

回退到合法 Teacher 输出时直接计为一致。这里的最优点来自同一 October validation 的回顾性阈值搜索，仍需独立数据检验。

## 数据与复现

- [validation 指标](validation_metrics.csv)、[逐条 Student 预测与五类概率](predictions.csv)、[分标签指标和置信度分箱](analysis.json)
- [门控阈值全量汇总](gate_threshold_metrics.csv)、[分节点的预算搜索](gate_budget_summary.csv)、[代表性阈值和预算最优点的逐条路由](gate_predictions.csv.gz)
- [13 个节点参数](models/)、[运行配置、版本和源文件哈希](config.json)
"""
    (output / "note.md").write_text(doc, encoding="utf-8")
