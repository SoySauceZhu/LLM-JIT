# Qwen3.5-0.8B：与 SmolLM2、Gemma 的 300 条配对预实验

2026-11-23。同一批 2024 年 1 月 CFPB 房贷投诉开发样本，按类别比例抽样，seed 42。Teacher、两个原装 Student 均使用 v3 共享提示词；本次没有微调，也没有重新调用 Teacher API。

**Qwen 准确率 30.33%（91/300），五类 macro-F1=0.1774，有效输出率 100.00%。** 相比 SmolLM2 v3，准确率变化 +14.33 个百分点，macro-F1 变化 +0.0476。

## 设置与可比性

- 样本数 A=26、B=19、C=172、D=68、E=15。已核对三组预测的 ID、顺序及 CFPB 参考标签，并核对 Qwen 与 Teacher 冻结清单中的正文哈希、截短信息和完整 v3 提示词。
- 正文继续按固定 SmolLM2 tokenizer 保留最多前 2,000 和后 1,000 token；两条长投诉沿用原截短正文。模型分别使用自己的聊天模板和 tokenizer。
- Qwen 模型为 `Qwen/Qwen3.5-0.8B`，revision `2fc06364715b967f1860aea9cf38778875588b17`，加载纯文本骨干，显式 `enable_thinking=False`；MPS GPU、FP32、贪心生成，最多 16 token，无效回复重试一次。
- SmolLM2 对照沿用已完成的 v3 预实验：CPU、FP32、四线程、贪心生成及 16-token 上限。Gemma 对照沿用同批 v3 API 记录。
- 三组自由生成均严格接受去除首尾空白后的单字母 A–E，耗尽重试的失败计错，分母均为 300。五类 macro-F1 等权；首 token 五选一只作独立诊断。

## 总体结果

| 模型 | 有效输出 | 与 CFPB 一致 | 准确率 | macro-F1 | 与 Gemma 一致 |
|---|---:|---:|---:|---:|---:|
| Gemma v3 | 300/300 | 240/300 | 80.00% | 0.6788 | — |
| SmolLM2-135M v3 | 293/300 | 48/300 | 16.00% | 0.1298 | 40/300（13.33%） |
| Qwen3.5-0.8B v3 | 300/300 | 91/300 | 30.33% | 0.1774 | 93/300（31.00%） |

全部预测多数类 C 的准确率为 57.33%。

## 各类别与输出偏向

| CFPB 类别 | 样本数 | Gemma 答对 | SmolLM2 答对 | Qwen 答对 | Qwen 召回率 |
|---|---:|---:|---:|---:|---:|
| A 申请或再融资 | 26 | 15 | 14 | 0 | 0.00% |
| B 签约或交割 | 19 | 5 | 5 | 1 | 5.26% |
| C 贷款账户管理 | 172 | 161 | 21 | 32 | 18.60% |
| D 还款困难或救济 | 68 | 47 | 8 | 56 | 82.35% |
| E 征信及相关服务 | 15 | 12 | 0 | 2 | 13.33% |

Qwen 预测分布：`{"E": 12, "D": 226, "C": 46, "A": 4, "B": 12}`。相对 SmolLM2，274 条预测发生变化，82 条由错变对、39 条由对变错，净变化 +43 条正确。

## 运行记录与限制

正式 MPS 运行记录了 300 次生成，累计尝试耗时约 471.3 秒；不包含模型下载与加载。输入 token（含重试）=327074，输出 token=600。PyTorch 2.10.0，Transformers 5.5.4。
首 token A–E 五选一诊断：准确率 30.33%，macro-F1=0.1774；未替换自由生成预测。

开始运行时发现 Qwen 的模型配置 EOS 为文本结束标记，而 tokenizer 的 EOS 为回合结束标记。初始诊断运行生成的字母后带有回合结束标记和换行，导致严格解析失败。已统一使用两种结束标记作为停止条件，并从新账本重新开始正式实验；没有事后从失败回复中提取字母计分。初始记录保留在 `output/qwen35_dev_n300_s42_v3_eos_diagnostic/`。
修复后另有 CPU 部分运行，完成 12 条后转为完整 MPS 运行，保留在 `output/qwen35_dev_n300_s42_v3/`。CPU/MPS 重叠样本中 11/12 条预测相同；差异见 comparison.json。两个 Student 的运行设备不同，速度不是受控的同硬件比较，分数差异也不能完全归因于模型。

这批样本属于反复使用的开发集，其中 25 条曾参与 v2 提示词开发。本次没有调整 v3 分类指令；结果不能视为未见测试集性能，也不能据此判断微调后的上限。

## 复现

```bash
python3 code/student_baseline.py --student-model qwen3.5-0.8b --device mps --split development --size 300 --seed 42 --prompt-version v3 --teacher-run output/gemma_dev_n300_s42_v3 --output output/qwen35_dev_n300_s42_v3_mps --run
```

需要可用的 Metal GPU；在当前沙箱内不可见。模型已缓存时可以设置 `HF_HUB_OFFLINE=1`。使用 `--device cpu` 时须另选输出目录，不能将不同设备的预测混入同一账本。

[Qwen 逐条回复](predictions.csv) · [指标](metrics.json) · [配置与冻结输入](config.json) · [三模型配对预测](paired_predictions.csv) · [比较数据](comparison.json) · [原始账本](../../../output/qwen35_dev_n300_s42_v3_mps/run.sqlite) · [SmolLM2 v3](../smollm2-135m-few-shot-v3/note.md) · [Gemma v3](../gemma/v3/note.md)
