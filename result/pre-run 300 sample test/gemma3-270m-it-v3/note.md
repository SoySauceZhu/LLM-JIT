# Gemma 3 270M IT：300 条配对预实验

2026-11-23。同一批 2024 年 1 月 CFPB 房贷投诉开发样本，按类别比例抽样，seed 42；所有模型均使用 v3 共享提示词。本次评估原装、未经微调的 Gemma 3 270M IT，复用已有 Teacher、SmolLM2 和 Qwen 结果。

**准确率 29.00%（87/300），五类 macro-F1=0.1631，有效输出率 86.67%。** 与 SmolLM2 v3 相比，准确率变化 +13.00 个百分点；与 Qwen v3 相比变化 -1.33 个百分点。

## 设置与可比性

- 模型：`google/gemma-3-270m-it`，revision `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`。模型文件通过用户已授权的 Hugging Face 账号下载，本地推理。
- A=26、B=19、C=172、D=68、E=15。已核对四组预测的 ID、顺序和 CFPB 参考标签，并核对 Gemma 3、Qwen 与 Teacher 清单中的正文哈希、截短信息及完整提示词。
- 正文沿用固定 SmolLM2 tokenizer 的截短规则：超过 3,000 token 时保留前 2,000 和后 1,000；两条长投诉与旧实验保持相同正文。
- 使用 Gemma 3 原生 tokenizer 和聊天模板。模板将系统指令合并到第一条 user 消息，并将 assistant 角色渲染为 model；共享指令、五示例和实际投诉内容保持不变。
- MPS GPU、FP32、四线程、贪心生成、16-token 上限、无效回复重试一次。停止条件包括 tokenizer/model 的 EOS 和原生 `<end_of_turn>`；只移除终止 token，随后严格解析单字母 A–E。
- 最终失败计错，分母均为 300；macro-F1 对五类等权。首 token 五选一仅作诊断，不替换失败回复。

## 总体结果

| 模型 | 有效输出 | 与 CFPB 一致 | 准确率 | macro-F1 | 与 Teacher 一致 |
|---|---:|---:|---:|---:|---:|
| Gemma 4 31B Teacher | 300/300 | 240/300 | 80.00% | 0.6788 | — |
| SmolLM2-135M | 293/300 | 48/300 | 16.00% | 0.1298 | 40/300（13.33%） |
| Qwen3.5-0.8B | 300/300 | 91/300 | 30.33% | 0.1774 | 93/300（31.00%） |
| Gemma 3 270M IT | 260/300 | 87/300 | 29.00% | 0.1631 | 93/300（31.00%） |

全部预测多数类 C 的准确率为 57.33%。

## 类别表现

| CFPB 类别 | 样本数 | Teacher 答对 | SmolLM2 答对 | Qwen 答对 | Gemma 3 答对 | Gemma 3 召回率 |
|---|---:|---:|---:|---:|---:|---:|
| A 申请或再融资 | 26 | 15 | 14 | 0 | 4 | 15.38% |
| B 签约或交割 | 19 | 5 | 5 | 1 | 0 | 0.00% |
| C 贷款账户管理 | 172 | 161 | 21 | 32 | 58 | 33.72% |
| D 还款困难或救济 | 68 | 47 | 8 | 56 | 25 | 36.76% |
| E 征信及相关服务 | 15 | 12 | 0 | 2 | 0 | 0.00% |

Gemma 3 预测分布：`{"D": 121, "A": 36, "C": 103, "FAIL": 40}`。
相对 SmolLM2，233 条预测发生变化，其中 64 条由错变对、25 条由对变错，净变化 +39 条正确。
相对 Qwen，196 条预测发生变化，其中 55 条由错变对、59 条由对变错，净变化 -4 条正确。

## 格式失败与诊断

40 条最终回复无效。无效回复分布：`{"": 40}`。逐次原始回复保存在 SQLite 中，CSV 保留首次和末次回复。
首 token A–E 五选一诊断：准确率 30.67%，macro-F1=0.1726；与自由生成分开报告。

## 环境、耗时与限制

累计 340 次生成，推理尝试耗时约 121.2 秒，不包含下载和加载；输入 token（含重试）=380394，输出 token=605。PyTorch 2.10.0，Transformers 5.5.4，huggingface-hub 1.11.0。
启动时发现全局 huggingface-hub 2.4.0 与 Transformers 5.5.4 的版本约束不兼容。该次在模型加载前停止，0 次推理，记录保存在 `output/gemma3_270m_dev_n300_s42_v3_mps_env_diagnostic/`。随后创建项目 `.venv`（复用系统包），在其中安装兼容的 huggingface-hub 1.11.0，从新账本完成正式实验；全局包保持原样。
Gemma 3 与 Qwen 均为 MPS/FP32，SmolLM2 对照为 CPU/FP32，Teacher 为托管 API。设备差异会影响速度，也可能影响预测，不能把分数差异完全归因于模型。
这是反复使用的开发集，其中 25 条曾参与 v2 提示词开发。本次未调整 v3 分类指令，结果不代表未见测试集性能，也不说明微调后的表现。

## 复现

```bash
source .venv/bin/activate
python3 code/student_baseline.py --student-model gemma3-270m-it --device mps --split development --size 300 --seed 42 --prompt-version v3 --teacher-run output/gemma_dev_n300_s42_v3 --output output/gemma3_270m_dev_n300_s42_v3_mps --run
```

需要有该模型访问权限的 Hugging Face 账号，以及可访问 Metal GPU 的运行环境。已缓存模型时可设置 `HF_HUB_OFFLINE=1`。

[逐条回复](predictions.csv) · [完整指标](metrics.json) · [冻结配置](config.json) · [四模型配对预测](paired_predictions.csv) · [比较数据](comparison.json) · [原始账本](../../../output/gemma3_270m_dev_n300_s42_v3_mps/run.sqlite) · [Qwen 报告](../qwen3.5-0.8b-v3/note.md)
