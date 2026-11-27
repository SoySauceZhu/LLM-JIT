# Student Accuracy 与 Teacher Consistency 未达到 90% 的综合分析

本报告综合 October validation 的学习曲线、Gemma 超参数实验、confidence 实验和投诉原文复核，解释 Student 的参考标签准确率（accuracy）与 Teacher 一致率（consistency）为什么仍有差距。

**当前证据支持多种因素共同作用：任务属于类别高度重叠的领域内细粒度分类；消费者参考标签与 v3 主诉判定规则并不完全一致；Student 对少数类和复杂边界的学习仍不充分；Gemma 还存在输出失败。** Confidence 能帮助定位这些问题，但不能单独确认错标或证明模型容量不足。“数据集比较难”可以作为有依据的解释，不能作为排除其他原因的结论。

另一个确定的限制是：**在当前冻结 Teacher、reference 和同一批全量样本上，accuracy 与 consistency 同时达到 90% 在数学上不可行。** 两项指标需要分别解释、分别设定改进目标。

## 1. 分析范围与指标口径

训练使用 1–9 月 history 的 Teacher 标签，October validation 共 1,013 条。Accuracy 统计 Student 与 CFPB reference 的匹配；consistency 统计 Student 与冻结 DeepSeek Teacher 的合法标签匹配。严格自由生成的无效输出按不匹配计，分母保持全部 1,013 条。

| 运行 | Reference accuracy | Teacher consistency | Macro-F1 | 有效输出率 |
|---|---:|---:|---:|---:|
| Gemma 270M，原 N8000 | 74.83% | 83.51% | 0.6393 | 97.14% |
| Gemma 270M，原 N9000 | 74.53% | 84.30% | 0.6274 | 97.14% |
| Gemma N9000，超参组 r8/a16/lr=1e-4 | 75.22% | 84.11% | 0.6495 | 97.14% |
| Gemma N9000，L40 BF16、B/E 加权配置 | 75.42% | 84.50% | 0.6378 | 97.14% |
| Qwen3 0.6B，N7000 | 76.70% | 87.56% | 0.6567 | 100.00% |
| Qwen3 0.6B，N8000 | 77.10% | 88.75% | 0.6512 | 100.00% |
| DeepSeek Teacher | 78.58% | — | 0.6833 | 99.90% |

这些行用于描述已经测到的方案，不构成只改变模型参数量或类别权重的受控对照。Qwen 与原 Gemma 的训练硬件、精度、micro-batch 等条件不同；加权 Gemma 也同时改变了运行配置。

Confidence 结果目前覆盖原 Gemma 学习曲线的 12 个节点，没有 Qwen confidence。下文 confidence 错误拆分以 **Gemma N9000 首 token A–E 五选一**为准：accuracy 为 **769/1,013=75.91%**，consistency 为 **870/1,013=85.88%**。这与上表严格生成的 74.53%、84.30% 是不同口径，不能混用。

本报告讨论的 90% 是进一步的质量目标。原实验预设选择规则是距 Teacher macro-F1 不超过 0.02，即本次 validation 上约 **0.6633**；不能事后把“两个指标均达到 90%”写成原实验已经预注册的门槛。

来源：[实验设计](Experiment%20Design.md)、[Gemma 学习曲线](result/validation/gemma3-270m-it-v3-lora-curve/note.md)、[超参实验](result/validation/gemma3-270m-it-v3-n9000-lora-sweep/note.md)、[Qwen 学习曲线](result/validation/qwen3-0.6b-v3-lora-curve/note.md)、[加权 Gemma 原始指标](output/lora_eval/gemma3-270m-it_v3_n9000_bf16_mixed_be1.5_torch28_mb4_validation/metrics.json)。

## 2. 两个目标本身存在冲突：模仿 Teacher 不等于匹配 reference

设 Student 输出为 S，Teacher 输出为 T，reference 为 Y。若某条样本 T≠Y，S 不可能同时与 T、Y 一致。因此对相同样本、同一 Student 输出和当前失败计分规则，有：

`Accuracy(S,Y) + Consistency(S,T) ≤ 1 + Accuracy(T,Y)`。

Teacher 对 reference 匹配了 796/1,013 条，所以右侧为 **178.58%**。由此得到：

- **两项同时 ≥90% 不可行。** 它们的和至少需要 180%，已经超出上界。
- 若 consistency 至少为 90%，accuracy 的上界为 **88.58%**；若 accuracy 至少为 90%，consistency 的上界同样为 88.58%。这是必要约束，不保证上界可以达到。
- 完全复制 Teacher 的标签与失败，reference accuracy 会是 78.58%。但 **78.58% 不是任意 Student 的 accuracy 上限**，Student 可以在 Teacher 错误或分歧的样本上匹配 reference，只是这种改进会降低 consistency。
- Teacher 自己的低 accuracy 不能解释 Student 为什么无法达到 90% consistency。即使 Teacher 相对 reference 有很多分歧，Student 仍可能复现其行为；当前 Teacher 的 1 条失败只使合法匹配率上限为 1,012/1,013=99.90%。

所以，“更好地模仿 Teacher”与“更接近 reference”并非完全相同的优化方向。修改 reference 本身也不会改变 S 与 T 的一致率。

## 3. 逐条拆分：哪些差距来自 Teacher/reference，哪些是额外蒸馏损失

下表使用与 confidence 对应的 Gemma N9000 五选一标签。

| 样本关系 | 条数 | 平均 confidence | Confidence ≥95% 的条数 |
|---|---:|---:|---:|
| S=T=Y | 724 | 93.86% | 507 |
| S=T≠Y：师生共同偏离 reference | 146 | 84.77% | 59 |
| T=Y≠S：Teacher/reference 一致，Student 不同 | 72 | 74.03% | 12 |
| S=Y≠T：Student/reference 一致，Teacher 不同 | 45 | 75.50% | 11 |
| 三者为不同的合法类别 | 25 | 68.22% | 1 |
| Teacher 无效，Student 也未匹配 reference | 1 | 85.79% | 0 |

**Accuracy：244 条未匹配 reference，其中 146 条已经匹配 Teacher，占 59.84%。** 这部分不能通过让 Student 更忠实地模仿同一个 Teacher 来改善，需要复核标签和任务规则。其余 98 条包含 Student 的额外分歧，也可能同时涉及 reference 或 Teacher 的问题。

**Consistency：143 条未匹配 Teacher，其中 45 条反而匹配 reference。** 因此，不能把全部 Teacher 不匹配都理解为 Student 的语义分类错误。72 条 Teacher/reference 一致而 Student 不同，是优先研究 Student 学习不足的样本，但两个标签一致也不等于经过独立裁决的真值。

从 Teacher accuracy 出发，Student 在 Teacher 匹配 reference 的样本中丢掉 72 条，在 Teacher 未匹配 reference 的样本中追回 45 条：

`796 − 72 + 45 = 769`。

这是五选一 Student 与 Teacher 的 accuracy 相差 **2.67 个百分点**的直接计数解释。错误集合可以精确拆分，但不能把这些集合直接等同为“错标率”“模型能力错误率”等因果比例。

来源：[Confidence 报告](result/validation/gemma3-270m-it-v3-confidence/note.md)、[派生统计与原始文件哈希](result/validation/gemma3-270m-it-v3-confidence/analysis.json)。

## 4. 各种原因及现有证据

### 4.1 领域内细粒度分类与多诉求叙事，提高了决策难度

所有样本都已属于房贷领域，模型还需要区分 A 申请/再融资、B 交割、C 账户管理、D 还款困难/救济、E 征信。一条真实投诉可以同时描述“交割出错 → 原贷款未结清 → 被记逾期 → 征信受损”。各阶段共享付款、费用、利率、逾期等词汇，单纯识别话题或关键词不一定能确定类别。

冻结 v3 要求识别消费者最希望解决的问题，而不是看到某个词就归类。模型必须区分业务阶段、起因、后果和最终诉求；有时原文本身也包含多个同等重要的诉求，单标签要求会放大分歧。

直接证据是：Gemma N9000 的 Teacher C→Student D 有 28 条，D→C 有 33 条，合计 **61/143=42.66%** 的 Teacher 不匹配。Qwen N8000 仍有 53 条 C/D 双向分歧，占其 114 条全量不匹配的 **46.49%**（114 条中包含 1 条 Teacher 失败）。相似边界在两个 Student 中都出现，说明问题并不限于某一个 Gemma checkpoint。

这支持“类别重叠和多诉求判断是重要困难来源”。但本实验没有在相同模型、训练规模、类别分布和评估协议下比较粗粒度新闻分类，因此不能声称已经证明该数据集比某个新闻数据集难，也不能推断关键词方法在这里一定无效。需要基线或跨任务对照才能检验这些说法。

### 4.2 Reference 来源与 v3 分类口径不完全一致，主要影响 accuracy

根据已有数据审计，reference 来自消费者选择的 CFPB 类别；v3 则要求仅根据公开正文判断主要诉求。二者可能因为消费者选择、公开文本信息缺失、多诉求或规则边界不同而产生分歧。

已有 [数据集误差分析](Dataset_Error_Analysis.md) 特别指出 B/C 的口径差异：v3 强调完成交割交易时的问题，又把现有贷款费用、托管和结清归 C；原始分类中的一些交割后费用等问题仍属于 B。两套规则对“交割时产生、交割后才请求修正”的投诉可能给出不同标签。该审计还发现原文、ID 和类别映射与原始档案一致，因此目前没有证据把主要差距归因于简单的标签映射或行关联错误。

原文复核中，10413024 的 reference 是 D，但消费者明确表示可以正常还款，争议是重复收取保险费用；师生均选 C，Student confidence 为 99.96%。这是 reference 复核的有力候选。另一方面，10541399 同时包含还款未入账和要求确认不向征信机构报告的问题，师生高置信选 C，reference E 仍有文本依据，不能简单认定错标。

**可以确认存在值得复核的 reference 分歧，但没有独立标注金标准，不能给出全体验证集的真实错标率。** 单纯更改 reference 会改变 accuracy，不会直接提高 consistency。

### 4.3 Teacher 不是语义金标准，Student 会继承其决策偏差

Teacher 对 reference 的匹配率为 78.58%，其中既可能有 reference/口径问题，也有 Teacher 自己的判断错误。已有正文审计中的 10418366，核心是买家承接 VA 贷款的交易手续拖延，reference B 有直接依据，但 Teacher 选 C。不能把 Teacher 与 reference 的全部 217 条不匹配都说成 reference 标错。

Student 学习的是 Teacher 接受的 A–E 标签，而不是 reference。它可能成功复现 Teacher 的系统性偏差：N9000 中 146 条师生共同偏离 reference 就体现了这种风险，但这些样本并不全是已确认的 Teacher 错误。

当前监督目标只有答案字母和原生结束符，没有 Teacher 的类别概率分布或边界解释。它提供了“这条选哪类”，没有显式提供“为何不选相邻类”的信息。这是值得检验的训练因素；现有数据并不能证明改用软标签、解释或对比样本必然提升结果。

来源：[训练实现](code/student_lora.py)、[v3 分类规则](code/prompts/mortgage_v3.toml)、[Teacher 错例审计](Dataset_Error_Analysis.md)。

### 4.4 类别不平衡与边界覆盖不足，可能限制少数类学习

| Teacher 类别 | N9000 训练标签数 | 训练占比 | Validation Teacher 样本数 | Gemma N9000 五选一类内一致率 |
|---|---:|---:|---:|---:|
| A | 720 | 8.00% | 85 | 76.47% |
| B | 246 | 2.73% | 43 | 48.84% |
| C | 5,387 | 59.86% | 598 | 92.81% |
| D | 2,187 | 24.30% | 232 | 85.34% |
| E | 460 | 5.11% | 54 | 57.41% |

训练 Teacher 标签中 C+D 合计 **84.16%**，B 只有 246 条。验证时，Teacher B 的 22 条不匹配全部去了 C（14）或 A（8）；Teacher E 的 23 条不匹配中，14 条去了 C。模型在这些边界上有把样本归入 C 的倾向。

这是“少数类监督较少、边界覆盖可能不足”的证据，但频率与困难程度同时变化，不能只凭表格认定类别不平衡造成了全部差距。B/E 加权运行的严格一致率为 84.50%，相比原 N9000 的 84.30% 改变不大；该运行同时改变了硬件、精度和 micro-batch，因此也不能据此单独确认加权有效或无效。

Validation reference 中 C 占 55.48%，固定预测 C 就有 55.48% accuracy，却不能完成五类任务。解释总体 accuracy 时需要同时看各类表现；macro-F1 对五类等权，也解释了为什么 accuracy 小幅变化时 macro-F1 可能明显波动。

### 4.5 模型与训练方案确实影响结果，但尚不能归因于参数量

相同 N8000 history 前缀下，Qwen 的严格 consistency 为 **88.75%**，原 Gemma 为 **83.51%**，多匹配 Teacher 53 条。其中 23 条来自 Gemma 的空回复样本，另 30 条净改善发生在双方均有合法输出的样本上。这说明已测试的另一套 Student 方案可以改善输出和分类，性能不是完全被数据集固定住的。

但 Qwen 与 Gemma 同时有模型架构、预训练、分词器及训练运行设置的差别，不能把这 5.23 个百分点全归因为“0.6B 比 270M 大”。Qwen 仍在 B/C、C/D 上分歧，说明更换 Student 也没有消除任务边界问题。

优化配置同样有实测影响。固定 Gemma N9000、rank 8、alpha 16，学习率从 2e-5 到 1e-4，严格 accuracy 从 **71.08% 到 75.22%**、consistency 从 **79.86% 到 84.11%**。原 2e-4 的训练 loss 最低、一致率为 84.30%，但 reference accuracy 和 macro-F1 并非最好。训练 loss 更低不等于两个验证目标都更好。

当前只试了有限的 LoRA 配置、固定训练轮数和单个训练种子；没有足够证据断言已经达到模型能力上限。另一方面，已有超参和规模实验没有把两个指标推到 90%，也没有依据承诺单纯扩大 rank 或继续增加同分布样本就能达到目标。

来源：[Qwen 配对比较](result/validation/qwen3-0.6b-v3-lora-curve/note.md)、[Gemma 超参结果](result/validation/gemma3-270m-it-v3-n9000-lora-sweep/note.md)。

### 4.6 输出失败额外拉低 Gemma 得分，但不足以解释全部差距

原 Gemma 学习曲线和超参组都在同样的 29 条样本上输出失败。N9000 改为首 token 五选一后，accuracy 从 74.53% 到 75.91%，consistency 从 84.30% 到 85.88%，分别恢复 14 和 16 条匹配，距离 90% 仍有明显差距。

即使用冻结 Teacher 回填这 29 条、其他预测不变，也只能恢复 28 条有效一致（另一条 Teacher 也失败），consistency 为 **882/1,013=87.07%**。假设 29 条全改成 reference，accuracy 为 **784/1,013=77.39%**。这两种假设分别衡量两个目标，不是同一个可同时实现的修复结果。

Qwen 所有学习曲线节点输出均有效，而最佳 consistency 仍为 88.75%，进一步表明分类分歧独立于格式失败存在。29 条失败的底层成因仍需解码与终止符诊断，不能仅据 confidence 把它们认定为纯模型容量问题。

### 4.7 Confidence 未校准，会让错误显得“非常确定”

当前 confidence 是 A–E 首 token logits 内部 softmax 的最大值。它是模型在这五个候选中的相对确定程度，未经过正确率校准。

N9000 confidence ≥95% 的 590 条样本，平均 confidence 为 **98.80%**，但 reference accuracy 只有 **87.80%**，Teacher consistency 为 **95.93%**。该组 72 条 reference 错例中，59 条师生一致、12 条 Teacher/reference 一致而 Student 不同、1 条三方不同。因此，“高置信错误就说明 reference 标错”不成立。

另一方面，143 条 Teacher 不匹配中，89 条（62.24%）的 confidence <80%，低置信确实能定位大量分歧；但其中有 26 条 Student 匹配 reference，所以“低置信分歧就说明 Student 判断错误”也不成立。

29 条空回复还有一个特殊问题：A–E 全词表总概率仅 **5.37×10⁻⁷ 至 7.39×10⁻⁶**，在五类内归一化后却能得到 **80.55%–91.38%** confidence。判断能否可靠输出，应同时检查实际输出有效性和 `label_mass`。

**校准不足主要影响 confidence 的解释和分流可靠性，不能当成独立解释全部 argmax 错误的原因。** 对概率做常规温度校准不会自动改变最大类别，也不会凭空提高全量 accuracy 或 consistency。

### 4.8 信息缺失与时间变化是待分离因素，不能过度归因

投诉 10611493 的正文只有“See attached. This is in response to prior closed case # XXXX”，模型没有附件和此前案件内容；Teacher 最终无效。它表明部分记录提供的信息确实不足，但尚未系统统计这类样本的比例。

History 与 validation 按月份划分，因此类别分布、投诉内容随时间变化可能增加泛化难度。训练前缀扩大时，样本组成和优化步数同时变化，当前学习曲线不能单独测出时间漂移的影响。需要固定样本量、训练步数并控制时间组成的实验。

正文截断只发生在 validation 的 **2/1,013 条（0.20%）**。即使完全纠正这两条，对总体比例的直接改善最多也是 0.20 个百分点，不能解释十几个百分点的差距。已有审计核对了训练/验证 ID 无重叠、预测配对与输入哈希；目前没有证据把主要原因归为数据管线错配。

## 5. 三个代表性边界案例

以下均为 Gemma N9000，与原始归档预测对应。解释基于正文与 v3 规则，不代表已经获得独立人工金标准，也不能反推模型内部实际使用了哪些线索。

| 投诉 ID | Reference / Teacher / Student | Confidence | 原文要点与边界解释 |
|---|---|---:|---|
| **10375556** | B / B / C | **98.42%** | 再融资交割文件更正未传达，造成放款延迟、旧贷款未结清，后续出现费用和逾期。Teacher B 更符合交易未完成的核心问题；Student C 可能被后续账务问题带偏。是高置信 Student 错判的有力例子。 |
| **10430047** | D / D / C | **99.48%** | 消费者按 reinstatement 报价付款，部分资金来自房贷救济项目，但服务商以入账时间为由要求追加款项，否则退款。D 对应恢复贷款状态的安排，C 对应付款入账争议；两者有依据，高 confidence 没有反映这种歧义。 |
| **10611716** | B / E / C | **97.44%** | 房贷已还清，仍被上报逾期，消费者明确只想删除征信逾期记录。按最终诉求 E 更合理；reference B 和 Student C 都不贴近主诉。这里 reference 与 Student 的问题同时存在。 |

另一个对照是 10373136：reference/Teacher 均为 B，Student=C、confidence **42.78%**，正文明确描述 VA 贷款承接手续和交割延期。它显示边界问题也会以低置信形式出现，而不只是模型过度自信。

原文和预测见 [逐条复核表](result/validation/gemma3-270m-it-v3-confidence/review_cases.csv)。

## 6. 能确认什么，仍需验证什么

| 解释 | 证据支持范围 | 不能据此声称 |
|---|---|---|
| Teacher/reference 目标冲突 | 固定标签下两项之和 ≤178.58%，严格成立 | Student accuracy 永远不可能高于 Teacher |
| 细粒度边界和多诉求困难 | 混淆集中于相邻类，有多个原文案例，两个 Student 均出现 | 数据集难度解释了全部差距；已证明比新闻数据集更难 |
| Reference 与规则差异 | 有合理分歧、疑似错标和规则不一致案例 | 146 条师生共同分歧全部是 reference 错标 |
| 少数类学习不足 | B/E 类内一致率低、训练支持少、错误去向集中 | 类别不平衡是唯一原因；加权一定解决问题 |
| 模型与优化方案影响 | Qwen 与 Gemma、不同学习率有实测差异 | 差异只由参数量决定；已经测出模型能力上限 |
| 输出失败 | 29 条失败及恢复匹配数可直接计数 | 修复格式就能使两个全量指标超过 90% |
| Confidence 校准不足 | 高 confidence 与实测正确率有明显差距 | 高置信必然正确，或校准会自动提高分类准确率 |
| 时间漂移、输入缺失 | 时间切分与缺失信息案例客观存在 | 已测出它们对全量误差的贡献比例 |

这些因素彼此重叠。例如，B/C 的规则歧义可能同时使 Teacher/reference 分歧、减少 B 训练标签、增加 Student 的分类困难。不能把各类案例数量相加，宣称得到了互斥的原因占比。

所有主要比较来自同一 October validation，多个 checkpoint 的最高值也是在这批数据上选出的；训练主要只有 seed 42。这些限制影响对未来表现和方案差异的置信程度，本身不是解释某条错例的原因。后续仍应冻结选择与规则后评价保留测试集；若旧 test 已被用于新一轮选择，应另留独立评估数据。

## 7. 改进应按两个目标分别推进

**如果目标是低成本复现 Teacher，应优先提高 consistency。** 先修复输出失败，再针对 history 中 B/C、A/B、C/D、C/E 边界补充覆盖、重采样或设计受控训练对照。保持 validation 与训练样本隔离，并比较类内一致率，防止只把 C 的误判转移到其他类。模型对照需尽量固定训练协议，并重复训练种子。

**如果目标是正确投诉分类，应先明确 reference 的裁决规则。** 对高置信师生共同分歧、Student 单独分歧、低置信但匹配 reference，以及全一致对照分层盲审，区分错标、歧义、信息不足和模型错误。保留原始 reference 与裁决标签两套结果，不能通过自动改标签制造能力提升。规则若改变，Teacher 标签与 Student 训练目标也需要一致更新。

**如果目标是用 confidence 控制服务质量，可以单独研究选择性接管。** 现有 Gemma N9000 中，要求严格输出有效、`label_mass≥0.5`、confidence≥80%，可接管 **783/1,013=77.30%** 的样本，接管部分 consistency 为 **94.76%**、accuracy 为 **84.04%**。其余回退冻结 Teacher 的离线模拟，全量 consistency 为 **95.85%**，accuracy 为 **78.38%**。这是组合系统的表现，并不是 Student 全量能力超过 90%；Teacher 回退的一致性按构造获得，实际成本与新调用波动尚未计入。

当前实验设计把 confidence 路由列为范围之外，以上是后续实验方向，不应混入本轮独立 Student 的正式成绩。阈值需要在独立校准数据上选择，并报告覆盖率、回退率、成本和保留测试集结果。

**如果要证明“数据集难度”是性能差距的重要来源，应增加对照。** 可以在当前数据上测试词级/字符级 TF-IDF 加线性分类器，并分别用相同 Teacher 标签和评估口径比较；另选类别区分更明显的任务时，控制训练规模、类别分布与指标。这样才能测量简单词汇线索是否足够，以及当前任务相对更细粒度的决策要求，而不是仅用主题名称推断难度。

## 8. 可直接用于实验报告的总结

> 本实验中的性能限制来自任务定义、标签口径与 Student 学习能力的共同作用。房贷投诉属于同一领域内的细粒度分类，各类别共享大量词汇，单条投诉也可能同时涉及交割、账户管理、还款救济和征信问题。模型需要识别消费者的主要诉求，而非仅判断文本主题。Gemma N9000 的首 token 五选一结果中，C/D 边界分歧占全部 Student–Teacher 不匹配的 42.66%，支持类别重叠和多诉求歧义是重要困难来源。同时，Student 的 244 条 reference 错例中有 146 条已经与 Teacher 一致，表明准确率差距还涉及 Teacher 与 reference 的不一致，不能全部归为 Student 未学会任务。B/E 类内一致率较低、不同训练方案的表现差异以及 Gemma 的输出失败，也说明模型与训练过程仍有改进空间。由于缺少独立裁决标签、跨任务受控比较和完整的因素消融，本研究不把全部差距归因于数据集难度，也不声称已经达到模型能力上限。进一步地，在当前 Teacher 对 reference 的匹配率为 78.58% 时，同一全量样本上 Student accuracy 与 Teacher consistency 之和至多为 178.58%，因此两项同时达到 90% 与冻结标签的关系本身不相容。后续应分别针对 Teacher 行为复现和经裁决的分类正确性设定目标并评价改进。

## 依据与复现

- [原始实验设计](Experiment%20Design.md)：监督来源、时间切分、指标和预设选择规则。
- [数据集与 Teacher 误差分析](Dataset_Error_Analysis.md)：参考标签来源、规则差异、原文和映射审计。
- [Gemma 学习曲线](result/validation/gemma3-270m-it-v3-lora-curve/note.md)、[Gemma 超参实验](result/validation/gemma3-270m-it-v3-n9000-lora-sweep/note.md)、[Qwen 学习曲线](result/validation/qwen3-0.6b-v3-lora-curve/note.md)：跨规模和方案证据。
- [Confidence 完整分析](result/validation/gemma3-270m-it-v3-confidence/note.md)、[派生统计](result/validation/gemma3-270m-it-v3-confidence/analysis.json)、[复核样本](result/validation/gemma3-270m-it-v3-confidence/review_cases.csv)：概率、分组及案例证据。
- [Confidence 分析脚本](code/analyze_student_confidence.py)：运行 `MPLCONFIGDIR=/tmp/llm-jit-matplotlib python3 code/analyze_student_confidence.py` 可复算 12 个节点的统计与图。综合报告中的训练方案比较另取自各运行的 `metrics.json` 和上述实验报告。

本报告基于已归档实验和本地原文审计进行综合解释，没有重新训练模型、调用 Teacher、修改 reference 或改变实验选择规则。
