# Reassessing Layer Pruning in LLMs: New Insights and Methods

> **arXiv:** [2411.15558](https://arxiv.org/abs/2411.15558) (2024.11.23)
> **作者:** Yao Lu, Hao Cheng et al. (浙江工业大学 / HKUST-GZ)
> **代码:** https://github.com/yaolu-zjut/Navigation-LLM-layer-pruning
> **模型权重:** [Llama-3.1-6.3B-It-Alpaca](https://huggingface.co/YaoLuzjut/Llama-3.1-6.3B-It-Alpaca) / [Llama-3.1-6.3B-It-Dolly](https://huggingface.co/YaoLuzjut/Llama-3.1-6.3B-It-Dolly)

---

## 💡 个人理解 (核心要点)

这是一篇**"反共识"性质的实证基准研究**——作者花了数千 GPU 小时，系统性地推翻了 LLM 层剪枝领域三个被广泛接受的"常识"，得到一个极其简洁的最佳实践：

> **从尾部剪掉 25% 的层 → 只微调 `lm_head` + 剩余最后 3 层**，就能把 Llama-3.1-8B 压成 6.3B，性能超越 ChatGLM2-6B、Vicuna-7B、Qwen1.5-7B、Baichuan2-7B 等同尺寸社区模型，且训练 token 仅为从头训练的 10⁻⁶。

三个被推翻的"常识"：
1. **"复杂的层选择指标更好"** → 错。简单的 reverse-order（直接剪最后几层）在 4 个模型上平均超过次优 PPL 指标 5.30%。
2. **"LoRA 是剪枝后恢复的最佳微调法"** → 错。partial-layer fine-tuning（只训最后几层 + lm_head）显著优于 LoRA/QLoRA，且训练更快。
3. **"迭代剪枝优于一次性剪枝"** → 错。对 LLM 而言迭代剪枝无收益甚至有害（灾难性遗忘），这一点与传统 CNN 剪枝结论相反。

**关键启示**：LLM 层剪枝领域存在"过度设计"倾向，复杂指标（BI、Taylor、Magnitude）的鲁棒性反而不如直接尾部剪除；同时，剪枝改变了模型结构，使得"全模型微调时 LoRA≈partial-layer"这一经验在剪枝后失效——partial-layer 微调能更好适应新的参数分布。这一结论提示：**剪枝后的微调策略应与剪枝本身耦合设计**，而非套用通用 PEFT 经验。

---

## 📄 论文基本信息

**类型:** 实证基准 / Best Practice 研究（ICLR 2025 投稿）
**任务:** LLM 层剪枝（Layer Pruning）—— 直接移除整个 Transformer block，降低深度
**动机:** LLM 部署资源消耗大；层剪枝因 Llama 等模型 block 输入输出维度一致而操作简单；但"最佳实践"不清晰，社区过度追求复杂指标。

**与现有研究的差异：** 现有工作（ShortGPT、Shortened LLaMA、BlockPruner、LaCo 等）致力于提出新的复杂剪枝方法；本文"退一步"，系统比较已有方法，回答三个核心问题。

---

## 🔬 三个核心问题与洞察

### Q1: Layer Selection —— 复杂指标是否必要？

**实验设置:** 4 个模型（Vicuna-7B-v1.5、Qwen1.5-7B、Gemma2-2B-It、Llama-3.1-8B-It），7 个指标（Random、Reverse-order、PPL、Magnitude-l1/l2、Taylor、BI），25% 与 50% 剪枝率，统一用 LoRA 恢复。

**7 个指标定义:**
- **Random:** 随机选层（baseline）
- **Reverse-order:** 直接剪最后几层（最简单）
- **Magnitude-l1/l2:** $I^n = \sum_k \|W_k^n\|_p$，权重范数小 = 不重要
- **Taylor:** $I^n = \sum_k |\frac{\partial \mathcal{L}}{\partial W_k^n} W_k^n|$（省略二阶项）
- **PPL:** 移除单层后 perplexity 变化小的剪除
- **BI (Block Influence):** $\mathrm{BI}_i = 1 - \mathbb{E}\frac{X_i^T X_{i+1}}{\|X_i\|\|X_{i+1}\|}$，衡量层对输入的改变程度

**结果:**
- Reverse-order 在 25% 剪枝率下跨 4 个模型稳定领先，平均超过次优 PPL **5.30%**
- 50% 剪枝率下结论依然成立
- Magnitude 系列表现接近 Random，几乎失效

> **🟢 Insight #1: Reverse-order 简单且万无一失，跨模型、跨剪枝率提供稳定可靠结果。**

### Q2: Fine-Tuning —— LoRA 家族是否最佳？

**比较方法:**
- **LoRA:** $W_0 + \Delta W = W_0 + BA$，rank=8，注入所有层
- **QLoRA:** LoRA + 量化
- **Partial-layer Fine-tuning:** 冻结前面层，只训 `lm_head` 或 `lm_head + 最后 1/2/3 层`

**结果（25% 剪枝率 + reverse-order）:**
- QLoRA 比 LoRA 略差
- **Partial-layer 显著优于 LoRA**，且 `lm_head + last three layers` 最优
  - Llama-3.1-8B-It: LoRA 0.5268 → partial(last 3) **0.5807**（+5.39%）
- **关键反差:** 对原始（未剪枝）Llama-3.1-8B-It，LoRA 与 partial-layer 表现相当（0.6354 vs 0.6337）。说明 LoRA 的优势在剪枝后**消失**。

**作者解释:** 剪枝造成的结构改变和参数减少，使 partial-layer 微调能更有效适应新参数分布，充分释放剪枝潜力。

**训练成本对比（Llama-3.1-8B 剪 8 层，2×A100）:**

| 方法 | 可训参数 | GPU 显存 | 训练时间 (2 epoch) |
|---|---|---|---|
| LoRA | 15.73M | 45.83G | 10440s |
| QLoRA | 15.73M | 14.26G | 17249s |
| lm_head only | 525.34M | 39.82G | 6953s |
| lm_head+last 3 | 1179.68M | 48.02G | 7931s |

- Partial-layer 可训参数多 75×，但显存相当、训练时间更短（因不引入 LoRA 矩阵的前向/反向开销）
- QLoRA 显存低但训练慢、效果差

> **🟢 Insight #2: Partial-layer 微调可作为 LoRA 的替代方案，性能恢复更好且训练时间更短（显存充足时）。**

### Q3: Pruning Strategy —— 迭代是否优于一次性？

- **One-shot:** 评分一次，直接剪到目标比例
- **Iterative:** 评分→剪→微调→合并，循环到达目标

**结果:** 与传统 CNN 剪枝（迭代显著受益）相反，LLM 迭代剪枝**无收益甚至有害**。

**作者解释:** 过多训练导致**灾难性遗忘**（catastrophic forgetting）。表征相似性可视化（Figure 5）显示不同策略产生差异显著的表示。叠加迭代的高计算开销，整体不划算。

> **🟢 Insight #3: 综合性能收益与计算开销，迭代剪枝对 LLM 无益。**

---

## 📊 敏感性分析

### 1. 校准样本数量（针对 BI / Taylor 等数据驱动指标）
- 用 1/5/10/30/50 个样本计算指标
- **发现:** 样本数显著影响剪枝结果与模型复杂度
- **结论:** 评估数据驱动剪枝指标时，**性能稳定性应作为关键准则**（不能只看某一样本数下的最优表现）

### 2. SFT 数据集选择
对 Llama-3.1-8B 剪 8 层后用不同 SFT 数据微调（lm_head + last 3 layers）:

| SFT 数据集 | Avg Acc |
|---|---|
| **Dolly-15k** | **0.5977** |
| Alpaca-cleaned | 0.5807 |
| MMLU (train) | 0.4165 |

- Dolly-15k 最优，Alpaca 次之，MMLU 最差（过拟合特定任务风格，损害泛化）
- **结论:** SFT 数据集对剪枝模型性能影响显著，需进一步探索最适配的数据集

### 3. 剪枝率
- 随剪枝层数增加，所有数据集性能下降并趋于收敛
- MMLU、CMMLU、ARC-c 对层变化**高度敏感**，下降更快
- 约 16 层后模型"损坏"，故论文最大剪枝率设为 16 层（50%）

---

## 🏆 最终模型：Llama-3.1-6.3B-It

**配方:** Llama-3.1-8B-It + reverse-order 剪 8 层 + partial-layer (lm_head + last 3) 微调

**两个版本:**
- **Llama-3.1-6.3B-It-Alpaca**（Alpaca-cleaned，12.74M tokens）
- **Llama-3.1-6.3B-It-Dolly**（Dolly-15k，14.96M tokens）

**对比结果（Avg Acc / 8 个常识推理数据集）:**

| 模型 | 参数 (训练 token) | Avg Acc |
|---|---|---|
| ChatGLM2-6B | 6.24B (1.4T) | 0.3034 |
| Vicuna-7B-v1.5 | 6.74B (370M) | 0.5484 |
| Baichuan2-7B | 7.51B (2.6T) | 0.5599 |
| Qwen1.5-7B | 7.72B (18T) | 0.5973 |
| LLaMA3-8B | 8.03B (15T+) | 0.6093 |
| Gemma2-7B | 8.54B (6T) | 0.6061 |
| Llama-3.1-8B-It (原模型) | 8.03B (15T+) | **0.6299** |
| ShortGPT (BI) | 6.29B (12.74M) | 0.4080 |
| Shortened LLaMA (PPL) | 6.29B (12.74M) | 0.4772 |
| Shortened LLaMA (Taylor) | 6.29B (12.74M) | 0.4796 |
| **Llama-3.1-6.3B-It-Alpaca** | 6.29B (12.74M) | 0.5807 |
| **Llama-3.1-6.3B-It-Dolly** | 6.29B (14.96M) | **0.5977** |

- 两个剪枝模型**超越** ChatGLM2-6B、Vicuna-7B、Baichuan2-7B
- Dolly 版本**超越** Qwen1.5-7B（用 10⁶× 更少训练 token）
- 相比 SOTA 层剪枝方法：比 ShortGPT 好 ~19%，比 Shortened LLaMA 好 10%+
- 在 MMLU 上甚至部分超越 LLaMA3-8B、Gemma2-7B

**资源统计:** 6.29B 参数，368.65G MACs，23984MiB 显存，210.35s 延迟（单 A100，WikiText2 测试集，64 token 句子）

---

## 🧪 实验设置

- **评测:** 8 个常识推理数据集（MMLU、CMMLU、PIQA、HellaSwag、WinoGrande、ARC-e、ARC-c、OpenbookQA）零样本 + WikiText2/PTB 困惑度
- **校准数据:** BookCorpus 随机 10 样本，序列长度 128（计算 Taylor/BI）
- **微调默认:** Alpaca-cleaned + LoRA，2 epoch，batch 64，lr=1e-5，100 warmup steps
- **硬件:** 2× NVIDIA A100 (40G) + 4× NVIDIA RTX A5000 (24G)
- **基准模型:** Vicuna-7B-v1.5、Qwen1.5-7B、Gemma2-2B-It、Llama-3.1-8B-It
- **总规模:** 7 个指标 × 4 个 LLM × 6 个微调方法 × 5 个剪枝策略 × 10 个数据集

---

## 📝 评价

**优点:**
- 实验规模宏大（数千 GPU 小时），结论可信度高
- 三个洞察均"反共识"，对社区有纠偏价值
- 给出极简可复现的 best practice，工程价值高
- 开源模型与代码

**局限:**
- 仅覆盖层剪枝，未涉及宽度剪枝 / 权重剪枝
- SFT 数据集选择问题未深入解决
- partial-layer 微调为何在剪枝后突然变强，理论解释偏弱（仅归因于"适应新参数分布"）

**对未来研究的启示:**
- 剪枝后的微调策略应针对结构改变重新设计，而非照搬通用 PEFT
- 数据驱动剪枝指标需报告跨样本数稳定性
- LLM 剪枝可能不需要 CNN 时代的迭代范式
