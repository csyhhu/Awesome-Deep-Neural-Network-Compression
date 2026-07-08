# GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

- **原文链接**: [https://arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)
- **作者**: Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, Omar Khattab
- **机构**: UC Berkeley, Stanford, Notre Dame, Databricks, MIT
- **会议**: ICLR 2026 投稿
- **关键词**: 提示优化 (Prompt Optimization)、复合 AI 系统、进化算法、Pareto 优化、反思学习 (Reflective Learning)、样本效率

---

## 1. 概述

本文提出 **GEPA（Genetic-Pareto）**，一种面向复合 AI 系统的高样本效率提示优化器。GEPA 的核心理念是：**自然语言的反思性学习可以比强化学习（GRPO）更高效地优化 AI 系统**。

作者观察到，现代 LLM 通过 GRPO 等强化学习方法进行优化通常需要成千上万次 rollout，样本效率极低。而 LLM 在 rollout 过程中产生的自然语言轨迹（推理链、工具调用、编译器错误信息等）本身就蕴含丰富的诊断信号。GEPA 通过让 LLM **反思**这些自然语言轨迹来发现问题和更新提示，结合 **遗传进化** 和 **Pareto 前沿选择** 策略，实现了极高的样本效率。

**核心结论**：在 Qwen3 8B 上，GEPA 以不到 GRPO 15% 的 rollout 预算（GRPO 为 24,000 次），实现了 +12.1% 的聚合性能提升（GRPO 为 +5.7%）。在 GPT-4.1 Mini 上，GEPA 提升 +14.2%，超过 MIPROv2 的 +7.01%。

---

## 2. 问题定义

### 2.1 复合 AI 系统

一个复合 AI 系统被形式化定义为 $\Phi = (M, C, \mathcal{X}, \mathcal{Y})$：

- **$M = \langle M_1, \ldots, M_{|M|} \rangle$**：语言模块集合，每个模块 $M_i = (\pi_i, \theta_i, \mathcal{X}_i, \mathcal{Y}_i)$
  - $\pi_i$：提示（包括指令和少样本示例）
  - $\theta_i$：底层 LLM 权重
  - $\mathcal{X}_i, \mathcal{Y}_i$：输入/输出模式
- **$C$**：控制流（代码），决定模块的调用顺序和条件逻辑
- **$\mathcal{X}, \mathcal{Y}$**：全局输入/输出模式，$\mathcal{Y}$ 中包含执行轨迹 `trace`

### 2.2 样本高效优化目标

给定训练集 $\mathcal{D}_\text{train}$、评估指标 $\mu$ 和 rollout 预算 $B$，优化目标为：

$$\langle \Pi^*, \Theta^* \rangle_\Phi = \arg\max_{\langle \Pi, \Theta \rangle_\Phi} \mathbb{E}_{(x, m) \sim \mathcal{T}} \left[ \mu\big( \Phi(x; \langle \Pi, \Theta \rangle_\Phi),\, m \big) \right], \quad \text{s.t.} \quad \#\text{rollouts} \leq B$$

**核心挑战**：如何在有限的 rollout 预算下，从每次昂贵的 rollout 中提取最大的学习信号？

---

## 3. GEPA 方法

GEPA 的名称来自三个核心设计原则：**(Re)flection**（反思）、**(Ge)netic**（遗传）、**(Pa)reto**（帕累托）。

### 3.1 遗传优化循环

GEPA 维护一个**候选池 $\mathcal{P}$**，其中每个候选是系统参数 $\langle \Pi, \Theta \rangle_\Phi$ 的实例化。优化循环如下：

1. **选择候选**：从候选池中选择一个有前景的候选（通过 Pareto 采样）
2. **选择模块**：通过轮询策略选择一个模块进行优化
3. **小批量 rollout**：在 $\mathcal{D}_{\text{feedback}}$ 的小批量上执行选中的候选
4. **反思性提示更新**：利用反馈函数 $\mu_f$ 收集执行轨迹和文本反馈，让 LLM 反思并生成更新的提示
5. **局部验证**：如果新候选在小批量上优于父候选，则将其加入候选池并记录谱系
6. **Pareto 评估**：在全量 $\mathcal{D}_{\text{pareto}}$ 上评估新候选

### 3.2 反思性提示突变 (Reflective Prompt Mutation)

这是 GEPA 最核心的创新：

- **执行轨迹作为诊断信号**：复合 AI 系统执行过程中产生的自然语言轨迹（模块的输入、输出、推理链）提供了对系统行为的细粒度可见性
- **评估轨迹作为额外诊断信号**：许多评估指标在产生标量奖励前会经历复杂步骤（编译、执行、分析等），产生丰富的文本轨迹。GEPA 将这些**评估轨迹**也纳入反思过程
- **反馈函数 $\mu_f$**：扩展评估指标 $\mu$，使其在返回标量分数的同时返回 `feedback_text`，可用于指导有针对性的提示更新

反思元提示的核心结构：
```
当前指令
↓
小批量示例的（输入、输出、反馈）
↓
LLM 反思分析 → 生成更新后的指令
```

### 3.3 Pareto 候选选择

为避免贪心选择最优候选导致的局部最优，GEPA 采用 Pareto 启发式搜索策略：

1. **构建实例级 Pareto 前沿**：对每个训练实例 $i$，记录所有候选中的最高分 $s^*[i]$
2. **筛选 Pareto 最优候选**：保留在至少一个实例上达到最优的候选
3. **修剪严格支配候选**：移除被其他候选严格支配的候选（即在所有实例上都不优于某候选）
4. **按频率加权采样**：按候选在 Pareto 前沿中出现的频率进行概率采样

这一策略确保 GEPA 在探索和利用之间取得平衡，避免在局部最优上浪费预算。

### 3.4 System-Aware Merge（交叉策略）

GEPA+Merge 引入了一种系统感知的交叉策略：

- 从候选池中选择两个共享共同祖先、但优化了互补模块的候选
- 合并时，对不同模块分别选择每个谱系中更高性能的版本
- Merge 仅在识别到互补策略时才被调用，保持稀疏性

---

## 4. 实验评估

### 4.1 基准测试

| 基准 | 任务类型 | 复合 AI 系统 |
|------|---------|-------------|
| **HotpotQA** | 多跳推理 QA | HoverMultiHop（3跳检索 + 答案生成） |
| **IFBench** | 指令遵循 | 2阶段系统（回答 + 约束重写） |
| **PUPA** | 隐私感知委托 | PAPILLON（查询重写 + 响应重写） |
| **HoVer** | 事实验证 | 多跳检索 |
| **AIME-2025** | 数学推理 | 单步 Chain-of-Thought |
| **LiveBench-Math** | 数学推理 | 单步 Chain-of-Thought |

### 4.2 模型

- **Qwen3 8B**（开源）：temperature=0.6, top-p=0.95, top-k=20
- **GPT-4.1 Mini**（商业）：temperature=1.0

### 4.3 对比方法

- **GRPO**（强化学习基线，LoRA 微调，24,000 rollouts）
- **MIPROv2**（联合指令+少样本优化的 SOTA 提示优化器）
- **Trace / TextGrad**（文本梯度优化）
- **SelectBestCandidate**（消融基线，始终选择最优候选）
- **BeamSearch(N=4)**（APO 使用的搜索策略）

### 4.4 核心实验结果

#### 观察 1：反思性提示进化在样本效率上远超强化学习

| 对比 | GEPA 优势 |
|------|----------|
| vs GRPO（24k rollouts） | 平均 +6%，最高 +20%，使用最多 35x 更少 rollouts |
| 样本效率 | GEPA 仅需 243-1179 train rollouts 即可匹配 GRPO 最优验证分数（最高 78x 效率） |

#### 观察 2：纯指令优化超越联合指令+少样本优化

- GEPA 在所有基准和模型上均超越 MIPROv2
- 聚合提升：GEPA +13.33% vs MIPROv2 +5.64%
- 在 AIME-2025 上，GEPA 提升 +12% 准确率
- GEPA 生成的提示更短（最高 9.2x），计算效率更高

#### 观察 3：Pareto 候选选择策略至关重要

- GEPA Pareto 采样优于 SelectBestCandidate（消融）最多 8.17%
- SelectBestCandidate 容易陷入局部最优：一次改进后停滞，耗尽预算
- Pareto 采样通过探索多样化策略最终收敛到更高性能

#### 观察 4：指令优化比少样本示例更高效

- GEPA 提示比 MIPROv2 少最多 9.2x 的 token 长度
- 高性能往往与更短的提示相关

#### 观察 5：System-Aware Merge 提供额外增益

- GEPA+Merge 最高提供额外 +5% 的提升
- 效果因模型和时间而异：GPT-4.1 Mini 收益更大
- Merge 的有效性依赖于调用时机和预算分配

#### 观察 6：跨模型泛化

- Qwen3 8B 优化的 GEPA 提示在 GPT-4.1 Mini 上直接使用，达到 +9.00% 聚合提升
- 甚至优于在 GPT-4.1 Mini 上原生优化的 MIPROv2、TextGrad、Trace

### 4.5 扩展应用

**推理时搜索（代码优化）**：
- **NPU Kernels**：GEPA 将 AMD NPU 向量利用率从 4.25% 提升至 30.52%（GPT-4o）
- **CUDA Kernels**：GEPA 将 KernelBench 超过 PyTorch 基准的任务比例从接近 0% 提升至 20%+

**对抗性提示搜索**：
- 在 AIME-2025 上（GPT-5 Mini），GEPA 通过注入琐事分散注意力，将 pass@1 从 76% 降至 10%
- 发现了 LLM 在琐事+严格格式化约束交互中的脆弱性

---

## 5. 方法深度分析

### 5.1 为什么自然语言反思比权重更新更高效？

1. **高信号密度**：自然语言反馈（编译器错误、评估析因）比标量奖励包含更丰富的诊断信息
2. **利用 LLM 先验**：现代 LLM 天然擅长理解和推理自然语言，反思过程充分利用了这一能力
3. **大行为更新**：一次反思性提示更新可以带来巨大的行为变化，而权重更新需要大量梯度步骤

### 5.2 GEPA 伪代码概要

```
输入：系统 Φ, 训练集 D_train, 评估指标 μ, 反馈函数 μ_f, 预算 B
1. 将 D_train 分割为 D_feedback 和 D_pareto
2. 初始化候选池 P = [Φ]
3. 初始化 Pareto 前沿
4. While 预算未耗尽:
   a. SelectCandidate(P, S) → 通过 Pareto 采样选择候选
   b. SelectModule(Φ_k) → 轮询选择模块
   c. 在 minibatch 上收集反馈和轨迹
   d. UpdatePrompt → LLM 反思并生成新提示
   e. 若 minibatch 性能提升，在 D_pareto 上评估并更新候选池
5. 返回 D_pareto 上平均分最高的候选
```

---

## 6. 局限性与未来工作

1. **提示 vs 权重学习的边界**：在数据充裕或重量级微调可行时，权重更新可能优于提示优化
2. **反馈工程**：识别哪些执行/评估轨迹能提供最强的学习信号是一个重要方向
3. **验证集效率**：GEPA 大部分 rollout 预算花在候选验证上，通过更小的 Pareto 验证集或动态采样可进一步提升效率
4. **Merge 策略优化**：Merge 的有效性依赖于预算分配和调用时机，需进一步研究自适应策略
5. **少样本示例集成**：GEPA 目前仅优化指令，集成少样本示例可能进一步提升性能
6. **提示+权重联合优化**：将 GEPA 的语言反思与 RL 的权重更新相结合可能是更优方案

---

## 7. 总结

GEPA 通过三个设计原则（反思、遗传进化、Pareto 选择）实现了极高的样本效率，在多个基准上显著超越了 GRPO 强化学习方法和 SOTA 提示优化器 MIPROv2。这项工作表明，**语言层面的反思性学习可能是优化 AI 系统的一个可扩展且高效的替代方案**，尤其适用于 rollout 昂贵的实际场景。同时，GEPA 在推理时搜索和对抗性提示发现方面也展现了令人鼓舞的潜力。

---

## 讨论 Q&A

### Q1: GEPA 优化的是模型还是 Prompt？

**答**：GEPA 优化的是 **Prompt（提示词/指令）**，而非模型权重。论文将复合 AI 系统参数分解为 $\langle \Pi, \Theta \rangle$：
- $\Pi$：提示（包括指令和少样本示例）— **GEPA 优化此项**
- $\Theta$：底层 LLM 权重 — **冻结，不做任何修改**

具体来说：
1. **模型冻结**：GEPA 不对底层 LLM（如 Qwen3 8B、GPT-4.1 Mini）做任何微调或权重更新。
2. **每次调用 LLM 时的 Prompt 被优化**：GEPA 维护一个候选提示池，通过遗传进化+反思迭代改写指令文本，产物是一个更优的自然语言提示。
3. **与 GRPO 的对比**：GRPO 通过 LoRA 微调模型权重（需 24,000 rollouts），而 GEPA 仅优化文本层面的 Prompt（仅需几百 rollouts）就超越了 GRPO。这正是其样本效率高的根本原因——自然语言的"一步反思"可以产生比梯度下降的"多步权重更新"更大的行为变化。

### Q2: 候选池中的"候选"具体是什么样子？给一个具体例子

以论文中的 HotpotQA 多跳问答系统为例，该系统有 3 个模块：

| 模块 | 功能 |
|------|------|
| $M_1$: `query_gen` | 第一跳查询生成 |
| $M_2$: `summarize` | 第一跳结果摘要 |
| $M_3$: `second_hop_query` | 第二跳查询生成 |

**初始候选池** $\mathcal{P}$ 只有一个候选（Seed）：

```
候选0（祖先）:
├── M1.query_gen:       "Given the question, produce a search query."
├── M2.summarize:       "Summarize the retrieved passages."
└── M3.second_hop_query: "Given the fields question, summary_1, produce the fields query."
                         ↑ 只有1句话，约15个词
```

**第 1 轮迭代**：Pareto 采样选中候选0 → 轮询选中模块 $M_3$ → 小批量 rollout → 收集反馈 → LLM 反思 → 生成新 Prompt：

```
反馈示例:
  ✗ 问题: "What is the population of Madeira archipelago?"
    summary_1: "Porto Moniz is a civil parish with population 1,955..."
    生成的 query: "Porto Moniz population"  ← 太窄了，只搜了小教区
    → 反思: "应推断更大范围的实体，搜索 Madeira 而非 Porto Moniz"

  ✓ 问题: "Which NFL player is younger, Billy Truax or Lance Rentzel?"
    summary_1: "Billy Truax was born July 15, 1943..."
    生成的 query: "Lance Rentzel birth date"  ← 正确找到了缺失信息
```

LLM 根据反馈生成候选1的 $M_3$ 新 Prompt（约 400 词的结构化指令），加入候选池：

```
候选池 P = [候选0, 候选1]

候选1（父=候选0）:
├── M1.query_gen:       "Given the question, produce a search query."    (不变)
├── M2.summarize:       "Summarize the retrieved passages."             (不变)
└── M3.second_hop_query: "You will be given two input fields: question and summary_1.
    Your task: Generate a new search query optimized for the second hop...
    Key Observations: First-hop docs often cover one entity...
    The query should target missing, logically linked documents...
    Avoid merely paraphrasing the original question..."                (400+词)
```

**第 2 轮**：Pareto 选中候选1（在多个实例上表现最好）→ 轮询选中 $M_2$ → 反思突变 → 候选2：

```
候选池 P = [候选0, 候选1, 候选2]

候选2（父=候选1）:
├── M1.query_gen:       "Given the question, produce a search query."    (不变)
├── M2.summarize:       "You are the first-hop summarization module...    (新，优化后)
    Extract direct answers, identify missing clues..."
└── M3.second_hop_query: [继承候选1的优化版本]                            (继承)
```

**关键要点**：
- 每个候选是**所有模块 Prompt 的快照**，每次突变只改一个模块
- 遗传链使候选2 可以继承候选1 优化过的 $M_3$，积累学习成果
- Pareto 选择在不同训练实例上有特长的候选间轮换，避免在某个"看似最好"的候选上反复无效突变
- Seed prompt 只有 1 句话，最终优化产物是一份包含具体策略、反例和领域知识的约 400 词结构化指令

### Q3: HotpotQA 为什么有 3 个模块？是任务限定好的吗？

**答**：不是。HotpotQA 任务本身只定义了"多跳问答"——给定一个需要跨多个文档推理的问题，返回答案。系统用几个模块、每个模块做什么，完全是**系统设计者的架构选择**。

HotpotQA 是 **2-hop** 数据集（每个问题需要从两篇 Wikipedia 文档提取信息），论文中选择的流水线架构自然拆分为 3 个需要 LLM 调用的步骤：

```
问题 → [M1: 第1跳查询生成] → 检索 → [M2: 摘要] → [M3: 第2跳查询生成] → 检索 → 答案
```

但完全可以用不同的模块划分：

| 架构设计 | 模块数 | 说明 |
|---------|--------|------|
| 端到端 | 1 | 直接把问题和所有检索文档喂给 LLM |
| 论文的 R-R-R 流水线 | 3 | 查询→摘要→查询，每一步独立优化 |
| 更细粒度 | 5+ | 增加查询改写、答案验证、置信度评估等模块 |

**GEPA 的通用性**：论文选 HotpotQA 恰好展示它可以优化多模块复合系统，但 GEPA 对模块数没有限制——1 个模块、10 个模块都可以优化。论文在 DROP、GSM8K、MATH 等其他任务上也做了实验，各自的模块划分各不相同。

### Q4: 如何知道当前 Prompt 生成的 Query 正确与否？

**答**：**没有直接评分 Query**。整个反馈机制分两层：

**第一层（可观测）**：只有**最终答案**与标准答案对比产生的标量分数。系统运行完整流水线：

```
问题 → M1 → 检索 → M2 → M3 → 检索 → 最终答案
```

评估函数 $\mu$ 对比**最终答案**和 HotpotQA 的 **ground truth**，给出 0 或 1。这个分数是针对整个系统的，无法直接判断是哪个模块出错。

**第二层（推断）**：优化器 LLM 通过**反思进行隐式信用分配**（这是 GEPA 的核心创新）。LLM 被喂入：
1. 当前模块的 Prompt
2. 执行轨迹（该模块的输入/输出）
3. 最终得分（0 或 1）
4. 文本反馈（如 evaluator 的 rubric 输出）

然后 LLM 推断失败原因，追溯到模块级责任：

```
"最终得分=0。M3 的输入 summary_1 提到了小教区 Porto Moniz，
 但问题问的是 Madeira 群岛整体。
 → M3 生成的 query 只搜索了小教区，没有推断更大实体。
 → 需要在 Prompt 中加入'从 summary_1 推断更高层实体'的指引。"
```

因此：
- **得分 0.0** 来自 $\mu$（最终答案 vs 标准答案）→ **真实可观测**
- **"query 太窄了"** 来自优化器 LLM 的反思推理 → **推断的信用分配**

没有任何组件直接评分了"这个 query 好不好"——那是 LLM 根据最终失败反推的诊断。这恰恰是 GEPA 利用 LLM 语言理解能力替代精确梯度信号的关键设计。

### Q5: 是用最终得分当中间输出的得分吗？有个专门 LLM 负责 Prompt 优化吗？

**答（两个子问题）**：

**(a) 中间模块没有独立分数——确实用系统最终得分覆盖。**
- 评估函数 $\mu$ 只计算**系统最终输出**对比 ground truth 的标量分数。算法中的 `σ = mean(s)`、`σ'` 都是**系统级**得分。
- 中间模块（$M_1/M_2/M_3$ 产生的 query、摘要）**没有自己的 numeric score**。
- 模块级责任归属完全靠 **LLM 反思做隐式信用分配**，而非独立分数。
- 例外：`μ_f` 可提供**模块级文本反馈**（e.g. 多跳系统中每跳后给出 textual feedback），但这只是文本，判断候选是否保留仍只看系统级 `σ' > σ`。

**(b) 有两类角色不同的 LLM：**
| 角色 | 说明 | 权重 |
|------|------|------|
| 系统工人 LLM $\Theta_\Phi$ | 执行各模块（生成 query、摘要） | 冻结，不被优化 |
| 反思/优化 LLM (Reflection LM) | 看轨迹+反馈，**改写**工人 LLM 的 Prompt | 不参与被优化 |

"生成候选1 的 M3 新 prompt"的 LLM 就是 **反思 LLM**，与工人 LLM 是**不同调用（不同 prompt）**，不是同一个。

**反思 LLM 的输入被显著增强**，包含：
```
[固定的 meta-prompt] + [目标模块当前 Prompt π_j] + [该模块执行轨迹 (X_j, Y_j)]
+ [聚合文本反馈 F] + [最终得分 s]
```
其中 `meta-prompt` 是论文作者**预先写好的固定模板**（Appendix: gepa_meta_prompt），用于指示反思 LLM "如何反思、如何提出改进指令"。**这个 meta-prompt 在优化过程中不会被修改**——它是 GEPA 算法本身的固定超参，GEPA 优化的对象 Π 只含模块 Prompt（π_1~π_|M|），不含 meta-prompt。

一句话总结：一个固定的"教练 prompt"指挥反思 LLM，后者拿着更丰富的诊断信息（轨迹+反馈+分数）去改写工人 LLM 的指令；教练 prompt 本身不在优化范围。

### Q6: 综合理解——整条优化流程的主干对吗？

**答**：用户综合理解（seed → 队列 → rollout → 得分 → 另一个 LLM 突变 → 循环）**抓住了主干，但有 5 处需修正/补充**：

1. **是候选池（population），不是 FIFO 队列**：候选被选中后不弹出、不删除，新候选会被加入池（只增不减、不断生长的种群）；"pop out"更接近 Pareto 采样（按每实例表现频率加权随机选一个去进化），选完候选仍在池里。
2. **Seed = 整套系统配置**：种子是基础系统 $\Phi$（所有模块用默认 prompt），候选是完整 $\Pi$ 快照（$\pi_1,\pi_2,\pi_3$ 全包含），不是单步 prompt。
3. **每次只突变一个模块（round-robin）**：每轮只让反思 LLM 改写一个模块的 prompt，其余从父候选原样继承，是局部突变而非整系统重写。
4. **有接受门槛**：新候选先在 minibatch 上验证，仅当 $\sigma' > \sigma$ 才加入池；变差则丢弃，下一轮重来。
5. **终止条件是预算耗尽**：循环跑到 rollout 预算 $B$ 用尽才停，返回 $D_{pareto}$ 上平均分最高的候选，没有"任务完成"信号。

**修正后的完整流程**：
```
初始化: 候选池 P = [种子系统Φ]（所有模块默认/1句话prompt）
while 预算 B 未耗尽:
  ① 采样: 从 P 按 Pareto 频率选候选 Φ_k（不弹出，留池）
  ② 选模块: round-robin 选一个模块 M_j
  ③ rollout: 在 minibatch 跑 Φ_k → 轨迹 + 最终得分 + 文本反馈
  ④ 反思突变: [固定meta-prompt + M_j当前prompt + 轨迹 + 反馈 + 得分]
              → 送入【反思LLM】→ 生成 M_j 新 prompt π_j'
  ⑤ 组装: Φ' = 复制Φ_k，仅把 M_j 换成 π_j'
  ⑥ 验证: Φ' 在 minibatch 跑得 σ'；if σ'>σ: 加入P，并在完整D_pareto评估、更新Pareto前沿
返回: P 中 D_pareto 平均分最高的候选 Φ*
```

### Q7: 整个过程有训练的参数吗？推理时如何使用？

**答（两个子问题）**：

**(a) 没有梯度意义上的可训练权重参数。**
- GEPA 优化的"参数"是 Prompt 文本 $\Pi$（指令 + few-shot 演示），是**离散自然语言**，不是连续张量；底层 LLM 权重 $\Theta$ **全程冻结**。
- 对比 GRPO：GRPO 用策略梯度更新 LoRA 权重（有连续可训练参数），GEPA 只让反思 LLM 改写文本。
- 过程中只有**用户设定的超参数**（预算 B、minibatch b、Pareto 集大小 n_pareto、round-robin 策略、固定 meta-prompt），这些不被学习。

**(b) 推理时：反思 LLM 仅在离线优化阶段使用，部署时不需要。**
- 优化产物 = 优化后的 prompt 集 $\Pi^*$（即 $\Phi^*$ 中各模块的指令文本，如第二跳那个 400 词指令）。保存为配置即可。
- 部署时直接跑 $\Phi^*(x)$：控制流、模块数、检索器完全不变，仅把每个模块的 prompt 从默认替换为 $\Pi^*$，工人 LLM 用优化后 prompt 执行。
- **推理开销 = 原始系统一次前向**（每模块一次 LLM 调用），与 seed 版本相同，无额外延迟/成本。
- 对比 GRPO：GRPO 推理需载入微调后权重；GEPA 仍用原始模型 + 更强 prompt。

一句话：训练期用反思 LLM 离线"写"出更好 prompt；部署期把这套 prompt 当普通配置加载，系统照常运行，无需任何额外机制。
