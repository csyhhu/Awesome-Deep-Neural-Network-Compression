# SoTo: (1D) Ordered Tokens Enable Efficient Test-Time Search

**会议**：ICML 2026  
**作者**：Zhitong Gao, Parham Rezaei, Ali Cy, Mingqiao Ye, Nataša Jovanović (EPFL), Jesse Allardice, Afshin Dehghan (Apple), Amir Zamir (EPFL), Roman Bachmann\*, Oğuzhan Fatih Kar\* (EPFL, Apple)  
**链接**：https://arxiv.org/abs/2604.15453  
**项目页面**：https://soto.epfl.ch  
**关键词**：tokenization, test-time scaling, autoregressive model, search

---

## 1. 核心动机

自回归（AR）生成模型依赖 tokenization 将原始数据转换为可建模的离散单元。传统方法将图像编码为 **2D 空间网格 token**（如 VQGAN），这些 token 按 raster-scan 顺序生成，仅对应局部像素区域，中间状态缺乏全局语义信息。

本文提出：**token 结构对 test-time search 的有效性有决定性影响**。近期出现的 **1D ordered tokenizer**（如 FlexTok、Semanticist）通过 nested dropout 训练，将图像压缩为**由粗到细（coarse-to-fine）的 1D 序列**——早期 token 捕获全局语义结构，后续 token 逐步细化细节，且任意前缀均可解码为有效图像。

**核心假设**：粗到细的 1D 有序结构使中间 token 前缀具有可验证的全局语义，从而天然适合 test-time search。

---

## 2. 方法框架：SoTo（Search-over-Tokens）

论文提出了一个系统化的三组件分析框架：

### 2.1 搜索算法
- **Best-of-N sampling**：独立生成 N 条序列，选 verifier 得分最高的。简单可并行，但不利用中间状态。
- **Beam Search**：维护 k 个部分序列，每步扩展 top-M 候选，保留最高分前缀。依赖中间状态的语义信息质量。
- **Lookahead Search**：Beam Search 的扩展，先 rollout 部分序列到更完整图像再评分，适合早期 token 语义弱的场景（如 2D grid token）。

### 2.2 验证器（Verifier）
三类验证器引导搜索：
- **图文对齐**：CLIPScore、ImageReward、HPSv2、PickScore、CycleReward、Grounded-SAM（规则验证器）、likelihood
- **图图对齐**：DreamSim（感知相似度）
- **图像质量**：LAION Aesthetic Score
- **集成验证器**：通过 rank-based aggregation 组合多个信号

### 2.3 AR 先验
三个层次的先验强度：
1. **条件先验**（text-conditional AR model）：最强引导
2. **无条件先验**（unconditional AR model）：中等引导
3. **均匀先验**（no AR model）：无引导，纯搜索

---

## 3. 关键发现

### 3.1 Token 结构影响搜索效率

| Token 结构 | Beam Search 增益 | Best-of-N 增益 | 最优策略 |
|-----------|:---------------:|:-------------:|---------|
| 1D Ordered (FlexTok) | **+13.05 CLIPScore** | 中等 | Beam Search |
| 2D Grid | +2.53 CLIPScore | 中等 | Best-of-N |

- 1D ordered token 的**前缀解码图像已含全局语义**，使 beam search 在早期就能可靠修剪搜索空间
- 2D grid token 的中间状态只含局部空间信息（如左上角像素），对 verifier 无用
- Lookahead search 可部分弥补 2D grid token 的不足，但计算成本大幅增加
- **每种 token 结构用其最优算法比较，1D ordered token 仍显著优于 2D grid token**

### 3.2 无训练图像生成（Training-Free Generation）

**仅通过搜索 token 空间即可生成合理图像**，无需训练 AR 模型：
- 使用 beam search + 均匀先验 + CLIP/ImageReward 验证器
- 单物体生成准确率 79%，双物体 32%（GenEval）
- 无条件 AR 先验进一步提升性能
- 这一行为在 2D grid token 或无序 1D token 中**无法实现**

### 3.3 零样本多模态控制

FlexTok + DreamSim verifier 可实现训练外信号引导：
- 在 DreamBench++ 上，搜索使 concept preservation（DINO-I）提升 **+18.4**
- Janus 仅提升 +5.9（即使使用 lookahead search）
- 无需任何微调即可实现图像引导的文本到图像生成

### 3.4 模型规模与推理计算的关系

- **530M 参数的 AR 模型 + 充足 test-time compute 可超越 3.4B 参数模型**（有限推理计算）
- 大模型随推理计算增加表现出更强的 scaling 行为
- 最优模型大小与推理计算满足**幂律关系**：`model_size ∝ compute^0.44`

### 3.5 验证器分析

- 所有验证器均能改善生成质量
- **ImageReward** 和 **HPSv2** 作为通用验证器表现最强
- **集成验证器**在各指标上平均排名最优（通常排第二，综合排第一）
- 不同验证器侧重不同方面：CLIPScore 偏向语义对齐，Aesthetic Score 偏向美感，Grounded-SAM 偏向空间关系

### 3.6 理论分析

论文提供了形式化证明：
- **搜索间隙**（search gap）被 tokenizer 的中间重建误差约束：Δ ≤ 2L·ε_t₀ + η_t₀
- 1D ordered tokenizer 通过 nested dropout 显式最小化中间重建误差
- 类似于 PCA 的信息压缩原理，早期 token 捕获主导方差分量，ε_t 随 t 快速下降
- 2D grid tokenizer 仅在完整序列（t=T）时施加重建约束，中间阶段 ε_t 大

---

## 4. 实验设置

- **主要模型**：FlexTok d18–d28 + 3.4B AR model；对比 Janus-1.3B（2D grid）
- **数据集**：COCO Karpathy validation、GenEval、DreamBench++
- **评估指标**：CLIPScore、ImageReward、GenEval accuracy、DINO-I 等
- **推理计算**：NFE（Number of Function Evaluations）衡量

### 泛化性验证

- **Semanticist**（另一 1D ordered tokenizer，class-to-image）：Beam search 提升 CLIPScore **+10.42** vs LlamaGen-L（2D grid）的 +3.51
- **Infinity**（scale-wise AR，多尺度 2D 生成）：Beam search 提升 +6.2，介于 Janus（+5.3）和 FlexTok（+9.6）之间，说明**语义粗到细排序最有效**

---

## 5. 局限性与未来工作

1. **搜索算法非定制化**：当前使用通用算法，针对 ordered token 结构设计专用搜索策略可进一步提升效率
2. **验证器可靠性**：compute budget 大时可能出现"verifier hacking"，验证器盲点被利用
3. **Detokenization 瓶颈**：流式 detokenizer 的多步去噪是搜索的主要计算开销
4. **泛化性**：需在更多架构和模态（文本、视频、多模态）上验证
5. **训练-推理计算权衡**：更大规模下搜索的有效性尚不明确

---

## 6. 核心启示

1. **Token 结构是 test-time scaling 的关键设计维度**，不亚于搜索算法和验证器的设计
2. 1D ordered tokenizer 的 coarse-to-fine 结构使前缀 token 具有全局语义可验证性，天然适配 beam search
3. 足够好的 token 结构甚至可以实现**无训练的生成**（pure search）
4. **小模型 + 搜索 > 大模型 + 无搜索**，为推理时计算分配提供了新的 trade-off 视角
5. Tokenization 不应仅被视为表示学习选择，也是实现高效 test-time search 的机制

---

## 7. 讨论 Q&A

### Q1: Test-Time Search 是什么？

**Test-time search（测试时搜索）** 是一种在模型推理阶段使用的技术：不满足于模型一次性生成，而是主动探索多个候选生成路径，并借助一个"裁判"（verifier）来挑选最优结果。

与**传统 AR 推理**的对比：

| | 传统 AR 推理 | Test-Time Search |
|---|---|---|
| 生成方式 | Greedy decoding / 单次采样 | 探索多条路径，择优 |
| 计算量 | 固定（每 token 一次前向） | 可变，随搜索预算增大 |
| 质量 | 受限于模型一次性输出 | 可通过额外计算提升 |
| 比喻 | 闭着眼睛走一步算一步 | 边走边看多个方向，挑最好的走 |

**数学形式**（论文公式 2）：

$$\hat{\mathbf{x}} = \arg\max_{\mathbf{x}}\; g(\mathbf{x}, c) \quad \text{s.t.} \quad x_t \in \mathcal{K}(p_\theta(\cdot \mid x_{<t}, c))$$

即在 AR 先验 $p_\theta$ 约束的候选集合中，搜索使 verifier $g$ 得分最高的 token 序列 $\hat{\mathbf{x}}$。三个执行组件：
1. **AR 先验**：限制搜索空间范围（从 text-conditional → unconditional → uniform 依次减弱）
2. **搜索算法**：Best-of-N / Beam Search / Lookahead Search，决定探索策略
3. **Verifier**：评估生成质量，引导搜索方向（如 CLIP 判断图文匹配度）

**为什么需要时序搜索？** 本文的 search 是在 token-by-token 生成过程中逐步进行的——Beam search 每生成几个 token 后就 detokenize 查看中间图像、请 verifier 打分，再决定保留哪些候选。因此**中间状态的可验证性是 search 有效的前提**：1D ordered token 的 coarse-to-fine 结构让每步都能被可靠评估，而 2D grid token 的早期状态毫无语义（只画了左上角像素），搜索自然失效。

**在更广语境中**：LLM 的 Chain-of-Thought + Best-of-N、Tree-of-Thought、Process Reward Model (PRM) 等都是 test-time search 的实例，核心逻辑一致——用推理计算换取质量，中间状态的可验证性决定搜索效率。

---

### Q2: LLM 中也有 Test-Time Search 吗？

**有的，LLM 领域的 test-time search 研究比图像生成更早、更成熟。** SoTo 论文大量借鉴了 LLM 的 search 思想，两者共享相同的核心框架——"先验 + 搜索算法 + 验证器"。

**LLM 中 Test-Time Search 的主要形态**：

| 技术 | 搜索算法 | Verifier | 代表工作 |
|---|---|---|---|
| Self-Consistency | Best-of-N（采样 N 条 CoT 路径） | 多数投票（答案一致性） | Wang et al., 2022 |
| Tree-of-Thought (ToT) | Beam Search / BFS / DFS（推理步骤级别） | LLM 自评 | Yao et al., 2023 |
| PRM + Beam Search | Beam Search over reasoning steps | Process Reward Model（对中间步骤打分） | Lightman et al., 2023 |
| MCTS + LLM | 蒙特卡洛树搜索 | Value network 或 rollout | AlphaCode, rStar |
| OpenAI o1 / DeepSeek-R1 | 隐式搜索（内部 CoT 长链 + 自验证） | 内置 reward / self-verification | o1, R1 tech report |
| Best-of-N with RM | 生成 N 条完整回答，选最优 | Reward Model | Llama 2 |

**三组件框架的对应**：

| 组件 | SoTo（图像 AR 生成） | LLM 对应 |
|---|---|---|
| AR 先验 | text-conditional / unconditional AR model | 预训练 LLM（如 GPT-4） |
| 搜索算法 | Best-of-N / Beam Search / Lookahead | ToT, MCTS, Beam Search |
| Verifier | CLIP, ImageReward, HPSv2... | PRM, ORM, LLM-as-judge |

**关键差异——中间状态的可验证性**：

- **LLM 天然可验证**：CoT 推理的每一步都是自然语言——"先算 3+5=8，再乘以 2 得 16"，PRM 可以轻松判断中间步骤是否正确。
- **图像 AR 生成不一定可验证**：2D grid token 的中间解码只显示"左上角有一片灰色"，verifier 无从判断最终图像质量。**这正是 SoTo 的核心贡献**——通过 1D ordered token 让图像生成的中间状态也变得可验证，从而将 LLM 领域已验证的 search 技术成功迁移到图像生成。

**论文直接引用的 LLM search 工作**：Wei et al., 2022 (CoT)；Yao et al., 2023 (ToT)；Lightman et al., 2023 (PRM)；Snell et al., 2024 (test-time compute scaling laws)；Brown et al., 2024 (large-scale test-time compute)。

论文指出，LLM 领域的现有工作主要聚焦于更好的搜索算法、更强的验证器以及 scaling laws，但**少有关注模型本身（token 结构）应具备什么特性才能受益于 test-time scaling**——这恰是 SoTo 的独特贡献。

---

### Q3: 现在 LLM 常用的生成方式和 Test-Time Search 有什么区别？

**一句话总结：常用生成方式是"局部贪心，一步到位"；Test-time search 是"全局搜索，择优而取"。**

**一、LLM 常用生成方式（无搜索/轻量搜索）**

日常使用 ChatGPT 或调用 API 时最常用的方式，每步只做一次局部最优决策：

| 方法 | 工作机制 | 特点 |
|---|---|---|
| Greedy Decoding | 每步选概率最高的 1 个 token | 确定性，最快，容易重复/退化 |
| Temperature Sampling | logits 除以温度 T 后采样，T↑→随机，T↓→确定 | 引入随机性，增加多样性 |
| Top-k Sampling | 只从概率最高的 k 个 token 中采样 | 截断低概率尾部，过滤"垃圾"token |
| Top-p (Nucleus) | 从累积概率 ≥ p 的最小 token 集合中采样 | 自适应截断，比固定 k 更灵活 |
| Repetition Penalty | 对已出现的 token 降权，防止重复 | 简单启发式惩罚，非搜索 |

核心特征：**每生成一个 token，只根据当前状态向前看一步，不回溯、不比较、不"后悔"。**

**二、Test-Time Search（全局搜索）**

核心区别：**同时探索多条路径，用额外计算换取更好的全局结果。**

| 方法 | 工作机制 | 与常用生成的核心差异 |
|---|---|---|
| Best-of-N (BoN) | 采样 N 条完整回答，用 RM 选最优 | **事后比较**，而非事中选择 |
| Beam Search + Verifier | 每步保留 k 个候选 beam，verifier 打分筛选 | **每步比较**多条路径，择优劣汰 |
| MCTS | 对候选分支做"探索-模拟-回溯" | 对未走路径做**模拟预判** |
| Tree-of-Thought | 每步推理生成多个候选，LLM 自评筛选 | 对**中间步骤**做搜索，非仅最终答案 |

**三、本质对比**

> 常用生成 ≈ 走在一条路上，每到一个岔路口看一眼就选一条往前走，不回头。
> Test-time search ≈ 每到一个岔路口，往每条路都走一段，把死路淘汰，只保留最有希望的路继续走。

| 维度 | 常用生成方式 | Test-Time Search |
|---|---|---|
| 决策粒度 | 局部（每个 token 独立选） | 全局（考虑完整序列的最优性） |
| 后见之明 | 无（选了就锁死） | 有（差的路径会被淘汰） |
| 计算开销 | O(1) per token，固定 | O(k) per token 或更高，随预算线性增长 |
| 质量保证 | 靠概率分布"运气" | 在搜索预算内逼近全局最优 |
| 是否需要 verifier | 不需要额外 verifier | **必须**有 verifier 来比较候选 |
| 适用场景 | 开放对话、创意写作（多样性重要） | 数学推理、代码生成（正确性重要） |
| 代表 | ChatGPT 默认采样、Claude 对话 | o1 隐式搜索、AlphaCode MCTS、R1 |

**四、为什么不是所有 LLM 都用 search？**

1. **计算成本**：BoN 的推理开销是普通采样的 N 倍
2. **Verifier 瓶颈**：search 上限取决于 verifier 质量，RM 可能被 reward hack
3. **多样性损失**：search 倾向于选"最安全的"答案，不适合创意任务
4. **延迟敏感**：实时对话不允许等 beam search 跑完

因此工业界采用**分层策略**：日常对话用快速采样（top-p + temperature），复杂推理自动触发 search（o1 的做法），代码/数学显式使用 MCTS 或 rejection sampling 训练出 search 行为（如 DeepSeek-R1）。

---

### Q4: Test-Time Search 中每一次 decoding 用什么方式？是 Greedy Decoding 吗？

**不是。Beam search 的每一步既不等于 greedy decoding，也不等于 random sampling。** 需要区分"候选怎么产生"和"最终怎么选择"两个阶段。

**一、两步流程**

| 阶段 | 做什么 | 谁来决定 |
|---|---|---|
| **候选产生** | 从模型输出分布中选出哪些 token 是"值得考虑的" | 模型 $p_\theta$（取 top-k 最高概率 token） |
| **选择保留** | 从所有候选路径中保留哪些继续扩展 | verifier $g$ + 模型累积概率（加权排序，保留 top-b beams） |

**二、不同搜索方法的候选产生方式**

| 搜索方法 | 候选产生方式 | 说明 |
|---|---|---|
| Beam Search（经典） | 模型 top-k 概率最高的 token（**确定性**） | 无 verifier，累积 log-likelihood 排序 |
| Beam Search + Verifier | 同上 top-k，但 verifier 参与最终筛选 | 模型给候选，verifier 定去留 |
| Best-of-N (BoN) | 每条序列用 **temperature sampling** 独立生成 | 这是普通随机采样，重复 N 遍 |
| MCTS | PUCT 公式（模型 prior + 探索 bonus） | 平衡高概率节点和探索不足的节点 |

**三、SoTo 论文的具体做法**

SoTo 的搜索公式直接说明了这一点：

$$\hat{\mathbf{x}} = \arg\max_{\mathbf{x}}\; g(\mathbf{x}, c) \quad \text{s.t.} \quad x_t \in \mathcal{K}(p_\theta(\cdot \mid x_{<t}, c))$$

- $\mathcal{K}(\cdot)$ 从模型概率分布中**取 top-k 个最高概率的 token**——这是确定性 top-k 截断
- $g(\mathbf{x}, c)$ 是 verifier，对每条候选路径打分决定保留哪些 beam

流程：模型输出分布 → 取 top-k 高概率 token（确定性）→ verifier 打分 → 保留 top-b beams

**四、常见误解澄清**

> ❌ "Beam search 每步用 greedy decoding 扩展"
>
> ✅ Beam search 每步看的是 top-k 个 token（如 k=10），然后保留 b 条最优路径（如 b=4）。**如果只取 greedy 的那 1 个 token，beam 就退化成 greedy decoding，beam width 毫无意义。**

| 方法 | 每步考虑的 token 数 | 每步保留的路径数 |
|---|---|---|
| Greedy Decoding | 1（概率最高的那个） | 1 |
| Top-k Sampling | k 个中随机选 1 个 | 1 |
| Beam Search（b=4, k=10） | 10（模型 top-10） | 4 |
| Beam Search + Verifier | 10（模型 top-10）→ verifier 重排 | 4 |

**五、为什么 BoN 和 Beam Search 的候选产生方式不同？**

- **BoN 用 sampling**：因为每条序列独立生成到终点再比较，中间不需要 beam 维护。sampling 保证多样性，让 N 条序列覆盖足够广的解空间。
- **Beam search 用 top-k**：因为需要在每步对多个 beam 做公平比较，随机采样会引入噪声，使 beam 之间的比较不可靠。确定性 top-k 保证每步候选稳定、可复现。

---

### Q5: Beam Search 每次选 10 个（k=10），beam width=4，为什么 4 步后路径数不是 10⁴？

**因为每一步都做了剪枝（pruning）。** Beam search 不是"先扩展完所有可能性再选"，而是**每步扩展后立即剪枝回 beam-width 条路径**。

**推演过程（k=10, b=4）：**

```
Step 1:
  模型输出分布 → 取 top-10 候选 token
  10 个候选 → 打分 → 保留 top-4 beams
  → 剩余路径：4（不是 10）

Step 2:
  4 beams × 10 candidates = 40 个候选
  40 → 打分 → 剪枝保留 top-4
  → 剩余路径：4（不是 10²=100）

Step 3:
  4 × 10 = 40 → 剪枝 → 4
  → 剩余路径：4（不是 10³=1000）

Step 4:
  4 × 10 = 40 → 剪枝 → 4
  → 剩余路径：4（不是 10⁴=10000）
```

**结论：路径数永远是 beam-width = 4，不会指数爆炸。** 10 只是每步"看一眼"的候选范围，看完就扔。

**10⁴ 对应的是什么？** 那是**穷举搜索（Exhaustive Search）**——每步保留所有候选、不做剪枝：

| 搜索方式 | 每步保留 | 4 步后路径数 |
|---|---|---|
| Greedy（b=1） | 只保留第 1 | 1 |
| Beam Search（b=4） | 剪枝到 4 | **4** |
| Exhaustive | 全部保留 | **10⁴ = 10000** |

Beam search 是 greedy 和 exhaustive 之间的折中——比 greedy 多看几条路，比 exhaustive 少指数级计算。

**计算量对比：**

- Beam Search：每步 b·k = 4×10 = 40 次扩展，总 O(T·b·k)，关于 T **线性**
- Exhaustive Search：总 O(k^T)，关于 T **指数爆炸**

这就是为什么 beam search 在实际中可用、而穷举搜索完全不可行的根本原因。"每步剪枝"是 beam search 最核心的机制。

---

### Q6: Beam search 结束后，也是通过 verifier 选最好的一条路吗？

**取决于哪种 search 方式。Beam search + verifier 中，verifier 从第一步就参与，不是等到最后才出场。**

**三种典型情况：**

**一、Beam Search + Verifier（SoTo 的做法）——verifier 贯穿全程**

```
Step 1: 10 候选 → verifier 打分 → 保留 top-4
Step 2: 40 候选 → verifier 打分 → 保留 top-4
...
Step T: 40 候选 → verifier 打分 → 保留 top-4（生成完毕）
```

到终点时，4 条 beam 的**累积得分已包含 verifier 的全程评分**。得分最高的 beam 自然胜出——**不需要额外的"最终选择"步骤**，verifier 的筛选在每一步都已经完成了。

**二、经典 Beam Search（无外部 verifier）**

用模型自身的累积 log-probability 排序，选累积概率最高的 beam（通常做长度归一化）。这里没有外部 verifier。

**三、Best-of-N —— verifier 只在最后出场**

```
独立生成 N 条完整序列（sampling，互不比较）
     ↓
N 条全部完成后 → verifier 一次性打分 → 选最高分
```

**对比总结：**

| | Verifier 何时介入 | 最终选择 |
|---|---|---|
| Beam Search + Verifier（SoTo） | **每一步**都打分剪枝 | 得分最高的 beam 自然胜出，无需额外选 |
| 经典 Beam Search | 无外部 verifier | 累积模型概率最高的 beam |
| Best-of-N | **只在最后** | N 条中 verifier 得分最高的 |

**直觉类比：**

> Beam Search + Verifier = 选秀节目全程淘汰制——每轮打分，差的当场淘汰，走到最后的选手本身就经过了全程验证。
>
> Best-of-N = 所有人独立完成比赛，最后统一打分排名，只选冠军。

SoTo 在实验中对比了这三种搜索方式，发现 beam search 在效率上优于 BoN——因为 **1D ordered token 让中间每一步的"半成品图像"都能被 verifier 可靠评估**，所以每步都可以做有意义的剪枝，而 BoN 必须把每条序列生成完才知道好坏，中间步骤全浪费了。

---

### Q7: 本文的核心是 test-time search 框架，不是 beam search 吧？

**是的，这是一个重要的区分。** SoTo 的核心贡献是提出了一个**通用的 test-time search 框架**，beam search 只是框架中搜索算法的一种实例化。

**SoTo 的 Test-Time Search 框架（三组件）：**

| 组件 | 角色 | 论文实验的选项 |
|---|---|---|
| **AR 先验** $p_\theta$ | 约束搜索空间 | text-conditional / unconditional / uniform |
| **搜索算法** | 如何探索候选路径 | **Best-of-N / Beam Search / Lookahead Search** |
| **Verifier** $g$ | 评价生成质量 | CLIP / ImageReward / HPSv2 / DINO-I / DreamSim |

论文实验了**三种搜索算法**：

| 搜索算法 | 论文章节 | 说明 |
|---|---|---|
| Best-of-N | §7.2 | 采样 N 条完整序列，verifier 最后选最优 |
| Beam Search | §7.2 | 每步保留 top-b beams，verifier 全程打分剪枝 |
| Lookahead Search | §7.3 | Beam search 扩展——每步多前瞻几步再打分，更准确但更昂贵 |

此外论文还实验了**三种 AR 先验强度**（text-conditional → unconditional → uniform）和**多种 verifier**（CLIP、ImageReward、HPSv2、DINO-I、DreamSim），展示了框架的通用性。

**之前 Q&A 为什么以 beam search 为例？** 因为讨论 search 的具体 mechanics（候选产生、剪枝、最终选择）时，beam search 是最完整、最能体现这些机制的载体。但这不意味着 SoTo = beam search。

**SoTo 的真正贡献是框架层面的**：证明了 1D ordered token 结构让 AR 图像生成的中间状态变得可验证，从而使 LLM 领域已有的多种 test-time search 技术（Best-of-N、Beam Search、Lookahead Search）都能有效迁移到图像生成。这是此前 2D grid token 无法做到的。

---

### Q8: 本文对不同搜索算法和验证器都做了实验吗？结果如何？

**是的，实验覆盖非常全面。** 论文从搜索算法、AR 先验、verifier、模型大小、零样本控制五个维度进行了系统实验。

**一、搜索算法对比（最核心实验，Fig. 1）**

| 搜索算法 | 1D Ordered (FlexTok) | 2D Grid | 关键发现 |
|---|---|---|---|
| Best-of-N | 有效 | 有效 | 两者 scaling 趋势相似 |
| **Beam Search** | **大幅提升** | 几乎无效 | 差异最显著的算法 |
| Lookahead Search | 有效 | 部分恢复 | 2D grid 需完整 rollout 才能打分，代价高 |

Beam search 是区分两种 token 结构最敏感的算法：1D ordered 每步都有语义、verifier 可靠打分，beam search 最高效；2D grid 早期只有局部像素，beam search 几乎无效。**对 2D grid 来说 Best-of-N 反而是最优策略。**

**二、与 Janus (2D grid 代表) 对比**

| | 无搜索 | +Beam Search |
|---|---|---|
| FlexTok (1D) | 略低 | **大幅反超** |
| Janus (2D) | 略高 | 几乎不涨 |

**三、泛化到其他有序模型**

- **Semanticist** (1D ordered, ImageNet)：beam search 收益显著优于 2D baseline LlamaGen
- **Infinity** (scale-wise ordered, COCO)：受益于搜索但不如 FlexTok
- 结论：有序就有收益，语义级粗到细排序收益最大

**四、模型大小 Scaling Law**

| 发现 | 细节 |
|---|---|
| 小模型 + 搜索 〉大模型 | 530M + sufficient search 超越 3.4B 无搜索 |
| 幂律关系 | 最优模型大小 ∝ (推理计算)^0.44 |
| Pareto 前沿 | 推理计算越大，最优模型越大 |

**五、AR 先验消融（Generation by Search）**

| 先验 | GenEval 单物体 | GenEval 双物体 |
|---|---|---|
| Uniform（纯搜索，无 AR 模型） | **79%** | **32%** |
| Unconditional AR | 更高 | 更高 |
| Text-conditional AR | 最高 | 最高 |

**不训练 AR 模型，纯靠 verifier 在 token 空间搜索，即可生成 reasonable 图像！**

**六、零样本多模态控制（DreamSim verifier）**

| | DINO-I 提升 |
|---|---|
| FlexTok (1D) | **+18.4** |
| Janus (2D) | +5.9 |

只需换 verifier，无需任何微调，即可实现训练时未见过的控制信号。

**七、Verifier 全面对比（8 种 + Ensemble + Oracle）**

测试了 CLIPScore、Aesthetic Score、CycleReward、HPSv2、ImageReward、Grounded SAM、PickScore、likelihood 共 8 种 verifier，以及 Ensemble 和 Oracle：

| Verifier | 特点 |
|---|---|
| **Ensemble（集成全部 8 种）** | 平均排名最优，最稳健 |
| **ImageReward** | 个体最强，通用性好 |
| **HPSv2** | 个体第二 |
| Oracle (GenEval GT) | 上界，验证 search 理论上限很高 |

**所有 verifier 都一致优于 baseline**，search 对 verifier 选择鲁棒。人类偏好模型（ImageReward、HPSv2）是最佳通用 verifier。

---

### Q9: 本文没有提出新 tokenizer，只是在已有 1D tokenizer 上验证 test-time search 效率？

**是的，这个理解完全准确。** SoTo 没有提出任何新组件——tokenizer、AR 模型、搜索算法、verifier 全部是已有的。它的贡献是"连接与发现"型的。

**一、论文用了什么（全是已有的）**

| 组件 | 来源 | 论文是否提出 |
|---|---|---|
| 1D ordered tokenizer | FlexTok, Semanticist, Infinity（已有工作） | ❌ |
| AR 生成模型 | 各 tokenizer 配套的预训练模型 | ❌ |
| 搜索算法 | Best-of-N, Beam Search, Lookahead（经典方法） | ❌ |
| Verifier | CLIP, ImageReward, HPSv2 等（已有工作） | ❌ |

**二、论文真正贡献了什么**

SoTo 做的是**连接两个原本独立的研究领域**：

```
之前：Tokenizer 设计 ⟂ Test-time search（互不相关）

SoTo：Tokenizer 设计 → 决定 → Test-time search 效率（首次建立因果联系）
```

具体贡献：
1. **问题定义**：首次系统研究 token 结构对 test-time search 的影响
2. **理论分析**：证明搜索间隙（search gap）被中间重建误差约束——1D ordered token 的 nested dropout 训练天然最小化该误差
3. **实验发现**：1D ordered token 的 coarse-to-fine 结构让中间状态天然可验证，使 beam search 效率远超 2D grid
4. **新能力揭示**：纯搜索生成（无 AR 训练）、零样本多模态控制——用现有组件组合出此前未发现的能力

**三、类比理解**

> 有人发明了新材料（FlexTok 等 1D tokenizer），大家只拿它盖普通房子（直接 AR 生成）。
>
> SoTo 的贡献是发现：这种材料的特殊结构（粗到细排列）让它特别适合建摩天大楼（test-time search），而传统砖块（2D grid token）根本建不了。材料不是论文发明的，但"这种材料能建摩天大楼"的洞察是核心贡献。

这种"重新审视已有组件、发现隐藏能力"的研究，在顶会中同样有重要价值——它改变了社区对 tokenizer 设计的认知：tokenization 不应仅被视为表示学习问题，同时也是实现高效 test-time search 的机制。

---

### Q10: 1D vs 2D tokenizer 的讨论只适用于图像？文本天然是 1D 的

**是的。文本天然是 1D 有序的，不需要额外设计；这个问题只出现在有空间/时间结构的多维模态中。**

**一、为什么文本没有这个问题？**

文本本身就是一维语义序列，中间状态天然可验证：

```
"The cat sat on the __"  → 很容易猜到 "mat"
"那只猫坐在 ___"         → 语义完整，可以打分
```

LLM 的 beam search 之所以一直有效，正是因为文本的线性性保证了每一步的中间状态都有意义——不需要额外设计 token 结构。

**二、为什么图像有这个问题？**

图像是二维空间，必须被"线性化"为一维序列。线性化方式决定了中间状态的质量：

| Tokenizer | 线性化方式 | 中间状态（前 25% token） |
|---|---|---|
| 2D Grid (VQGAN) | 左上→右下 raster-scan | 左上角一片像素，无全局语义 |
| 1D Ordered (FlexTok) | Nested dropout 粗→细 | 整张图的模糊缩略图，全局语义完整 |

**三、其他模态呢？**

| 模态 | 天然维度 | 需要专门设计？ | 原因 |
|---|---|---|---|
| 文本 | 1D | ❌ | 天然有序，每步语义完整 |
| 音频 | 1D 时间 | ❌ | 天然有序，贴近文本 |
| 代码 | 1D | ❌ | 天然有序 |
| 图像 | 2D 空间 | ✅ | Raster-scan 破坏中间语义 |
| 视频 | 3D 时空 | ✅（可能收益更大） | 逐帧 raster-scan 问题更严重 |
| 3D 点云/体素 | 3D 空间 | ✅ | 扫描顺序中间无结构 |

**四、核心洞察的泛化**

SoTo 的洞察本质上不是"图像的问题"，而是**"如何将高维结构线性化为一维序列，同时保持中间状态可验证"**这一通用问题：

- 文本/音频/代码天然不需要线性化，自动满足条件
- 图像/视频/3D 必须线性化，简单 raster-scan 会破坏中间语义
- 1D ordered tokenizer（nested dropout 训练出粗到细排序）是解决这一问题的通用方法

这意味着 SoTo 框架的适用范围远超图像——任何需要从高维空间压缩为 1D 序列的模态，都可以从"设计可验证中间状态的 tokenizer"中获益。

---

### Q11: 详细介绍 FlexTok

FlexTok 是 SoTo 论文使用的**核心 1D ordered tokenizer**，发表于 ICML 2025，与 SoTo 来自同一研究组（EPFL + Apple），SoTo 是其自然延续工作。

**一、核心能力：可变长度的 1D 有序 token 序列**

FlexTok 将一张 256×256 图像压缩为 **1~256 个离散 token** 的 1D 序列，token 数量可自由选择：

```
1 个 token  → 模糊但语义正确（"这是海边日落"）
8 个 token  → 大致轮廓清晰
64 个 token → 细节丰富
256 个 token → 近乎无损重建
```

这与传统 2D grid tokenizer（如 VQGAN 固定 16×16=256 个 token）有本质区别。

**二、架构设计**

| 组件 | 实现 |
|---|---|
| **编码器** | ViT，将图像编码为最大长度的 latent 序列 |
| **量化** | 可学习码本（codebook），每个 token 对应一个离散 embedding |
| **解码器** | **Rectified Flow 模型**（扩散模型变体，直线轨迹采样），将任意长度 token 序列重建为图像 |
| **AR 生成** | 标准 GPT-style decoder-only Transformer，逐 token 预测 |

**三、关键训练技巧：Nested Dropout**

这是 FlexTok 实现"1D ordered"的核心机制：

```
训练时：
  每轮随机选择一个有效长度 ℓ ∈ {1, 2, ..., 256}
  只取前 ℓ 个 token 送入解码器
  丢弃剩余的 (256 - ℓ) 个 token
```

**效果**：迫使编码器将最重要的信息**挤压到前面的 token**：
- Token 1~8：全局语义（物体类别、颜色基调、布局）
- Token 9~32：大结构（轮廓、主要物体）
- Token 33~128：中等细节（纹理、边缘）
- Token 129~256：高频细节

任意前缀都是该图像的一个"有效但粗糙"的表示——**这正是 SoTo 能对中间状态做 verifier 打分的根本前提。**

**四、性能亮点**

| 指标 | 数据 |
|---|---|
| ImageNet 重建 | 仅 8~128 token 达到 FID < 2，匹配 SOTA 但 token 用量大幅减少 |
| 文生图 | 扩展至 text-conditional 生成，证明框架通用性 |
| 对比 TiTok | 同为"少 token"方案，FlexTok 支持**可变长度**，TiTok 固定长度 |

**五、与 SoTo 的关系**

FlexTok 回答了"如何设计一个好的 tokenizer"，SoTo 回答了"这个好 tokenizer 能带来什么额外能力"：

```
FlexTok (ICML 2025): "我们做出了可变长度的粗到细 1D tokenizer"
          ↓
SoTo (ICML 2026):   "这个 tokenizer 的粗到细结构，恰好让
                     test-time search 变得极其高效"
```

两者来自同一团队，SoTo 是 FlexTok 的自然延伸——发现了已有工具中隐藏的"test-time scaling"潜力。

**六、为什么叫 "Flex"Tok？**

- **Flex**ible length：token 数量 1~256 可自由选择
- **Flex**ible quality-efficiency tradeoff：少 token = 快但粗糙，多 token = 慢但精细
- 相比固定长度的 2D grid，"灵活性"是其最核心的差异化优势
