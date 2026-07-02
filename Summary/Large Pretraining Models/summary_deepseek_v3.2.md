# DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models

- **论文链接**: [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)
- **作者**: DeepSeek-AI
- **发布时间**: 2025年12月
- **关键词**: 稀疏注意力、强化学习后训练、智能体、MoE、高效推理

---

## 1. 核心贡献

DeepSeek-V3.2 旨在缩小开源模型与闭源前沿模型（如 GPT-5、Gemini-3.0-Pro）之间的性能差距。论文提出了三大技术创新：

1. **DeepSeek Sparse Attention (DSA)**：高效的稀疏注意力机制，大幅降低长序列计算复杂度。
2. **可扩展强化学习框架**：稳定的 GRPO 训练框架，后训练计算投入超过预训练成本的 10%。
3. **大规模智能体任务合成流水线**：系统化生成 1800+ 环境和 85000+ 提示，驱动智能体能力泛化。

此外，论文还推出了高计算量变体 **DeepSeek-V3.2-Speciale**，在 IMO 2025、IOI 2025、ICPC WF 2025 和 CMO 2025 中均获得金牌级别的表现。

---

## 2. DeepSeek Sparse Attention (DSA)

### 2.1 核心架构

DSA 是 DeepSeek-V3.2 相比 V3.1-Terminus 唯一的架构变更，通过继续训练引入。DSA 由两个组件构成：

**Lightning Indexer（闪电索引器）**：
- 计算查询 token $\mathbf{h}_t$ 与之前 token $\mathbf{h}_s$ 之间的索引分数 $I_{t,s}$：

$$I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU}(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I)$$

- 索引器头部数 $H^I$ 较小，可实现 FP8 运算，计算成本极低。

**细粒度 Token 选择机制**：
- 根据 Top-k 索引分数，仅检索对应的 KV entries：

$$\mathbf{u}_t = \text{Attn}(\mathbf{h}_t, \{ \mathbf{c}_s \mid I_{t,s} \in \text{Top-k}(I_{t,:}) \})$$

- DSA 基于 MLA（Multi-Head Latent Attention）的 MQA 模式实现，每个 latent vector 在所有 query head 间共享。

### 2.2 继续预训练

分两个阶段：

| 阶段 | 策略 | Token 量 | 学习率 | 关键设置 |
|------|------|----------|--------|----------|
| **Dense Warm-up** | 冻结主模型，仅训练索引器，用 KL 散度对齐索引器输出与主注意力分布 | 2.1B (1000步) | $10^{-3}$ | 冻结所有参数仅训练索引器 |
| **Sparse Training** | 引入 token 选择，优化全部参数；索引器与主模型分离优化 | 943.7B (15000步) | $7.3 \times 10^{-6}$ | 每 query 选 2048 个 KV tokens |

Dense Warm-up 的索引器损失（全局对齐）：

$$\mathcal{L}^{I} = \sum_t D_{KL}(p_{t,:} \| \text{Softmax}(I_{t,:}))$$

Sparse Training 的索引器损失（仅选中的 token 集内对齐）：

$$\mathcal{L}^{I} = \sum_t D_{KL}(p_{t,\mathcal{S}_t} \| \text{Softmax}(I_{t,\mathcal{S}_t}))$$

索引器从计算图中 detach，仅由 $\mathcal{L}^{I}$ 驱动训练；主模型仅由语言建模损失优化。

### 2.3 效率与性能对比

- **复杂度**：从 $O(L^2)$ 降至 $O(Lk)$，其中 $k \ll L$。
- **性能持平**：在标准基准、ChatbotArena Elo 分数、长上下文评估（AA-LCR、Fiction.liveBench）中，V3.2-Exp 与 V3.1-Terminus 性能相当。
- **推理成本**：长上下文场景下实现显著端到端加速。

### 2.4 Lightning Indexer 执行细节

#### 执行步骤
1. **降维投影**：从隐藏状态 $\mathbf{h}_t \in \mathbb{R}^{d}$ 通过投影矩阵 $W_q^I \in \mathbb{R}^{d \times d^I}$ 和 $W_k^I \in \mathbb{R}^{d \times d^I}$ 得到索引器专用的 $\mathbf{q}^{I}_{t, j} \in \mathbb{R}^{d^I}$ 和 $\mathbf{k}^{I}_{s} \in \mathbb{R}^{d^I}$
2. **计算索引分数**：$I_{t, s} = \sum_{j=1}^{H^I} w_{t, j}^I \cdot \text{ReLU}\left(\mathbf{q}^{I}_{t, j} \cdot \mathbf{k}^{I}_{s}\right)$
3. **Top-k 选择**：选择索引分数最高的 $k$ 个 token（V3.2 中 $k = 2048$）

#### 关键设计
- **降维**：索引器维度 $d^I \ll d$（推测 $d^I \approx 16$ 或 $32$，相比主模型 $d_h = 128$）
- **ReLU 激活**：使评分稀疏化，负值被截断为 0
- **FP8 实现**：索引器计算可使用 FP8，进一步降低计算成本
- **Head 数少**：$H^I$ 较小，减少计算量

#### 与 MiniMax Index Branch 的核心区别
| 维度 | DeepSeek-V3.2 DSA | MiniMax MSA |
|------|-------------------|-------------|
| **选择粒度** | Token 级 | 块级（块大小 128） |
| **Head 共享** | 所有 query head 共享单个 Top-k 索引 | 每 GQA 组独立 Top-k |
| **索引器架构** | Multi-head ReLU-based，可学习权重 $w_{t,j}^I$ | 单 head dot-product，无激活函数 |
| **降维** | 是，$d^I \ll d$ | 是，$d_{\text{idx}} \ll d_h$ |
| **本地上下文** | 未明确强调 | 强制包含本地块 |
| **硬件协同设计** | 未详细展开 | 非常详细（Exp-free TopK、KV-outer 等） |

### 2.5 与 MiniMax Sparse Attention (MSA) 的全面对比

#### 架构设计对比
| 方面 | DeepSeek-V3.2 DSA | MiniMax MSA |
|------|-------------------|-------------|
| **基于架构** | MLA (Multi-head Latent Attention) + MQA 模式 | GQA (Grouped Query Attention) |
| **选择粒度** | Token 级（精细） | 块级（连续，硬件友好） |
| **索引器 Head 数** | $H^I$ 较小（具体值未公开） | 每 GQA 组 1 个索引 head |
| **共享策略** | 所有 query head 共享 Top-k 索引 | 每 GQA 组独立 Top-k |
| **评分函数** | $\sum w \cdot \text{ReLU}(q \cdot k)$ | $\frac{q \cdot k}{\sqrt{d_{\text{idx}}}$ |
| **激活函数** | ReLU（稀疏化） | 无（直接 dot-product） |
| **降维** | 是（$d^I \ll d$） | 是（$d_{\text{idx}} \ll d_h$） |

#### 训练策略对比
| 方面 | DeepSeek-V3.2 DSA | MiniMax MSA |
|------|-------------------|-------------|
| **热身阶段** | Dense Warm-up（1000 步，2.1B tokens） | Indexer Warmup（40B tokens） |
| **KL 损失对齐** | 两个阶段：全局对齐 → 仅选定 token 对齐 | 始终在选定 token 上对齐 |
| **梯度截断** | Indexer 输入从计算图 detach | Indexer 输入 + Teacher 分布都 detach |
| **本地块** | 未明确强调 | 强制包含本地块 |
| **稀疏训练** | 15000 步（943.7B tokens） | 剩余预训练（约 3T tokens） |

#### 效率对比
| 方面 | DeepSeek-V3.2 DSA | MiniMax MSA |
|------|-------------------|-------------|
| **理论复杂度** | 主模型 $O(Lk)$，索引器 $O(L^2)$ | 主模型 $O(NkB_k)$，索引器 $O(H_{kv}d_{\text{idx}}N^2)$ |
| **实际加速** | 未明确给出数字 | 14.2× prefill，7.6× decoding（1M 上下文） |
| **硬件优化** | 未详细展开 | 非常详细（Exp-free TopK、KV-outer、LSE 融合等） |
| **开源内核** | 有（inference 实现） | 有（完整推理内核） |

#### 适用场景对比
| 方面 | DeepSeek-V3.2 DSA | MiniMax MSA |
|------|-------------------|-------------|
| **优势** | Token 级精细选择，性能损失小 | 块级连续访问，实际加速比高 |
| **劣势** | Token 级选择可能导致不规则内存访问 | 块级粒度较粗，可能选入无关 token |
| **适用** | 通用稀疏注意力，性能优先 | 超长上下文部署，加速优先 |

### 2.6 Index Branch 如何学习主 Attention 分布

#### 共同目标
在不考虑 MQA/GQA 的情况下，**MSA 和 DSA 的 Index Branch 都在学习主 attention 的分布**。两种方法都使用 KL 散度对齐 Index Branch 和 Main Branch 的分布。

#### 关键区别：如何定义"主 Attention 的结果"

| 维度 | MSA | DSA |
|------|-----|-----|
| **Teacher 分布来源** | Main Branch 的 softmax 输出 | 主模型的注意力分数 |
| **聚合方式** | 在概率级别平均（先 softmax，再平均） | 在分数级别求和（先求和，再 softmax） |
| **对齐粒度** | 块级别（选定的块内的 token） | Token 级别（所有 token） |
| **训练阶段** | 始终在选定的块上对齐 | 两个阶段：全局对齐 → 选定 token 对齐 |

#### 详细公式对比

**MSA**（概率级别平均）：
- Teacher 分布 $P^{(r)}_{i,j}$：对所有 query head 的 Main Branch 分布在概率级别取平均
$$P^{(r)}_{i,j} = \frac{1}{G}\sum_{\ell \in \mathcal{H}_r} \frac{\exp(S^{(\ell)}_{i,j})}{\sum_{u \in \mathcal{I}} \exp(S^{(\ell)}_{i,u})}$$
- KL 损失：$\mathcal{L}_{\text{KL}} = \frac{1}{NH_{kv}} \sum_{i=1}^{N}\sum_{r=1}^{H_{kv}} \text{KL}(P^{(r)}_{i,\cdot} \| P^{\text{idx},(r)}_{i,\cdot})$
- **特点**：对齐的是组级别的注意力模式（块级别）

**DSA**（分数级别求和）：
- Teacher 分布 $p_{t,:}$：对所有 head 的注意力分数在分数级别求和，然后 L1 归一化
$$p_{t,s} = \frac{\sum_{\text{all heads}} S^{(\ell)}_{t,s}}{\sum_{u} \sum_{\text{all heads}} S^{(\ell)}_{t,u}}$$
- KL 损失（Dense Warm-up）：$\mathcal{L}^{I} = \sum_t D_{\text{KL}}(p_{t,:} \| \text{Softmax}(I_{t,:}))$
- **特点**：对齐的是所有 head 共享的注意力模式（token 级别）

#### 为什么这样设计？

**MSA 的设计考量**：
1. **块级别选择**：Indexer 选择块，Main Branch 在块内计算精确注意力
2. **组级别对齐**：由于块是共享的，对齐组级别的分布更合理
3. **训练稳定性**：仅在选定的块上计算 KL 损失，避免对齐无关 token

**DSA 的设计考量**：
1. **Token 级别选择**：更精细的选择，需要对齐所有 token
2. **分数级别求和**：所有 head 的注意力模式可能不同，求和可以捕获"至少一个人头关注的 token"
3. **两阶段训练**：先全局对齐（学习大致模式），再稀疏对齐（适应稀疏选择）

#### 梯度截断：单向对齐

两种方法都使用**梯度截断**，使得 Index Branch 的梯度不会影响主模型：

- **MSA**：Index Branch 输入 $X$ 被 detach，Teacher 分布 $P$ 也被 detach
- **DSA**：Indexer 输入从计算图 detach（"detach the indexer input from the computational graph"）

**这意味着**：
- Index Branch 通过 $\mathcal{L}_{\text{KL}}$ 学习主 attention 分布
- 但主模型**不通过** Index Branch 的梯度更新
- 主模型只通过语言建模损失 $\mathcal{L}_{\text{LM}}$ 更新

这是一个**单向对齐**的设计：Indexer 适配主模型，但主模型不改变以适应 Indexer。

#### 总结

> 在不考虑 MQA 或者 GQA 下，MSA 和 DSA 的 Index，都是在学习主 attention 的结果吗？

**是的，但学习方式略有不同**：

1. **MSA**：学习"Main Branch 的注意力分布是什么"（块级别）
2. **DSA**：学习"主模型的注意力分数分布是什么"（token 级别）

**本质相同**：都是让 Index Branch 的预测分布尽可能接近主模型的注意力分布。

**实现不同**：
- MSA 用 $\text{KL}(P_{\text{main}} \| P_{\text{idx}})$，其中 $P_{\text{main}}$ 是组平均的 softmax 分布
- DSA 用 $\text{KL}(p_{\text{main}} \| \text{Softmax}(I))$，其中 $p_{\text{main}}$ 是所有 head 求和的 L1 归一化分布

---

## 3. 后训练策略

### 3.1 专家蒸馏 + 混合 RL 训练

流水线包含：
- **专家蒸馏**：针对 6 个领域分别训练专家模型（数学、编程、通用逻辑推理、通用智能体、智能体编程、智能体搜索），支持 thinking/non-thinking 两种模式。
- **混合 RL 训练**：将推理、智能体和人类对齐数据合并到一个 RL 阶段中，采用 GRPO 算法。

### 3.2 GRPO 稳定性扩展

四项目标稳定性技术：

**① 无偏 KL 估计**
- 修正 K3 估计器，使用重要性采样比获得无偏 KL 梯度估计：

$$D_{KL}(\pi_\theta \| \pi_{ref}) = \frac{\pi_\theta}{\pi_{old}} \left( \frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1 \right)$$

- 解决 $\pi_\theta \ll \pi_{ref}$ 时梯度噪声过大的问题。

**② Off-Policy Sequence Masking**
- 对产生较大策略偏移（KL 散度超过阈值 $\delta$）的负优势序列应用掩码，将其从梯度更新中排除：
$$M_{i,t} = \begin{cases} 0 & \hat{A}_{i,t} < 0 \text{ 且 } \frac{1}{|o_i|}\sum_{t}\log\frac{\pi_{old}}{\pi_{\theta}} > \delta \\ 1 & \text{otherwise} \end{cases}$$

**③ Keep Routing**
- MoE 模型在推理和训练框架间可能存在路由不一致，训练时保存并强制使用推理阶段采样的专家路由路径。

**④ Keep Sampling Mask**
- 保留推理时的 top-p/top-k 截断掩码并在训练时应用，保持 $\pi_{old}$ 和 $\pi_\theta$ 的动作空间一致。

### 3.3 工具使用中的思考（Thinking in Tool-Use）

**上下文管理**：
- 仅在新用户消息出现时丢弃历史推理内容；工具调用之间的推理内容保留。保留工具调用及结果的历史。

**冷启动 (Cold-Start)**：
- 通过精心设计的系统提示，将推理模板和智能体工具调用模板融合，使模型能在 `<think></think>` 中执行工具调用。

**大规模智能体任务合成**：

| 任务类型 | 任务数量 | 环境类型 | 提示来源 |
|----------|----------|----------|----------|
| Code Agent | 24,667 | 真实（GitHub Issues） | 提取 |
| Search Agent | 50,275 | 真实（搜索 API） | 合成 |
| General Agent | 4,417 | 合成 | 合成 |
| Code Interpreter | 5,908 | 真实（Jupyter） | 提取 |

关键合成方法：
- **Search Agent**：多智能体流水线（采样长尾实体 → 构建 QA → 多候选生成 → 验证过滤）
- **Code Agent**：从 GitHub 挖掘百万 issue-PR 对，自动搭建可执行环境，支持 Python/Java/JS/TS/C/C++/Go/PHP
- **General Agent**：自动合成 1,827 个任务导向环境（环境+工具集+任务+验证器），采用"难求解但易验证"原则

---

## 4. 主要实验结果

### 4.1 综合基准测试

| 基准 | GPT-5 High | Gemini-3.0 Pro | Kimi-K2 Thinking | DeepSeek-V3.2 Thinking |
|------|:---:|:---:|:---:|:---:|
| MMLU-Pro | 87.5 | **90.1** | 84.6 | **85.0** |
| GPQA Diamond | 85.7 | **91.9** | **84.5** | 82.4 |
| HLE | 26.3 | **37.7** | 23.9 | **25.1** |
| LiveCodeBench | 84.5 | **90.7** | 82.6 | **83.3** |
| AIME 2025 | 94.6 | **95.0** | **94.5** | 93.1 |
| HMMT Feb 2025 | 88.3 | **97.5** | 89.4 | **92.5** |
| SWE Verified | **77.2** | 76.2 | 71.3 | **73.1** |
| SWE Multilingual | **68.0** | - | 61.1 | **70.2** |
| Terminal Bench 2.0 | 35.2 | **54.2** | 35.7 | **46.4** |
| BrowseComp | **54.9** | - | 60.2 | **51.4/67.6*** |
| MCP-Universe | 47.9 | **50.7** | 35.6 | **45.9** |
| MCP-Mark | **50.9** | 43.1 | 20.4 | **38.0** |
| $\tau^2$-Bench | 80.2 | **85.4** | 74.3 | **80.3** |

> 开源模型中的最佳结果用粗体标出。

### 4.2 DeepSeek-V3.2-Speciale 竞赛成绩

| 竞赛 | 总体成绩 | 奖牌 |
|------|----------|------|
| IMO 2025 | 35/42 | 🥇 金牌 |
| CMO 2025 | 102/126 | 🥇 金牌 |
| IOI 2025 | 492/600 | 🥇 金牌 |
| ICPC WF 2025 | 10/12 题 | 🥇 金牌 |

Speciale 在推理基准上与 Gemini-3.0-Pro 持平或超越：

| 基准 | Gemini-3.0 Pro | V3.2 Speciale |
|------|:---:|:---:|
| AIME 2025 | 95.0 | **96.0** |
| HMMT Feb 2025 | 97.5 | **99.2** |
| HMMT Nov 2025 | 93.3 | **94.4** |
| IMOAnswerBench | 83.3 | **84.5** |

### 4.3 合成数据的泛化能力

- 仅用合成 General Agent 数据做 RL 训练，模型在 $\tau^2$-Bench、MCP-Mark、MCP-Universe 上显著超越只做 Code+Search RL 训练的版本。
- 合成任务具有足够难度（V3.2-Exp 仅 12%，GPT-5 Thinking 达 62%）。

---

## 5. 搜索智能体的上下文管理

为突破 128K 上下文限制，提出三种测试时上下文管理策略：

| 策略 | 描述 | BrowseComp 得分 | 平均步数 |
|------|------|:---:|:---:|
| **Summary** | 总结溢出轨迹后重启 | 60.2 | 364 |
| **Discard-75%** | 丢弃前 75% 的工具调用历史 | - | - |
| **Discard-all** | 丢弃所有历史工具调用（类似新上下文工具） | **67.6** | 较少 |
| **Parallel-fewest-step** | 并行采样 N 条轨迹，选最短的 | 对比基线 | - |

Discard-all 以最低效率和良好的可扩展性实现了与并行扩展相当的性能。

---

## 6. 局限性与未来工作

1. **世界知识广度不足**：总训练 FLOPs 少于闭源前沿模型（如 Gemini），计划通过扩展预训练计算来弥补。
2. **Token 效率不足**：需要更长生成轨迹才能匹敌 Gemini-3.0-Pro 的输出质量。
3. **复杂任务仍有差距**：在极具挑战性的任务上仍落后于前沿模型。
4. **V3.2 的推理过程存在冗余自验证**：导致上下文长度超出 128K 限制。
5. 未来将优化推理链的**智能密度**，并继续改进基础模型和后训练配方。

---

## 7. 总结

DeepSeek-V3.2 是开源 LLM 领域的重要进步：通过 DSA 稀疏注意力实现高效长序列处理；通过可扩展 GRPO 训练框架（无偏 KL 估计、off-policy 序列掩码、keep routing、keep sampling mask）稳定扩展 RL 计算；通过大规模智能体任务合成显著提升工具使用泛化能力。其高计算量变体 V3.2-Speciale 在多个国际顶级竞赛中获金牌，标志着开源模型在推理领域达到了闭源前沿水平。
