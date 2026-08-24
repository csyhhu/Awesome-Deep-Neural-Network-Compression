# Kwai Keye-VL-2.0-30B-A3B 技术报告 论文总结

> **论文标题**：Kwai Keye-VL-2.0 Technical Report
> **作者团队**：快手 Keye Team（核心贡献者 21 人 + 主要/活跃/支持贡献者若干）
> **arXiv ID**：2606.10651
> **模型规模**：30B 总参数 MoE，仅 3B 激活参数
> **核心定位**：开源多模态 MoE 基础模型，面向长视频理解与智能体（Agentic Intelligence）

---

## 0. 综合理解
本文是Multimodal LLMs的经典工作，介绍了视频、文本输入，文本输出的训练框架，流程。

## 1. 研究背景与核心问题

在将多模态大模型从短视觉感知推进到长程智能体推理时，存在**两大核心瓶颈**：

1. **基础设施瓶颈**：扩展到小时级视频（256K 上下文）会带来 KV Cache 灾难性扩张和计算开销爆炸，传统稠密注意力迫使模型牺牲时间连续性进行激进抽帧。
2. **算法困境（多模态对齐困境）**：在注入复杂视频理解与工具使用能力时，容易引发**灾难性遗忘**，损害模型原有的 STEM、数学、语言推理能力。

Keye-VL-2.0 通过**架构创新**与**对齐创新**两个维度协同解决上述问题。

---

## 2. 模型架构（四大核心组件）

模型遵循标准 MLLM 范式，包含四个核心组件：

| 组件 | 设计 |
|------|------|
| **视觉编码器 (ViT)** | 继承自 Keye-VL-1.5-8B 的 SigLIP-400M-384-14，支持原生分辨率 |
| **语言解码器 (LLM)** | 基于 Qwen3-30B-A3B-Thinking-2507 |
| **MLP 投影器** | 随机初始化，在 Stage 0 训练以对齐视觉-语言表征空间 |
| **稀疏注意力模块** | GQA 兼容的 DSA 设计：全局 MQA 索引 + 分组 GQA 聚合 |

### 2.1 原生分辨率视觉编码器（Native-Resolution ViT）

- 摒弃动态切块（dynamic tiling）方法（如 InternVL3、MiniCPM-V），采用 NaViT 风格的原生分辨率建模，保留原始宽高比与全局结构。
- **自适应位置编码**：对固定绝对可学习位置嵌入进行插值以适应可变分辨率。
- **2D RoPE**：增强空间建模与高分辨率图像的外推能力。
- **Sequence Packing**：结合 NaViT 的 Patch n' Pack 机制与 FlashAttention，无 padding 浪费。
- **分布对齐 ViT 预训练**：在 SigLIP 损失下训练，使用与下游 MLLM 相同的数据分布（500B tokens，含 DataComp、LAION、CC12M、PD12M、COCO 等）。

### 2.2 统一视觉编码（Unified Visual Encoding）

- **动态分辨率图像编码**：图像直接由动态分辨率 ViT 编码，token 数按原始像素分配。
- **动态分辨率视频编码**：每帧视为独立高分辨率图像；**关键创新**——在每帧视觉 token 前置自然语言时间戳，为 LLM 提供时间锚点。
- **自适应视频像素预算**：按时长分级（256s/512s/1024s/2048s 对应缩放因子 0.125/0.25/0.5/1.0），压缩冗余短视频，保留长视频视觉证据。

### 2.3 面向长上下文多模态建模的 DSA（核心创新之一）

将 **DeepSeek Sparse Attention (DSA)** 集成到 GQA-based MLLM 主干（区别于以往基于 MLA 的 DSA 系统）。

> ** 读者理解与点评**：
> **读者原话**："indexer采用MQA，所有Query共享一个kv, 执行index并找出该Query下需要计算的token. 然后实际计算时使用GQA，这个Query对选中的Token的多个KV计算Attention结果"
>
> **点评（部分正确，需修正两点）**：
> - ✅ **前半句正确**：indexer 采用 MQA 风格，所有 query head 共享**一个 key**（注意是 key，不是 kv；indexer 主要算 key 用于评分，不涉及 value）。
> - ⚠️ **"多个 KV"说法需修正**：GQA 不是"一个 query 对多个 KV"，而是"query head 分成 G 组，每组内共享同一组 KV，不同组有不同 KV"。Top-k 集合 $\Omega_t$ 是所有组共用的，但每个组在该集合上做的是各自的 KV 计算。
> - **正确描述应为**：indexer 用 MQA 共享 key 算分找出 Top-k token 集合 $\Omega_t$；实际计算时回到 GQA，所有组复用 $\Omega_t$，但每个组 $g$ 用自己的 KV $c_{s,g}$ 在该集合上算 attention。

#### (1) MQA 风格的 Lightning Indexer
全局索引评分：

$$I_{t,s} = \sum_{j=1}^{H^{I}} w_{t,j}^{I} \cdot \mathrm{ReLU}(q_{t,j}^{I} \cdot k_{s}^{I})$$

通过共享 key head 大幅降低计算与显存开销，配合 FP8 实现和 ReLU 评分函数，在数十万 token 序列上保持高效。Top-k 形成 sparse index set $\Omega_t$。

#### (2) GQA 稀疏聚合
对 GQA 第 $g$ 组应用同一稀疏索引集 $\Omega_t$：

$$u_{t,g} = \mathrm{Attn}(h_{t,g}, \{c_{s,g} \mid s \in \Omega_t\})$$

设 $k=2048$，将 $O(L^2)$ 降到 $O(Lk)$，保留 GQA 表征结构同时避免全上下文稠密注意力。

#### (3) 两阶段训练（Dense Warm-up + Sparse Adaptation）
- **稠密预热**（约 2B 多模态 token）：冻结主模型，仅初始化 indexer，通过 KL 损失让其覆盖所有 GQA 组的注意力分布：

$$\mathcal{L}_{\mathrm{warmup}}^{I} = \sum_t \sum_{g=1}^{G} \mathbb{D}_{KL}(p_{t,:,g} \parallel \mathrm{Softmax}(I_{t,:}))$$

- **稀疏适配**：解冻全部参数切换至稀疏模式，KL 仅在 Top-k 集合上计算；indexer 输入从计算图 detach 以减少梯度干扰：

$$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{NTP}} + \lambda \mathcal{L}_{\mathrm{sparse}}^{I}$$

---

## 3. 四阶段预训练课程（Pre-Training）

| 阶段 | 上下文长度 | 数据规模 | 核心目标 |
|------|-----------|----------|---------|
| **Stage 0** | - | - | 仅训练 Projector，对齐视觉-语言表征（ViT/LLM 冻结） |
| **Stage 1** | 32K | ~1T tokens | 全参数预训练，建立稳定视觉-语言对齐、图像感知、视频理解、OCR |
| **Stage 2** | 64K | ~2T tokens | 多任务能力注入（OCR、Math/STEM、Caption、GUI、Grounding/Counting、QA、电商、中文扩展） |
| **Stage 3** | 256K | - | 长上下文扩展（长视频/长文档/多文档/长程智能体轨迹）；长短样本 1:1 混合 |

**视频训练课程**：Stage 1（15s, 24K tokens）→ Stage 2（15min, 64K tokens, 引入 TVG 数据）→ Stage 3（2h, 180K tokens）。
- **Scene-Wise Dense Caption**：将 dense video captioning 重构为带时间戳的结构化场景描述。
- **Diverse TVG Data**：受 ETBench 启发，覆盖 Referred Action Recognition、Video Highlight Detection、Extractive Video Summarization、Temporal Event Matching。

**数据清洗**：Hash + CLIP 联合去重；双队列异步机制将 CPU 预处理与 GPU 推理解耦，生产吞吐提升 3–5×。

---

## 4. 后训练（Post-Training）

### 4.1 SFT 与 Synthetic CoT
- SFT 语料约 500B tokens，40% 为纯文本以保留指令跟随与文本推理能力。
- 多模态指令混合：Video、Perception、Reasoning、Agent、Long-context 互补发展。
- **Synthetic CoT**：从高质量 QA 构造推理轨迹，经 query/response/process 三级质检；数学任务用 Doubt2Clean 二次审查清洗 27 个数据集。
- 视频任务采用 `\think` 阶段验证候选时间区间，输出格式 `[[mm, mm], ...]`。

### 4.2 强化学习（RL）

#### (a) Synthetic-Data RL
基于程序生成的图像差异识别任务：两图差异由可控编辑 $\mathcal{E}$ 生成，奖励通过规则化验证（无需学习型奖励模型）。
- 感知任务：匈牙利匹配 + IoU 奖励 $R_{\mathrm{perc}}$。
- 结构化任务：几何/坐标几何/化学公式/电路图 DSL，规则匹配 + 软距离相似度。
- 引入 **difference-irrelevant re-rendering perturbations**（颜色抖动、布局扰动、slot shuffling、semantic no-op、视角变化）防止模型依赖像素级差异。

#### (b) General RL
使用 **GSPO (Group Sequence Policy Optimization)** 算法，目标函数：

$$\mathcal{J}_{\mathrm{GSPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(s_i(\theta)\hat{A}_i, \mathrm{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\right)\right]$$

其中 $s_i(\theta)$ 是序列级重要性采样比率（序列长度归一化的 log-ratio 指数）。奖励系统包含：Format Reward、Outcome Reward、Process Reward、**ContextRL Reward**（与已验证参考解对比，减少"答案对但推理错"的假阳性）。

#### (c) Specialized RL（5 个领域专家）
均从 General RL 检查点出发，用于后续蒸馏：
- **Grounding Expert**：Hungarian 匹配 + 最小/平均 IoU + 重复框惩罚。
- **Spatial Expert**：生成式模型 judge 评分 $\{-1,0,1\}$。
- **Math Expert**：符号等价奖励 + 格式门控。
- **Counting Expert**：精确数值匹配。
- **OCR Expert**：归一化文本匹配（大小写/空格/标点归一化）。

#### (d) Video RL
约 31K 视频样本，冻结 ViT 与 Projector，GSPO 优化。覆盖 TVG（TimeIT, DiDeMo, Charades-STA）、temporal dense captioning（LLM-as-Judge）、FrameForge 合成视频（时间戳定位/计数/前后推理）。使通用视频 benchmark 性能提升约 1 个百分点。

#### (e) Agentic RL（核心创新之二）
共享训练协议：环境接地奖励 + 轨迹级验证 + **colocated partial-rollout 机制**（未完成轨迹缓存并在后续 rollout 步骤中恢复，完成组立即用于 GSPO 更新）。
- **Coding RL**：Online Judge（编译+隐藏测试） + Software Engineering（Docker 仓库级 issue 解决，多 reviewer agent + integration agent 验证补丁）。
- **Tool Use RL**：150+ 模拟 API 域，工具与参数名随机化避免记忆依赖。
- **Search RL**：多跳检索 + 证据验证 + 答案合成。

### 4.3 Cross-Modal Multi-Teacher On-Policy Distillation (MOPD)（核心创新之二）

为解决多任务能力冲突（如推理 RL 后响应过短、Agent 训练后工具调用格式过重）：
- **13 个 RL 训练的领域教师**（safety、纯文本数学、指令跟随、code、visual STEM、OCR、grounding、counting、video、tool use 等）。
- 每个样本动态路由到最匹配的教师。
- **SPRR (Segmented Prompt-Response Re-tokenization)**：分别处理 prompt 与 response，确保教师 log-prob 与学生 response token 严格对齐。
- **Top-k Overlap 优势估计**：

$$\Omega_{i,t} = \mathrm{TopK}(\pi_{\mathrm{T}}^{r(i)}(\cdot \mid s_{i,t})) \cap \mathrm{TopK}(\pi_{\theta}(\cdot \mid s_{i,t}))$$

仅在被师生双方都认为合理的局部分布上计算，避免极低概率 token 的不稳定比较。
- Token 类别感知优势缩放（down-weight 格式 token，up-weight 感知/推理 token）。
- 局部化重复惩罚（仅在崩塌点 $\tau_i$ 之后施加）。

---

## 5. 高效训练与推理基础设施

### 5.1 预训练系统
- **ExtraIO**：水平可扩展的 I/O 服务，与训练异步解耦。
- **ViT-LM 异构并行**：ViT 与 LM 共置于同一 GPU 组但各自采用独立分片策略，避免将 ViT 绑定到 LM PP0 导致的不均衡；recompute-or-offload 策略将 ViT 激活显存降至近零。
- **两级负载均衡**：多模态 token 级 + LM 样本级，端到端吞吐提升约 **20%**。
- **DSA 变长序列优化**：基于 FlashInfer 与 TileLang，相对开源基线 **2×** 加速。
  - Top-k 显存优化：score 存储从 $T \times T$ 降到 $T \times \text{max\_seq}$，用 `flashinfer.top_k_ragged_transform`。
  - 短序列优化：当 $i < \text{top}k$ 时仅迭代因果可注意 KV，**1.5×** 加速。
  - Indexer Loss：在 sparse-attention backward kernel 内恢复 attention score，复用 FlashAttention 风格重计算。

### 5.2 后训练系统
- **RL 中 DSA 适配**：确定性 Top-k（`flashinfer.topk` 替代 `torch.topk`，**2–3×** 加速且保持确定性）；分块 DSA indexer 降低峰值显存。
- **OPD 系统**：异构多专家教师调度、多模态对齐验证（图像 token 数、视频帧采样、chat template、mRoPE 位置严格对齐）、三种 Top-k 蒸馏模式（Overlap/Student-Only/Teacher-on-Student）。

### 5.3 GQA+DSA 高效推理
- **Chunk ViT**：视频帧分块顺序处理后合并，降低峰值显存且不改变输出。
- **稀疏注意力优化**：相邻 query 去重 Top-k KV 集合 + MMA Thread Layout-Aware Mask；128K 上下文、topk=2048 时，16 个相邻 query 仅需约 8K 有效 KV token。
- **Decode 优化**：128K 上下文下 prefill 成本降低 **3×** 以上，decode 成本降低 **5×** 以上（H800 平台）。

---

## 6. 综合评估

### 6.1 视频理解（关键亮点）
| Benchmark | Keye-VL-2.0 | Qwen3.5-35B-A3B | Qwen3-VL-235B-A22B | GPT-5-mini |
|-----------|-------------|------------------|----------------------|------------|
| LongVideoBench | **74.1** | 61.6 | 70.5 | -- |
| Video-MME-v2 ACC (64/512) | **35.3 / 42.4** | 32.6 / 28.5 | 33.3 / 36.8 | -- |
| ActivityNet-TimeLens (mIoU) | **58.5** | 53.2 | 52.1 | -- |
| QVHighlights-TimeLens (mIoU) | **70.1** | 65.7 | 64.6 | -- |
| Charades-TimeLens (mIoU) | **58.4** | 49.1 | 47.8 | -- |
| MLVU | 82.8 | 85.6 | 83.8 | 83.3 |
| Video-MMMU | 80.0 | 80.4 | 80.0 | **82.5** |

**亮点**：在 LongVideoBench、Video-MME-v2 准确率、三个 TimeLens 子集上均取得最佳；验证了 scene-wise dense caption、多样化 TVG 数据、tIoU 中心 Video RL 的有效性。

### 6.2 Agentic 能力
- **Code Agent**：LiveCodeBench v6 **64.2**（最佳）、OJBench **71.5**（最佳）、SWE-bench Verified 62.0（次佳，仅次于 Qwen3.5 63.5）。
- **Tool Use**：τ²-Bench **82.6**（最佳）、VitaBench **33.1**（最佳）、BFCL-V4 65.7（次佳）。

### 6.3 通用 Vision-Language
- **HallusionBench**：**73.5**（最佳，体现强抗幻觉能力）。
- **WeMath**：**75.5**（最佳）。
- 在 OCRBench、MMMU、MMStar 等基准上保持竞争力（部分次佳于 Qwen3.5-35B-A3B）。

---

## 7. 关键贡献总结

1. **首个将 DSA 适配到 GQA-based 多模态架构**，实现 256K 无损长上下文处理。
2. **Cross-Modal MOPD**：通过动态路由 + Top-k Overlap 蒸馏 + On-Policy rollouts，将异构 RL 教师能力无缝融合到统一 MoE 主干，解决多任务对齐困境。
3. **完整工程化栈**：ExtraIO、ViT-LM 异构并行、FlashInfer/TileLang DSA kernel、Chunk ViT 推理，使小时级视频理解在实际部署中可行。
4. **Context-RL + Video-RL**：bucket advantage scaling 稳定长序列决策树，系统性减少视觉幻觉。
5. **开源模型 checkpoint**，推动社区向可扩展、鲁棒的多模态智能体应用发展。

---

## 8. 未来工作

1. **业务深度落地**：将细粒度长视频感知与图像文本理解集成到生成式推荐、内容生态治理、商业定向等产品管线。
2. **Video × Agent 工作流**：从被动理解视频转向主动协调生产闭环（大规模视频检索、亮点片段提取、自动剪辑包装、营销文案生成）。
3. **基础设施持续强化**：DSA 计算优化、可扩展数据飞轮、Context-RL 后训练，向原生多模态建模与端到端深度融合演进。

---

## 9. 讨论记录（Q&A）

### Q1：共享 key head 是 MQA 的定义吧？Indexer 一般是用一个低维度的投影进行快速的 Approximate Attention 计算，检索出临近的 token，再进行高维度的原本计算？

**A1**：理解框架基本正确，但需补充三点关键细节。

- **MQA 定义确认**：Shazeer 2019 提出，所有 query head 共享同一个 key/value head。本文 indexer 中 $k_s^I$ 无下标 $j$（而 $q_{t,j}^I$ 有），证实是 MQA 风格。

- **比"低维投影"更精细**：本文 indexer 是多 head 加权和，而非单投影：
  $$I_{t,s} = \sum_{j=1}^{H^{I}} w_{t,j}^{I} \cdot \mathrm{ReLU}(q_{t,j}^{I} \cdot k_{s}^{I})$$
  有 $H^I$ 个 indexer head，每个用 $w_{t,j}^I$ 加权，是"低秩多视角"检索，表达力强于单投影。

- **ReLU 而非 Softmax**：避免长序列数值不稳定，配合 FP8 实现更友好，评分可加和便于跨 head 聚合。

- **检索"注意力临近"而非"位置临近"**：Top-k 选的是与 query $h_t$ 语义相关的 token，不是位置接近的 token。这是 sparse attention 与 sliding window attention 的本质区别。

- **后续高维计算保留 GQA 结构**：Top-k 集合 $\Omega_t$ 复用到所有 GQA 组 $u_{t,g} = \mathrm{Attn}(h_{t,g}, \{c_{s,g} \mid s \in \Omega_t\})$，其中 $c_{s,g}$ 带下标 $g$ 说明回到 GQA 正常高维计算，不是全 MHA，也不是 indexer 的 MQA。

---

### Q2：本文都已经使用了 MQA，为什么还会提及 GQA？

**A2**：本文并非"全用 MQA"，而是"indexer 用 MQA，主干用 GQA"，两者职责不同。

- **背景澄清**：本文 LLM 主干是 Qwen3-30B-A3B-Thinking-2507，原生基于 GQA。DSA 是在 GQA 主干之上添加的稀疏检索模块。

- **两阶段分工**：

| 阶段 | 机制 | 目的 | 参数量 |
|------|------|------|--------|
| Indexer（检索） | MQA 风格（共享 key） | 轻量快速找 Top-k | 小（低秩） |
| Aggregation（聚合） | GQA（分组 KV） | 高质量注意力计算 | 大（保留主干） |

- **为何不全用 MQA**：
  1. **质量考虑**：MQA 的 KV cache 小但模型质量略低于 GQA（Ainslie 2023 结论）。Qwen3 主干已用 GQA 训好，强行换 MQA 会破坏表征。
  2. **职责分离**：indexer 只需"大致对"的 Top-k，不需要细粒度表征；真正注意力计算需要 GQA 的多组 KV 保留主干表征能力。
  3. **Top-k 集合复用**：indexer 算一次 $\Omega_t$，所有 GQA 组复用，避免每组各自检索。

- **类比**：像数据库查询——先用轻量级索引（MQA indexer，类似 B-tree）快速定位相关行，再用完整行数据（GQA KV）做精细计算。

- **与纯 MLA-based DSA 的区别**：DeepSeek 原版 DSA 基于 MLA（本身有低秩压缩）；本文要在 GQA 主干上加 DSA，需单独设计 MQA-style indexer 承担"低秩检索"角色，这是本文的工程贡献。

---

### Q3：介绍整体结构，如 VAE/Text Encoder -> DiT，之后输出如何与 LLM 交互，最终输出什么？

**A3**：需要重要澄清——Keye-VL-2.0 不是 Diffusion 模型，没有 VAE 和 DiT，它是 Multimodal LLM (MLLM)，采用自回归生成。

- **结构澄清**："VAE → Text Encoder → DiT" 是 Diffusion 模型（如 SD3、Sora）的结构。Keye-VL-2.0 完全不同，是理解型而非生成型模型。

- **实际结构**：

```
图像/视频帧
    ↓
ViT (Vision Encoder, SigLIP-based)
  - 原生分辨率（无固定 384×384 裁剪）
  - 2D RoPE + 自适应位置编码
  - Sequence Packing (NaViT Patch n' Pack)
    ↓ 视觉 token 序列
    ↓ （视频帧前插入自然语言时间戳文本）
    ↓
MLP Projector (2×2 patch-merge)
  - Stage 0 单独训练，对齐视觉到 LLM 表征空间
    ↓ 对齐后的视觉 token
    ↓
LLM (Qwen3-30B-A3B MoE, GQA 主干)
  + DSA 稀疏注意力模块
    ↓ 与文本 token 拼接后自回归处理
    ↓
输出：autoregressive text tokens
  （含 <think> 推理过程 + <answer> 最终答案）
```

- **与 LLM 交互方式**：
  1. **视觉 token 与文本 token 序列维度拼接**：不是 cross-attention，而是直接拼成长序列送入 LLM，如 `[视频帧1 token, 时间戳文本, 视频帧2 token, 时间戳文本, ..., 用户问题文本]`。
  2. **时间戳是文本形式**：每帧视觉 token 前置自然语言时间戳（如 "00:15:32"），让 LLM 在原生语言空间感知时间——把时间信息"翻译"给 LLM，而非设计专门时间编码。
  3. **输出格式**：thinking-oriented policy，输出形如：
     ```
     <think>验证候选时间区间 [00:15, 00:18]...</think>
     <answer>最终答案 + [[mm, mm], ...] 支持区间</answer>
     ```

- **与 Diffusion 模型对比**：

| 维度 | Diffusion (VAE+DiT) | MLLM (ViT+LLM) |
|------|---------------------|----------------|
| 输入 | 噪声 + 条件 | 视觉 token + 文本 token |
| 主干 | DiT 去噪迭代 | LLM 自回归生成 |
| 输出 | 像素/latent | 文本 token |
| 训练 | 噪声预测 loss | NTP loss + RL |
| 任务 | 图像/视频生成 | 理解、问答、推理 |

---

### Q4：RL/SFT/Pretraining 有什么区别？与一般 LLM 训练范式是否类似？

**A4**：整体范式类似（Pretraining → SFT → RL 三阶段），但多模态场景下每一阶段都有显著特化。

- **与一般 LLM 范式对照**：

| 阶段 | 一般 LLM 范式 | Keye-VL-2.0 范式 | 关键差异 |
|------|--------------|------------------|---------|
| Pretraining | NTP 学语言/世界知识 | 4 阶段课程（Stage 0-3） | Stage 0 Projector 初始化（多模态特有）；上下文分阶段扩展（32K→64K→256K） |
| SFT | 指令跟随 | 指令跟随 + Synthetic CoT | 40% 纯文本保留语言能力；多模态指令混合 |
| RL | RLHF 对齐人类偏好 | 6 类 RL + MOPD 蒸馏 | 5 个 specialized experts；MOPD 多教师整合 |

- **各阶段细节**：
  1. **Pretraining 多了两个子目标**：
     - Stage 0：仅训 MLP Projector 建立视觉→LLM 表征映射（LLM 无此步骤）
     - Stage 3：上下文 64K→256K 扩展，长短样本 1:1 混合以保常规输入性能

  2. **SFT 平衡显式推理 vs 简洁回答**：
     - Synthetic CoT：强教师生成推理轨迹 + query/response/process 三级过滤 + 数学任务 Doubt2Clean 二次审查
     - 长 think 推理数据 + 直接答案数据混合，避免简单任务啰嗦

  3. **RL 不仅学特殊领域，更承担能力整合角色**（与 LLM RLHF 最大差异）：

```
General RL checkpoint
    ↓ 分支训练
    ├── Grounding Expert ─┐
    ├── Spatial Expert    │
    ├── Math Expert       │── MOPD 蒸馏 → 主模型
    ├── Counting Expert   │
    └── OCR Expert       ─┘
```
     - Specialized RL 训练的 5 个专家不直接上线，而是作为 MOPD 教师把能力蒸馏回主模型
     - "先分后合"解决多任务冲突（同时训好 OCR+数学+视频定位难，分别训好再蒸馏更容易）

- **对"RL 学特殊领域知识"理解的修正**：
  - ✅ 对：Specialized RL 学特殊领域（Grounding/Spatial/Math/Counting/OCR）
  - ⚠️ 补充：RL 还承担**能力整合**职责（通过 MOPD 把多专家蒸馏回主模型）
  - ⚠️ 补充：Agentic RL 学的是**环境交互能力**（Code/Tool/Search），是"行为"而非"知识"

---

### Q5：MLP Projector 起什么作用？为什么需要额外 Projector 建立视觉→LLM 映射？ViT 做不到这点吗？Stage 0 如何训练（如何确定 label）？

**A5**：这是 MLLM 架构的经典问题，本文采用 LLaVA 系列的"ViT + Projector + LLM"三段式设计。

#### (1) Projector 的作用：跨表征空间"翻译器"

ViT 与 LLM 是**独立预训练**的，它们的表征空间完全不兼容：

| 模块 | 训练目标 | 表征空间特性 |
|------|---------|------------|
| **ViT (SigLIP)** | 对比学习（图像 vs 文本对比） | 对比性视觉表征，强调"区分不同图像" |
| **LLM (Qwen3)** | NTP（下一 token 预测） | 语言空间，强调"生成连贯文本" |

两者表征空间在维度、分布、几何结构上都不同。直接把 ViT 特征喂给 LLM，相当于让一个只懂中文的人直接读英文——信息虽在但无法解码。**Projector 是可学习的变换矩阵（通常是一个 MLP），把 ViT 特征"翻译"到 LLM 能理解的语言空间**。

#### (2) 为什么 ViT 自己做不到？

理论上 ViT 末端可以加一个 head 直接输出 LLM 兼容特征，但实际有几个问题：

1. **保留 ViT 预训练能力**：ViT 已经在 500B tokens 上训好（SigLIP 对比学习），重新训练或修改末端会破坏已有视觉表征。保持 ViT 冻结 + 单独训 Projector 是最低风险方案。

2. **训练目标不一致**：ViT 的训练目标是 SigLIP 对比损失，与 LLM 的 NTP 目标差距大。如果让 ViT 直接输出 NTP 兼容特征，需要修改 ViT 训练目标——但这又破坏对比学习能力。

3. **维度匹配问题**：ViT 输出维度（如 SigLIP 的 1152）与 LLM 输入维度（如 Qwen3 的 hidden_size）通常不同，需要一个投影层做维度变换。这个投影层本身就是 Projector 的一部分。

4. **本文的 2×2 patch-merge 设计**：Projector 还承担了 **token 数压缩**功能——通过 2×2 patch 合并，把 4 个 patch token 合成 1 个，将视觉 token 数减少 4 倍。这是 ViT 本身做不到的"后处理"。

**结论**：ViT 不是"做不到"，而是"不该做"。保持 ViT 专注视觉表征、Projector 专注跨空间映射，是职责分离的最优解。

#### (3) Stage 0 如何训练？Label 是什么？

Stage 0 训练目标非常简单——**NTP loss（next-token prediction）**，但只更新 Projector 参数：

```
输入: [图像] + [描述文本 "A cat sitting on a mat"]
              ↑ caption 作为条件      ↑ caption 作为预测目标

流程:
1. 图像 → 冻结的 ViT → 视觉特征
2. 视觉特征 → 可训练的 Projector → 对齐后的视觉 token
3. 视觉 token + caption 文本 token 拼接 → 冻结的 LLM
4. LLM 在 caption 部分计算 NTP loss
5. 反向传播只更新 Projector 参数
```

**Label 来源**：就是 caption 文本本身。这是**自监督**的思想——不需要额外标注，caption 既是输入也是 label。具体：

- 输入序列：`<image_tokens> A cat sitting on a`
- 目标序列：`A cat sitting on a mat`
- Loss：仅在 caption token 位置计算 cross-entropy（图像 token 位置被 mask 不参与 loss）

**数据类型**：
1. **Image-text caption 数据**：单图 + 详细描述（建立图像→语言直接映射）。
2. **Image-text interleaved 数据**：多图 + 文本交错（模拟真实文档结构，如图文并茂的网页）。

**为什么冻结 ViT 和 LLM**：
- ViT 已有好视觉表征，不需重训。
- LLM 已有好语言能力，如果一起训会破坏语言表征（catastrophic forgetting）。
- 只训 Projector 是"最小干预"——用最少参数调整建立两个已训好模块的桥梁。

---

### Q6：ViT 之前如何对视频 patchify？图像+文本如何在 LLM 拼接？

#### (1) ViT 之前的视频 patchify 流程

视频处理是**帧级 patchify**，不是 3D patchify：

```
视频 (e.g., 2小时, 30fps)
    ↓ 时序采样（按 fps 提取关键帧）
    ↓ 例如: 提取 512 帧
    ↓
帧 1 (e.g., 1920×1080)
    ↓ 标准 ViT patchify:
    ↓ 划分为 14×14 像素的 patch
    ↓ → (1920/14) × (1080/14) ≈ 137 × 77 = 10,549 patches
    ↓ 每个 patch 经 linear projection → 1 个 visual token
    ↓
帧 1 的 visual token 序列 [t_1, t_2, ..., t_10549]
    ↓ 前置时间戳文本 "00:00:00"
    ↓
[时间戳文本 token, t_1, t_2, ..., t_10549]
    ↓
... 对所有 512 帧重复 ...
    ↓
最终: [时间戳1, 帧1 tokens, 时间戳2, 帧2 tokens, ..., 时间戳512, 帧512 tokens]
```

**关键点**：
1. **帧级而非 3D**：每帧独立 patchify，不跨帧做 3D 卷积。这与论文的"frame-as-image formulation"一致——视觉通路对图像和视频统一处理。
2. **原生分辨率**：不同帧可能分辨率不同（如横竖屏切换），patch 数动态变化。这与固定 384×384 的传统 ViT 不同。
3. **patch 大小固定**：14×14 像素（继承自 SigLIP-400M-384-14），但 patch 数随分辨率变化。
4. **时间戳是文本**：时间戳 "00:00:00" 通过 LLM 的 tokenizer 编码为文本 token，不是视觉 token。这是关键——时间信息走文本通路，让 LLM 用原生语言空间处理时间。
5. **2×2 patch-merge**：Projector 把相邻 4 个 patch token 合成 1 个，将 token 数压缩 4 倍（如 10,549 → 2,637），显著降低后续 LLM 计算量。

#### (2) 图像+文本如何在 LLM 拼接

**核心原则：视觉 token 与文本 token 在序列维度直接拼接，送入 LLM**。不是 cross-attention，不是 fusion module，就是简单的序列拼接。

**典型输入序列结构**（以视频问答为例）：

```
<system_prompt>
你是视频理解助手...

<user>
<video>
[时间戳1文本][帧1视觉tokens][时间戳2文本][帧2视觉tokens]...[时间戳N文本][帧N视觉tokens]
</video>

请回答: 这个视频里的人在哪里跑步？
</user>

<assistant>
.ctx
我需要查看视频中跑步的场景...
验证候选区间 [00:15, 00:18], [02:30, 02:45]...
.
<answer>他在公园的林间小道上跑步，时间大约在视频的第15到18秒。支持区间: [[00:15, 00:18]]</answer>
</assistant>
```

**拼接细节**：

| 组件 | 处理方式 | Token 类型 |
|------|---------|-----------|
| 时间戳 "00:15:32" | LLM tokenizer | 文本 token |
| 帧 1 视觉特征 | ViT → Projector | 视觉 token |
| 用户问题 | LLM tokenizer | 文本 token |
| 系统提示 | LLM tokenizer | 文本 token |

**为什么直接拼接就有效**：
1. **LLM 的自注意力天然处理混合序列**：self-attention 不区分视觉 token 和文本 token，所有 token 互相 attend。文本 token 可以 attend 视觉 token 获取视觉信息，视觉 token 也可以 attend 文本 token 获取任务上下文。
2. **位置编码统一**：所有 token 共享同一套位置编码（本文用 3D RoPE / mRoPE），让 LLM 知道每个 token 在序列中的位置。
3. **时间戳作为锚点**：时间戳文本 token 让 LLM 知道"这帧对应 00:15:32"，从而能在输出中引用具体时间。

**与传统 cross-attention 方案对比**：

| 方案 | 代表模型 | 优点 | 缺点 |
|------|---------|------|------|
| 序列拼接 | LLaVA, Keye-VL, Qwen-VL | 简单统一，复用 LLM 自注意力 | 序列长，计算开销大 |
| Cross-attention | Flamingo, BLIP-2 | 视觉与文本解耦，序列短 | 需要额外 cross-attn 层，架构复杂 |

本文选序列拼接方案，配合 DSA 稀疏注意力解决长序列计算开销问题——这是 DSA 在本文中的核心价值：**让序列拼接方案在 256K 上下文下也变得可行**。
