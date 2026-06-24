# ZAYA1-8B Technical Report 论文总结

**论文信息**

- **标题**：ZAYA1-8B Technical Report
- **作者**：Robert Washbourne 等（Zyphra）
- **arXiv**：[2605.05365](https://arxiv.org/abs/2605.05365)
- **提交日期**：2026-05-06

---

## 一、核心贡献（一句话）

Zyphra 发布 **ZAYA1-8B**：面向推理的 MoE 模型（约 **0.76B 激活 / 8.4B 总参**），在 AMD 全栈上完成预训练与中后期训练；在数学与代码基准上可与远大于自身的开源推理模型竞争，并通过 **Markovian RSA** 测试时算力（TTC）在 AIME'25、HMMT'25 上逼近 Gemini-2.5 Pro、DeepSeek-V3.2、GPT-5-High 等前沿系统。

---

## 二、模型架构（MoE++）

| 配置项 | 数值 |
|--------|------|
| 架构 | Decoder-only MoE（Zyphra MoE++） |
| 激活 / 总参数 | 0.76B / 8.4B（对外常写作 0.7B / 8B） |
| 层数 / 隐藏维 | 40 / 2048 |
| 专家数 / 路由 | 16 专家，**Top-1**，无 residual expert |
| 注意力 | **CCA**（Compressed Convolutional Attention）+ CCGQA |
| KV 压缩 | Query 2×，KV-cache 相对全 MHA 约 8× |
| 分词器 | Gemma3，词表 262,272 |
| 训练硬件 | AMD MI300X + Pollara 400 网络 |

相对常见 MoE Transformer 的三项主要改动：

1. **CCA**：在压缩潜空间做序列混合，降低训练/prefill FLOPs 与 KV 显存，利于长上下文 midtrain 与 RL。
2. **ZAYA1 Router**：用 **MLP 路由器**（非线性层）替代线性 router；经 **EDA**（Exponential Depth Averaging）融合上一层路由表示；配合 PID 启发的 **bias balancing** 稳定专家负载。
3. **Residual Scaling**：对残差流与层输出分别学习缩放/偏置，控制深度方向残差范数增长，开销极小。

---

## 三、训练流程

### 3.1 预训练与中训练

| 阶段 | 上下文 | RoPE base | Token 量 | 重点 |
|------|--------|-----------|----------|------|
| 基座预训练 P1 | 4K | 10K | 8T | 广域网页、代码、数学、多语 |
| 基座预训练 P2 | 4K | 10K | 4T | 加强代码、数学、推理、指令数据 |
| 32K 中训练 | 32K | 1M | 1.2T | 长 CoT 推理、代码、数学、长上下文 |
| SFT | 131K | 5M | 660B | 对话模板、推理、代码、IF、TTC 轨迹 |

中训练 / SFT 数据以 **长链式思维（Long-CoT）** 为主（中训练约 86%，SFT 约 75%）。优化器为 **Muon**（含 AdamW RMS matching）。

### 3.2 Answer-Preserving（AP）Trimming

从预训练起即混入推理数据；当 CoT 超过当前上下文时：

- 优先 **截断最后一段推理的尾部**，保留推理开头与 **完整答案**；
- 多轮对话可丢弃较早轮的 thinking 块，仅留答案；
- 若答案本身超长则丢弃样本。

与 Qwen3 / ScaleRL 等在 **推理或 RL rollout 时** 强制结束 thinking 不同，AP-trimming 在 **建数据阶段** 完成，使短上下文阶段也能学到“有结论的推理”。

### 3.3 后训练：SFT + 四段 RL 级联

```
SFT → (1) 推理热身 → (2) RLVE-Gym 400 任务 → (3) 数学+代码+TTC（两相）→ (4) 行为 RL
```

**RL 算法骨架（各阶段共用）**

- **PipelineRL**：rollout 与 trainer 异步，rollout 工人数约为 trainer 的 2–5×。
- **DPPO Binary-TV**：用总变差阈值掩码高散度 token，替代 PPO clip；δ=0.1。
- **Dr-GRPO（SMTSN）**：按序列求和再对 batch 平均，减轻对长回复的隐式偏好。
- **MaxRL 优势**：Â = (rᵢ − r̄) / r̄，可验证阶段不用 GRPO 的标准差归一化。
- **无 reward 内 KL**：信任域仅靠 DPPO；避免 PipelineRL 下 signed log-ratio 导致的长度偏置。
- **Muon 零动量**：矩阵参数每步仅用当前 batch 梯度做 Muon 更新；embedding/LM head 仍用 AdamW。
- **难度缩放的长度奖励**：在组内至少 2 个正确样本时，鼓励更短且正确的回答。

**各 RL 阶段要点**

| 阶段 | 步数 | 数据/环境 | 作用 |
|------|------|-----------|------|
| 推理热身 | 232 | 数学、谜题、TTC 格式 | 适应长 rollout、引入聚合 prompt |
| RLVE-Gym | 400 | 400 个可验证环境 | Thompson Sampling + IRT 式难度调度，维持约 50% 通过率 |
| Math+Code+TTC | 384+464 | 奥赛数学、竞赛编程、合成 CodeI/O / CodeARC / 反例任务 | 主能力阶段；含专家/自聚合 TTC prompt |
| 行为 RL | 384 | HelpSteer + IF 门控 RM | 聊天风格与指令遵循（不主攻数学/TTC） |

**工程稳定性**

- **Router replay**：训练时复用 vLLM rollout 的专家路由，消除 Top-1 MoE 的离散路由不匹配。
- **数值对齐**：LM head、CCA、RMSNorm、router softmax、残差加等在 rollout 与 trainer 间统一 FP32 子集；KL≈1.3×10⁻⁴，r>0.9996。
- **退化检测**：LZ77 可压缩性 + 稀有 token 比例；触发则 reward 置零。

---

## 四、Markovian RSA（测试时算力）概要

结合 **RSA**（多候选递归聚合）与 **Markovian Thinker**（有界状态传递）。默认 **N=16, C=4, T=2**；配置常写作 **β/τ**（如 40K/4K）。

- Round 0：对问题 q 并行生成 N 条推理（每条 ≤β），只保留最后 **τ** token 为 tail。
- Round t≥1：随机抽 C 个 tail 拼聚合 prompt，再生成 ≤β 的新推理并截 tail。
- 聚合 prefill 上界：**|q| + Cτ + O(1)**，与历史总长度解耦。

**Headline（40K/4K）**：AIME'25 **91.9%**，HMMT'25 **89.6%**（单次 rollout 为 88.3% / 82.7%）。

详细算法、训练对齐、调参与服务画像见 **第九节 Q1**。

---

## 五、主要实验结果

### 5.1 同量级开源模型（单次采样，Zyphra harness）

| 基准 | ZAYA1-8B | Qwen3-4B-Thinking | Qwen3.5-4B | Gemma-4-E4B |
|------|----------|-------------------|------------|-------------|
| AIME'26 | **89.1** | 79.0 | 84.5 | 50.3 |
| HMMT'26 Feb. | **71.6** | 53.6 | 63.6 | 32.1 |
| LiveCodeBench-v6 | **64.8** | 54.9 | 55.8† | 54.2 |
| MMLU-Pro | 74.2 | 74.3 | **79.7** | 70.2 |
| BFCL-v4 / τ² | 40.5 / 36.3 | 49.7 / 52.9 | 45.2 / **82.1** | 31.7 / 37.7 |

知识/聊天有胜有负；**Agentic** 弱——本版未做专门 agentic RL。

### 5.2 更大开源模型（同 harness）

| 模型 | Active | Total | AIME'26 | HMMT'26 | LCB-v6 |
|------|--------|-------|---------|---------|--------|
| ZAYA1-8B | 0.7B | 8B | **89.1** | **71.6** | **64.8** |
| Nemotron-3-Nano | 3B | 30B | 90.1 | **75.5** | 64.6 |
| Qwen3-Next-80B-Think | 3B | 80B | 90.2 | 79.3 | **67.8** |
| Intellect-3 | 12B | 106B | 86.3 | 72.3 | 66.8 |
| Mistral-Small-4-119B | 6B | 119B | 86.4 | 70.6 | 57.9 |
| Arcee-Trinity-Mini | 3B | 26B | 59.6 | 36.9 | 33.3 |

### 5.3 后训练增益（SFT → 最终）

AIME'26 **+20.8**，HMMT'26 **+32.4**，LCB-v6 **+10.0**，IFEval **+19**，IFBench **+22**（约 **1480 步**可验证 RL）。

### 5.4 TTC 与前沿对比

Markovian RSA（40K/4K）下接近或超过 DeepSeek-R1-0528、Qwen3-235B-Thinking 等公开数字；与 Gemini-2.5 Pro、DeepSeek-V3.2、GPT-5-High 差距缩小（部分为外部报告数）。

---

## 六、讨论与局限（作者观点）

1. **RL 样本效率高**：能力主要在预训练/中训练植入；短 RL 主要改变采样分布。
2. **MoE + PipelineRL** 必须 router replay + 数值对齐。
3. **数据/Verifier SNR** 与在线难度策展是算法的一部分。
4. **小激活 MoE + TTC**：有效成本 ≈ 激活参数 × 生成 token。
5. **局限**：无 agentic RL；知识榜弱于超大模型；8B 级 AMD 栈验证；架构因果消融不完整。

---

## 七、与本仓库主题的关联

本文 **不是** 剪枝/量化/蒸馏类压缩论文，而是 **小激活 MoE + 长上下文训练 + TTC** 的系统级参数效率方案。评估需同时看 **激活参数 × 生成 token** 与单次 forward 的上下文上界。

---

## 八、关键术语速查

| 术语 | 含义 |
|------|------|
| MoE++ | Zyphra 的 MoE 架构族（含 CCA、ZAYA1 router 等） |
| AP-trimming | 训练数据截断长 CoT 尾部、保留答案 |
| Markovian RSA | 有界 tail 传递的递归自聚合 TTC |
| PipelineRL | rollout 与训练异步流水线 RL |
| Router replay | RL 训练复用推理时的专家选择 |
| EDA | Exponential Depth Averaging，路由表示跨层融合 |

---

## 九、阅读延伸：讨论问答实录

以下整理自阅读本报告后的追问与回答，与上文「论文原文摘要」互补。

---

### Q1：详细介绍 Markovian RSA

**问**：详细介绍下 Markovian RSA。

**答**：

#### 1.1 动机：三种 TTC 形态的对比

| 形态 | Decode | 聚合/上下文状态 | 痛点 |
|------|--------|-----------------|------|
| 单条长 CoT | BS=1，位置持续增长 | 整条历史 | KV 膨胀，难批处理 |
| Full-chain RSA | BS=N | C 条整链，各 ≤β | Prefill ∝ **Cβ**（40K×4≈160K） |
| **Markovian RSA** | BS=N | C 条 **tail**，各 ≤τ | Prefill ∝ **Cτ**（4K×4≈16K） |

核心：**β 控制单候选思考深度，τ 控制轮间记忆宽度**，二者解耦。

#### 1.2 超参数

| 符号 | 含义 | 默认 |
|------|------|------|
| N | 每轮候选池大小 | 16 |
| C | 每次聚合抽几条 tail | 4 |
| T | 聚合轮数（不含 round 0 探索） | 2 |
| β | 单条候选最大生成 token | 8K / 16K / 40K |
| τ | 携带 tail 长度（τ≤β） | 常 4K |

#### 1.3 算法步骤

**Round 0**：从 q 并行生成 N 条推理 \(y_i\)，每条 ≤β；取 \(\text{tail}_\tau(y_i)\) 构成种群。

**Round t≥1**（重复至 T）：

1. 从上一轮 N 个 tail 中 **均匀随机** 抽 C 个；
2. 拼聚合 prompt（q + C 段 tail +「综合并改进」类指令，**无需 verifier**）；
3. 生成新推理 y'（≤β），再取 tail 进入本轮种群（生成 N 个新候选需重复 N 次，每次抽样可不同）。

**结束**：从最终轮输出按常规定式抽答案；论文主结果多为 **最终轮候选准确率的平均**（非 best-of-N）。

#### 1.4 特例（统一框架）

| 设置 | 等价 |
|------|------|
| T=0 | N 路并行；有外部 selector 才是 Best-of-N |
| τ=β | Full-chain RSA |
| C=1 | 并行 Markovian/Delethink 续写（无跨候选聚合） |

**vs PaCoRe**：PaCoRe 前传「结论段」；Markovian RSA 固定取推理 trace **最后 τ token**。论文 hybrid：有答案段则传答案，否则传 partial chain（LCB 上 71.1% vs 纯 Markovian 69.2%）。

#### 1.5 两种「成本」

- **Serving 轮廓（友好）**：每步 prefill ≤ |q|+Cτ；decode batch=N；**无阶段 attend 全历史**。40K/4K 时 carried state 约 **16K vs full-chain 160K（~90%↓）**。
- **总生成 token（仍大）**：\(D(q)=\sum_s n_s \bar{g}_s(q)\)。AIME/HMMT 实测约 **440K/题（16K/4K）**、**740K/题（40K/4K）**（不含 prompt/prefill/复制 tail）。

#### 1.6 训练对齐（ZAYA 能用好的原因）

- **SFT**：每题 n 条教师 rollout → 抽 C 条取 tail → 教师写聚合轨迹为 label（离线、可无 verifier）。
- **RL**：Expert-aggregation / **Self-aggregation**（用当前或前一 checkpoint rollout 作候选）；梯度仍是单条 rollout + 可验证 reward。
- **对比**：同一 Markovian RSA 套在 **Qwen3-4B-Thinking** 上增益明显弱于 ZAYA——作者归因 **midtrain/SFT/RL 含聚合轨迹**，非方法对任意模型即效。

#### 1.7 实验要点

| 配置 (T,N,C,β,τ) | AIME'25 | HMMT'25 |
|------------------|---------|---------|
| C=1 基线 (4,16,1,8K,4K) | 82.1 | 75.0 |
| (2,16,4,8K,4K) | 86.5 | 80.8 |
| (2,16,4,16K,4K) | 88.8 | 87.1 |
| **(2,16,4,40K,4K)** | **91.9** | **89.6** |
| (2,16,4,40K,8K) | 90.8 ↓ | 89.2 ↓ |

- **加长 β** 优于加长 τ（同预算下优先「每候选想更深」）。
- **C=1→C=4**（8K/4K）：AIME +4.4、HMMT +5.8 → 跨候选聚合在 bounded carryover 之上仍有独立收益。
- **部署**：日常 **16K/4K**；刷榜 **40K/4K**；轻量 **N=8, β=8K** 曾报约 **0.4×** 墙钟 vs 单条长 CoT（实现相关）。

---

### Q2：该论文在模型架构上有什么创新吗？

**问**：该论文在模型架构上有什么创新吗？

**答**：

#### 2.1 定位

- **有架构层面的改动**，但不是全新 Transformer；在 **MoE++** 上相对 DeepSeek/Qwen 类 MoE 标榜 **三项改动**。
- **颠覆式创新：否**；**系统集成 + 参数效率：是**。

#### 2.2 三项主改动（创新度分级）

**（1）CCA — 采用 + 规模化验证**

- **非本文首发**；先前工作已在 small scale 验证 PPL。
- 机制：低秩投影；短卷积 + 分组头卷积 preconditioner；一半 value head 1-token 延迟；Q/K mean skip；key 用 **线性温度 T**（非 exp(T)）稳 logit。
- **CCGQA**：8 query / 2 KV，query 2× 压缩 → **~8× KV-cache vs 全 MHA**。
- **本文贡献**：8B 总参 + 推理/131K/长 RL rollout 上仍有效；使长上下文训练在有限算力下可行。

**（2）ZAYA1 Router — 最具辨识度**

| 组件 | 作用 |
|------|------|
| Down-project | D→R=256 |
| **EDA** | \(r_l \leftarrow r_l + \gamma r_{l-1}\) |
| 3-layer MLP | 出 expert logits |
| Top-1 + bias \(b_l\) | 无 residual expert |
| **PID 式 bias 更新** | \(\nabla b_{l,e}=p_{l,e}-1/E\)，AdamW 更新 |

- 参数匹配消融：**边际参数加 router 优于加 expert/attention**。
- 配套：**Top-1 only**、路由熵更低更自信；附录与 LFM2/OLMoE/Qwen3-30B-A3B 比 **专家子空间重叠无异常**。

**（3）Residual Scaling — 轻量增量**

- 对 residual 与 layer output 分别 \(\alpha x+\beta\)；对标 Qwen attention gating，**更省参/FLOP**；控深度残差范数。

#### 2.3 重要配方（非新模块）

- 16 experts、FFN 2× 扩维；**40 层**（vs Qwen3-4B 36 层、OLMoE 16 层）；50% RoPE；Gemma3 tokenizer。

#### 2.4 与常见 MoE 的差异一句话

**CCA 替 MLA/GQA + 非线性跨层 router 替线性 router + 残差缩放 + Top-1 无 shared expert**。

#### 2.5 不算架构创新的部分

Markovian RSA、AP-trimming、RL 级联、AMD 栈、router replay 等属 **训练/推理/系统**。

---

### Q3：本文在小模型相关的工作还有什么？

**问**：本文在小模型相关的工作还有什么？

**答**：

论文 **不是小模型综述**，但贯穿 **「<1B 激活也要强推理」**。除架构与 Markovian RSA 外，还包括：

#### 3.1 实证：按激活参数量对标

**同量级 dense thinking（~4B）**

- 数学/Code 全面领先 Qwen3-4B-Thinking、Qwen3.5-4B、Gemma-4-E4B。
- Qwen3-4B 亦作 **TTC 对照**（同流程下 ZAYA 更能吃满聚合）。

**小激活、大总参 MoE**

- Nemotron-3-Nano（3B/30B）、Qwen3-Next-80B-Think（3B/80B）等：数学密度 ZAYA 仍突出；IF/知识常落后。
- Arcee-Trinity-Mini（3B/26B）数学/Code 远低。

**标杆叙事**

- 单次：**≥ DeepSeek-R1-0528**（37B act / 671B total）部分数学/Code。
- TTC：**逼近** Gemini-2.5 Pro、DeepSeek-V3.2、GPT-5-High 等（注意协议差异）。
- 文中有 **active-param scaling** 图（气泡面积为总参）。

#### 3.2 附录：小 MoE 专家诊断

与 **LFM2-8B-A1B、OLMoE-1B-7B、Qwen3-30B-A3B** 比 expert subspace overlap → ZAYA **非 expert 塌缩 outlier**。

#### 3.3 为小算力服务的训练配方

| 方法 | 与小模型关系 |
|------|----------------|
| AP-trimming | 4K 预训练也能用长教师 CoT |
| Front-loading reasoning（NVIDIA Synergy） | 推理数据必须早灌 |
| CCA + 131K SFT | 有限集群可做长上下文 |
| ~1480 步 RL | post-train 算力友好 |
| 长度奖励（ShortRL/ALP 混合） | 压小模型啰嗦 CoT |
| PipelineRL + router replay | 小 MoE 长 rollout RL 前提 |
| Muon 零动量 | 引用 RL 稀疏子网络更新；省 optimizer state |
| 1.8B 消融 | router 熵收敛（架构验证规模） |

#### 3.4 TTC 作为小模型的「第二缩放轴」

- RSA、Markovian Thinker、PaCoRe 等被引用/对比。
- 讨论：**激活参数 × 生成 token**；小激活 MoE 适合多路并行 rollout。
- 展望：**小推理核 + TTC + 检索**，不必全能力进参数。

#### 3.5 前作与系列

- **prior-zyphra-work**（AMD 全栈预训练）、**MoE++**；ZAYA1-8B 为 **ZAYA-1 家族首个/最小**。
- 目标：**reasoning performance per active parameter**。

#### 3.6 本文少覆盖的「小模型」传统话题

- 参考文献有 MiniCPM、JetMoE 等，**正文未系统对比**。
- 未主打：通用 chat 全能、蒸馏/量化/剪枝、多语、agentic（BFCL、τ² 弱）。

#### 3.7 归纳公式

```
小激活 MoE（CCA + 强 router + 深 40 层）
  + 预训练即推理（AP-trimming + Long-CoT）
  + 短密 verifiable RL
  + 训练对齐的 Markovian RSA
→ 0.7B 激活打 4B～100B+ 对手；TTC 再换精度而不换更大 backbone
```

---

## 十、文档修订说明

| 版本 | 内容 |
|------|------|
| 初版 | 基于 arXiv:2605.05365 TeX 的中文结构化摘要 |
| 增补 | 合并对话 Q1（Markovian RSA 详解）、Q2（架构创新）、Q3（小模型相关工作） |

*TeX 本地缓存：`~/.cache/knowledge/2605.05365/`。*
