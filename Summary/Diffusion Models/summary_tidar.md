# TiDAR（Think in Diffusion, Talk in Autoregression）论文总结

- **论文**: TiDAR: Think in Diffusion, Talk in Autoregression  
- **arXiv**: [2511.08923](https://arxiv.org/abs/2511.08923)（NVIDIA Tech Report，2025-11-12）
- **说明**：前半为论文阅读总结；后半为 **对话问答整理**，以及 **speculative decoding** 与 **rejection sampling** 的独立介绍（含例子）。

## 核心动机：把“免费 token slots”吃满

自回归（AR）LLM 的解码通常是 **memory-bound**：每一步只生成 1 个 token，GPU 大量算力闲置；但同一次前向里多塞一些 token 位置，延迟在一定范围内几乎不涨（论文称这些额外位置为 **free token slots**）。

扩散语言模型（dLM）能并行解码多个 token，看似能更好利用 free token slots，但一旦在单步里并行生成多个 token，就会引入 **token 之间的独立性假设**（从边缘分布分别采样），导致质量下降；因此像 Dream、LLaDA 这类模型在“速度-质量”上仍难与强 AR 模型匹敌。

TiDAR 的目标是：**计算上像扩散一样并行“草稿”，采样上像 AR 一样保证质量**，并且把两者做到 **同一次 forward** 里，最大化硬件利用率。

## 方法概览：单模型、单次前向的“扩散草稿 + AR 采样”

TiDAR 是一种 **序列级混合架构**，在一次生成 step（一次 forward）里同时做两件事：

- **Talking（高质量输出）**：对“上一轮草稿 token”用 AR 的链式联合分布进行 **rejection sampling**（拒绝采样式验证/采样），决定本轮接受多少 token。
- **Thinking（并行草稿）**：同时用扩散式（更准确说是“掩码预测”的边缘分布）对“下一轮”要用的 token 进行 **并行预草稿（pre-draft）**。

关键点是：它通过 **结构化 attention mask** 在同一序列里实现“前缀因果（causal）+ 草稿块内双向（bidirectional）”的混合注意力，从而在一个模型里同时得到：

- **AR 模式**：用于联合分布的高质量采样/验证（因果注意力）
- **Diffusion 模式**：用于边缘分布的并行草稿（块内双向注意力）

并且 TiDAR 支持 **exact KV cache**：因果部分的 KV 可缓存；在 rejection sampling 接受长度小于草稿长度时，对应的 KV 可被驱逐（evict），避免重复计算。

## 训练：双模式同训 + “Full Mask”简化扩散目标

TiDAR 训练时把输入序列扩展为两段（长度约翻倍）：

- **前半段（clean/prefix）**：因果 mask，做标准 next-token prediction（NTP）损失（标签右移）。
- **后半段（diffusion section）**：块内双向 + 依赖前缀，做扩散式掩码预测损失（标签不移位）。

与许多 dLM 的“随机掩码策略”不同，TiDAR 在 diffusion 段采用 **Full Mask**：把 diffusion 段 token 全部置为 `[mask]`。作者给出的收益：

- **更密集的 diffusion loss**（每个位置都有监督，而不是只监督被随机 mask 的位置）
- **更容易平衡 AR loss 与 diffusion loss**（两段损失项数一致，权重更好设）
- **匹配 one-step diffusion 草稿** 的推理设定，提升训练-推理一致性

## 推理与实现细节：并行化与服务友好

- **one-step diffusion drafting**：草稿只做一步“从全 mask 预测”，用作高容量 drafter，提高 acceptance rate。
- **mask 复用**：推理时通过重排 prefix 与草稿/预草稿部分，预先初始化大 attention mask，按 prefix 长度切片使用（配合 Flex Attention），减少运行时开销。
- **几乎无额外推理超参**：相对很多 dLM 需要调阈值/熵策略，TiDAR 主流程更固定；论文也讨论了“trust AR vs trust Diff”（将两路 logits 线性混合，系数 \beta）的灵活性，但整体较稳健。

## 实验结论（论文主张）

作者主要在 1.5B 与 8B 尺度上，从 AR（Qwen2.5 / Qwen3）做 continual pretraining 得到 TiDAR，并对比 AR、speculative decoding（EAGLE-3）、Block Diffusion、Dream、LLaDA 等。

- **吞吐提升**：  
  - TiDAR 1.5B 平均约 **7.45 tokens/NFE**，相对同尺度 AR 达到约 **4.71× tokens/s**。  
  - TiDAR 8B 平均约 **8.25 tokens/NFE**，相对同尺度 AR 达到约 **5.91× tokens/s**。
- **质量**：在代码/数学等生成任务上与同尺度 AR 接近，并显著优于 Dream/LLaDA 等扩散 LLM 的速度-质量折中；同时作者强调其是首批在吞吐上 **实测超过 speculative decoding** 的扩散相关方案（但两者用途不同：speculative decoding 可保证与 base model 完全一致输出）。
- **likelihood 评测**：TiDAR 的 AR 模式允许像标准 AR 一样高效算 likelihood（无需扩散模型常见的 Monte Carlo 估计），使得在多选题等 likelihood 指标上更可比且更高效。

## 局限与展望

- **batch size 场景**：论文主要强调 batch=1 的 latency-critical 吞吐；更大 batch 下仍可通过调整 block/draft 长度适配算力配置。
- **长上下文**：训练需要“序列翻倍”（追加 mask 段），长上下文训练/并行策略需要更专门的工程优化。
- **系统优化空间**：即便原生 PyTorch + Flex Attention 已有收益，但作者认为自定义 kernel、调度与 KV 管理可以进一步把 free token slots 榨干。

## 我理解的关键 takeaways

- **不是简单“把扩散和 AR 串起来”**，而是把“草稿（边缘分布并行预测）”和“采样/验证（AR 联合分布）”做成 **同一次 forward 的并行工作负载**，这是它能吃到硬件 free token slots 的核心。
- **Full Mask + one-step diffusion** 把扩散部分做成高效、高信噪比的“并行 drafter”，再用 AR rejection sampling 把质量兜住。

---

## 对话整理：用户问题与详细答复

以下按讨论顺序整理。**训练**与**推理**务必分开理解：训练是「在数据上同时学两种注意力形态」；推理才是「先并行提案、再 AR 式验证、再并行预草稿」的循环。

### Q1. 阅读并总结 TiDAR（arXiv:2511.08923）

**答复（展开）**：

TiDAR 针对的是 AR 与离散扩散 LLM 的互补短板：**AR** 因果结构利于质量与 likelihood，但 decode 常 **memory-bound**，单步只推进 1 token，算力吃不满；**扩散**可并行预测多位置，但若在单步内把多 token 当「独立边缘」处理，会损害 **序列一致性** 与下游指标。

TiDAR 在 **同一套权重、同一次 decoder forward** 内，用 **结构化 attention mask** 同时启用两条路径：

- **Diffusion / 掩码并行路径**：对一整块 `[MASK]` 做 **one-step** 并行预测，充当 **高容量 drafter**（Thinking）。
- **因果 AR 路径**：对已进入上下文的 proposal 用 **链式联合分布** 做 **rejection sampling**（Talking），决定本轮接纳多少 token。

再叠加 **exact KV cache**（因果段缓存；拒掉尾部 proposal 时 **evict** 对应 KV）与 **free token slots**（小 batch 下多塞 token 一次前向延迟未必线性涨）的系统观察，形成论文报告的 **tokens/s** 与 **tokens/NFE**。论文级公式、实验与 baseline 对比见本文档前半各节。

---

### Q2. TiDAR 主要以什么工作为基础？

**答复（展开）**：

可理解为在四条成熟脉络上 **合成**，而非单一 prior 的线性扩展：

1. **自回归 Transformer LLM**：从 Qwen 等 **continual pretrain**；推理用 **因果注意力 + NTP 式联合分布** 做质量守门；likelihood 可走 **纯 AR mask** 路径。
2. **离散扩散 / 掩码 LM**：并行「填 mask」的能力；TiDAR 用 **full-mask 扩散段 + one-step drafting** 降低训练-推理错配并提高草稿信噪比。
3. **Speculative decoding**：「先 proposal、再按 target 规则接受/纠正」；TiDAR 把 draft 与 verify **并进同一次 forward**，draft 不必是小模型。
4. **Block Diffusion / semi-AR**：**块间因果、块内双向** 与 **KV 可缓存** 的可行性；TiDAR 进一步服务 **单步 self-speculative** 推理。

---

### Q3. Block Diffusion 指的是哪篇论文？

**答复（展开）**：

TiDAR bib 中的 **Block Diffusion** 指 **Arriola 等**，*Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models*，**[arXiv:2503.09573](https://arxiv.org/abs/2503.09573)**（ICLR 2025 Oral）。其他文献若口语化使用「block diffusion」，默认宜对齐此文，除非另有定义。

---

### Q4. Block Diffusion 这篇论文在讲什么？（与 TiDAR 的关系）

**答复（展开）**：

**问题**：纯离散扩散 LM 常有 **固定生成长度**、**双向注意力下 KV 不自然**、**PPL 落后 AR** 等痛点。

**方法**：**块间按 AR 分解、块内在 block 上做离散扩散**；整体分布对 block **链式分解**，每块条件分布用扩散建模，从而 **变长生成**、**block-causal mask**、**KV 更可用**，并用 **专门训练算法** 与 **低方差噪声日程** 追 PPL。

**与 TiDAR**：Block Diffusion 提供 **混合注意力 + 局部并行扩散** 的积木；TiDAR 在此基础上走向 **推理期 self-speculation**（并行 draft + AR rejection + pre-draft）与 **单 forward 吞吐**。更长笔记见 **`summary_block_diffusion.md`**。

---

### Q5. Block Diffusion 的分析写完了吗？

**答复**：已写入同目录 **`summary_block_diffusion.md`**（中文；含目标、训练向量化、方差与 clipped schedule、实验与局限）。本文不再重复其全文推导。

---

### Q6. pre-draft、draft、speculative decoding、MTP 有何异同？

**答复（展开）**：

| 概念 | 它是什么 | 解决什么 | 常见误区 |
|------|----------|----------|----------|
| **draft** | 一组候选 token 的 **提案** | 把昂贵串行采样拆成「先猜后验」 | 以为 draft 本身是独立「模型类型」 |
| **pre-draft** | **提前**为下一步准备草稿，常与当前步验证 **时间重叠** | 减少「验完才起草」的空泡 | 不等于新范式；是调度/并行化技巧 |
| **speculative decoding** | **draft + 验证/接受规则**（常为 rejection sampling）的 **系统** | 提高 **tokens / 次 target 前向**；经典版可 **对齐 target 采样** | 把「打分排序」当成严格 spec |
| **MTP** | 结构上一次输出多步（多头/附加层/树） | 增强提案或训练信号 | **未必**有 verifier；**未必**与单步 AR 采样同分布 |

**关系一句话**：draft/pre-draft 是 **零件**；spec decoding 是 **装配与概率契约**；MTP 常提供 **AR 式串行提案**，不像整块 mask 那样天然「一步填多格」。

---

### Q7. draft / pre-draft + verifier 是否就等于 speculative decoding？

**答复（展开）**：

**多数情况下可以这么说**：存在「提议」与「以目标分布为准的纠正」即属 speculative **家族**。

更严谨时通常还要求：

1. **可写的接受机制**：不仅是启发式分数，而是能写清 **何时接纳 proposal、拒绝后如何从 target 续采样**（rejection 或其等价形式）。
2. **若声称与某 target 完全同分布**：需对应论文中的 **证明/构造**（如经典 spec 文）；若只是 rerank，则宜称 **启发式加速**。

**pre-draft** 不改变「是不是 spec」的本质，只改变 **时间线**：验证与下一步草稿 **并行**。

---

### Q8. 训练时序列扩成两段，是否为一次 forward 同时训 AR 与 diffusion？长度翻倍是否更费算力？

**答复（展开）**：

**第一问：是。** 拼接「干净段 + 全 mask 扩散段」，用 **不同 attention mask** 在同一次 forward 得到 **因果 logits（NTP，标签右移）** 与 **扩散 logits（对齐位置、标签不移位）**，使 **训练图与推理图同构**。

**第二问：训练通常更贵。** 有效长度约 **2×**，注意力与激活随 token 数增长；即便 FlashAttention，**总算力/显存**一般仍高于「只训长度 \(S\) 的纯 AR」。与论文 **limitations**（长上下文工程）一致。

**第三问：为何不与「推理 free slots」矛盾？** **free token slots** 主要针对 **小 batch decode、memory-bound**：多 slot **墙钟未必线性涨**；训练常在 **另一瓶颈 regime**（更重激活与带宽），故 **训练更贵** 与 **推理可摊薄** 可同时成立。

**训练 vs 推理（再强调）**：训练是 **teacher forcing**：真值序列 + mask 段监督；**不是**「前半自生成 1 token → 后半 diffusion 再选」——那是 **推理循环** 的误套。

---

### Q8a. 请介绍 speculative decoding，并给出一个具体例子

**答复**：见下文 **「Speculative decoding（推测解码）简介与例子（展开）」** 整节。

---

### Q9（推理）. 上一轮 pre-draft 是多个 token 吗？本轮 AR + rejection 是一次只出一个 token，还是对整段 proposal 同时处理？

**答复（展开）**：

**proposal 粒度**：draft / pre-draft 按 **block**，实验常见长度 **4 / 8 / 16**，故 proposal 通常是 **多 token 一块**，非单 token。

**验证是否「整段并行接受」**：论文用 **rejectively sampled** 且 **autoregressive manner**，与经典 spec 一致：

- **逻辑**：从左到右检查 draft 第 1、2、… 个 token；每步条件于 **已写入上下文的前缀**（含本轮已接纳的 proposal）。
- **接受长度**：记为 \(m\in\{0,\dots,k\}\)，可出现 **全接、只接前缀、第一个就不接**；故「本轮写进历史的 token 数」是 **随机变量**，不是固定「每步 1 个」。
- **为何拒绝后常丢后缀**：位置 \(j\) 拒绝后，前缀状态改变；**\(j\) 之后尚未验证的 draft** 条件于 **错误假设**，故 **整段废弃** 并自 \(j\) 起按 **target** 重采样/续写。

**实现**：TiDAR 称 **同一次 forward** 并行算 draft / verify / pre-draft；**语义**上验证链仍是 **AR 前缀式**。

---

### Q10. 是否要「原模型 + draft + verify」三个模型？verify「prefix + 3 token」是否仍很贵？

**答复（展开）**：

**通常不是三套独立权重**：经典 spec 是 **小 draft + 大 target**；**verify** 指 **target 的计算角色**（一次或树状前向），不是第三个 `verify.pt`。

也有 **单 backbone + MTP 头**：draft/verify **共享主干**。

**verify 成本**：**有**——上下文更长、算子更大；经济账是 **平均接纳 \(m\) 个 token / 次大模型前向**。接受率低则 **验证照付、推进少**，端到端可能变慢。

---

### Q11. rejection sampling 是什么？验证不通过是否等于「后面 draft 全丢」？

**答复（展开）**：

**一般**：从 **提议 \(q\)** 得候选，用 **随机阈值**（常配合 \(p/q\) 型接受率）使样本边际对齐 **目标 \(p\)**。

**在 spec decoding**：\(p\) 为 **target** 条件分布；\(q\) 为 **draft**。沿链验证时，**在首拒点之后、尚未验证的 draft 通常整段不再使用**，并在该点按 **target 修正分布**续写——这不是随意浪费，而是 **条件链断裂** 的必然后果。

---

### Q12. 「随机判定」具体怎么随机？为什么要随机？

**答复（展开）**：

**机制**：对每次判定采样 **\(u\sim\mathrm{Uniform}(0,1)\)**，与由 **\(p_{\text{tgt}}(\hat{x}\mid c)\)**、**\(p_{\text{dft}}(\hat{x}\mid c)\)** 等构造的 **接受概率**比较；拒绝则进入 **target 重采样** 分支（具体公式因论文版本略异）。

**为何随机**：离散大词表上，纯确定性规则（如「argmax 就接」）通常 **无法** 复现「逐步从 target 采样」的完整随机结构；\(u\) 用于 **校准接受频率**。

**工程**：\(u\) 来自 **PRNG**；可复现则固定 seed。

---

### Q13. 是否可理解为：TiDAR 把 spec decoding 的 draft 换成 diffusion，并用 pre-draft 拉高 GPU 吞吐？

**答复（展开）**：

**类比可用，但建议补三处限定**：

1. 不是仅「把小 draft 换成扩散」：TiDAR **共享同一 transformer**，用 **mask 切换模式**；draft 侧是 **高容量** 的。
2. **pre-draft 只是一块拼图**：还有 **同 forward 并行打包**、**free token slots**、**exact KV + evict**。
3. **与经典 spec 的契约不同**：经典 spec 常强调 **严格对齐某 target 输出**；TiDAR 是 **hybrid 加速-质量**；论文对比 EAGLE-3 时也提醒 **用途不同**（spec 可保证与某 base **完全一致**）。

**收束句**：**TiDAR ≈ spec 式提案-验证骨架 + 扩散式并行提案 + 单 forward 与缓存工程**。

---

## Speculative decoding（推测解码）简介与例子（展开）

### 1. 解决什么问题

AR decode 常 **memory-bound / 计算密度低**：每步加载整模与 KV，但有效 FLOPs 利用率不高。spec decoding 问：能否用 **可控额外计算**，换「**一次大模型前向推进更多 token**」。

### 2. 概念流程

1. **Draft**：更快组件一次给出 \(k\) 个候选（或一棵路径）。
2. **Verify**：**target** 在扩展上下文上算分布，按 **接受规则**（常为 rejection sampling）得 **接纳前缀长度** \(m\in\{0,\dots,k\}\)。
3. **Continue**：\(m<k\) 时在首拒点 **target 重采样** 并继续；\(m=k\) 则本轮最大化推进。

### 3. 具体例子

前缀：`"The weather is"`。draft：`nice` · `today` · `.`。

- **情况 A（\(m=3\)）**：三处均在拒绝采样意义下可接受 → `"The weather is nice today ."`，摊销最优。
- **情况 B（\(m=1\)）**：`nice` 接、`today` 拒 → 固定到 `"The weather is nice"`，下一位置从 **target** 重采；`today`、`.` 的后续原 draft **通常整段作废**。
- **情况 C（\(m=0\)）**：首 token 拒 → 推进极少，验证成本仍在，可能 **不加速** → **接受率** 是系统关键。

### 4. 误区：不是三模型

常见 **两套权重（小 draft + 大 target）** 或 **单模型多头**；**verify = target 的计算**，不是第三个独立训练网络。

### 5. 参考

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2302.01318)  
- [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2211.17192)

---

## Rejection sampling（拒绝采样）简介（展开）

### 1. 一般形式（直觉）

从 **目标 \(p\)** 采样困难时：从 **提议 \(q\)** 得候选 \(x\)，抽 **\(u\sim U(0,1)\)**，用 **接受概率 \(A(x)\in[0,1]\)**（常由 \(p,q\) 构造）决定接受或拒绝并重试，使长期样本边际对齐 \(p\)。

### 2. 在 spec decoding 中的对应

- **\(p\)**：**target** 对「下一 token」的条件分布；前缀随验证 **动态更新**。
- **\(q\)**：**draft** 的提议分布。

### 3. 「随机」随机什么

主要是 **独立的 \(u\)**。没有它，许多确定性规则无法同时满足 **速度** 与 **分布正确性**（尤其当要求与某 target 采样严格一致时）。

### 4. 与「后缀 draft 丢弃」的关系

**拒绝**改变前缀；**未再验证** 的后续 draft 所依赖的条件失效，故 **整段废弃** 是 **条件链断裂** 的结果，而非随意丢结果。

### 5. 与 TiDAR 的衔接

TiDAR 用 **AR 链式验证** 保质量，用 **扩散式并行填 mask** 提案；理解 rejection sampling 有助于区分 **「被提案的 token」** 与 **「被写进历史的 token」**，并理解论文为何要让 **pre-draft 覆盖 rejection 的多种接受长度**（conditioned on all possible outcomes of the rejection sampling）。

