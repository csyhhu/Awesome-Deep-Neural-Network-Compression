# Probe and Skip: Self-Predictive Token Skipping for Efficient Long-Context LLM Inference

> **论文信息**：Zimeng Wu*, Donghao Wang*, Chaozhe Jin, Jiaxin Chen†, Yunhong Wang（北京航空航天大学）
> arXiv: [2601.13155](https://arxiv.org/pdf/2601.13155)，ICML 2026 格式投稿
> 本地 TeX 源码：`knowledge/2601.13155/main.tex`
> **一句话总结**：提出免训练的 SPTS 框架，通过"先探测目标层影响、再决定跳过哪些 token"，对 MHA 和 FFN 同时做 token skipping，并配合多阶段延迟剪枝，在长上下文 LLM 推理中实现最高 2.46× TTFT / 2.29× E2E 加速且精度保持 SOTA。

---

## 0. 综合理解与点评（讨论区）

### 读者原始理解
本文在给定的 pretrained dit 模型上做 training-free important token selection. 核心做法是根据Attention MAP选择self-attention中的重要token；用validation set运行一遍FFN并计算变换幅度最大的token作为FFN中的重要token.

### 点评：方向正确，但有三处关键偏差

** 抓住的大方向**
- 整体定位准确：在**冻结的预训练模型**上做 training-free token selection。
- 抓住了"SA 和 FFN 用不同准则"这一核心差异。

** 需要修正的三点**

1. **模型类型笔误**：本文是 **LLM**（LLaMA-3.1-8B、Qwen-2.5-7B、openPangu-1B），不是 DiT（Diffusion Transformer）。虽然都是 Transformer 架构，但场景完全不同：本文是**长上下文 LLM 推理加速**，与扩散模型无关。

2. **LTP 描述混淆了"离线校准"与"在线推理"**：读者所述"用 validation set 运行一遍 FFN 并计算变换幅度最大的 token"**不是在线推理时做的**。实际流程是：
   - **离线阶段**：用 200 条校准文本跑一遍模型，统计 FFN 通道激活显著性 → 裁剪通道 → SVD 低秩分解，得到代理网络 $f(\cdot)$。校准集只用于"塑造代理网络"，不直接参与 token 选择。
   - **在线推理阶段**：用代理网络 $f(\cdot)$ 轻量预测每个 token 的变换幅度 $\|f(\bm{X}_n)\|$，再乘以 PAP 分数做条件化打分选 TopM。**不是真跑 FFN**，否则就违背了加速初衷。

3. **PAP 不是"根据完整 Attention Map"**：PAP 是 **Partial Attention Probing**——只算全部 token 的 Key 投影 + 仅最后一个 token 的 Query 投影，对所有 head 取 softmax 平均得到一个轻量贡献分。这是"部分注意力探测"，不是完整的 token-to-token attention map（后者代价太高，等于完整跑一遍 SA）。这也是本文强调 self-predictive 的关键：用目标层自身的轻量信号，而非跨层借用旧 attention map。

### 修正后的一句话理解
本文在**冻结的预训练 LLM** 上做 training-free 的重要 token selection：在 self-attention 中用**部分注意力探测**（仅最后一个 token 的 query 对所有 key 打分）选重要 token；在 FFN 中用**离线校准 + SVD 构建的低秩代理网络**在线预测变换幅度，并与注意力分相乘做条件化选 token；再用多阶段延迟剪枝缓解冗余干扰。

### 被忽略的关键点（建议补充）
- **MSDP 延迟剪枝**：stage 边界处用最新 MHA 分数剪掉冗余候选，这是本文区别于 FTP 等工作的关键设计之一，消融贡献 +1.69%。
- **即时 KV Cache 压缩**：PAP 只缓存活跃 token 的 K/V，32K 下省 58.6% 显存，是 E2E 加速 2.29× 的重要来源。
- **浅层全量、深层 skipping**：不是所有层都跳过，浅层保留完整计算以保底。

---

## 1. 研究背景与动机

### 1.1 问题设定
长上下文 LLM 推理（prefilling + decoding）计算开销随序列长度急剧增长。现有 token 级加速方法分为两类：

| 路线 | 代表工作 | 特点 | 缺陷 |
|---|---|---|---|
| Token 剪枝（缩短序列） | PyramidInfer、GemFilter、LazyLLM、SlimInfer | 直接丢弃 token / 回算被剪 token | 端到端加速与信息保留之间存在固有权衡，回算开销大 |
| Sparse Attention / FFN skipping | MInference、FlexPrefill、FTP | 选择部分 token 参与注意力或 FFN 计算，序列完整性不变 | 三大局限见下 |

### 1.2 现有方法的三大局限（论文核心 motivation）
1. **结构优化不充分**：多数方法只针对 MHA 或只针对 FFN 做优化。而实际推理中两个模块都占大头（MInference 等 sparse attention 几乎不碰 FFN；FTP 只做 FFN）。
2. **选择准则过时**：依赖前序层的注意力分数等"旧信号"来选择目标层要跳过的 token，无法反映 token 在**目标层**的真实贡献。
3. **冗余干扰**：深层仍保留全部上下文作为候选集，冗余 token 导致注意力分数平滑化（dilution），top-token 选择越来越不可靠。

### 1.3 关键实证观察（Fig. 3）
- **(a)** 深层中 token 经过 MHA、FFN 后与输入的余弦相似度持续升高 → 残差捷径保留了绝大部分信息，token skipping 在深层可行。
- **(b)** 从某层起停掉几乎所有上下文 token 的 MHA 更新几乎无损，但停掉最后一个 token 的更新则精度骤降 → **深层 MHA 的主要职能是把前文信息聚合进最后一个 token**。
- **(c)** 累积 90% 注意力质量所需 token 数固定（不随长度增长），但更高覆盖率要求越来越多 token → 长序列下注意力分数被稀释。
- **(d)** 相邻层间 top-token 集合的 Jaccard 一致性随序列变长而下降 → 冗余 token 干扰选择。

## 2. 方法：SPTS（Self-Predictive Token Skipping）

**核心原则**："一个 token 是否适合被跳过，取决于它**在目标层内部**将产生的影响"（self-predictive：用目标层自身的轻量探测信号做预测，而不是用旧层分数）。框架为纯推理期方法，**training-free**。

### 2.0 框架基础（Eq. 1–2）
残差结构下 $\bm{Y}=\bm{X}+\mathcal{F}(\bm{X})$。选定活跃子集 $\mathcal{T}_{\mathrm{active}}$ 后：
$$\hat{\bm{Y}}[\mathcal{T}_{\mathrm{active}},:] = \hat{\bm{X}}+\mathcal{F}(\hat{\bm{X}}), \quad \hat{\bm{Y}}[\text{其余}] = \bm{X}[\text{其余}]$$
被跳过的 token 不被永久丢弃——借残差连接仍留在序列里，可参与后续层。

### 2.1 PAP：Partial Attention Probing（面向 MHA）
- **理论依据**：Eq. 3 定义的目标是选 $M$ 个 token 使最后一个 token 的输出误差最小（NP-hard），用注意力机制作为代理信号。
- **做法**：进目标层前先为全部 token 算 key 投影 $\bm{K}$，但只为**最后一个 token** 算 query 投影 $\bm{q}$；对所有 head 的 softmax 注意力取平均得到每个 token 的贡献分（Eq. 4），TopK 选出活跃集。
- **省算力细节**：Q、V 只为活跃 token 计算；K 复用预计算的 $\bm{K}[\mathcal{T}_{\mathrm{active}},:]$，后续 causal attention 与输出投影都在缩减后的集合上进行。
- **附带收益 —— 即时 KV Cache 压缩**：只有活跃 token 的 K/V 进入缓存，被跳过的 token 若在后续层重新成为活跃则再入缓存。32K 上下文下 KV 显存节省 **58.6%**，同时 decoding 也因此提速（这正是 E2E 能到 2.29× 的原因之一）。相比解码期剪枝（prefill 无加速）和 prefill 期剪枝（信息早期就丢），这种"skip + 延迟剪枝"组合允许信息灵活回流（附录 Fig. 7 对比三种缓存管理方案）。

### 2.2 LTP：Low-rank Transformation Probing（面向 FFN）
- **问题形式**：FFN 是逐 token 独立运算（Eq. 6），最小化整序列表示偏差的最优解 = 选变换幅度 $\|\mathcal{F}(\bm{X}_n)\|$ 最大的 token（Eq. 7）。在线精确求解代价太高（等于多跑一遍 FFN）。
- **低秩代理网络 $f(\cdot)$（离线构建，两步瘦身）**：
  1. **数据驱动的通道裁剪**：用校准文本（Qasper 中采 200 条）收集 FFN 输入 hidden states 得校准集 $\mathcal{G}$；每 token 激活显著性 $z(\bm{x})=|\sigma(\bm{x}\bm{W}_g)\odot\bm{x}\bm{W}_u)|$；第 $j$ 维重要性 = 其显著性值 **top-$\rho$（ρ=0.2）部分的均值**（优先保留被高响应 token 持续激活的维度）；保留 top-$D_{\mathrm{low}}$ 个通道得 $\bm{W}'$。
  2. **SVD 低秩分解**：$\bm{W}'\approx \bm{U}\bm{V}$，秩 $r\ll \min(D,D_{\mathrm{low}})$。
- **条件化变换打分（解决"自变小事关重大"问题）**：有些 token 自身变换小但承载关键任务信息，仅按 $\|f(\bm{X}_n)\|$ 会误杀。故与 PAP 探测出的注意力贡献分相乘融合：
$$S_n^{\mathrm{FFN}} = C_n^{\mathrm{FFN}}\cdot S_n^{\mathrm{MHA}}$$
即"只有既不重要又几乎不变的 token 才被跳过"。消融显示比纯注意力准则平均高 **+0.54%**。

### 2.3 MSDP：Multi-Stage Delayed Pruning
- 每层活跃数采用**固定预算** $M_{\mathrm{fixed}}$ 而非固定比例（关键 token 数不随长度增长；Qwen/openPangu 每阶段预算更大因为候选集更多）。
- 把 skipping 层划分为多个 stage：stage 内候选集 $\mathcal{T}$ 固定；stage 结束层用最新的 $S^{\mathrm{MHA}}$ 分数剪掉一批候选（LLaMA 每阶段 -1K，Qwen/openPangu -2K，Eq. 11）。
- 作用：① 降低逐层探测的开销；② 移除冗余候选缓解注意力分数稀释（对应 Fig. 3(c)(d)）；③ 为长文本任务（HPQA、Count）带来明显增益，消融平均 **+1.69%**。

### 2.4 完整推理流程（算法 1）
浅层全量计算保底 → 从指定层起启用 skipping（进入 prefill 且该层在 skipping 层内）：PAP 探测+缩减 MHA+缓存压缩 → LTP 探测+条件打分+缩减 FFN → 到达 stage 边界执行延迟剪枝收缩序列。LayerNorm、位置编码等标准操作照常。

### 2.5 实现配置
| 模型 | 总层数 | skipping 起始层 | stage 边界 | 各阶段活跃预算 | stage 端剪枝 | $(D_{\mathrm{low}},r)$ |
|---|---|---|---|---|---|---|
| LLaMA-3.1-8B-Instruct | 32 | 10 | 13/18/23/28 | 9K/7K/4K/2K | 1K/阶段 | (512, 192) |
| Qwen-2.5-7B-Instruct | 28 | 9 | 12/16/20/24 | 13K/10K/7K/4K | 2K/阶段 | (1024, 256) |
| openPangu-Embedded-1B-V1.1 | 26 | 11 | 13/16/19/22 | 13K/10K/7K/4K | 2K/阶段 | (1024, 128) |

硬件：LLaMA/Qwen 用单卡 NVIDIA A800 80G；openPangu 用 Ascend 910B2 NPU（原生 CANN 运行时）。基线统一为 FlashAttention-2。

## 3. 实验结果

### 3.1 主实验（LongBench，17 个子集 / 6 大类）
对比 sparse attention（MInference、FlexPrefill）、token 剪枝（PyramidInfer、GemFilter、LazyLLM、SlimInfer）、token skipping（FTP），在对齐平均 TTFT 加速比的条件下比较：

| 模型 | Full Model Avg. | SPTS Avg. | 第二名 | SPTS TTFT 比 |
|---|---|---|---|---|
| LLaMA-3.1-8B | 47.98 | **47.80** | SlimInfer 47.56 (+0.24) | **1.68×**（次优 LazyLLM 1.63×） |
| Qwen-2.5-7B | 47.76 | **47.50** | SlimInfer 47.43 (+0.07) | **1.36×**（并列 FTP 1.36×） |
| openPangu-1B | 31.09 | **30.62** | GemFilter 29.25 (**+1.13**) | **1.32×** |

要点：SPTS 在三个模型上均为"精度最好 + 加速最大"；FTP 因使用过时且受冗余干扰的准则，在高加速率下掉点严重（LLaMA 上 42.06）；GemFilter/PyramidInfer 直接在 prefill 中剪 token 造成不可逆信息损失。

### 3.2 效率曲线（8K–32K 输入，各 20 条采样）
- TTFT/E2E 加速比随上下文变长单调上升；32K 下 LLaMA 达 **TTFT 2.46×、E2E 2.29×**。
- SlimInfer 虽 TTFT 尚可，但回算被剪 token 导致 decode 明显减速，E2E 差。

### 3.3 消融研究
- **主组件累计加速（TTFT@32K）**：Full 1.00× → +PAP 1.44× → +PAP+LTP 2.15× → +MSDP **2.46×**（8K 时为 1.35×，说明短上下文下收益变小）。
- **PAP 与 KV 显存**：32K 时从 16GB 降到 6.63GB（-58.6%）。
- **LTP 代理网络**：同配置（40% skip）下，仅注意力准则 46.31 → 条件化变换分 **46.85**。
- **延迟剪枝 DP**：46.11 → **47.80**（+1.69%，Count 任务上尤其显著 4.32→7.24）。
- **代理网络规模**：保留秩 $r$ 比 $D_{\mathrm{low}}$ 更关键；$(512,/)$ 这类两分解矩阵尺寸失衡的配置跨数据集波动大 → 平衡的低秩设计更优。
- **top-ρ 显著性统计**：w/ top-ρ 47.80 > w/o 47.56（大激活离群值背景下，聚焦持续高激活通道更有效）。
- **探测 query 长度**：只用最后 1 个 token（47.80）优于 4 个（47.53）和 64 个（47.00）——多 token 平均会引入语义歧义（如 Count/TREC 波动）；作者指出 task-aware 自适应探测是未来方向。

## 4. 结论与评述

### 贡献小结
1. 免训练框架 SPTS，首次对 MHA 与 FFN **同时**做 token skipping 并统一在一个 self-predictive 原则下。
2. PAP（部分注意力探测）+ 即时 KV Cache 压缩；LTP（低秩代理探测 + 条件化变换打分）；MSDP（多阶段延迟剪枝抑制冗余干扰）。
3. 两个硬件平台上的一致 SOTA 精度-效率权衡（2.46×/2.29×）。

### 与本项目（token 压缩教程）的关联
- 本文是 **KV Cache 压缩 × token dropping/skipping × 低秩近似** 三条技术线的交叉范例：LTP 的通道重要性估计直接复用了 Wanda 式激活显著性统计思想；延迟剪枝与 Cake/SnapKV 等解码期 eviction 形成"何时压缩"光谱上的另一端（prefill 期 + 可回流）。
- "self-predictive"思想可以视作把在线的路由式门控（ learned predictor，如 FTP-route/EA-MoE 类）替换成**零训练的结构化代理网络**，属于免训练动态稀疏的一种低成本实现路径。

### 可能的讨论点 / 局限
- 探测依赖最后一个 token 的注意力信号，对生成任务中 decoder-only 场景合理，但对非最后位置敏感的任务（如 Count/LCC 局部检索类）收益不一。
- 超参（stage 划分、预算、起始层）目前手工设定且不同模型差异较大，自适应分配是显式的未来方向。
- 仅覆盖标准 MHA；GQA/MQA（如现代模型的 kv-head 缩减）下 PAP 的探测成本结构会有变化。

---

## 5. 讨论记录

### Q1：本文对 Self-Attention 和 FFN 的 token selection 采取了不同方式：SA 中是用 Attention score 计算；FFN 中是训练了另一个模型去预估？但本文 claim 是 training-free？

**答**：SA 侧确实是直接用注意力分数做轻量探测（PAP）；FFN 侧**并不是"训练了另一个模型"**。LTP 构建的低秩代理网络 $f(\cdot)$ 是从原 FFN 权重通过**"统计校准 + 确定性分解"**两步得到的结构近似，全程没有梯度优化，因此 training-free 的 claim 成立。

#### 三种易混淆的"构建"方式对比

| 方式 | 是否算 training-free | 本文是否使用 |
|---|---|---|
| 梯度训练新模型（反向传播、optimizer、可学习参数） | ❌ 否 | 本文未使用 |
| 数据驱动的统计校准（用校准集算激活统计量，据此裁剪） | ✅ 是（PTQ / Wanda / SparseGPT 同类） | ✅ LTP 第一步 |
| 确定性线性代数分解（SVD 近似权重） | ✅ 是 | ✅ LTP 第二步 |

#### LTP 代理网络的具体构建过程（无梯度）

1. **通道裁剪（校准，非训练）**：用 200 条 Qasper 校准文本跑一遍模型，仅收集 FFN 前的 hidden states；统计每个中间通道的 top-ρ（ρ=0.2）激活显著性 $I_j$，保留 top-$D_{\mathrm{low}}$ 个通道得到 $\bm{W}'$。此步骤**没有任何权重被梯度更新**，校准数据仅用于确定"哪些通道经常被激活"，与 Wanda、SparseGPT 用校准集做敏感度统计属于同一类操作。
2. **SVD 低秩分解（确定性分解，非训练）**：对裁剪后的 $\bm{W}'$ 直接做 SVD，截前 $r$ 个奇异值得 $\bm{U}\bm{V}$。纯线性代数操作，无学习率、无迭代优化。

因此代理网络 $f(\cdot)$ 的参数**完全来自原 FFN 权重的重排与近似**，未引入任何新的可学习参数，也未微调原 LLM 权重。

#### 本文 "training-free" 的精确含义

同时满足三件事为零：
- 零梯度优化（no backprop / no optimizer step）
- 零新增可学习参数（代理网络参数是原权重的派生量）
- 零原模型微调（原 LLM 权重冻结）

#### 为什么不真训练一个 predictor？

这正是本文与 FTP-route 等"学习式路由"工作的核心区别。训练 predictor 需要额外数据与开销，且 predictor 在长上下文任务上的分布外泛化未必稳定；而 LTP 用**目标层自身权重的低秩近似**来预测该层真实变换幅度，属于**自预测（self-predictive）**——探测信号与目标层强绑定，这也是论文标题 "Self-Predictive" 的由来。

> 注：SA 侧 PAP 同样是"自预测"——只用目标层自身的 K 投影和最后一个 token 的 Q 投影算注意力分数，没有跨层借用旧信号，也没有训练任何参数。两个模块的 token selection 都遵循"探测目标层自身影响"的统一原则。
