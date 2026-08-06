# USV: Unified Sparsification for Accelerating Video Diffusion Models

> **Paper**: [arXiv:2512.05754](https://arxiv.org/abs/2512.05754)  
> **Authors**: Xinjian Wu, Hongmei Wang, Yuan Zhou, Qinglin Lu (Tencent Hunyuan / UCAS)  
> **Venue**: CVPR 2026  
> **Date**: 2026-08-03

---

## 读者综合理解与点评

> **读者理解**：本文把 key projection 后余弦相似度相近的 token 进行叠加取平均合并，计算不同层和时间步的 attention entropy 作为稀疏度分配指标，再配合 Distillation 进行学习。

**点评：基本正确，但有三处需要补充和澄清。**

**正确部分：**
- ✅ "key projection 后余弦相似度相近的 token 进行叠加取平均合并"——准确。具体是对所有 head 的 key 取平均得到描述子 $\mathbf{m}_i$，再用余弦相似度贪心选择最相似的 source token，以均值聚合方式合并到 destination token
- ✅ "计算不同层和时间步的 attention entropy 作为稀疏度分配指标"——准确。低熵层（注意力集中、冗余多）获得更强稀疏化，高熵层保留更多计算
- ✅ "配合 Distillation 进行学习"——准确。使用 DMD2 风格的稀疏蒸馏，冻结教师模型指导学生

**需要补充/澄清的三处：**

1. **Token 合并受限于局部 3D 块**：合并不是全局进行的，而是先将 $(T,H,W)$ 网格划分为不重叠的 3D 块（大小 $(s_t, s_h, s_w)$），仅在块内合并。这是**视频感知**的关键设计——保证合并遵循时空 tubelet，避免合并不相关的远距离区域

2. **VSA ≠ 熵感知策略网络，二者是"执行"与"调度"的关系**：
   - **VSA（Video Sparse Attention）**：一种稀疏注意力的**计算方式/内核**，通过二值掩码 $\mathbf{M}^{(l)}$ 决定哪些 Q-K 对参与计算——它是"怎么算稀疏注意力"
   - **熵感知策略网络**：一个**参数无关的调度器**，根据注意力熵决定每层每步的稀疏率 $\rho^{\text{attn}}_{t,l}$——它是"每层该多稀疏"。它不基于 pruning 损失学习，而是直接用注意力熵这一内容信号重新分配预算

3. **熵感知策略同时调度两个维度**：它不仅控制注意力稀疏率 $\rho^{\text{attn}}_{t,l}$，还控制 token 合并率 $\rho^{\text{token}}_{t,l}$，在统一预算下协调两者的分配

---

## 核心思想

视频扩散模型（VDM）的推理瓶颈在于两个维度：**全局时空注意力的二次复杂度** 和 **大量迭代去噪步数**。现有加速方法（如稀疏注意力、步数蒸馏）通常单独优化某一维度，随着该维度趋近极限，其他瓶颈会成为主导，导致**收益递减**。

**USV** 提出一个端到端可训练的统一框架，通过**联合协调**三个维度的稀疏化来打破收益递减的天花板：

1. **注意力稀疏化（Attention Sparsity）**：决定哪些 token 对可以互相注意
2. **Token 稀疏化（Token Sparsity）**：通过 token merging 减少每层处理的时空 token 数量
3. **采样稀疏化（Sampling Sparsity）**：通过蒸馏减少去噪步数

三者并非独立的"trick"，而是由一个**动态稀疏化策略网络**统一协调，形成协同优化的闭环。

---

## 方法架构

### 基础框架：FastVideo

USV 以 FastVideo 为基础，继承了两个核心组件：

- **VSA（Video Sparse Attention）**：可学习的视频稀疏注意力核，通过二值掩码 $\mathbf{M}^{(l)}$ 对 Q-K 对进行筛选
- **DMD2 稀疏蒸馏**：将长步数教师模型蒸馏为少步数学生模型，使用冻结的真实分数网络和可训练的虚假分数网络进行分布匹配

### 新增组件 1：Token Merging 模块

这是 USV 的核心创新之一，显式减少 token 级冗余：

1. **3D 二部图划分**：将 $(T, H, W)$ 网格划分为大小为 $(s_t, s_h, s_w)$ 的不重叠 3D 块，每块指定一个**目标 token（destination）**，其余为**源 token（source）**

2. **基于注意力键的相似度度量**：
   - 对所有 head 的 key 取平均得到描述子 $\mathbf{m}_i$
   - 用余弦相似度计算源与目标的相似性
   - 贪心选择最相似的 $r$ 个源 token 合并到目标 token

3. **Merge 算子**：通过均值聚合将源 token 信息合并到目标 token，减少后续注意力计算的 token 数量

4. **Unmerge 算子**：注意力计算完成后，将合并后的 token 还原到原始稠密网格，保持逐 token 预测精度

5. **视频感知性**：限制在局部 3D 块内合并，天然遵循连贯的时空"tubelet"，避免合并不相关的远距离区域，保持时序一致性

### 新增组件 2：熵感知动态稀疏化策略

不同层和时间步的注意力熵差异很大——深层的注意力高度集中（低熵、高冗余），而浅层则更分散。USV 利用这一观察：

1. **注意力熵计算**：
   - 计算每层每步的注意力图的边际分布熵 $h_{t,l} \in [0, 1]$
   - 低熵 → 注意力集中 → 冗余多 → 可更大胆地稀疏

2. **熵感知分配**：
   - 以手工设计的全局稀疏率为基准
   - 根据熵值计算重要性权重 $w_{t,l} = (1 - h_{t,l})^\gamma$
   - 将稀疏预算重新分配到各层：低熵层获得更强稀疏化，高熵层保留更多计算

3. **参数无关**：该调度器无需额外可学习模块或损失项，仅用注意力熵驱动内容自适应的稀疏分配

### 训练策略

- 两阶段训练：先用固定稀疏率预热学生模型，再启用 token 稀疏和动态策略
- 蒸馏目标保持教师模型行为，预算损失约束计算量
- 推理时仅使用稀疏学生模型和策略网络

---

## 📊 实验结果

### 主结果对比（Wan2.1-1.3B, 480p, 81帧 ≈ 131K tokens）

| 方法 | 稀疏率 | 步数 | VBench Total | VBench Quality | VBench Semantic | E2E 加速 | DiT 加速 |
|------|--------|------|-------------|----------------|-----------------|---------|---------|
| Wan2.1 (dense) | 0× | 50 | 78.8 | 83.5 | 60.2 | 1.0× | 1.0× |
| FastWan (baseline) | 0.80 | 3 | 80.8 | 84.5 | 66.2 | 20.3× | 73.0× |
| **USV (ours)** | **0.95** | 3 | **80.7** | **84.8** | 64.1 | **22.7×** | **83.3×** |

- USV 实现了 **83.3× 去噪加速** 和 **22.7× 端到端加速**
- 在更高稀疏率（0.95 vs 0.80）下仍保持甚至略优的 VBench 分数
- 相比 FastWan 额外获得 12-15% 的加速提升

### 消融实验

逐步开启三个稀疏维度的效果：

- **仅 VSA（注意力）**：1.7× E2E 加速，质量基本不变
- **仅 DMD（步数）**：18.2× E2E 加速，主导性加速来源
- **VSA + DMD（FastWan）**：20.3× E2E，质量最佳
- **Token Merging + DMD**：速度快但质量下降（缺乏协调的稀疏化破坏信息流）
- **USV（三者联合）**：22.7× E2E，83.3× DiT，质量保持最佳

关键发现：**单独的 token merging 若无注意力稀疏配合会导致质量下降**，但三者协同可获得超加性收益。

### 动态策略的重要性

- 反转静态策略（将高稀疏分配给后期步）导致灾难性时序闪烁和纹理破坏
- USV 的熵感知动态策略自适应调整每层每步的稀疏率，保持纹理保真和运动连续性

### 序列长度扩展性

随着序列长度增加（99K → 131K → 193K tokens）：
- E2E 加速：20× → 23× → 32×
- DiT 加速：65× → 83× → 96×
- **统一稀疏化从更长的时空上下文中获益更多**

---

## 💡 关键贡献与启示

1. **问题诊断**：首次系统分析了视频扩散模型中孤立加速策略的收益递减机制
2. **统一框架**：首次在端到端可训练框架中联合协调注意力、token 和采样三个维度的稀疏化
3. **动态策略**：基于注意力熵的参数无关调度器，实现内容自适应、时间步自适应、层级自适应的稀疏分配
4. **超加性收益**：证明多维度协同设计可获得远超各部分之和的加速效果
5. **实用价值**：在实际大规模视频生成基准上验证了高加速比下的视觉质量保持

---

## 🔗 相关工作参考

- **FastVideo** (2024)：USV 的基础框架，结合 VSA + DMD2
- **VSA** (2025)：可学习视频稀疏注意力，层次化粗到细方案
- **DMD2** (2024)：改进的分布匹配蒸馏，少步采样
- **Blade** (2025)：块稀疏注意力 + 步数蒸馏的联合优化
- **VMoBA** (2025)：视频扩散的混合块注意力

---

## 📝 总结

USV 的核心价值在于**"统一"和"动态"**：

- **统一**：不再将注意力稀疏化、token 合并、步数蒸馏视为独立技巧，而是在单一优化目标下协同编排
- **动态**：通过熵感知策略网络，实现每层、每步、每输入的自适应稀疏分配，而非固定的静态模式

实验表明，这种多维度协同设计能够在极端加速比（83× 去噪加速）下保持视觉质量，为高效视频生成提供了一条切实可行的路径。未来工作可探索将 USV 应用于其他生成任务。

---

##  深度问答 (Q&A)

### Token Merging 模块

#### Q1: 3D 二部图划分中，每个 3D 块是否只有 1 个目标 token，其余都是 source token？

**是的。** 论文明确描述：将 $(T, H, W)$ 网格划分为大小为 $(s_t, s_h, s_w)$ 的不重叠 3D 块，**每个块指定且仅指定一个 destination token**（集合 $\mathcal{B}$ 中每块恰好一个），其余全部作为 source tokens（集合 $\mathcal{A}$）。

具体来说：
- 若每个 3D 块包含 $s_t \times s_h \times s_w$ 个 token
- 则其中 1 个是 destination，$(s_t \times s_h \times s_w - 1)$ 个是 source
- 总块数 = 总 token 数 / $(s_t \times s_h \times s_w)$
- 这在每个 3D 块内部形成了一个二部图结构

这种设计的好处是：每个块内的合并操作是**局部**的，不会跨块合并，从而保持视频的时空连贯性。

#### Q2: 描述子（descriptors）在后续流程中起到什么作用？

描述子 $\mathbf{m}_i$ 是**所有注意力头的 key 的均值**，其作用是：

1. **相似度度量的基础**：通过对每个 token 的 key 向量在所有 head 上取平均，得到一个紧凑的表示。用它来计算 source-destination 之间的余弦相似度，决定哪些 token 最冗余
2. **内容感知的合并决策**：不是随机合并，而是基于注意力空间中的语义相似度来决定"谁和谁可以合并"
3. **计算高效**：直接利用已有的 Q/K 投影结果，无需额外计算

简言之，描述子是 token 合并的**"决策依据"**——它告诉我们哪些 token 在注意力计算中是可替代的（相似度高的 token 在 self-attention 中会产生相似的 K/V 表示，合并后对计算结果影响最小）。

#### Q3: Token Merging 的具体操作是怎样的——prune + overlay？如何保证 unmerge？

**不是简单 prune + overlay，而是基于均值聚合的合并。** 具体流程：

**Merge 操作：**
- 对于每个 destination token $j$，找到所有要合并到它的 source tokens 集合 $\mathcal{M}(j)$
- 更新 destination 特征为**均值聚合**：
  $$\tilde{\mathbf{z}}_j = \frac{\mathbf{z}_j + \sum_{i \in \mathcal{M}(j)} \mathbf{z}_i}{1 + |\mathcal{M}(j)|}$$
- 这是**等权平均**，不是简单叠加。信息被压缩到更少的 token 中

**Unmerge 操作（关键！）：**
- 合并时**必须存储映射关系**：$\mathcal{U}$（未合并的 source）、$\mathcal{M}$（已合并的 source）、$\mathcal{B}$（所有 destination）、以及每个 source 的目标映射 $j^*(i)$
- 注意力计算完成后，通过映射还原稠密网格：
  - 未合并 source $\in \mathcal{U}$ → 直接取自身更新后的值
  - 所有 destination $\in \mathcal{B}$ → 取自身更新后的值
  - 已合并 source $\in \mathcal{M}$ → 取其对应 destination $j^*(k)$ 的更新后的值

**核心思想**：信息从 source 汇聚到 destination（merge），计算完成后再从 destination 分发回 source（unmerge）。这不是简单的"删掉再拼回"，而是一个**信息压缩-计算-分发**的可逆过程。存储的映射表就是逆变换的依据。

#### Q4: 实验中对比了哪些其他 token merge 方法？当前 SOTA 是什么？

**USV 论文中未与其他 token merge 方法进行直接对比。** 消融实验仅测试了"有/无 token merging"对 USV 整体性能的影响。

关于当前领域中 token merging/caching 相关方法的横向对比：

| 方法 | 类型 | 特点 | 适用场景 |
|------|------|------|----------|
| **ToMe** (Bolya+ 2023) | Token Merging | 基于注意力相似度的贪心合并，CV 领域开创性工作 | ViT 图像分类 |
| **Dynamic ToMe** | Token Merging | 动态调整每层合并率 | ViT 下游任务 |
| **DiffSparse** (ICLR 2026) | Token Caching | 可学习的分层缓存分配，代价矩阵+动态规划 | 图像/视频扩散 |
| **USV Token Merging** (CVPR 2026) | Token Merging | 3D 局部二部图+视频感知约束+熵感知动态调度 | 视频扩散 |
| **VEDA** (ICML 2026) | Sparse Attention | Tile 级稀疏注意力蒸馏，非 token merging | 视频扩散 |

**USV 的 token merging 独特性**在于：
1. **视频感知**：限制在局部 3D 块内合并（而非全局），天然遵循时空 tubelet，保持时序一致性
2. **与注意力稀疏协同**：消融证明单独 token merging 反而降低质量，必须与注意力稀疏配合才能获得收益
3. **熵感知动态调度**：合并率不是固定的，由熵感知策略网络根据每层每步的注意力自适应调整

**当前视频生成领域的 token 级加速 SOTA**：USV 整体框架在视频生成（Wan2.1-1.3B）上实现了 22.7× E2E 加速，DiffSparse 在图像生成（PixArt-α）上实现了 1.91× 加速。两者技术路线不同——USV 侧重多维度稀疏化协同设计，DiffSparse 侧重学习最优 token 缓存分配。

### 训练策略

#### Q5: 学生模型（Student）比 Teacher 模型更小吗？

**不是。Student 和 Teacher 使用完全相同的骨干网络（Wan2.1-1.3B）。**

两者的区别在于**计算模式**而非模型大小：

| 维度 | Teacher | Student |
|------|---------|---------|
| 注意力模式 | 全注意力 | VSA 稀疏注意力 |
| Token 数量 | 全部 N 个 | N-r 个（合并后） |
| 去噪步数 | 50 步 | 3 步 |
| 总参数量 | 相同 | 相同 |

加速来自三个维度的**稀疏化**（减少计算量），而非**压缩模型**（减少参数量）。这与模型压缩中的"剪枝"和"量化"不同——这里的稀疏化是**运行时**的动态决策，而非**静态**的模型结构改变。

### 整个 Pipeline

#### Q6: Distillation、稀疏度分配、Token Merge 三者是 decoupled 的吗？能否独立替换？

**部分解耦，但协同设计是关键。** 首先需要澄清一个重要概念区分：

- **VSA（Video Sparse Attention）**：一种稀疏注意力的**计算内核**，通过二值掩码 $\mathbf{M}^{(l)}$ 决定哪些 Q-K 对参与计算——它是"怎么算稀疏注意力"
- **熵感知策略网络**：一个**参数无关的调度器**，根据注意力熵决定每层每步的稀疏率——它是"每层该多稀疏"

两者的关系是**执行者与调度者**：熵感知策略计算注意力熵 → 分配稀疏率 $\rho^{\text{attn}}_{t,l}$ → VSA 按此稀疏率执行稀疏注意力计算。

修正后的架构关系图：

```
┌─────────────────────────────────────────────────────────────┐
│                   熵感知动态策略 (parameter-free)             │
│  计算注意力熵 h_{t,l} → 分配稀疏预算到每层每步                │
│  输出: ρ^attn_{t,l} (注意力稀疏率) + ρ^token_{t,l} (合并率)  │
└──────┬──────────────────────────────────┬────────────────────┘
       │ ρ^attn_{t,l}                    │ ρ^token_{t,l}
       ▼                                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  VSA 稀疏注意力 │  │ Token Merging │  │ Distillation  │
│  (执行: 按掩码  │  │ (执行: 按合并率 │  │ (DMD2: 少步   │
│   计算注意力)   │  │   合并 token)  │  │   采样训练)   │
└──────────────┘  └──────────────┘  └──────────────┘
  可替换为其他       可替换为其他       可替换为其他
  稀疏注意力内核     token merge 方法   蒸馏方法
```

**各组件的独立性分析：**

1. **Distillation (DMD2)**：最独立。可替换为其他少步蒸馏方法（如 Consistency Model、Progressive Distillation 等），不影响另外两个组件
2. **VSA 稀疏注意力内核**：较独立。可替换为其他稀疏注意力方法（如 VEDA、VMoBA 等），只要能接受外部指定的稀疏率
3. **Token Merging**：较独立。可替换为其他合并方法（如 ToMe），只要能接受外部指定的合并率
4. **熵感知策略**：是粘合剂。它同时为 VSA 和 Token Merging 输出稀疏率/合并率，协调两者的预算分配

**关键洞察：**

- 三者**可以部分独立替换**——例如换一个 token merge 方法，只要它能接受熵感知策略输出的合并率即可
- 但**协同优化是必要的**——消融实验表明：
  - 仅 Distillation + Token Merging（无 VSA 注意力稀疏）→ 质量下降
  - 仅 VSA + Distillation → 有效但未达最优
  - 三者联合 → 获得超加性收益
- **熵感知策略是协调者**——它不参与前向计算，仅根据注意力熵重新分配预算，确保三个维度在统一计算预算下协调工作

简言之：**组件可以独立替换，但必须在统一策略下协同训练才能获得最佳效果。** 这正是 USV "Unified" 一词的核心含义——不是简单的组件堆叠，而是有机的协同设计。

#### Q7: $(T, H, W)$ 网格是什么？是 VAE 压缩后的结果吗？3D 块如何定义？

**是的，$(T, H, W)$ 是 VAE 压缩后的潜空间维度，不是原始视频像素维度。**

视频扩散模型的完整数据流：

```
原始视频像素 (T_raw × H_raw × W_raw × 3)
    │
    ▼  VAE 编码器 (空间 8× 压缩, 时间 4× 压缩)
潜空间表示 (T × H × W × C)       ← 这里的 (T,H,W)
    │
    ▼  展平为 token 序列
DiT 输入: Z ∈ ℝ^{N×d}, N = T×H×W
    │
    ▼  DiT backbone (L 层 transformer)
去噪预测
```

以 Wan2.1-1.3B @ 480p, 81 帧为例：
- 原始视频：81 帧 × ~480 × ~848 像素
- VAE 压缩后（空间 8×、时间 4×）：~21 × ~60 × ~106
- $N = T \times H \times W \approx 21 \times 60 \times 106 \approx 131\text{K}$ tokens（与论文一致）

**3D 块的定义**：在这个 $(T, H, W)$ 潜空间网格上，按块大小 $(s_t, s_h, s_w)$ 划分为不重叠的 3D 块。例如若 $(s_t, s_h, s_w) = (2, 4, 4)$，则每块包含 32 个 token，其中 1 个是 destination，31 个是 source。

**为什么在潜空间而非像素空间定义块？**
1. DiT 的所有计算都在潜空间进行，token 就是潜空间的 patch
2. 潜空间已压缩了冗余信息，块大小 $(s_t, s_h, s_w)$ 直接对应 DiT 处理的 token 邻域
3. 3D 块天然对应视频中的局部时空"tubelet"（一小段连续帧的局部区域），符合视频冗余的物理分布

#### Q8: 本文需要多少样本进行重新训练？

**论文未明确报告训练样本数量。** 这是一个信息缺失点。

论文中关于训练的描述仅提到：
- **训练数据集**：Vchitect-T2V-Dataverse（一个大规模文本-视频对数据集）
- **初始化**：从预训练的 Wan2.1-1.3B 模型初始化（**不是从零训练**）
- **训练规模**：与 FastWan baseline 保持相同的 global batch size、优化步数和总推理 FLOPs 预算
- **硬件**：NVIDIA H20 GPU
- **优化器**：AdamW + cosine learning rate decay + warm-up
- **训练策略**：两阶段（Stage 1: 固定稀疏率预热；Stage 2: 启用 token 稀疏和动态策略）

**与其他方法的训练成本对比：**

| 方法 | 训练样本量 | 训练成本 | 模型权重 |
|------|-----------|---------|---------|
| **USV** | 未报告（大规模视频数据集） | 未报告（H20 GPU） | 微调（从 Wan2.1 初始化） |
| **DiffSparse** | 10,000 captions | 4-10 GPU·h (MI250) | **冻结**（仅训练代价矩阵 C） |
| **VEDA** | 未明确（校准集小） | 1B: 23K steps; 14B: 10K steps | Stage1 冻结, Stage2 微调 |

**关键区别**：USV 是**全量微调**学生骨干网络 $f_\theta$（从预训练模型初始化），而 DiffSparse 仅训练轻量代价预测器（模型权重冻结）。因此 USV 的训练成本理论上远高于 DiffSparse，但论文未给出具体数字。

#### Q9: VSA 是本文提出的吗？既然已经做了 token merge（减少了 token 数量），为什么还需要稀疏注意力？

**VSA 不是本文提出的。** VSA 来自 FastVideo 框架（引用自 Zhang et al. 2025 的独立工作），USV 将其作为基础组件继承使用。

**核心问题：token merge 已经减少了 token 数量，为什么还需要 VSA 稀疏注意力？**

答案是：**两者作用在不同层面，是正交的、互补的，不是替代关系。**

| 维度 | Token Merging | VSA 稀疏注意力 |
|------|--------------|----------------|
| **作用对象** | 注意力矩阵的**行/列数** | 注意力矩阵的**非零元素密度** |
| **做什么** | 减少 token 数量 N → N-r | 在剩余 token 中，跳过不重要的 Q-K 对 |
| **代价变化** | $\mathcal{O}(N^2) \to \mathcal{O}((N\!-\!r)^2)$ | $\mathcal{O}(N^2) \to \mathcal{O}(N \cdot k),\ k \ll N$ |
| **比喻** | 缩小矩阵尺寸 | 稀疏化矩阵内容 |

**两者结合的效果：**

```
原始全注意力:  N × N 矩阵，全部计算         → O(N²)
仅 Token Merge: (N-r) × (N-r) 矩阵，全部计算  → O((N-r)²)  仍是稠密的！
仅 VSA:       N × N 矩阵，每行只算 k 个       → O(N·k)
两者结合:      (N-r) × (N-r) 矩阵，每行只算 k 个 → O((N-r)·k)  ← 最优
```

**具体说明：**

1. **Token merge 后的注意力仍然是稠密的**：合并后剩余 N-r 个 token，它们之间仍然两两计算注意力。如果 N-r 仍然很大（如 131K 合并 20% 后仍有 ~105K），$(N-r)^2$ 的二次复杂度依然是瓶颈

2. **VSA 在合并后的 token 上进一步稀疏化**：它通过二值掩码 $\mathbf{M}^{(l)}$ 决定每个 query 只关注哪些 key，将每行的计算量从 $O(N-r)$ 降到 $O(k)$，其中 $k \ll N-r$

3. **为什么单独 token merge 会降低质量（消融实验结论）**：token merge 是有损压缩（信息被均值聚合到更少的 token 中）。如果没有 VSA 配合，模型被迫在"更少但信息已损失"的 token 上做全注意力——既损失了信息，又没有获得足够的稀疏化收益来弥补。加上 VSA 后，模型可以**选择性地关注重要 token 对**，在信息损失和计算节省之间找到更优平衡

**稀疏率 $\rho^{\text{attn}}$ 影响的不是 token merging 的数量**，两者是独立的控制信号：

- $\rho^{\text{token}}_{t,l}$ → 控制 **token merging 数量**（合并多少 source token 到 destination）
- $\rho^{\text{attn}}_{t,l}$ → 控制 **VSA 掩码稀疏度**（注意力矩阵中多少 Q-K 对被跳过）

两者都由熵感知策略网络根据注意力熵分配，但作用于不同的稀疏维度。

#### Q10: VSA 和 Token Merge 在 USV 中如何交互？详细的数据流是怎样的？

**VSA 在 token merge 之后运行，作用于合并后的 token 序列。** 完整的数据流如下：

```
输入: Z ∈ ℝ^{N×d}  (N = T×H×W 个 token，来自 VAE 压缩的潜空间)
  │
  │  ┌─── Token Merge (受 ρ^token_{t,l} 控制) ───┐
  │  │  1. 3D 二部图划分: N 个 token → 若干 3D 块  │
  │  │  2. 计算描述子 m_i = avg_k(所有 head 的 key) │
  │  │  3. 贪心合并最相似的 r 个 source → destination │
  │  │  4. 输出: Z̃ ∈ ℝ^{(N-r)×d}                   │
  │  │     (仅包含 destination + 未合并的 source)     │
  │  └────────────────────────────────────────────┘
  │
  ▼
  Z̃ ∈ ℝ^{(N-r)×d}  (合并后的 token 序列，更短)
  │
  │  ┌─── VSA 稀疏注意力 (受 ρ^attn_{t,l} 控制) ───┐
  │  │  1. Coarse Stage: 立方体级 mean pooling        │
  │  │     → 立方体注意力 → 预测 critical 立方体     │
  │  │  2. Fine Stage: 在选中的 K 个立方体内部       │
  │  │     做 token 级稀疏注意力                     │
  │  │  3. 门控融合: O = O_c ⊙ G_c + O_f ⊙ G_f      │
  │  │  4. 输出: Z̃' ∈ ℝ^{(N-r)×d}                  │
  │  │     (注意力计算后的更新 token)                 │
  │  └────────────────────────────────────────────┘
  │
  ▼
  Z̃' ∈ ℝ^{(N-r)×d}  (更新后的合并 token)
  │
  │  ┌─── Unmerge ──────────────────────────────────┐
  │  │  1. 查映射表: 源位置 → 目标位置               │
  │  │  2. 合并的 source → 取对应 destination 的值    │
  │  │  3. 未合并的 source → 取自身的值               │
  │  │  4. 输出: Ẑ ∈ ℝ^{N×d}                        │
  │  │     (恢复到原始稠密 token 网格)               │
  │  └────────────────────────────────────────────┘
  │
  ▼
  Ẑ ∈ ℝ^{N×d}  (稠密 token，继续下一层)
```

**为什么"单独 token merge 会降低质量"——基于消融数据的事实与分析：**

### 论文实际消融数据

| 配置 | Attn 稀疏 | Steps 蒸馏 | Token Merge | VBench Total | VBench Quality | E2E 加速 |
|------|-----------|------------|-------------|-------------|---------------|---------|
| Dense Wan | ✗ | ✗ | ✗ | 78.8 | 83.5 | 1.0× |
| + VSA only | ✓ | ✗ | ✗ | 79.0 | 83.2 | 1.7× |
| + DMD only | ✗ | ✓ | ✗ | 79.9 | 84.8 | 18.2× |
| FastWan (VSA+DMD) | ✓ | ✓ | ✗ | **80.8** | 84.5 | 20.3× |
| **+ merge + DMD** | ✗ | ✓ | ✓ | **77.1** | **80.8** | 20.2× |
| **USV (三者)** | ✓ | ✓ | ✓ | **80.7** | **84.8** | **22.7×** |

### 关键观察（事实）

1. **merge + DMD（无 VSA）**：Quality 从 83.5 降至 80.8，Total 从 78.8 降至 77.1 → **token merge 单独使用确实降低质量**
2. **USV（merge + VSA + DMD）**：Quality 84.8（超过 dense 的 83.5），Total 80.7 → **加入 VSA 后不仅恢复，甚至超越 dense baseline**
3. **merge + DMD vs VSA + DMD**：两者 E2E 加速相近（20.2× vs 20.3×），但前者 Quality 低 3.7 分 → **token merge 替代 VSA 作为加速手段时质量显著下降**

### 论文给出的解释（原文）

> "Introducing token merging without attention sparsity yields faster inference but suffers noticeable quality loss, suggesting that **uncoordinated sparsity may disrupt information flow**."

这是一段**定性描述**，没有机理分析或理论证明。论文并未深入解释为何 VSA 能"修复" token merge 的质量损失。

### 我的分析（标注为个人推断，非论文观点）

以下是我对这一现象的可能解释，但**论文未讨论这些**：

1. **Token merge 改变了注意力的统计分布**：Token merge 减少了 token 数量，使得剩余 token 的注意力权重分布发生变化（更集中于少数 key）。VSA 的稀疏注意力恰好利用了这种分布变化——它在注意力权重更集中的情况下，更容易识别"critical tokens"

2. **协同正则化效应**：Token merge 减少了模型参数的有效数量，VSA 的稀疏化起到了类似 dropout 的正则化作用。两者组合可能形成一种正则化机制，防止模型在信息压缩后过拟合

3. **熵感知策略的桥梁作用**：论文的熵感知策略网络同时调整 $\rho^{\text{attn}}$ 和 $\rho^{\text{token}}$。这意味着 VSA 的注意力稀疏率和 token merge 的合并率是**联合优化**的，而非独立设置——这可能是两者协同的关键

### 关于"Transformer 应该自动学会信息选择"的讨论

用户提出的质疑完全合理。Transformer 确实可以通过 softmax 权重自动学习关注重要 token。那么稀疏注意力的价值是什么？

- **计算效率视角**：即使 Transformer 能学会信息选择，全注意力的 $\mathcal{O}(N^2)$ 计算量仍是瓶颈。稀疏注意力是在**不显著损失质量**的前提下降低计算量的工程手段
- **训练信号视角**：强制的稀疏模式可能作为一种**归纳偏置**，引导模型学习更鲁棒的注意力模式。但这需要消融实验验证（如对比"学习的稀疏模式"vs"随机稀疏模式"）
- **论文未探讨**：本文没有进行"Transformer 自动选择 vs 强制稀疏"的对比实验，也没有分析 VSA 稀疏模式与 Transformer 自发学习的注意力模式的差异

### 论文未测试的配置

用户的问题非常关键。论文**没有**进行以下实验：

- ❌ Token merge + 其他稀疏注意力方法（如 STA、Sparge、MoBA 等）
- ❌ Token merge + 随机稀疏注意力（验证是否是 VSA 特有的机制）
- ❌ Token merge + 固定模式稀疏注意力（如滑动窗口）
- ❌ VSA + 其他 token merge 方法（如 ToMe 替代）

因此，**"VSA 和 token merge 的协同效应是否是 VSA 独有的"这一问题，本文并未回答**。

> 关于 VSA 的详细技术细节，参见 [VSA 论文摘要](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_vsa_video_sparse_attention.md)

---

## 🔍 扩展分析：Token Merge 与稀疏注意力的设计空间

### Token Merging 的起源与演进

USV 的 token merge 模块基于 **ToMe (Token Merging)** 框架，由 Meta AI 提出（ICLR 2023 Oral）。

#### ToMe 核心思想（Meta, arXiv:2210.09461）

**动机**：ViT 中存在大量冗余 token，token pruning 虽然加速但需要训练且信息损失大。

**核心方法**：
1. **二部图划分**：将 token 分为 source 和 destination 两组
2. **贪心合并**：用轻量级匹配算法（基于 token 相似度）将 r 个最相似的 source 合并到各自的 destination
3. **信息保留**：destination token 通过聚合（均值/加权）接收 source 的信息，而非直接丢弃
4. **无需训练**：ToMe 是 training-free 的，可直接应用于预训练 ViT

**结果**：ViT-L@512 吞吐量提升 2×，准确率仅降 0.2-0.3%

#### ToMe for Stable Diffusion（arXiv:2303.17604）

**改进**：
- 针对扩散模型的 step-by-step denoising 特性调整合并策略
- 引入 step-aware schedule：后期步骤合并更多 token（因为图像细节已确定）
- 可在不训练的情况下减少 60% token，加速 2×

#### USV 对 ToMe 的扩展

USV 的 token merge 模块与原始 ToMe 有以下关键区别：

| 维度 | 原始 ToMe | USV Token Merge |
|------|-----------|-----------------|
| **应用场景** | ViT 分类/图像 | 视频 DiT（时空 3D） |
| **划分方式** | 全局二部图 | 3D 局部块划分 |
| **相似度度量** | token 内容相似度 | key projection 后的描述子余弦相似度 |
| **与其他模块协同** | 无（独立使用） | 与 VSA、DMD2 联合训练 |
| **训练要求** | Training-free | End-to-end 训练 |

### 视频扩散中的稀疏注意力方法全景

以下是目前视频扩散中主要的稀疏注意力方法，可与 token merge 组合：

| 方法 | 年份 | 类型 | 训练 | 稀疏模式 | 加速比 | 质量影响 |
|------|------|------|------|---------|--------|---------|
| **STA** | 2024 | 固定模式 | Training-free | 滑动窗口 | ~2× | 轻微下降 |
| **Sparge** | 2024 | 固定模式 | Training-free | 轴向稀疏 | ~2× | 轻微下降 |
| **VSA** | NeurIPS 2025 | 可训练 | End-to-end | 层次化（coarse→fine） | 1.7-6× | 持平/略优 |
| **VMoBA** | ICLR 2026 | 可训练 | End-to-end | Block 稀疏 | 1.5× | 持平/略优 |
| **SVOO** | 2026 | Training-free | Offline profiling | Q-K co-clustering | 1.93× | 轻微下降 |
| **SALAD** | 2026 | 训练+推理 | 轻量微调 | 线性+稀疏混合 | 1.52-2.03× | 持平 |
| **Light Forcing** | 2026 | 可训练 | End-to-end | Chunk-aware 增长 | 2.3× | 持平 |

### 关键分析：Token Merge + 稀疏注意力的组合空间

#### 已验证的组合（USV 论文）

```
Token Merge + VSA + DMD2 → 22.7× E2E, 80.7 Total
```

#### 未验证但值得探索的组合

| 组合 | 可行性分析 | 预期效果 |
|------|-----------|---------|
| **Token Merge + STA** | STA 是固定滑动窗口模式，无法与 token merge 产生协同。Token merge 改变 token 分布，但 STA 的固定窗口无法自适应 | 可能比 VSA 差，因为 STA 无法适应 token merge 后的分布变化 |
| **Token Merge + VMoBA** | VMoBA 是可训练的 block 稀疏注意力，能与 token merge 联合训练。Block 级稀疏天然支持 token merge 后的 block 划分 | 可能与 VSA 相当，因为 VMoBA 的 block 稀疏与 token merge 的局部块划分互补 |
| **Token Merge + SVOO** | SVOO 是 training-free，无法与 token merge 联合训练。Offline profiling 基于 dense 模型，token merge 改变了注意力统计 | 不确定。Training-free 方法与训练式 token merge 存在 train-test mismatch |
| **Token Merge + SALAD** | SALAD 是线性注意力+稀疏混合，理论上与 token merge 正交。但 SALAD 的线性注意力需要投影，与 merge 后的 token 兼容性待验证 | 理论上可能有优势，因为线性注意力和 token merge 都减少了计算复杂度 |
| **Token Merge + 随机稀疏** | 用于验证协同效应是否为 VSA 特有机制 | 如果随机稀疏也能恢复质量 → 说明是 token merge 本身有效；如果不能 → 说明 VSA 确实有协同作用 |

#### USV 论文的关键局限分析

1. **仅测试了 VSA 一种稀疏注意力**：论文没有验证 token merge 与其他稀疏注意力方法的组合，因此无法得出"VSA 与 token merge 的协同是最优的"这一结论

2. **缺乏消融实验验证协同机理**：
   - 没有测试"token merge + 随机稀疏注意力"来验证协同是否为 VSA 特有
   - 没有测试不同 token merge 率与不同 VSA 稀疏率的交互曲线
   - 没有可视化分析 token merge 前后注意力分布的变化

3. **熵感知策略的设计局限**：
   - 策略网络是 parameter-free 的，仅基于注意力熵分配预算
   - 没有学习 token merge 与 VSA 之间的最优预算比例
   - 固定使用 $(4,4,4)$ 立方体划分，没有自适应调整

### 回答你的核心问题

> "本文还有进行 Token Merge 叠加其他稀疏加速的实验吗？"

**没有。** 论文仅测试了 token merge + VSA 这一种组合。从设计空间来看：

- 如果 token merge + STA 能恢复质量 → 说明协同效应来自 token merge 减少 token 数量本身，VSA 不是必要的
- 如果 token merge + 随机稀疏也能恢复质量 → 说明协同效应来自"token 数量减少 + 稀疏计算"这一结构性组合，与具体稀疏模式无关
- 如果只有 token merge + VSA 能恢复质量 → 说明 VSA 的可训练层次化稀疏模式确实与 token merge 有独特的协同效应

这三种假设的验证将大大增强论文的结论力度。遗憾的是，论文没有进行这些实验。

---

## 📖 DMD2 蒸馏详解与替代方法

### DMD2 (Distribution Matching Distillation 2) 核心原理

DMD2 是 USV 框架中的蒸馏组件，用于将 50 步 denoising 压缩为 3 步。

**基本思想**：
1. **教师模型**：原始 Wan2.1（50 步全注意力）
2. **学生模型**：与教师相同架构，仅 3 步采样
3. **蒸馏目标**：最小化学生输出分布与教师输出分布之间的距离

**关键技术**：
- **Distribution Matching**：在分布级别（而非单样本级别）匹配学生和教师的输出
- **Progressive Distillation**：逐步减少步数（50→25→12→6→3），每步都从上一阶段的学生模型初始化
- **Reflected Diffusion**：使用反射扩散过程提高训练稳定性

**局限**：
- 训练成本高（需要多次渐进蒸馏）
- 存在 covariate shift 问题（学生推理时的输入分布与训练时不同）
- 与稀疏注意力结合时可能引入额外的 train-test mismatch

### DMD2 的替代方法

| 方法 | 年份 | 核心思想 | 步数 | 质量 | 多样性 | 训练成本 |
|------|------|---------|------|------|--------|---------|
| **Progressive Distillation** | 2022 | 逐步减半步数 | 4-8 | 中等 | 好 | 高 |
| **LCM** | 2023 | Consistency Model | 1-4 | 中-高 | 好 | 中 |
| **DMD2** | 2024 | Distribution Matching | 3-4 | 高 | 好 | 高 |
| **DDIL** | 2024 | Imitation Learning + DAgger | 3-4 | 高 | 好 | 中 |
| **sCM** | 2025 | Continuous-time Consistency | 1-4 | 中 | 好 | 高 |
| **rCM** | 2025 | Score-regularized sCM | 1-4 | **高** | **好** | 高 |
| **TADA** | 2025 | Training-free ODE solver | 5-30 | 高 | 好 | 无 |

#### rCM (Score-regularized Continuous-time Consistency)

NVIDIA + 清华大学 2025 年的最新工作（arXiv:2510.08431），在 Wan2.1 上**匹配或超越 DMD2**：

- 使用 score distillation 作为 long-skip regularizer，解决 sCM 的 fine-detail 生成问题
- 在 1-4 步设置下，质量与 DMD2 相当，但**多样性更好**
- 已验证在 Cosmos-Predict2（NVIDIA）和 Wan2.1（1.3B-14B）上
- **潜在应用**：可作为 USV 中 DMD2 的直接替代

**如果将 USV 中的 DMD2 替换为 rCM**：
- 可能获得更好的生成质量和多样性
- 训练成本相当（rCM 需要 Jacobian-vector product 计算）
- 与 VSA + token merge 的兼容性需要验证

---

## 💻 开源实现状态

| 项目 | 状态 | 链接 | 说明 |
|------|------|------|------|
| **VSA (FastVideo)** | ✅ 已开源 | https://github.com/hao-ai-lab/FastVideo | 包含 VSA 稀疏注意力和 DMD2 蒸馏 |
| **ToMe** | ✅ 已开源 | https://github.com/facebookresearch/ToMe | Meta 官方实现 |
| **ToMe-SD** | ✅ 已开源 | https://github.com/dbolya/tomesd | Stable Diffusion 适配 |
| **CA-ToMe** | ✅ 已开源 | https://github.com/omidiu/ca_tome | 自适应 token merging |
| **SVOO** | ✅ 已开源 | https://github.com/Mutual-Luo/SVOO | Training-free 稀疏注意力 |
| **EasyCache** | ✅ 已开源 | https://github.com/H-EmbodVis/EasyCache | 训练-free 缓存加速 |
| **USV** | ❌ 未开源 | — | 仍在审稿阶段（CVPR 2026 Findings） |
| **VMoBA** | ⚠️ 代码未公开 | — | 论文已发表但代码未发布 |
| **rCM** | ⚠️ 代码未公开 | — | NVIDIA 内部实现 |

---

## 🌐 视频扩散加速方法全景（2024-2026）

### 按加速维度分类

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     视频扩散加速方法全景                                 │
├─────────────┬──────────────┬──────────────┬─────────────────────────────┤
│ 1. 注意力   │ 2. Token     │ 3. 采样/步   │ 4. 缓存/计算重用            │
│ 稀疏化      │ 减少         │ 蒸馏         │                             │
├─────────────┼──────────────┼──────────────┼─────────────────────────────┤
│ STA (固定)   │ ToMe (合并)  │ PD (渐进)    │ TeaCache (静态)             │
│ Sparge (固定)│ CA-ToMe (自) │ LCM (一致)   │ EasyCache (动态)            │
│ VSA (可训)   │ MPTM (提示)  │ DMD2 (分布)  │ HetCache (异构)             │
│ VMoBA (块)   │              │ sCM (连续)   │                             │
│ SVOO (训练free)│            │ rCM (分数)   │                             │
│ SALAD (线性)  │              │ TADA (求解)  │                             │
│ HASTE (头)   │              │ TurboDiff    │                             │
│ LightForcing │              │              │                             │
└─────────────┴──────────────┴──────────────┴─────────────────────────────┘
```

### 关键方法对比（Wan2.1-1.3B @ 480p, 81帧）

| 方法 | 维度 | 类型 | 加速比 | 质量影响 | 开源 |
|------|------|------|--------|---------|------|
| **Dense Wan** | — | 基准 | 1× | — | ✅ |
| **VSA (alone)** | Attn | 可训练 | 1.7× | 持平 | ✅ |
| **DMD2 (alone)** | Steps | 蒸馏 | 18.2× | 持平 | ✅ (FastVideo) |
| **FastWan** | Attn+Steps | 可训练+蒸馏 | 20.3× | 持平 | ✅ (FastVideo) |
| **SVOO** | Attn | Training-free | 1.93× | 轻微下降 | ✅ |
| **EasyCache** | Cache | Training-free | 2.1-3.3× | 轻微下降 | ✅ |
| **SALAD** | Attn | 微调 | 1.52-2.03× | 持平 | ❌ |
| **VMoBA** | Attn | 可训练 | 1.5× | 持平 | ❌ |
| **rCM** | Steps | 蒸馏 | 15-50× | 持平/略优 | ❌ |
| **USV** | Attn+Token+Steps | 端到端 | **22.7×** | 持平/略优 | ❌ |

### 关键研究方向与开放问题

1. **多维度协同设计**：USV 证明了 Attn + Token + Steps 三维协同的有效性，但仍有开放问题：
   - 是否存在最优的三维预算分配比例？
   - 缓存（Cache）作为第四维度能否进一步提升？
   - 不同维度的组合是否对不同架构（非 DiT）有效？

2. **Training-free 方法的局限**：SVOO、EasyCache 等 training-free 方法加速比有限（~2×），且无法与训练式方法（如 VSA+DMD2）组合。核心挑战是 training-free 方法难以适应 token merge 后的分布变化。

3. **rCM 作为 DMD2 替代**：rCM 在质量和多样性上超越 DMD2，将其集成到 USV 框架中可能是下一个重要方向。需要解决的问题：
   - rCM 与 VSA/token merge 的兼容性
   - rCM 的训练成本是否与 DMD2 相当
   - rCM 在更长序列（>100K tokens）上的表现

4. **自适应稀疏模式**：当前方法使用固定的 $(4,4,4)$ 立方体或固定的 Top-K。自适应调整：
   - 立方体尺寸随分辨率变化
   - Top-K 随时间步/层动态调整（熵感知策略的增强版）
   - Token merge 率与内容复杂度挂钩

5. **长视频扩展**：当前所有方法验证的最长序列为 193K tokens（~5秒 720p）。对于更长的视频（>1分钟）：
   - 稀疏化收益是否单调递增？
   - 缓存方法（EasyCache）与稀疏化方法的组合是否有效？
   - 时间维的稀疏化是否需要特殊处理？
