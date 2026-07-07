# T⁵: Linearizing Vision Transformer with Test-Time Training

- **论文链接**: [arXiv:2605.02772](https://arxiv.org/abs/2605.02772)
- **作者**: Yining Li, Dongchen Han, Zeyu Liu, Hanyi Wang, Yulin Wang, Gao Huang (清华大学)
- **发表**: ICML 2026
- **代码**: [github.com/LeapLabTHU/Transformer-to-TTT](https://github.com/LeapLabTHU/Transformer-to-TTT)

---

## 1. 核心问题与动机

Softmax Attention 的 O(N²) 复杂度严重限制了大模型（尤其是高分辨率图像生成）的扩展性。尽管已有多种线性复杂度注意力机制，但**从零训练线性模型成本极高**，而将预训练 Softmax Transformer "线性化"（继承预训练权重）是一个更实际的目标。

**核心挑战**：Softmax Attention 与标准线性注意力之间存在本质的**表示空间鸿沟**（representational gap），导致权重无法有效迁移，通常需要大量重训练或复杂蒸馏策略。

---

## 2. 核心方法

本文从两个互补角度解决线性化问题：

### 2.1 结构对齐（Structure Alignment）

**关键洞察**：Softmax Attention 本质上是一个**两层动态 MLP**（以 K 为权重、V 为输出权重，Softmax 为非线性激活），而标准线性注意力仅等价于**单层动态线性变换**，缺乏中间非线性。

$$ \text{Attn}(q, K, V) = \sigma(q K^\top) V \quad \leftrightarrow \quad \text{TTT}(q) = \sigma(q W_1') W_2' $$

作者发现 **Test-Time Training (TTT)** 的两层内模型结构与 Softmax Attention 具有**直接的结构对应关系**，使得 TTT 能够：
- 完全继承预训练 Softmax 注意力权重（Q/K/V/O 投影、MLP、Norm 层等）
- 利用可学习的内模型参数快速适应 Softmax 表示空间

**实验验证**（DeiT-Tiny）：
| 模型 | 冻结模式 Acc | 微调 Acc |
|------|:---------:|:------:|
| Linear Attn | 3.71 | 63.30 |
| Linear + Proj_QK | 24.39 | 66.23 |
| TTT-2Layer | 65.98 | 68.14 |
| TTT-SwiGLU | **67.33** | **69.25** |

非线性内模型深度越大，冻结性能越好 → 验证了非线性对弥合 Softmax 鸿沟的关键作用。最终选择 **TTT-SwiGLU** 作为默认架构。

### 2.2 表示对齐（Representation Alignment）

#### (a) 平移不变性对齐 — Key Instance Normalization

Softmax Attention 对 Key 的常数平移具有**不变性**，而 TTT 缺乏此特性。分析表明：Key 偏移会在 TTT 内模型梯度中引入 O(δ¹) 和 O(δ²) 项，积累后导致梯度爆炸（NaN）。

**实验测量**：预训练 ViT 的 Key 偏移比率 ≈ 0.5（随机初始化仅 0.07），证实了系统性 Key 偏置的存在。

**解决方案**：对 Key 应用 Instance Normalization（跨 token 维度）：
$$\hat{k}_i = \frac{k_i - \bar{k}}{\sqrt{\frac{1}{N}\sum (k_j - \bar{k})^2 + \varepsilon}}$$

消融实验显示：去掉标准差缩放影响极小（71.19% → 71.15%），但去掉均值减法立即导致 NaN，证明**中心化是关键**。LayerNorm/RMSNorm 因作用于 token 级别而非序列级别，均失败。

#### (b) 局部性对齐 — Depthwise Convolution on Q, K

Softmax Attention 天然学习强局部表示，而 TTT 倾向于全局建模。作者提出通过**梯度归因**计算隐式注意力分数 $A_{\text{implicit}}(i,j) = \partial o_i / \partial v_j$ 来分析 TTT 的局部性。

**解决方案**：对 Q、K 应用 Depthwise Convolution（DWC$_{QK}$），可选地结合 Neighbourhood Attention (NAT)：
$$\hat{q} = q + \text{DWC}(q), \quad \hat{k} = k + \text{DWC}(k)$$

DWC$_{QK}$ 在三种局部增强策略中效果最优（Acc: 69.25 → 71.19）。

---

## 3. 实验与结果

### 3.1 图像分类（ImageNet-1K）

| 模型 | Epochs | 训练比例 | Top-1 | FLOPs |
|------|:------:|:-------:|:-----:|:-----:|
| DeiT-T | 300 | 100% | 72.05 | 1.25G |
| **T⁵-T** | 30 | 10% | 71.19 | 1.34G |
| **T⁵⁺-T** | 30 | 10% | **72.06** | 1.39G |
| DeiT-S | 300 | 100% | 80.24 | 4.59G |
| **T⁵-S** | 30 | 10% | 79.00 | 4.77G |
| **T⁵⁺-S** | 30 | 10% | 79.64 | 4.86G |

仅需 10% 训练轮数即可恢复基线性能。与其他线性化方法对比（均为 30 epoch）：
- Linear: 63.30, LiT: 69.52, CLEAR: 68.66, Hedgehog: NaN, **T⁵: 71.19**

### 3.2 类别条件图像生成（DiT-XL/2）

| 模型 | Epochs | 训练比例 | FID↓ | 蒸馏 | 替换比例 |
|------|:------:|:-------:|:----:|:---:|:------:|
| DiT-XL/2 | 1400 | - | 2.27 | - | - |
| LiT-XL/2 | 280 | 20% | 2.32 | ✓ | 100% |
| Hyena-X | ~28 | 2% | 2.74 | ✗ | 50% |
| **DiT⁵-XL/2** | **8** | **0.57%** | **2.48** | ✗ | 100% |

仅需 **8 epoch** 微调（原始训练的 0.57%），无需蒸馏或算子激活，即可接近 DiT-XL/2 性能。

### 3.3 文本到图像生成（Stable Diffusion 3.5-Medium）

替换约 50% Transformer 块（13 个 image self-attention + 最后 5 个 joint attention），在 4×H20 GPU 上微调约 1 小时（3000 步）：

| 方法 | DPG-Bench | GenEval | 1024px 延迟 | 2048px 延迟 |
|------|:--------:|:------:|:---------:|:---------:|
| SD3.5-Medium | 83.83 | 0.66 | 25s | 231s |
| SD3.5-FT | 82.74 | 0.70 | 25s | 231s |
| **SD3.5-T⁵** | **84.43** | **0.69** | **19s** | **157s** |

- 1024 分辨率加速 **1.32×**，2048 分辨率加速 **1.47×**
- 生成质量持平甚至优于原始 SD3.5

---

## 4. 消融研究

### 权重继承策略（DiT-S/2）
| 策略 | FID↓ |
|------|:---:|
| Softmax 基线 | 68.40 |
| 仅继承 MLP | 89.71 |
| 仅继承 Attention | 93.18 |
| **全继承** | **68.52** |

→ 全权重继承远优于部分继承，说明 TTT 结构对 Softmax 权重的强兼容性。

### NAT 的影响
- 有 NAT: FID 68.52, 无 NAT: FID 72.98
- NAT 是可选增强，非核心组件；文本到图像实验中完全移除 NAT。

---

## 5. 贡献总结

1. **架构发现**：首次将 TTT 识别为与 Softmax Attention 结构兼容的线性复杂度架构，实现**完整权重继承**
2. **表示对齐**：揭示 Key 平移不变性和局部性偏置两个关键表示属性，提出 Key InstanceNorm + DWC$_{QK}$ 实现对齐
3. **极简线性化流程**：无需蒸馏、无需多阶段训练、无需算子初始化，**仅 1 小时微调**即可线性化 SD3.5-Medium

---

## 6. 关键启示

- 预训练 Transformer 线性化的核心不在于训练策略（蒸馏等），而在于**架构层面的结构兼容性**
- TTT 的两层内模型结构是弥合 Softmax 鸿沟的关键
- Softmax 的平移不变性使得预训练 Key 携带显著偏置，线性化时必须显式处理
- 该方法与知识蒸馏、多阶段训练等策略**正交互补**，可组合使用获得进一步提升

---

## 7. 讨论与问答

### Q1: Test-Time Training 是什么？

TTT（Test-Time Training）将序列建模转化为**在线学习问题**，核心理念是：

**传统 Softmax Attention**：无损缓存所有 K、V（O(N²) 复杂度，O(N) 显存），query 在缓存上做全局加权。
**TTT**：用一个紧凑的**内模型（inner model）**来"压缩" K、V 信息，推理时通过梯度下降让内模型学习从 K 重建 V，然后用学习到的参数处理 query。因为内模型参数量固定（与 N 无关），复杂度降为 O(N)。

#### TTT 包含两种参数：
| 参数类型 | 何时确定 | 说明 |
|---------|---------|------|
| 可学习参数 W | 训练阶段反向传播优化 | 类似普通网络权重 |
| 快速权重 Δ | 推理时每条序列动态计算 | 在 W 基础上梯度下降，W' = W - Δ |

#### 推理流程（MLP 内模型 + L2 损失）：
1. 内模型：`f_W(x) = σ(x W₁) W₂`
2. 自监督损失：`L_inner = Σᵢ ||f_W(kᵢ) - vᵢ||²`（学习从 K 重建 V）
3. 快速权重更新：`W₁' = W₁ - ∇L_inner`，`W₂' = W₂ - ∇L_inner`
4. 输出：`TTT(q) = σ(q W₁') W₂'`

#### 两个关键自由度：
- **内模型架构**：单层线性 → 退化为 Linear Attention；两层 MLP → 更强表达能力
- **自监督损失**：内积损失 或 L2 重建损失

#### 为何本文选择 TTT？
Softmax Attention 本质是"两层动态 MLP"：`σ(q K^⊤) V`，TTT 的两层内模型结构 `σ(q W₁') W₂'` 与之天然对应，因此可以**完整继承预训练 Softmax 权重**，这是其他线性注意力无法做到的。

### Q2: 为什么内模型（用 K→V 训练）直接对 q 做计算就可以？

这个问题触及 TTT 的核心工作原理。内模型在 K→V 上训练后，对 q 做前向传播能产生有意义输出的原因在于：

#### 直觉层面：泛化而非记忆

- **Softmax Attention**：q 显式地和每个 kⱼ 算相似度，按相似度加权取 vⱼ → 这是**精确的、无需学习的查找**（O(N²)）
- **TTT**：用 (K,V) 数据训练内模型，学到 K→V 的映射规律，然后对新输入 q 做**泛化预测** → 如果 q 和某些 kⱼ 相似，f(q) 自然趋近 vⱼ；如果 q 落在多个 k 之间，f 会做**插值**

#### 数学层面

以两层线性内模型 + L2 损失为例，一步梯度下降后的输出可展开为：

$$f_{W'}(q) \approx q \cdot (K^\top V) + \text{高阶交互项}$$

其中 `K^⊤ V` 就是 Linear Attention 的 KV 状态矩阵。高阶项（来自多层结构和非线性激活）赋予 TTT **超越 Linear Attention 的表达能力**，使其能更好地近似 Softmax Attention 的行为。

#### 形象类比

- **Softmax Attention**：把脑子里所有单词都翻出来，逐一和"苹果"算相似度，再用相似度加权各自的解释 → O(N²)，精确但昂贵
- **TTT**：用这些单词-解释对现场训练一个小网络，网络学会了"单词→解释"的映射规律，然后把"苹果"喂进去直接出答案 → O(N)，规律已压缩在参数里

#### 关键前提（为什么能泛化到 q）

1. **Q 和 K 共享同一语义空间**：在自注意力中，Q、K 来自同一组 token（或共享投影矩阵），q 和 k 分布在同一表示空间，学到的 K→V 映射自然适用于 q
2. **内模型有足够非线性**：单层线性 → 只能学到全局线性映射（退化为 Linear Attention）；两层 MLP → 能学到复杂的非线性映射，这正是论文 Table 1 验证的趋势（内模型越深、冻结性能越好）

#### 数学推导（详细）

**设定**：$K \in \mathbb{R}^{N \times d}$，$V \in \mathbb{R}^{N \times d}$，内模型 $f_W$，损失 $\mathcal{L}(W) = \sum_i \|f_W(k_i) - v_i\|^2$

---

**情况一：单层线性内模型 → 退化为 Linear Attention**

内模型 $f_W(k) = k W$，$W \in \mathbb{R}^{d \times d}$

$$\mathcal{L}(W) = \|KW - V\|_F^2$$

梯度：

$$\nabla_W \mathcal{L} = 2K^\top KW - 2K^\top V$$

从 $W_0 = 0$ 一步梯度下降（学习率 $\eta$）：

$$W' = 0 - \eta \cdot (-2K^\top V) = 2\eta K^\top V$$

对 query $q$ 的输出：

$$f_{W'}(q) = q \cdot W' = 2\eta \cdot qK^\top V$$

而 **Linear Attention** 的核心计算正是 $qK^\top V$（分子）。**结论：单层线性 TTT 精确等价于未归一化的 Linear Attention**。

---

**情况二：两层 MLP 内模型 → 连接 Softmax Attention**

内模型 $f_W(k) = \sigma(k W_1) W_2$，$W_1 \in \mathbb{R}^{d \times h}$，$W_2 \in \mathbb{R}^{h \times d}$

令 $H = \sigma(KW_1) \in \mathbb{R}^{N \times h}$ 为隐藏表示。

**(a) 对 $W_2$ 的梯度**：

$$\nabla_{W_2} \mathcal{L} = 2 \sum_{i} H_i^\top \cdot (H_i W_2 - v_i) = 2 H^\top (H W_2 - V)$$

**(b) 对 $W_1$ 的梯度**：

$$\nabla_{W_1} \mathcal{L} = 2 K^\top \left[ (H W_2 - V) W_2^\top \odot \sigma'(K W_1) \right]$$

其中 $\odot$ 是逐元素乘法。

**(c) 一步梯度下降后的输出**：

$$W_1' = W_1 - \eta\nabla_{W_1}\mathcal{L}, \quad W_2' = W_2 - \eta\nabla_{W_2}\mathcal{L}$$
$$\text{TTT}(q) = \sigma(q W_1') W_2'$$

**(d) 结构对比**：

$$\text{Softmax Attn}(q, K, V) = \sigma_q(q \cdot \underbrace{K^\top}_{W_1^{\text{dyn}}}) \cdot \underbrace{V}_{W_2^{\text{dyn}}}$$

$$\text{TTT}(q) = \sigma(q \cdot \underbrace{W_1'}_{\text{固定大小 } h}) \cdot \underbrace{W_2'}_{\text{固定大小 } h}$$

| 维度 | Softmax Attention | TTT |
|------|------------------|-----|
| 第一层权重 | $K^\top \in \mathbb{R}^{d \times N}$ | $W_1' \in \mathbb{R}^{d \times h}$ |
| 非线性 | Softmax（行归一化） | SiLU / SwiGLU（逐元素） |
| 第二层权重 | $V \in \mathbb{R}^{N \times d}$ | $W_2' \in \mathbb{R}^{h \times d}$ |
| 复杂度 | O(N²) | **O(N)** |

---

**(e) 泰勒展开：为什么两层比一层强？**

**一层线性**（展开到一阶）：

$$f_{W'}(q) = 2\eta \sum_{j=1}^{N} (q \cdot k_j) \cdot v_j$$

→ 每个 $v_j$ 的权重仅是 $q \cdot k_j$，只能做**线性加权**。

**两层 MLP**（展开到二阶）：

$$f_{W'}(q) \approx 2\eta \sum_{j} (q \cdot k_j) \cdot v_j + 4\eta^2 \sum_{j} \sum_{t} (q \cdot k_j)(k_t \cdot k_j) \cdot v_t + \cdots$$

→ 高阶项捕捉了 **token 间两两交互**（$k_t \cdot k_j$），使 TTT 能建模 Softmax Attention 中多个 key 竞争/协作的复杂模式。

---

#### 小结

| 层次 | 数学等价关系 |
|------|------------|
| 单层线性 TTT | $= qK^\top V$，精确等价于未归一化的 Linear Attention |
| 单层线性 + 归一化 | $= \frac{qK^\top V}{qK^\top \mathbf{1}}$，精确等价于 Linear Attention |
| 两层 MLP TTT | 结构与 Softmax Attn 同构：$\sigma(qW_1)W_2$，非线性捕获 token 间高阶交互 |

核心洞察：**TTT 的梯度下降本质是对 (K,V) 做隐式非参数回归，然后把学到的函数作用到 q**。一层 = 线性回归，两层 = 非线性回归——这和 Softmax Attention 用 Softmax 做非线性加权在数学上同构。

### Q3: Test-Time Training 的原始文章是哪篇？之前用于什么方向？

TTT 有两条发展线，来自同一研究组（Yu Sun 等人），T⁵ 论文直接继承的是第二篇。

---

#### 1. 原始 TTT 概念（ICML 2020）

> **"Test-Time Training with Self-Supervision for Generalization under Distribution Shifts"**
> Yu Sun et al., arXiv:1909.13231

- **思想**：训练时同时学主任务 + 自监督辅助任务（如旋转预测）；测试时每个样本先用自监督任务更新模型，再预测
- **应用方向**：图像分类的**分布外泛化**（distribution shift），如在干净图片训练、在损坏图片上测试时自动适应
- **局限**：每次测试要更新整个模型，不是为序列建模设计的

---

#### 2. TTT 序列建模层（2024）— T⁵ 论文直接继承

> **"Learning to (Learn at Test Time): RNNs with Expressive Hidden States"**
> Yu Sun et al., arXiv:2407.04620

- **核心创新**：将 TTT 改造为序列建模层——隐藏状态 = 内模型权重，更新规则 = 一步自监督学习
- **关键突破**：提出 TTT-Linear 和 TTT-MLP，可**直接替换 Self-Attention**，保持 O(N) 线性复杂度
- **应用方向**：语言建模（长上下文处理），对标 Mamba、Linear Attention

---

#### 两条线对比

| | TTT 2020 (ICML) | TTT 2024 |
|------|------|------|
| 全称 | Test-Time Training | Learning to (Learn at Test Time) |
| 定位 | 通用测试时适应方法 | 序列建模层（Attention 替代） |
| 更新对象 | 整个模型参数 | 内模型权重（作为隐藏状态） |
| 复杂度 | 与原始模型相同 | **O(N) 线性** |
| 原始应用 | 图像分类（分布偏移） | 语言建模（长序列） |

---

#### TTT 后续扩展（T⁵ 论文提及）

| 工作 | 方向 |
|------|------|
| LACT, Atlas | 语言建模 |
| TTT-Video | 视频生成 |
| TTT3R | 3D 重建 |
| ViT³ (Han et al., 2025) | 视觉骨干网络设计 → 为 T⁵ 奠定基础 |

T⁵ 的定位：在 ViT³ 的视觉 TTT 基础上，解决**预训练 Softmax Transformer → TTT 高效转换**的问题。

### Q4: TTT 中的 W 参数是在训练时学习，还是在推理时学习？

**答案：两个阶段都学，但学的内容和目的不同。**

---

#### 阶段一：训练时（Training Time）— 学习基础权重 W

内模型的基础权重 $W = (W_1, W_2)$ 在**训练阶段**通过标准反向传播学到：

1. 前向传播经过 TTT 层（包括内模型的梯度更新步骤）
2. 计算下游任务损失（分类交叉熵 / 扩散去噪损失等）
3. 梯度回传，更新 $W_1$、$W_2$

训练完成后的 $W$ 是一个**良好的初始化点**，编码了通用的 K→V 映射规律。

---

#### 阶段二：推理时（Test-Time）— 学习快速权重 Δ

推理时，对于每条输入序列，$W$ 被临时的梯度下降更新为 $W'$：

$$W' = W - \eta \cdot \nabla_W \mathcal{L}_{inner}(W)$$

其中 $\mathcal{L}_{inner} = \sum_i \|f_W(k_i) - v_i\|^2$，用当前序列的 K、V 做**自监督学习**。

| | 训练阶段 | 推理阶段（Test-Time） |
|------|------|------|
| 更新参数 | W（基础权重） | W' = W - Δ（快速权重） |
| 优化目标 | 下游任务损失 | 自监督损失（K→V 重建） |
| 梯度来源 | 任务标签/扩散目标 | 序列自身的 K、V |
| 更新范围 | 整个数据集 | **每条序列独立计算** |
| 持久性 | 保存为模型参数 | **用完即弃**，不跨序列传递 |
| 谁在学 | 开发者训练模型 | **模型自己在推理时现场学** |

---

#### 形象类比

- **训练时的 W**：出发前学的"通用旅行手册"——覆盖常见场景，但不够具体
- **推理时的 W'**：到了目的地后，现场看了几条街名和地标（K→V），快速更新了脑内路线模型——针对当下环境做即时适配

---

#### T⁵ 论文中的特殊性

T⁵ 的 W 参数**完全继承自预训练 Softmax Transformer 的 QKV 投影权重**（通过重参数化转换），然后推理时靠 Key InstanceNorm 和 DWC 保证梯度更新稳定。因此：

- **训练阶段**：权重直接从预训练模型搬过来，几乎不需重新训练（冻结 or 极少微调）
- **推理阶段**：TTT 机制逐序列做自监督梯度更新，实现 O(N) 线性注意力

这正是 T⁵ 的核心贡献——**训练阶段用预训练权重直接初始化 W，推理阶段 TTT 自动做序列级适配**，省去了从头训练的代价。

### Q5: TTT 的入参只有 q 吗？

**从输出计算的角度：是的，TTT 只接收 q。但完整机制需要 K 和 V 在前一步更新内模型。**

---

#### 两阶段机制

TTT 不是"只有 q 作为输入"，而是**分两步**，每步输入不同：

**阶段 1：内模型更新（输入 K、V）**

$$\mathcal{L} = \sum_i \|f_W(k_i) - v_i\|^2, \quad W' = W - \eta \cdot \nabla_W \mathcal{L}$$

用序列自身的 K、V 做自监督训练，将 K→V 的映射信息"压缩"进 $W'$。

**阶段 2：输出计算（输入只有 q）**

$$\text{TTT}(q) = \sigma(q W_1') W_2'$$

$W'$ 里已经包含了整条序列的信息，所以只需把 q 喂进去。

---

#### 对比 Softmax Attention

| | 输入 | 说明 |
|------|------|------|
| Softmax Attn(q, K, V) | q, K, V **同时传入** | 显式计算 q 与每个 k 的相似度 |
| TTT(q) | **只有 q** | K、V 已通过梯度更新"编码"进 $W'$ |

核心设计：**K、V 的信息不是通过注意力矩阵存储的，而是通过梯度下降压缩进内模型权重里**。输出时不需要再显式访问 K 和 V。

---

#### 形象类比

- **Softmax Attention** = 去图书馆查资料：每次有人问问题（q），都要把书架上所有书（K）翻一遍，按相关度加权抄答案（V）→ 每次都遍历全部
- **TTT** = 雇一个研究助理（内模型）：先把所有资料（K,V）喂给助理学习一遍，之后有人来问问题（q），助理直接用学到的东西回答 → 不需要再见原始资料

资料（K,V）只用来训练助理，回答问题（q）时助理一个人就够了。这也是 TTT 能做到 **O(N) 线性复杂度**的根本原因——q 不需要和每个 k 算相似度。

### Q6: 推理时是否需要跑一遍 Full Attention 来指导内模型更新？

**不需要。无论训练还是推理，内模型的自监督更新都不依赖 Full Attention。**

---

#### 澄清一个常见误解

**误解**：内模型的更新需要 Full Attention 的输出作为 Ground Truth

**实际**：内模型的自监督目标是 **V 本身**，不是 Attention 的输出

$$\mathcal{L}_{inner} = \sum_i \| \underbrace{f_W(k_i)}_{\text{内模型预测}} - \underbrace{v_i}_{\text{V 本身}} \|^2$$

V 是从输入 token 直接通过线性投影得到的，和 K、Q 一样——**不涉及任何 Attention 计算**。

---

#### 实际流程

```
输入 X → W_Q, W_K, W_V 投影 → Q, K, V
                                   │
          ┌────────────────────────┘
          ↓
   内模型训练：输入 K，预测 V̂, 目标 = V
          ↓
   W' = W - η·∇L （梯度更新，O(N)）
          ↓
   q → σ(q·W₁')W₂' → TTT 输出 （O(N)）
```

全程 O(N)，不涉及 Softmax Attention。

---

#### 具体数值示例

输入 token 投影后得到：

$$K = \begin{bmatrix} 0.5 & 0.3 \\ -0.2 & 0.8 \\ 0.7 & 0.1 \end{bmatrix}, \quad V = \begin{bmatrix} 1.2 & -0.4 \\ 0.3 & 0.9 \\ -0.5 & 1.1 \end{bmatrix}$$

内模型自监督训练：
1. 输入 $k_1$，预测 $\hat{v}_1 = f_W(k_1)$，目标 $v_1 = [1.2, -0.4]$
2. 输入 $k_2$，预测 $\hat{v}_2 = f_W(k_2)$，目标 $v_2 = [0.3, 0.9]$
3. 输入 $k_3$，预测 $\hat{v}_3 = f_W(k_3)$，目标 $v_3 = [-0.5, 1.1]$

损失 $\mathcal{L} = \sum_i \|\hat{v}_i - v_i\|^2$，梯度下降更新 W。**V 在这里就是 Ground Truth，和 Full Attention 没任何关系。**

---

#### TTT vs Softmax Attention：输入相同，聚合方式不同

| | Softmax Attention | TTT |
|------|------|------|
| Q, K, V 来源 | 输入 × 投影矩阵 | **完全一样** |
| 信息聚合 | $\text{Softmax}(QK^\top)V$ | K→V 自监督训练 + q 经内模型输出 |
| 需要 Full Attention？ | 它自己就是 | **不需要** |

两者输入完全一致（Q, K, V 都来自输入投影），只是聚合方式不同。TTT 是独立、自洽的计算机制。

---

#### 训练/转换阶段：Full Attention 只用于提供初始权重

```
预训练 Softmax 模型
        ↓
  提取 W_Q, W_K, W_V
        ↓  重参数化
  TTT 内模型的 W₁, W₂（直接继承）
        ↓  冻结或极少微调（0.5%~10% 训练轮数）
  可用的 TTT 模型
```

微调阶段模型以 TTT 架构端到端训练，内模型的梯度更新步骤可微（在计算图中反向传播），**全程不涉及 Full Attention**。Full Attention 模型唯一的贡献就是提供了高质量的初始权重。

### Q7: TTT 的训练和推理机制总结

**训练阶段**：内模型通过反向传播学习基础权重 W，目标是让内模型具备 K→V 的通用映射能力。$W_1$、$W_2$ 作为模型参数保存下来。

**推理阶段**：对于每条输入序列：
1. 用当前序列的 K、V 对内模型做一步梯度下降 → 得到 $W' = W - \eta\nabla_W\mathcal{L}$
2. 将 query q 输入更新后的内模型 → $\text{TTT}(q) = \sigma(q W_1') W_2'$

**关键澄清**：最终输出不是"梯度乘 q"，而是 **q 乘上被梯度更新过的权重**。以单层线性为例，$W' = 2\eta K^\top V$，则 $\text{TTT}(q) = q W' = 2\eta \cdot qK^\top V$——在这个特例下，输出确实等价于 q 乘上梯度累积结果 $K^\top V$。但对于两层 MLP（T⁵ 实际使用），非线性激活使过程更复杂，本质是"q 输入一个学完了整条序列信息的新模型"。

```
训练阶段                        推理阶段
────────                        ────────
数据集 (K,V) → 反向传播学 W      单条序列的 K,V
               ↓                           ↓
           W 保存下来         内模型 f_W 做一步梯度下降
               ↓                           ↓
                              学出 W'（包含本条序列信息）
                                          ↓
                                    q → f_W'(q)
                                          ↓
                                    模拟 Attention
```

### Q8: 除了将 TTT 用于 ViT，本文还有哪些贡献？

本文的核心贡献远不止"把 TTT 套到 ViT 上"，而是从理论到实践建立了一套完整的**预训练 Transformer 线性化方法论**：

---

#### 1. 理论贡献：揭示 Softmax Attention 与 TTT 的结构同构性

这是本文最根本的洞察。作者**第一次**明确指出：

$$\text{Softmax Attn}(q,K,V) = \sigma_q(q K^\top) V \quad \longleftrightarrow \quad \text{TTT}(q) = \sigma(q W_1') W_2'$$

两者都是"非线性 × 第一权重矩阵 × 第二权重矩阵"的两层动态 MLP 结构。这个发现的意义在于：
- 解释了**为什么其他线性注意力（单层线性）无法直接继承预训练权重**（结构不匹配）
- 为完整权重继承提供了**理论依据**，使线性化真正变得"极简"

这不是简单的"把 X 用到 Y"，而是一个**被此前所有人忽略的架构级别发现**。

---

#### 2. 表示分析贡献：发现预训练 Softmax 模型的两个关键属性

作者通过系统分析发现，预训练 Softmax 模型拥有两个**结构性特征**，线性化时必须处理：

**(a) Key 的系统性偏置（平移非零均值）**

- **发现**：预训练 ViT 的 Key 偏移比率 ≈ **0.5**（随机初始化仅 0.07），说明预训练模型自发学会了在 Key 空间引入显著偏置
- **后果**：该偏置在 TTT 梯度中引入 O(δ¹) 和 O(δ²) 项，积累后导致梯度爆炸（NaN）
- **意义**：这不仅是一个工程修复，更是对 Softmax Attention 工作机理的**分析性洞察**——模型通过 Key 偏置来控制注意力的"焦点"

**(b) Softmax Attention 的局部性偏置**

- 通过**梯度归因**（$\partial o_i / \partial v_j$）计算 TTT 的隐式注意力分数，量化分析了 TTT 与 Softmax 在局部建模上的差距
- 这种分析方法本身也是一个方法论贡献

---

#### 3. 方法论贡献：建立"结构对齐 + 表示对齐"的通用线性化框架

本文提出的不是一个具体的模型，而是一套**可推广到其他架构的方法论**：

| 步骤 | 内容 | 普适性 |
|------|------|:---:|
| 结构对齐 | 选择与目标 Attention 结构同构的线性架构 | 可用于任何 Attention 变体 |
| 平移不变性对齐 | Key InstanceNorm | 通用的 Softmax → 线性转换 |
| 局部性对齐 | DWC 或 NAT | 可选的局部增强 |

这套框架的哲学是：**线性化问题的核心不是"怎么训练更好"，而是"怎么让架构匹配"**。这与之前蒸馏派（LiT、CLEAR）的思路有本质区别。

---

#### 4. 工程贡献：将方法扩展到生成模型（DiT、SD3.5）

这不是简单的模型替换，而是证明了：

- **DiT-XL/2**：仅 **0.57%** 训练轮数（8 epoch），**无需蒸馏**，全层替换
- **SD3.5-Medium**：仅 **~1 小时** 在 4×H20 上微调，部分替换（~50% 层块），1024px 加速 1.32×，2048px 加速 **1.47×**，且质量**不降反升**

这证明了 T⁵ 的方法在高分辨率扩散模型中**能直接产生部署级别的加速收益**，而不只是学术实验。

---

#### 5. 与现有工作的本质区别

| | 蒸馏派（LiT, CLEAR） | 混合派（Hedgehog, Hyena） | **T⁵** |
|------|:---:|:---:|:---:|
| 核心思路 | 用 Softmax 输出监督线性模型 | 混合使用不同注意力 | **架构同构 + 直接权重继承** |
| 训练成本 | 高（需蒸馏） | 中等 | **极低（0.5%~10%）** |
| 理论基础 | 弱（经验驱动） | 中 | **强（结构对等性）** |
| 全层替换 | ✓ | ✗（仅 50%） | **✓** |

T⁵ 的独特性在于：它不是"又一个线性化方法"，而是**重新定义了这个问题的解法**——从"蒸馏逼近"转向"结构继承"。

### Q9: 总结 — T⁵ 本质上做了什么？

本文可以浓缩为一句话：

> **用 TTT 替换 Softmax Attention，加 Key InstanceNorm 防梯度爆炸，加 DWC 补局部性，极少量微调收尾。**

| 问题 | 诊断 | 解法 | 复杂度 |
|------|------|------|------|
| 权重没法迁移 | 单层线性 vs 两层 MLP 结构不匹配 | 换 TTT（架构替换） | 零额外参数 |
| 推理时直接 NaN | Key 有系统性偏置（偏移比 ≈ 0.5） | Key InstanceNorm（去均值） | 1 行代码 |
| 局部性不如 Softmax | TTT 天然偏全局 | DWC_QK（深度卷积） | 微量参数 |

三个"技巧"的价值不在复杂度，而在**精准度**——每个都打在要害上。

#### 这篇论文的定位：诊断型 / 整合型贡献

CV/ML 领域大致有两种好论文：
- **发明型**：提出全新组件（如 Attention、BatchNorm）
- **诊断型 / 整合型**：把问题分析透，用已有组件精准组合

T⁵ 属于后者。它的力量在于：
1. **分析深度**：从 Softmax Attn 的结构属性推导出"TTT 为什么合适""Key 偏置为什么导致 NaN"
2. **极简实现**：三个改动，代码量极少，效果全面碾压蒸馏/混合方案
3. **通用性**：ViT → DiT → SD3.5 全部有效，说明抓住了本质而非特例

T⁵ 证明的核心命题是：**线性化 Transformer 不需要蒸馏、不需要复杂训练策略，只要架构选对、两个关键属性对齐，就够了。**
