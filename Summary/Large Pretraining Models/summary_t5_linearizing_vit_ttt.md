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
