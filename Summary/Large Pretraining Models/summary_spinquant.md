# SpinQuant: LLM Quantization with Learned Rotations

## 论文信息
- **论文链接**: [https://arxiv.org/abs/2405.16406](https://arxiv.org/abs/2405.16406)
- **发表会议**: ICLR 2025
- **作者**: Zechun Liu\*, Changsheng Zhao\*, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort (Meta)
- **代码**: [https://github.com/facebookresearch/SpinQuant](https://github.com/facebookresearch/SpinQuant)

## 1. 研究动机

LLM 的推理成本（内存、延迟、功耗）是实际部署的核心瓶颈。PTQ 通过将权重/激活量化到低精度来缓解这一问题，但**异常值（outlier）**的存在导致量化范围被少数极端值支配，压缩了大部分正常值的有效表示精度。

此前的工作（如 QuIP#、QuaRot）提出用**随机旋转矩阵**消除 outlier——旋转操作将少数通道上的极端值"打散"到所有通道，使分布趋于均匀。然而，SpinQuant 发现了一个关键问题：**不同随机旋转矩阵产生的量化效果差异极大**，在 LLaMA-2 7B W4A4 设定下，最好的随机旋转比最差的**高出 13 个点**（zero-shot 准确率）。即使使用更好的随机 Hadamard 矩阵，方差也有 **6 个点**。

由此自然引出一个问题：**能否通过优化找到最优的旋转矩阵？**

## 2. 核心方法

### 2.1 旋转参数化

SpinQuant 在 Transformer 架构中插入四种旋转矩阵，利用**旋转不变性**保证全精度网络输出不变。其数学基础是对于任意正交矩阵 $\mathbf{R}$（$\mathbf{R}^\top\mathbf{R}=\mathbf{I}$）：

$$\mathbf{X} \cdot \mathbf{W} = (\mathbf{X}\mathbf{R}) \cdot (\mathbf{R}^\top\mathbf{W})$$

即：在激活上乘 $\mathbf{R}$、在权重上乘 $\mathbf{R}^\top$，输出不变。所有旋转类量化方法都建立在这一基础上。

#### R1：残差路径旋转（Residual Path Rotation，可吸收）

**插入位置与动机**：在 Embedding 输出后，对**残差流本体 $\mathbf{x}$** 乘以 $\mathbf{R}_1$，使其分布更均匀（消除 outlier）；在进入 Attention/FFN 等非线性模块之前，用 $\mathbf{R}_1^\top$ 逆转回来，保证数值等价。

$$\boxed{\tilde{\mathbf{x}} = \mathbf{x} \cdot \mathbf{R}_1}$$

- $\mathbf{R}_1 \in \mathbb{R}^{D_{\text{token}} \times D_{\text{token}}}$，**全局共享**（所有 Transformer 层用同一个）
- **关键技巧**：需将 RMSNorm 的 scale 参数 $\alpha$ 吸收进后续权重矩阵（参考 SliceGPT），使得 RMSNorm 剩余部分 $\mathbf{x}/\|\mathbf{x}\|$ 是旋转**等变的**（$\|\mathbf{xR}\| = \|\mathbf{x}\|$，旋转方向被保留）

**前置理解：一个 Transformer Block 的数据流**

在讨论 R1 如何被吸收之前，必须先明确各符号的含义。标准 pre-norm Transformer Block 的数据流为：

```
x  ← 残差流本体（跨层传递的隐藏状态，进入 block 时的值）
│
├──► bypass: x 被跳过，不做任何处理，等待加回
│
└──► RMSNorm(x) → 记作 x̂   ← 这才是 Q/K/V/gate/up projection 的输入
     │
     ├──► Q = x̂ · W_Q,  K = x̂ · W_K,  V = x̂ · W_V
     │
     └──► Attn_out = softmax(QK^T/√d) · V · W_out

x₁ = x + Attn_out     ← 残差加和

├──► bypass: x₁
│
└──► RMSNorm(x₁) → 记作 x̂₁
     │
     ├──► gate = x̂₁ · W_gate,  up = x̂₁ · W_up
     │
     └──► A = SiLU(gate) ⊙ up        ← FFN 中间激活（非 x 非 x̂）
          FFN_out = A · W_down

x₂ = x₁ + FFN_out     ← 最终残差，进入下一层
```

关键区分：
- $\mathbf{x}$：残差本体，是 $\mathbf{R}_1$ 旋转的**直接作用对象**
- $\hat{\mathbf{x}} = \text{RMSNorm}_{\text{no scale}}(\mathbf{x})$：残差经过 RMSNorm 后的结果，是 Q/K/V/gate/up 的输入
- $\mathbf{A}$：FFN 内部 SiLU 后的中间激活，本质是 `gate ⊙ up` 经非线性处理后的值
- $\mathbf{x}$ 和 $\hat{\mathbf{x}}$ **不是同一个量**，之间隔了一道 RMSNorm

R1 之所以能"穿透" RMSNorm 影响下游投影，靠的是 RMSNorm 去掉 per-channel scale 后的旋转等变性：

$$\hat{\tilde{\mathbf{x}}} = \text{RMSNorm}_{\text{no scale}}(\mathbf{x} \cdot \mathbf{R}_1) = \frac{\mathbf{x}\mathbf{R}_1}{\|\mathbf{x}\mathbf{R}_1\|} = \frac{\mathbf{x}\mathbf{R}_1}{\|\mathbf{x}\|} = \hat{\mathbf{x}} \cdot \mathbf{R}_1$$

**R1 吸收表**（以一个 Attention + FFN 块为例）：

| 权重 | 原始计算 | 插入 $\mathbf{R}_1$ 后的计算 | 吸收后 |
|------|---------|---------------------------|--------|
| $\mathbf{W}_{\text{emb}}$ | $\mathbf{X}_{\text{in}} \cdot \mathbf{W}_{\text{emb}}$ | $\mathbf{X}_{\text{in}} \cdot (\mathbf{W}_{\text{emb}}\mathbf{R}_1)$ | $\mathbf{W}_{\text{emb}}\mathbf{R}_1$ |
| $\mathbf{W}_Q$ | $\hat{\mathbf{x}} \cdot \mathbf{W}_Q$ | $\overbrace{(\hat{\mathbf{x}}\mathbf{R}_1)}^{\hat{\tilde{\mathbf{x}}}} \cdot \mathbf{R}_1^\top \cdot \mathbf{W}_Q$ | $\mathbf{R}_1^\top\mathbf{W}_Q$ |
| $\mathbf{W}_K$ | $\hat{\mathbf{x}} \cdot \mathbf{W}_K$ | 同上 | $\mathbf{R}_1^\top\mathbf{W}_K$ |
| $\mathbf{W}_V$ | $\hat{\mathbf{x}} \cdot \mathbf{W}_V$ | 同上 | $\mathbf{R}_1^\top\mathbf{W}_V$ |
| $\mathbf{W}_{\text{out}}$ | $\text{Attn} \cdot \mathbf{W}_{\text{out}}$ | $\text{Attn} \cdot (\mathbf{W}_{\text{out}}\mathbf{R}_1)$ | $\mathbf{W}_{\text{out}}\mathbf{R}_1$ |
| $\mathbf{W}_{\text{gate}}$ | $\hat{\mathbf{x}}_1 \cdot \mathbf{W}_{\text{gate}}$ | $(\hat{\mathbf{x}}_1\mathbf{R}_1) \cdot \mathbf{R}_1^\top \cdot \mathbf{W}_{\text{gate}}$ | $\mathbf{R}_1^\top\mathbf{W}_{\text{gate}}$ |
| $\mathbf{W}_{\text{up}}$ | $\hat{\mathbf{x}}_1 \cdot \mathbf{W}_{\text{up}}$ | $(\hat{\mathbf{x}}_1\mathbf{R}_1) \cdot \mathbf{R}_1^\top \cdot \mathbf{W}_{\text{up}}$ | $\mathbf{R}_1^\top\mathbf{W}_{\text{up}}$ |
| $\mathbf{W}_{\text{down}}$ | $\mathbf{A} \cdot \mathbf{W}_{\text{down}}$ | $\mathbf{A} \cdot (\mathbf{W}_{\text{down}}\mathbf{R}_1)$ | $\mathbf{W}_{\text{down}}\mathbf{R}_1$ |
| $\mathbf{W}_{\text{head}}$ | $\tilde{\mathbf{x}}_{\text{final}} \cdot \mathbf{R}_1^\top \cdot \mathbf{W}_{\text{head}}$ | —（最终输出，无需逆转回原始空间） | $\mathbf{R}_1^\top\mathbf{W}_{\text{head}}$ |

**规律总结**：
- 所有**输入来自 $\hat{\mathbf{x}}$ 的投影**（Q/K/V/gate/up/LM Head），左侧乘 $\mathbf{R}_1^\top$ 抵消 RMSNorm 传下来的 $\mathbf{R}_1$
- 所有**输出汇入残差流 $\tilde{\mathbf{x}}$ 的投影**（emb/out/down），右侧乘 $\mathbf{R}_1$ 将输出对齐到 $\mathbf{R}_1$ 空间
- 总效果：残差流 $\tilde{\mathbf{x}}$ 全程处于 $\mathbf{R}_1$ 空间；进入非线性模块前，RMSNorm 的旋转等变性将 $\mathbf{R}_1$ 传递到 $\hat{\tilde{\mathbf{x}}}$，再经由 $\mathbf{R}_1^\top$ 逆转回原始空间的 $\hat{\mathbf{x}}$ 进行计算；模块输出经 $\mathbf{R}_1$ 重新旋转，加回残差流

**验证（以 Q 投影为例）**：残差 $\mathbf{x}$ 被旋转后得到 $\tilde{\mathbf{x}} = \mathbf{x}\mathbf{R}_1$。RMSNorm（scale 已吸收）将旋转传递到下游：

$$\hat{\tilde{\mathbf{x}}} = \text{RMSNorm}_{\text{no scale}}(\tilde{\mathbf{x}}) = \text{RMSNorm}_{\text{no scale}}(\mathbf{x}\mathbf{R}_1) = \hat{\mathbf{x}}\mathbf{R}_1$$

进入 Attention 前用 $\mathbf{R}_1^\top$ 逆转，等价于在 $\mathbf{W}_Q$ 左侧乘 $\mathbf{R}_1^\top$：

$$\hat{\tilde{\mathbf{x}}} \cdot (\mathbf{R}_1^\top\mathbf{W}_Q) = (\hat{\mathbf{x}}\mathbf{R}_1) \cdot (\mathbf{R}_1^\top\mathbf{W}_Q) = \hat{\mathbf{x}} \cdot \mathbf{W}_Q \quad \checkmark$$

#### R2：注意力块内头级旋转（Head-wise Attention Rotation，可吸收）

**动机**：$\mathbf{R}_1^\top$ 在进入 Attention 前逆转了旋转，导致 Attention **内部**所有激活都在原始空间中，量化时暴露于 outlier 风险。R2 专门填补这个盲区。

**插入位置**：在 Value 矩阵计算后乘 $\mathbf{R}_2$，在 out-projection 权重前乘 $\mathbf{R}_2^\top$。两者之间只有 softmax 加权求和——该操作对 $V$ 是线性的，因此 $\mathbf{R}_2$ 能完美穿透。

$$\boxed{V_{\text{rot}} = V \cdot \mathbf{R}_2, \quad \text{Attn}_{\text{rot}} \cdot \mathbf{R}_2^\top \to \mathbf{W}_{\text{out}}^{\text{fold}}}$$

- $\mathbf{R}_2 \in \mathbb{R}^{D_{\text{head}} \times D_{\text{head}}}$，**每层独立**（不同层可学习不同的 $\mathbf{R}_2$），每个注意力头有独立的 $\mathbf{R}_2$

**原始 Attention 计算（无任何旋转）**：

```
(1) Q = x̂ · W_Q
(2) K = x̂ · W_K
(3) V = x̂ · W_V
(4) scores = Q · K^T / √d
(5) weights = softmax(scores)
(6) Attn = weights · V                    ← 对 V 行向量的线性加权组合
(7) Output = Attn · W_out
```

**插入 R2 后的计算（吸收前）**：

```
(1) Q = x̂ · W_Q                            ← 不动
(2) K = x̂ · W_K                            ← 不动
(3) V = x̂ · W_V  →  V_rot = V · R₂         ← R₂ 插入在 V 之后
(4) scores = Q · K^T / √d                  ← 不动
(5) weights = softmax(scores)              ← 不动
(6) Attn_rot = weights · V_rot             ← 自然得到 R₂ 空间的 Attn_rot
             = weights · (V · R₂)
             = (weights · V) · R₂
             = Attn · R₂                   ← R₂ 穿透了 softmax 加权求和
(7) Attn_rot · R₂^T · W_out  →  Output    ← R₂^T 逆转
    = (Attn · R₂) · (R₂^T · W_out)
    = Attn · W_out  ✓
```

**为何 $\mathbf{R}_2$ 能穿透 softmax？** softmax 加权求和在 $V$ 侧是线性的——对 $V$ 的每行独立加权后求和，旋转可直接提取：

$$\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) \cdot (V\mathbf{R}_2) = \left[ \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) \cdot V \right] \cdot \mathbf{R}_2 = \text{Attn} \cdot \mathbf{R}_2$$

**R2 吸收（折叠进权重）**：

| 步骤 | 吸收前（显式旋转） | 吸收后（权重内隐式） |
|------|------------------|-------------------|
| V 计算 | $\hat{\mathbf{x}} \cdot \mathbf{W}_V$，然后 $\cdot \mathbf{R}_2$ | $\hat{\mathbf{x}} \cdot \mathbf{W}_V^{\text{fold}}$ |
| Attn 输出 | $\text{Attn}_{\text{rot}} = \text{Attn} \cdot \mathbf{R}_2$（显式） | $\text{Attn}_{\text{rot}}$ 自然产生（$V$ 已在 $\mathbf{R}_2$ 空间） |
| 进入 W_out 前 | $\text{Attn}_{\text{rot}} \cdot \mathbf{R}_2^\top$，然后 $\cdot \mathbf{W}_{\text{out}}$ | $\text{Attn}_{\text{rot}} \cdot \mathbf{W}_{\text{out}}^{\text{fold}}$ |

| 权重 | 吸收表达式 | 吸收后的等效效果 |
|------|-----------|----------------|
| $\mathbf{W}_V^{\text{fold}}$ | $\mathbf{W}_V \cdot \mathbf{R}_2$ | $V$ 计算出来天然就在 $\mathbf{R}_2$ 空间，无需显式乘 $\mathbf{R}_2$ |
| $\mathbf{W}_{\text{out}}^{\text{fold}}$ | $\mathbf{R}_2^\top \cdot \mathbf{W}_{\text{out}}$ | 输入 $\text{Attn}_{\text{rot}}$ 乘权重时天然逆转了旋转 |

**吸收后推理时的完整 Attention 路径**：

```
(1) Q = x̂ · W_Q                    ← 不变
(2) K = x̂ · W_K                    ← 不变
(3) V = x̂ · W_V^fold               ← R₂ 被 W_V 吸收，V 天然在 R₂ 空间
(4) weights = softmax(QK^T/√d)
(5) Attn_rot = weights · V          ← Attn_rot 在 R₂ 空间（量化友好）
(6) Output = Attn_rot · W_out^fold  ← R₂^T 被 W_out 吸收，一步完成逆转+投影
           = (Attn · R₂) · (R₂^T · W_out)
           = Attn · W_out  ✓
```

**结论**：$\text{Attn} \cdot \mathbf{R}_2$ **不需要在运行时显式计算**。$\mathbf{R}_2$ 已完全折叠进 $\mathbf{W}_V$ 和 $\mathbf{W}_{\text{out}}$：

- $\mathbf{W}_V$ 吸收 $\mathbf{R}_2$ → $V$ 算出来就在 $\mathbf{R}_2$ 空间 → $\text{Attn}_{\text{rot}}$ 自然在 $\mathbf{R}_2$ 空间
- $\mathbf{W}_{\text{out}}$ 吸收 $\mathbf{R}_2^\top$ → 乘 $\mathbf{W}_{\text{out}}$ 时自动逆转旋转

**运行时实际发生的**：$V$ 和 $\text{Attn}_{\text{rot}}$ 处于 $\mathbf{R}_2$ 空间，分布均匀、量化友好；但推理代码中不出现任何显式的 $\mathbf{R}_2$ 或 $\mathbf{R}_2^\top$ 矩阵乘法，所有旋转都被权重吸收，**零推理开销**。

**与 R1 组合后的完整吸收**：

| 权重 | 完整吸收 |
|------|---------|
| $\mathbf{W}_V^{\text{fold}}$ | $\boxed{\mathbf{R}_1^\top \cdot \mathbf{W}_V \cdot \mathbf{R}_2}$ |
| $\mathbf{W}_{\text{out}}^{\text{fold}}$ | $\boxed{\mathbf{R}_2^\top \cdot \mathbf{W}_{\text{out}} \cdot \mathbf{R}_1}$ |

左侧 $\mathbf{R}_1^\top$：抵消 RMSNorm 传下来的残留旋转，回到原始空间
中间 $\mathbf{W}_V$ / $\mathbf{W}_{\text{out}}$：原始权重
右侧 $\mathbf{R}_2$ / $\mathbf{R}_2^\top$：R2 的插入与抵消
右侧 $\mathbf{R}_1$：将输出旋转回残差流的 $\mathbf{R}_1$ 空间

#### R1+R2 之后：为什么 K 没有被保护？（V 有，K 没有）

KV-cache 存储的是**投影后的原始 K 和 V**（$\hat{\mathbf{x}} \cdot \mathbf{W}_K$、$\hat{\mathbf{x}} \cdot \mathbf{W}_V$），不以 Attention 计算后的结果存入。两者的输入完全相同（$\hat{\tilde{\mathbf{x}}}$），路径却不对称：

$$
\begin{aligned}
K &= \hat{\tilde{\mathbf{x}}} \cdot \mathbf{W}_K^{\text{fold}} = (\hat{\mathbf{x}} \cdot \mathbf{R}_1) \cdot (\mathbf{R}_1^\top \cdot \mathbf{W}_K) = \hat{\mathbf{x}} \cdot \mathbf{W}_K \quad &\rightarrow \text{原始空间} \quad \times \\[6pt]
V &= \hat{\tilde{\mathbf{x}}} \cdot \mathbf{W}_V^{\text{fold}} = (\hat{\mathbf{x}} \cdot \mathbf{R}_1) \cdot (\mathbf{R}_1^\top \cdot \mathbf{W}_V \cdot \mathbf{R}_2) = \hat{\mathbf{x}} \cdot \mathbf{W}_V \cdot \mathbf{R}_2 \quad &\rightarrow \mathbf{R}_2\text{ 空间} \quad \checkmark
\end{aligned}
$$

**差异根源**：$\mathbf{W}_V^{\text{fold}}$ 比 $\mathbf{W}_K^{\text{fold}}$ 多吸收了一个 $\mathbf{R}_2$。V 之所以能吸收 $\mathbf{R}_2$，是因为下游有 $\mathbf{W}_{\text{out}}$ 来吸收 $\mathbf{R}_2^\top$；而 K 的下游是 $\mathbf{QK}^\top$ 点积——**没有可学习权重来吸收逆转矩阵**，因此 K 无法像 V 一样将旋转折叠进权重。

这就需要一个**不可吸收的方案**来给 K "补票"——即 R3。

#### R3：KV-Cache 在线 Hadamard 旋转（不可吸收）

**插入位置**：Key 矩阵计算完成后、存入 KV-cache 之前，对 Key 做在线 Hadamard 变换。**对称地，Query 在计算 $\mathbf{QK}^\top$ 之前也做同样的 Hadamard 变换**，在点积中抵消。

$$\boxed{K_{\text{cache}} = \text{Hadamard}(K), \qquad Q_{\text{rot}} = \text{Hadamard}(Q)}$$

**R3 的三重保护作用**：

| 保护对象 | 机制 |
|----------|------|
| **K（KV-cache 存储）** | K 经 Hadamard 后分布均匀，4-bit 量化友好 |
| **K（Score 计算中的 K）** | 从 cache 读出的 K 已在 Hadamard 空间，直接参与 score 计算 |
| **Q（Score 计算中的 Q）** | Q 在线做 Hadamard，也在 Hadamard 空间，量化友好 |
| **Score = QK^T** | $\text{Hadamard}(Q) \cdot \text{Hadamard}(K)^\top = Q\mathbf{H} \cdot \mathbf{H}^\top K^\top = QK^\top$，等价 |

R3 是一个"买一送多"的设计：一次在线 Hadamard 同时保护了 **K 缓存量化**、**Q 的激活量化**、**K 的激活量化**三处。在 W4A4KV4 极低比特场景下，Q 和 K 的激活量化精度与 V 同样关键——R3 正是填补了这个空白。

**为什么不可吸收**：K 的下游是 $\mathbf{QK}^\top$ 点积而非可学习权重，没有权重矩阵来吸收 $\mathbf{H}^\top$ 逆转矩阵。补偿只能转移到 Q 侧，在点积中在线抵消。

**开销**：快速 Walsh-Hadamard Transform（WHT），$O(d\log d)$，配合 Tensor Core 约 ~4% 延迟。

**可吸收 vs 不可吸收的本质区别**：

R1/R2 能完全吸收，R3/R4 必须在线计算，根本原因不在于旋转本身，而在于**旋转被夹在什么东西之间**：

```
可吸收模式（R1, R2）：              不可吸收模式（R3, R4）：

  权重A  →  [旋转]  →  计算  →  [逆转]  →  权重B        投影后激活  →  [旋转]  →  (点积/非线性)  →  权重
    ↑ 可折入A              ↑ 可折入B                         ↑ 无处吸收                    ↑ ？
```

| 场景 | 逆转侧是什么 | 能否吸收？ |
|------|------------|-----------|
| R1: 残差→Q/K/V/gate/up | 可学习权重（$\mathbf{W}_Q$ 等） | ✅ $\mathbf{R}^\top$ 左乘入权重 |
| R1: Attn/FFN→残差 | 可学习权重（$\mathbf{W}_{\text{out}}$ 等） | ✅ $\mathbf{R}$ 右乘入权重 |
| R2: V→Attn→W_out | 可学习权重（$\mathbf{W}_{\text{out}}$） | ✅ $\mathbf{R}_2$ 和 $\mathbf{R}_2^\top$ 分别入 $\mathbf{W}_V$ 和 $\mathbf{W}_{\text{out}}$ |
| **R3: K→QK^T** | **点积（非权重）** | ❌ 无处吸收，Q 侧在线同步抵消 |
| **R3: Q→QK^T** | **点积（非权重）** | ❌ 在线计算，每个 token 都要做 |
| **R4: SiLU(...)→W_down** | **被 SiLU 非线性阻断** | ❌ 旋转无法跨非线性滑动 |

核心条件：**旋转必须被夹在两个可学习权重之间，中间没有非线性（或只有线性操作），才能完全吸收。** 一旦下游是点积、非线性，就只能在线计算。

#### R4：FFN 块内在线 Hadamard 旋转（不可吸收）

**插入位置**：FFN 块中，SiLU 激活函数之后、down projection 权重之前，对中间激活做在线 Hadamard 变换。

$$\boxed{A_{\text{rot}} = \text{Hadamard}(\text{SiLU}(X \cdot \mathbf{W}_{\text{gate}}) \odot (X \cdot \mathbf{W}_{\text{up}}))}$$

**为什么不可吸收**：Hadamard 变换位于 **SiLU 非线性** 和 **down projection 权重** 之间。非线性不满足旋转不变性，因此无法像 R1/R2 那样通过 $\mathbf{R}^\top\mathbf{R}=\mathbf{I}$ 滑动进权重。必须在每次推理时在线计算。

**开销**：同样用 WHT，约 ~4% 延迟。R3 + R4 合计 ~8%。

#### 吸收模式的通用规律

以三层权重 $\mathbf{W}_a \to \mathbf{W}_b \to \mathbf{W}_c$ 为例（相邻权重间无非线性），插入旋转的完整吸收模式为：

$$\boxed{\mathbf{W}_a^{\text{fold}} = \mathbf{R}_a^\top \cdot \mathbf{W}_a \cdot \mathbf{R}_b, \quad \mathbf{W}_b^{\text{fold}} = \mathbf{R}_b^\top \cdot \mathbf{W}_b \cdot \mathbf{R}_c, \quad \mathbf{W}_c^{\text{fold}} = \mathbf{R}_c^\top \cdot \mathbf{W}_c}$$

每一层的左侧 $\mathbf{R}^\top$ 抵消传入旋转，右侧 $\mathbf{R}$ 预施加传向下一层的旋转。最后一层仅吸收左侧 $\mathbf{R}^\top$。

**两种方案**：

| 方案 | 包含的旋转 | 推理开销 | 适用场景 |
|------|-----------|---------|---------|
| **SpinQuant$_{no\,had}$** | $\mathbf{R}_1$（Cayley SGD学习）, $\mathbf{R}_2$（Cayley SGD学习） | **零**（全部吸收进权重） | W4A8KV8、W3A8 |
| **SpinQuant$_{had}$** | $\mathbf{R}_1,\mathbf{R}_2$（学习）+ $\mathbf{R}_3,\mathbf{R}_4$（固定Hadamard） | 在线 Hadamard ~8% | W4A4KV4（极低比特） |

#### 与 QuaRot 旋转矩阵的对比

| 维度 | QuaRot | SpinQuant |
|------|--------|-----------|
| 可吸收旋转数量 | 1 个（全局 $\mathbf{Q}$） | 2 个（$\mathbf{R}_1$ 全局共享 + $\mathbf{R}_2$ 逐层独立） |
| 在线 Hadamard 数量 | 每块 4 个 | 每块 2 个（仅 $\mathbf{R}_3$, $\mathbf{R}_4$） |
| $\mathbf{R}_1$ 来源 | 随机 Hadamard 矩阵 | **Cayley SGD 在 Stiefel 流形上学习** |
| $\mathbf{R}_2$ | 不存在 | **Cayley SGD 学习**（QuaRot 无此自由度） |
| 吸收后参数形式 | $\mathbf{Q}^\top\mathbf{W}$ 或 $\mathbf{Q}^\top\mathbf{WQ}$ | $\mathbf{R}_1^\top\mathbf{W}\mathbf{R}_1$ + $\mathbf{R}_2^\top\mathbf{W}_{\text{out}}\mathbf{R}_1$ 等更丰富的吸收形式 |
| 搜索空间 | Hadamard 矩阵的有限离散集合 | Stiefel 流形上的连续优化 |

QuaRot 本质上只利用了 SpinQuant 中 $\mathbf{R}_1$ 这一种自由度（用随机 Hadamard 填充），而 SpinQuant 进一步引入了 $\mathbf{R}_2$（V 矩阵 + out-projection 的头级旋转）并通过 Cayley SGD 学习最优值。这意味着 SpinQuant 的**搜索空间更丰富**，能在 Stiefel 流形上找到比任意随机 Hadamard 更好的量化友好基。

### 2.2 Cayley SGD：在 Stiefel 流形上学习最优旋转矩阵

#### 一句话概括

> 本文在 **激活量化前插入旋转矩阵**，通过旋转将 outlier 打散使得激活分布更均匀，从而降低量化误差。其中 **可吸收进权重的旋转（$\mathbf{R}_1, \mathbf{R}_2$）是 learnable 的**，在 **Stiefel 流形上用 Cayley SGD 保持正交性**进行优化；不可吸收的旋转（$\mathbf{R}_3, \mathbf{R}_4$）固定为 Hadamard。优化目标是量化网络在标定集上的 **模型最终损失（交叉熵）**，权重保持 FP16 不量化，权重量化误差在旋转确定后由 GPTQ 单独处理。

#### 问题设定

旋转矩阵 $\mathbf{R}$ 必须满足正交约束 $\mathbf{R}^\top\mathbf{R} = \mathbf{I}$，这意味着 $\mathbf{R}$ 生活在 **Stiefel 流形** $\mathcal{M}$（所有 $n \times n$ 正交矩阵构成的曲面）上，而非普通的 $\mathbb{R}^{n \times n}$ 欧氏空间。

优化目标是在这个流形上找到使量化损失最小的 $\mathbf{R}$：

$$\argmin_{\mathbf{R}_1, \mathbf{R}_2 \in \mathcal{M}} \mathcal{L}_Q(\mathbf{R}_1, \mathbf{R}_2 \mid W, X)$$

展开 $\mathcal{L}_Q$ 的具体含义：

$$\boxed{\mathcal{L}_Q(\mathbf{R}_1, \mathbf{R}_2 \mid W, X) = \mathcal{L}_{\text{CE}}\Big( f_{\text{quant}}\big(X_{\text{calib}}; \; \tilde{W}, \; Q_{\text{act}}(\cdot) \big), \; y_{\text{calib}} \Big)}$$

#### 展开：$\mathbf{R}$ 如何进入损失函数（以单一线性层为例）

考虑一个最简单的线性层 $Y = X \cdot W$，将旋转 $\mathbf{R}$ 吸收后的前向过程展开为：

$$Y = Q_{\text{act}}\!\big( X \cdot \mathbf{R} \big) \;\cdot\; \big( \mathbf{R}^\top \cdot W \big)$$

这里激活 $X \cdot \mathbf{R}$ 被量化函数 $Q_{\text{act}}(\cdot)$ 压缩到低比特，然后与吸收后的权重 $\mathbf{R}^\top W$ 相乘。注意数学上若没有 $Q_{\text{act}}$，$\mathbf{R}$ 和 $\mathbf{R}^\top$ 恰好抵消，$Y = X \cdot W$，$\mathbf{R}$ 对损失毫无影响。**量化是让 $\mathbf{R}$ 暴露于梯度的关键：**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{R}} = \underbrace{\frac{\partial \mathcal{L}}{\partial Y}}_{\text{\small 来自上层回传}} \cdot \frac{\partial Y}{\partial \big(Q_{\text{act}}(X\mathbf{R})\big)} \cdot \underbrace{\frac{\partial Q_{\text{act}}(X\mathbf{R})}{\partial (X\mathbf{R})}}_{\text{\small STE = 1（直通估计）}} \cdot \frac{\partial (X\mathbf{R})}{\partial \mathbf{R}} \;\;+\;\; \frac{\partial \mathcal{L}}{\partial Y} \cdot \frac{\partial Y}{\partial (\mathbf{R}^\top W)} \cdot \frac{\partial (\mathbf{R}^\top W)}{\partial \mathbf{R}}$$

两项来源：
1. **激活侧**（$X\mathbf{R}$ 项）：$\mathbf{R}$ 改变 $Q_{\text{act}}$ 的输入分布 → round 误差变化 → 输出变化
2. **权重侧**（$\mathbf{R}^\top W$ 项）：此处优化时 $W$ 为 FP16 不量化，该项无量化噪声，贡献较小

**扩展到整个网络**：$\mathbf{R}_1$ 和 $\mathbf{R}_2$ 出现在多个层、多个位置。以 Attention 块的 V 路径为例：

$$\begin{aligned}
V &= Q_{\text{act}}\big( X \cdot \tilde{W}_V \big), \quad \tilde{W}_V = \mathbf{R}_1^\top \cdot W_V \cdot \mathbf{R}_2 \\[4pt]
\text{Attn} &= \text{softmax}\!\big( \frac{QK^\top}{\sqrt{d}} \big) \cdot V \\[4pt]
\text{Out} &= Q_{\text{act}}\big( \text{Attn} \big) \cdot \tilde{W}_{\text{out}}, \quad \tilde{W}_{\text{out}} = \mathbf{R}_2^\top \cdot W_{\text{out}} \cdot \mathbf{R}_1
\end{aligned}$$

$\mathcal{L}_Q$ 对 $\mathbf{R}_2$ 求导：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{R}_2} = \underbrace{\frac{\partial \mathcal{L}}{\partial \text{Out}}}_{\text{\small V 路径输出损失}} \!\cdot\! \left[ \frac{\partial \text{Out}}{\partial \tilde{W}_{\text{out}}} \cdot \frac{\partial \tilde{W}_{\text{out}}}{\partial \mathbf{R}_2} \;+\; \frac{\partial \text{Out}}{\partial \text{Attn}} \cdot \frac{\partial \text{Attn}}{\partial V} \cdot \frac{\partial V}{\partial \tilde{W}_V} \cdot \frac{\partial \tilde{W}_V}{\partial \mathbf{R}_2} \right]$$

$\mathbf{R}_2$ 同时出现在 $\tilde{W}_V$（左端）和 $\tilde{W}_{\text{out}}$（右端）中，梯度沿两条路径回传并叠加。而量化函数 $Q_{\text{act}}$ 在 $V$ 和 $\text{Attn}$ 两个位置插入非线性，使得 $\mathbf{R}_2$ 的旋转效果无法被下游权重完全"抵消"——round 误差对 $\mathbf{R}$ 的偏导不为零。

> **本质**：若没有 $Q_{\text{act}}$，$\mathbf{R}$ 处处与 $\mathbf{R}^\top$ 配对抵消，$\frac{\partial \mathcal{L}}{\partial \mathbf{R}} \equiv 0$。量化打破了这种对称性——$\mathbf{R}$ 在 $Q_{\text{act}}$ 的一侧旋转激活分布，$\mathbf{R}^\top$ 在另一侧与权重相乘，两侧不再完全等价。这种"不对称"正是 $\mathbf{R}$ 可优化的根源。

#### 为什么普通 SGD 不行

在平坦的欧氏空间中，梯度下降是 $\mathbf{R}' = \mathbf{R} - \eta \nabla \mathcal{L}$。但这样更新后的 $\mathbf{R}'$ **通常会离开 Stiefel 流形**：

```
欧氏空间 SGD：    R' = R - ηG    →   R'^T R' ≠ I   ← 不再是正交矩阵！
```

要保持正交性，需要对更新后的矩阵做正交投影（如 SVD: $\mathbf{R}' \gets \mathbf{U}\mathbf{V}^\top$），但这样会破坏梯度方向，优化不稳定。

#### Cayley 变换：在流形上"滑行"

**Cayley 变换**是 Stiefel 流形上的一个**测地线更新**——它保证每一步更新后 $\mathbf{R}$ 仍然正交，且过程完全可微：

$$\boxed{\mathbf{R}' = \left(I - \frac{\eta}{2}\mathbf{Y}\right)^{-1} \left(I + \frac{\eta}{2}\mathbf{Y} \right) \mathbf{R}}$$

其中：
- $\mathbf{G} = \nabla_{\mathbf{R}} \mathcal{L}$：损失对 $\mathbf{R}$ 的欧氏梯度
- $\hat{\mathbf{G}} = \mathbf{G}\mathbf{R}^\top$：将梯度"拉回"到切空间
- $\mathbf{Y} = \hat{\mathbf{G}} - \hat{\mathbf{G}}^\top$：**反对称矩阵**（$\mathbf{Y}^\top = -\mathbf{Y}$），是切空间的元素
- $\eta$：学习率

**直觉理解**：

1. **$\mathbf{Y}$ 是反对称的** → $\frac{\eta}{2}\mathbf{Y}$ 是一个无穷小旋转的生成元。反对称矩阵对应李代数，通过 Cayley 变换映射到正交群（李群），类似于旋转矩阵的指数映射 $\exp(\mathbf{Y})$ 的有理近似。

2. **$(I - \frac{\eta}{2}\mathbf{Y})^{-1}(I + \frac{\eta}{2}\mathbf{Y})$ 本身就是一个正交矩阵** → 这个因子保证了对 $\mathbf{R}$ 的右乘变换仍在 Stiefel 流形上。

3. **小 $\eta$ 下一阶近似为黎曼梯度下降** → 对 Cayley 因子做一阶泰勒展开：$(I - \frac{\eta}{2}\mathbf{Y})^{-1}(I + \frac{\eta}{2}\mathbf{Y}) = I + \eta\mathbf{Y} + O(\eta^2)$，于是
   $$\mathbf{R}' \approx \mathbf{R} + \eta \mathbf{Y}\mathbf{R} = \mathbf{R} - \eta(\mathbf{G} - \mathbf{R}\mathbf{G}^\top \mathbf{R})$$
   其中 $\mathbf{G} - \mathbf{R}\mathbf{G}^\top\mathbf{R}$ 正是 Stiefel 流形上的**黎曼梯度**。小步长下，Cayley SGD 等价于在曲面上的梯度下降——每一小步都保持 $\mathbf{R}$ 正交。

```
比喻：假设你站在一个球面上（Stiefel 流形），普通 SGD 会让你"踩入球面内部"（违反约束）。
Cayley SGD 让你沿着球面"切线方向"滑行（反对称方向），并通过 Cayley 变换把你"贴回"球面，
保证每一步都在球面上。
```

#### 与 SpinQuant 的结合

| 旋转矩阵 | 是否学习 | 原因 |
|----------|---------|------|
| $\mathbf{R}_1$ | ✅ Cayley SGD | 可吸收，学习最优的残差流旋转方向 |
| $\mathbf{R}_2$ | ✅ Cayley SGD | 可吸收，学习最优的头级 V 路径旋转方向 |
| $\mathbf{R}_3$ | ❌ 固定 Hadamard | 在线计算，不可吸收，无法"离线优化"（因为它不进入权重） |
| $\mathbf{R}_4$ | ❌ 固定 Hadamard | 在线计算，不可吸收 |

**关键细节**：
- 优化时仅对**激活**做量化（权重保持 16-bit），权重量化误差交给后续 GPTQ 处理——这种解耦设计避免了端到端联合优化的训练不稳定
- 仅需 800 个 WikiText2 样本、100 次迭代，LLaMA-2 7B 约 **25 分钟**
- $\mathbf{R}_1 + \mathbf{R}_2$ 参数量仅占模型权重的 **0.26%**，优化成本极低

### 2.3 梯度分析：为何优化旋转有效

虽然全精度下网络输出对 $\mathbf{R}$ 不变（梯度为零），但量化引入非线性后梯度变为非零：

$$\frac{\partial \sum_{ij} Q(W\mathbf{R}^{-1}) Q(\mathbf{R}X)_{ij}}{\partial \mathbf{R}_{mn}} \neq 0$$

梯度来自两部分：(1) 量化前后旋转权重的差异；(2) 量化前后旋转激活的差异。这解释了为何仅在量化网络中旋转优化才有效果。

## 3. 实验评估

### 3.1 实验设置

- **模型**: LLaMA-2 (7B/13B/70B), LLaMA-3/3.2 (1B/3B/8B), Mistral-7B
- **评估**: WikiText2 困惑度 + 8 个 zero-shot 推理任务（BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-easy, ARC-challenge, OBQA）
- **量化设定**: W4A8KV16, W4A8KV8, W4A4KV16, W4A4KV4
- **流程**: Cayley SGD 优化旋转 → GPTQ 量化旋转后权重

### 3.2 主要结果

#### W4A4KV4（极端量化）

| 模型 | FP 精度 | SpinQuant$_{had}$ 精度 | 与 FP 差距 |
|------|---------|----------------------|-----------|
| LLaMA-2 7B | 66.9 | **64.0** | **-2.9** |
| LLaMA-2 13B | 68.3 | **66.9** | **-1.4** |
| LLaMA-2 70B | 72.9 | **71.2** | **-1.7** |
| LLaMA-3 8B | 69.6 | **65.5** | **-4.1** |
| Mistral-7B | 71.0 | **68.6** | **-2.4** |

对比此前方方法：在 W4A4KV4 下，SpinQuant$_{had}$ 比 LLM-QAT 提升 **19.1 点**，比 SmoothQuant 提升 **25.0 点**，比 QuaRot 提升 **2.0~28.6 点**。

#### W4A8KV8（仅权重量化到 4-bit）

- Mistral-7B：精度差距从 12.1 缩小至 **1.6** 点
- LLaMA-3 8B：精度差距仅 **1.0** 点
- 不需在线 Hadamard 的 SpinQuant$_{no\,had}$ 已足够

#### 与 QuaRot 的对比（RTN 公平比较，LLaMA-3 8B/70B）

| 方法 | W4A4KV16 (8B) | W4A4KV4 (8B) | W4A4KV16 (70B) | W4A4KV4 (70B) |
|------|:---:|:---:|:---:|:---:|
| QuaRot + RTN | 59.5 | 58.6 | 41.5 | 41.3 |
| SpinQuant$_{had}$ + RTN | **64.6** | **64.1** | **70.1** | **70.1** |
| QuaRot + GPTQ | 63.8 | 63.3 | 65.4 | 65.1 |
| SpinQuant$_{had}$ + GPTQ | **65.8** | **65.5** | **69.5** | **69.3** |

### 3.3 消融实验

**学习旋转 vs 随机旋转**：学习旋转在各种模型和比特设置下一致优于最优随机旋转，提升最高 **16.2 点**。

**旋转初始化**：优化前 Hadamard 优于浮点正交矩阵，但优化后初始选择不再显著——Cayley SGD 总能找到好的局部最优。

**与 GPTQ 兼容性**：旋转优化时只对激活做量化（不做权重），再搭配 GPTQ 效果最佳。

**校准数据鲁棒性**：减少到 128 个样本、使用 C4 替代 WikiText2 均对结果影响很小。

**量化层级 SNR 分析**：学习旋转将端到端量化 SNR 从不旋转的 **-2.9 dB** 提升至随机旋转的 **0.9 dB**，再提升至学习旋转的 **6.8 dB**（提升 9.7 dB）。有趣的是，SNR 提升并非均匀——少数层贡献了大部分改善。

## 4. 核心贡献

1. **首个学习旋转矩阵的量化方法**：将旋转矩阵的优化转化为 Stiefel 流形上的优化问题，用 Cayley SGD 高效求解
2. **揭示随机旋转的巨大方差**：不同随机旋转矩阵导致高达 13 点的性能差异，Hadamard 矩阵也有 6 点
3. **灵活的两种方案**：SpinQuant$_{no\,had}$（零推理开销，适合 W4A8）和 SpinQuant$_{had}$（极低比特，~8% 开销）
4. **与 QuaRot 的本质区别**：QuaRot 用随机 Hadamard → SpinQuant 学习最优旋转，在线 Hadamard 矩阵数量减半（每块 2 个 vs QuaRot 的 4 个）
5. **全面的实验验证**：7 个模型、4 种量化配置，在 W4A4KV4 下与 FP 精度差距缩小至 1.4~4.1 点

## 5. 与 QuaRot 的关系

| 维度 | QuaRot | SpinQuant |
|------|--------|-----------|
| **旋转来源** | 随机 Hadamard 矩阵 | Cayley SGD 学习得到的最优旋转矩阵 |
| **旋转方差** | 大（~6 点），依赖多次随机尝试 | 极小，优化过程稳定 |
| **在线 Hadamard 数量** | 每 Transformer 块 4 个 | 每 Transformer 块 2 个（仅 $\mathbf{R}_3$, $\mathbf{R}_4$） |
| **可吸收旋转** | $\mathbf{Q}$（全局，跨层共享） | $\mathbf{R}_1$（可学习，全局共享）+ $\mathbf{R}_2$（可学习，每层独立） |
| **架构修改** | 需要修改模型 forward pass | $\mathbf{R}_1/\mathbf{R}_2$ 可完全吸收进权重，无需架构修改 |
| **精度（LLaMA-3 8B W4A4KV4）** | RTN: 58.6, GPTQ: 63.3 | RTN: 64.1, GPTQ: 65.5 |
| **精度（LLaMA-3 70B W4A4KV4）** | RTN: 41.3, GPTQ: 65.1 | RTN: 70.1, GPTQ: 69.3 |

SpinQuant 可视为 QuaRot 的"升级版"：将 QuaRot 的随机 Hadamard 替换为学习得到的更优旋转矩阵，在保持兼容性的同时显著提升精度。

## 6. 局限与展望

- 旋转优化虽轻量（7B 约 25 分钟），但仍需要前向+反向传播的校准过程
- 未探索给定异常值分布下的闭式最优旋转矩阵
- 第一个 token 在某些激活层中表现出较大的多通道异常值，旋转后集中到该 token 的所有通道——进一步理解并消除这一现象可能带来更多提升
- 未测试向量量化等更复杂的量化网格方案
