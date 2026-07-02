# Performer: Rethinking Attention with Performers

- **论文链接**: [arXiv:2009.14794](https://arxiv.org/abs/2009.14794)
- **作者**: Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller
- **机构**: Google Research, University of Cambridge, DeepMind, Alan Turing Institute
- **发表**: ICLR 2021

---

## 一、核心动机

标准 Transformer 的注意力机制复杂度为 $\mathcal{O}(L^2 d)$，在长序列场景下不可行。现有解决方案（Reformer、Linformer、Sparse Transformer 等）大多通过**稀疏化、低秩近似或局部窗口**来降低计算量，但：

1. **不近似标准 softmax 注意力**，而是用简化的替代机制
2. **缺乏严格理论保证**（如稀疏模式有效性只能靠经验验证）
3. **通常需要额外约束**（如 Reformer 的 shared-QK）

Performer 的目标：**以线性复杂度无偏估计标准的 softmax 全秩注意力，不依赖稀疏性或低秩性先验**。

---

## 二、核心方法：FAVOR+

FAVOR+ = **F**ast **A**ttention **V**ia positive **O**rthogonal **R**andom features (+).

### 2.1 Kernelized Attention（FA 部分）

将注意力矩阵视为核函数：$\mathbf{A}(i,j) = \mathrm{K}(\mathbf{q}_i, \mathbf{k}_j)$。

如果核函数可以分解为随机特征映射：
$$\mathrm{K}(\mathbf{x}, \mathbf{y}) = \mathbb{E}[\phi(\mathbf{x})^{\top}\phi(\mathbf{y})]$$

则注意力计算可以重排为：
$$\widehat{\mathrm{Att}}_\leftrightarrow = \widehat{\mathbf{D}}^{-1} (\mathbf{Q}'((\mathbf{K}')^{\top} \mathbf{V})), \quad \widehat{\mathbf{D}} = \mathrm{diag} (\mathbf{Q}'((\mathbf{K}')^{\top} \mathbf{1}_L))$$

其中 $\mathbf{Q}', \mathbf{K}' \in \mathbb{R}^{L \times r}$ 是经过随机特征映射后的 query/key 矩阵，$r$ 是随机特征数量（$r \ll L$）。

**关键技巧**：先算 $(\mathbf{K}')^{\top} \mathbf{V}$（$\mathcal{O}(Lrd)$），再算 $\mathbf{Q}'$ 乘以上述结果（$\mathcal{O}(Lrd)$），避免构建 $L \times L$ 的注意力矩阵。

```
标准注意力: Q × K^T × V  → O(L²d)，内存 O(L²)
                                       ↑
                                 显式存储 L×L 矩阵
                                          
线性注意力: Q' × (K'^T × V) → O(Lrd)，内存 O(Lr)
                 ↑
            先算这个 L×r 的内积
```

| | 标准注意力 | FAVOR+ |
|---|---|---|
| 时间复杂度 | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L r d)$ |
| 空间复杂度 | $\mathcal{O}(L^2 + Ld)$ | $\mathcal{O}(Lr + Ld + rd)$ |

### 2.2 Positive Random Features（P 部分）

**核心问题**：softmax kernel $\mathrm{SM}(\mathbf{x}, \mathbf{y}) = \exp(\mathbf{x}^{\top}\mathbf{y})$ 如何分解为随机特征？

**朴素方法**（Trigonometric）：
- 利用 $\exp(\mathbf{x}^{\top}\mathbf{y}) = \exp(\frac{\|\mathbf{x}\|^2}{2}) \cdot \mathrm{K}_{\text{gauss}}(\mathbf{x}, \mathbf{y}) \cdot \exp(\frac{\|\mathbf{y}\|^2}{2})$
- 用 $\sin/\cos$ 近似 Gaussian kernel
- **致命缺陷**：当 $\mathrm{SM}(\mathbf{x}, \mathbf{y}) \to 0$ 时，MSE $\to \infty$（方差爆炸），导致训练不稳定甚至 NaN

**论文的解决方案**（Positive Random Features）：

**Lemma 1**：对于 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$，有：
$$\mathrm{SM}(\mathbf{x},\mathbf{y}) = \mathbb{E}_{\omega \sim \mathcal{N}(0,\mathbf{I}_d)}\left[\exp\left(\omega^{\top}\mathbf{x} - \frac{\|\mathbf{x}\|^{2}}{2}\right) \exp\left(\omega^{\top}\mathbf{y} - \frac{\|\mathbf{y}\|^{2}}{2}\right)\right]$$

**关键对比**：

| 特性 | Trigonometric | Positive（论文） |
|---|---|---|
| 特征值符号 | 有正有负 | **始终为正** |
| $\mathrm{SM} \to 0$ 时的方差 | $\to \infty$ | $\to 0$ |
| 训练稳定性 | 不稳定，可能 NaN | 稳定 |

使用 $\widehat{\mathrm{SM}}^{\mathrm{hyp+}}$（hyperbolic 变体）可利用 $f_1(u) = \exp(u)$ 和 $f_2(u) = \exp(-u)$ 进一步降低方差。

### 2.3 Orthogonal Random Features（O 部分）

为进一步降低方差，将独立采样的随机向量 $\omega_1, \dots, \omega_m$ 替换为**严格正交**的向量（通过 Gram-Schmidt 正交化，要求 $m \le d$）。

**理论保证**（Theorem 2）：对任意 $d > 0$，正交随机特征的 MSE 严格小于独立采样：
$$\mathrm{MSE}(\widehat{\mathrm{SM}}_{m}^{\mathrm{ort+}}) \le \mathrm{MSE}(\widehat{\mathrm{SM}}_{m}^{+}) - \frac{2(m-1)}{m(d+2)}\left(\mathrm{SM} - \exp(-\frac{\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2}{2})\right)^2$$

**Previous work** 仅在 $d \to \infty$ 的渐近意义上证明 ORF 的优势；本文**首次证明**了对任意维度 $d$，ORF 都能降低 softmax/Gaussian kernel 估计量的方差。

### 2.4 Unidirectional (Causal) 扩展

对于因果注意力（decoder），利用 **prefix-sum（前缀和）** 技巧：
$$\mathrm{tril}(\mathbf{Q}' (\mathbf{K}')^{\top}) \mathbf{C}_i = \mathbf{G}^{\mathrm{PS}}_{i,:,:} \times \mathbf{Q}'_i$$

其中 $\mathbf{G}^{\mathrm{PS}}_{i,:,:} = \sum_{j=1}^i \mathbf{K}'_j \mathbf{C}_j^{\top}$，可通过 $\mathcal{O}(L)$ 的并行前缀和高效计算。

---

## 三、完整算法流程

```
Input: Q, K, V ∈ R^{L×d}, isBidirectional

1. 计算随机特征映射: Q' = φ(Q), K' = φ(K), C = [V | 1_L]
   - 对每个 q_i: φ(q) = (h(q)/√m) · [f_1(ω_1^T q), ..., f_m(ω_m^T q)]
   - 其中 h(q) = exp(-||q||²/2), f_1 = exp (或添加 f_2 = exp(-u) 的 hyp+ 变体)
   - ω_i 为正交化后的随机向量 (ORF)

2. if isBidirectional:
        Buf₁ = (K')^T · C           ∈ R^{m × (d+1)}
        Buf₂ = Q' · Buf₁            ∈ R^{L × (d+1)}
   else (unidirectional):
        G_j = K'_j ⊗ C_j            ∈ R^{m × (d+1)}
        G^PS = prefix_sum(G)        ∈ R^{L × m × (d+1)}
        Buf₂[i] = G^PS_i · Q'_i     ∈ R^{d+1}

3. [Buf₃ | buf₄] = Buf₂, Buf₃ ∈ R^{L×d}, buf₄ ∈ R^L

4. Return diag(buf₄)^{-1} · Buf₃
```

---

## 四、广义注意力：超越 Softmax

FAVOR+ 不仅适用于 softmax kernel，还适用于任意可核化的注意力：

$$\phi(\mathbf{x}) = \frac{h(\mathbf{x})}{\sqrt{m}}(f_1(\omega_1^{\top}\mathbf{x}),...,f_m(\omega_m^{\top}\mathbf{x}))$$

- 取 $f = \mathrm{ReLU}$ 可获得 **Performer-RELU**，在蛋白质建模上取得最佳效果
- 可以在不改变架构的前提下灵活对比不同 attention kernel

---

## 五、实验与结果

### 5.1 计算效率

- 前向/反向传播接近线性时间复杂度
- 内存消耗呈次平方增长（无需存储显式 $L \times L$ 注意力矩阵）
- 在 V100 16GB GPU 上，Performer 可处理远超标准 Transformer 的序列长度

### 5.2 近似精度

- **ORF > IID**：正交特征比独立采样精度更高
- **Positive > Trigonometric**：正特征比 sin/cos 特征误差更低
- 上述两点验证了 FAVOR+ 的 PORF 机制

### 5.3 与预训练 Transformer 的兼容性

- 可将预训练 Transformer 权重直接迁移到 Performer
- 通过少量 fine-tuning 即可恢复精度（LM1B 数据集验证）
- 在更大的 PG-19 数据集上，Positive 特征 + 特征重采样（redrawing）至关重要

### 5.4 蛋白质序列建模（TrEMBL）

- 36 层模型、序列长度 8192
- Reformer 和 Linformer 准确率显著下降
- **Performer-RELU 取得最高准确率**
- Performer 的 softmax 近似与精确 Transformer 精度一致

### 5.5 ImageNet64（L=12288）

- Performer/6 层 ≈ Reformer/12 层
- Performer/12 层 ≈ Reformer/24 层
- Performer 在速度上可达到 Reformer 的 2 倍

---

## 六、理论保证

| 定理 | 内容 |
|------|------|
| Lemma 1 | Softmax kernel 存在 **正值** 随机特征映射无偏估计 |
| Lemma 2（MSE） | Trigonometric：$\mathrm{SM} \to 0$ 时 $\mathrm{MSE} \to \infty$；Positive：$\mathrm{SM} \to 0$ 时 $\mathrm{MSE} \to 0$ |
| Theorem 1 | 正则化 softmax kernel 是标准 softmax 的良好近似（误差 $\sim 2/d^{1/3}$） |
| Theorem 2 | ORF 对**任意维度 $d$** 严格降低正随机特征的 MSE |
| Theorem 3 | ORF 提供**指数级更紧**的尾部概率界 |
| Theorem 4 | ORF 取 $m = \Theta(d \log d)$ 即可保证注意力矩阵一致收敛 |

---

## 七、与其他方法的对比

| 方法 | 复杂度 | 近似类型 | Shared-QK 约束 | 理论保证 |
|------|--------|----------|----------------|----------|
| Standard Transformer | $\mathcal{O}(L^2 d)$ | 精确 | 无 | 无近似 |
| Reformer (LSH) | $\mathcal{O}(L \log L)$ | 稀疏 | 需要 | 概率性（哈希碰撞） |
| Linformer | $\mathcal{O}(L k d)$ | 低秩 | 无 | 有（但非无偏） |
| **Performer (FAVOR+)** | $\mathcal{O}(L r d)$ | **全秩无偏** | 无 | **严格理论保证** |

---

## 八、局限性

1. **$r$（随机特征数）需根据精度需求调整**：$r$ 越大近似越好，但计算量增加
2. **正交特征要求 $m \le d$**：低维嵌入场景受限
3. **特征需要周期性重采样**（redrawing）以维持多轮训练的近似质量
4. **受 embedding 范数影响**：查询/键的 $L_2$ 范数上界 $R$ 影响收敛速度

---

## 九、总结

Performer 通过 FAVOR+ 机制首次实现了 **线性复杂度下对标准 softmax 全秩注意力的无偏近似**，核心贡献：

1. **Positive Random Features**：解决传统 trigonometric 特征在 softmax kernel 值接近 0 时方差爆炸的关键问题
2. **Orthogonal Random Features**：首次证明对任意维度 ORF 都严格降低方差
3. **计算重排**：通过 $(\mathbf{K}')^{\top} \mathbf{V}$ 代替 $(\mathbf{Q}'(\mathbf{K}')^{\top}) \mathbf{V}$，避免构建 $L \times L$ 矩阵
4. **广义内核框架**：支持任意可核化注意力函数，包括 ReLU 等非线性

Performer 在蛋白质建模、图像生成、长文本建模等任务上取得了与标准 Transformer 相当或更好的性能，同时保持了线性复杂度，且可直接与预训练 Transformer 兼容。

---

## 十、Q&A

### Q1: 为什么可以进行计算重排？

**核心答案**：矩阵乘法结合律。

Performer 通过随机特征映射将 softmax 注意力矩阵近似分解为两个低维矩阵的乘积：

$$\mathbf{A} = \exp(\mathbf{Q}\mathbf{K}^\top) \approx \mathbf{Q}' \mathbf{K}'^\top$$

其中 $\mathbf{Q}', \mathbf{K}' \in \mathbb{R}^{L \times r}$ 是经过随机特征映射 $\phi$ 处理后的矩阵。由于矩阵乘法满足结合律，可以将计算顺序从：

$$\underbrace{\mathbf{Q}' \mathbf{K}'^\top}_{L \times L} \cdot \mathbf{V} \quad \longrightarrow \quad \mathbf{Q}' \cdot \underbrace{\left(\mathbf{K}'^\top \mathbf{V}\right)}_{r \times d}$$

| 计算顺序 | 中间矩阵大小 | 时间复杂度 |
|----------|-------------|-----------|
| 先算 $\mathbf{Q}'\mathbf{K}'^\top$ 再乘 $\mathbf{V}$ | $L \times L$ | $\mathcal{O}(L^2 d)$ |
| 先算 $\mathbf{K}'^\top \mathbf{V}$ 再乘 $\mathbf{Q}'$ | $r \times d$ | $\mathcal{O}(L r d)$ |

**为什么标准 Transformer 不能这样做**：标准 Transformer 的注意力矩阵 $\mathbf{A} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top)$ 包含逐元素非线性操作（exp + 归一化），无法分解成两个独立矩阵的乘积形式 $\phi(\mathbf{Q})\phi(\mathbf{K})^\top$，因此必须先计算完整的 $L \times L$ 矩阵。

---

### Q2: 随机特征映射是什么？

随机特征映射是 Performer 实现线性复杂度的**数学基石**。其核心思想是将 $\exp(\mathbf{q}_i^\top \mathbf{k}_j)$ 近似分解为两个低维向量的内积：

$$\exp(\mathbf{q}_i^\top \mathbf{k}_j) \approx \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)$$

其中 $\phi: \mathbb{R}^d \to \mathbb{R}^r$ 将 $d$ 维向量映射到 $r$ 维（$r \ll L$）。

**数学推导**（Lemma 1）：

利用高斯积分的归一化性质，可以证明：

$$\exp(\mathbf{x}^\top\mathbf{y}) = \mathbb{E}_{\boldsymbol{\omega} \sim \mathcal{N}(0, \mathbf{I}_d)}\left[\exp\left(\boldsymbol{\omega}^\top\mathbf{x} - \frac{\|\mathbf{x}\|^2}{2}\right) \cdot \exp\left(\boldsymbol{\omega}^\top\mathbf{y} - \frac{\|\mathbf{y}\|^2}{2}\right)\right]$$

由此定义随机特征映射：

$$\phi(\mathbf{x}) = \frac{1}{\sqrt{m}} \exp\left(-\frac{\|\mathbf{x}\|^2}{2}\right) \cdot \begin{bmatrix} \exp(\boldsymbol{\omega}_1^\top \mathbf{x}) \\ \exp(\boldsymbol{\omega}_2^\top \mathbf{x}) \\ \vdots \\ \exp(\boldsymbol{\omega}_m^\top \mathbf{x}) \end{bmatrix}$$

其中 $\boldsymbol{\omega}_i \overset{\text{iid}}{\sim} \mathcal{N}(0, \mathbf{I}_d)$。该估计是**无偏**的：$\mathbb{E}[\phi(\mathbf{x})^\top \phi(\mathbf{y})] = \exp(\mathbf{x}^\top \mathbf{y})$。

**为什么必须用 $\exp$ 而非 $\sin/\cos$**（Positive Random Features）：
- Trigonometric（$\sin/\cos$）特征：当 softmax kernel 值趋于 0 时，正负值抵消导致方差 $\to \infty$，训练不稳定
- Positive（$\exp$）特征：输出始终为正，$\text{SM} \to 0$ 时 $\text{MSE} \to 0$，训练稳定

**广义框架**：论文提出了更一般的随机特征映射形式，通过选择不同的 $f$ 和 $h$ 可覆盖多种 kernel：

| $f$ | $h(\mathbf{x})$ | Kernel 类型 |
|-----|-----------------|-------------|
| $f_1 = \exp$ | $\exp(-\|\mathbf{x}\|^2/2)$ | Softmax kernel（默认） |
| $f_1 = \exp, f_2 = \exp(-u)$ | $\frac{1}{\sqrt{2}}\exp(-\|\mathbf{x}\|^2/2)$ | Softmax hyp+（降方差） |
| $f_1 = \text{ReLU}$ | 1 | Performer-RELU（广义注意力） |

---

### Q3: 本文对 Q、K 分别处理的具体函数是什么？

Performer 用一个函数 $\phi$ **分别处理每个 $\mathbf{q}_i$ 和 $\mathbf{k}_j$**，使其内积近似 softmax，从而绕过先算 $L \times L$ 矩阵：

$$\boxed{\phi(\mathbf{x}) = \frac{\exp\left(-\frac{\|\mathbf{x}\|^2}{2}\right)}{\sqrt{m}} \cdot \begin{bmatrix} \exp(\boldsymbol{\omega}_1^\top \mathbf{x}) \\ \exp(\boldsymbol{\omega}_2^\top \mathbf{x}) \\ \vdots \\ \exp(\boldsymbol{\omega}_m^\top \mathbf{x}) \end{bmatrix}}$$

关键特性：
- 每个向量**独立**通过 $\phi$ 映射，不依赖 Q-K 交叉项
- 映射后 $\phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j) \approx \exp(\mathbf{q}_i^\top \mathbf{k}_j)$
- $\boldsymbol{\omega}_i$ 为随机采样并经正交化（ORF）的向量

**计算重排完整流程**：
```
Step 1: 分别映射
    Q' = φ(Q)    # L×d → L×m
    K' = φ(K)    # L×d → L×m

Step 2: 计算重排
    Buf = K'^T @ V    # m×d
    Out = Q' @ Buf    # L×d

Step 3: 归一化
    D = Q' @ (K'^T @ 1_L)
    Output = D^{-1} @ Out
```

---

### Q4: 这种做法有什么问题？为什么没有推广？

尽管 Performer 在数学上非常优雅，但存在以下问题导致其在工业界未广泛落地：

**1. 近似误差在长序列上累积**

$\phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)$ 只是 $\exp(\mathbf{q}_i^\top \mathbf{k}_j)$ 的 Monte Carlo 无偏估计，方差不可忽略。虽然 ORF 降低了方差，但：
- softmax 的指数归一化使某些 token 的注意力权重对误差极其敏感
- 少数关键 token 的注意力容易被噪声淹没
- 增加 $m$（随机特征数）可缓解，但会稀释速度优势

**2. FlashAttention 的降维打击（最关键原因）**

FlashAttention 走了一条完全不同的路——保持精确计算，优化 IO：

| | Performer | FlashAttention |
|---|---|---|
| 策略 | 降低计算复杂度（近似） | 保持精确计算，优化 IO |
| 数学上 | **近似** | **精确** |
| 复杂度 | $\mathcal{O}(Lrd)$ 理论更优 | $\mathcal{O}(L^2d)$ 但常数极小 |
| 实际场景 | 适合超长序列 (>100K) | 适合大多数场景 (≤32K) |

对于绝大多数实际应用（序列长度 2K–32K），FlashAttention 的精确 softmax + IO 优化比 Performer 的近似方案**更快且更准确**。Performer 的优势仅在极端长序列时体现，这种场景需求相对有限。

**3. GPU 硬件不友好**

$\phi$ 映射涉及多次逐元素 $\exp$、随机矩阵乘法、$L_2$ 范数计算，这些操作不如标准 matmul 在 GPU tensor core 上高效。FlashAttention 通过 tiling 利用 SRAM 将访存优化到极致，Performer 省了理论 FLOPs 但实际 wall-clock 速度优势不如预期。

**4. 训练-推理不一致**

随机向量 $\boldsymbol{\omega}_i$ 在训练时固定采样（或周期性 redraw），导致：
- 模型权重依赖于特定的 $\boldsymbol{\omega}_i$ 实现
- 推理时重新采样 $\boldsymbol{\omega}_i$ 精度下降
- redrawing 机制缓解过拟合但引入额外不确定性

**5. 产业界选择了其他路线**

| 路线 | 代表工作 | 状态 |
|---|---|---|
| 精确 + IO 优化 | FlashAttention 1/2/3 | **主力方案** |
| KV cache 压缩 | GQA / MQA | **已标准化**（LLaMA, Mistral 等） |
| 状态空间模型 | Mamba / Mamba-2 | 新兴替代方案 |
| 滑动窗口 | Mistral, Longformer | 长序列场景 |
| 线性注意力 | Performer, Linear Transformer | 学术影响力大，工业落地有限 |

**总结**：Performer 的近似误差、硬件不友好以及 FlashAttention 在实用场景下的绝对优势，使其停留在学术影响力层面，未成为工业标准。
