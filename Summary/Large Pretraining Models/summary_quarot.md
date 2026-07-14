# QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

- **论文链接**: [https://arxiv.org/abs/2404.00456](https://arxiv.org/abs/2404.00456)
- **发表会议**: NeurIPS 2024
- **作者机构**: ETH Zurich, EPFL, Microsoft Research, IST Austria, NeuralMagic
- **代码**: [https://github.com/spcl/QuaRot](https://github.com/spcl/QuaRot)

## 1. 研究动机

大语言模型（LLM）推理需要大量计算、内存和能源，尤其在 prefill 阶段。量化是解决这一问题的重要技术，但**激活值中存在大量异常值（outliers）**，激活值的某些通道会比其他通道大几个数量级，这使得激活值量化远比权重量化困难，尤其在 4-bit 情况下。此前的工作（如 SmoothQuant、QUIK）需要保留部分异常特征通道用更高精度表示，增加了系统复杂度。

本文的核心思路：利用**随机 Hadamard 变换**旋转模型输入，在不改变模型输出的前提下消除异常值，进而实现所有权重、激活值和 KV 缓存的端到端 4-bit 量化。

## 2. 核心方法

QuaRot 包含两个阶段。第一阶段在全精度下修改模型权重并插入在线 Hadamard 操作；第二阶段对权重进行量化并加入激活值的在线量化操作。

### 2.1 阶段一：权重修改与在线 Hadamard 变换

#### 阶段 1a：权重修改（Weight Modification）

利用**计算不变性**（Computational Invariance）原理，将 LayerNorm/RMSNorm 的缩放参数吸收进相邻权重矩阵，然后用随机 Hadamard 矩阵 $\mathbf{Q}$ 对权重矩阵进行预乘或后乘：

- 输入侧的权重矩阵（如 $\mathbf{W}_k$, $\mathbf{W}_q$, $\mathbf{W}_v$, $\mathbf{W}_\text{up}$, $\mathbf{W}_\text{gate}$）左乘 $\mathbf{Q}^\top$
- 输出侧的权重矩阵（如 $\mathbf{W}_\text{down}$, $\mathbf{W}_\text{out}$）右乘 $\mathbf{Q}$

因为 RMSNorm 具有旋转不变性 $\text{RMSNorm}(\mathbf{X}) = \text{RMSNorm}(\mathbf{X}\mathbf{Q}^\top)\mathbf{Q}$，所以此修改不影响模型输出。

关键效果：处理后的激活值 $\mathbf{X} \leftarrow \mathbf{XQ}$ 不再包含任何异常值（见原论文 Figure 1），分布变得均匀，极大利于量化。

#### 阶段 1b：FFN 激活值旋转

在 FFN 的下投影矩阵之前插入在线 Hadamard 操作（FP16 精度），并通过将 Hadamard 矩阵融合进 $\mathbf{W}_\text{down}$ 来隐式逆转：$\mathbf{W}_\text{down} \leftarrow \mathbf{H} \mathbf{W}_\text{down}$。最终下投影矩阵变为 $\mathbf{H} \mathbf{W}_\text{down} \mathbf{Q}$。

#### 阶段 1c：注意力值投影（Attention Value Projection）

利用注意力计算中 $\mathbf{W}_v$ 和 $\mathbf{W}_\text{out}$ 在每个头内隐式相乘的特性：
$$\mathbf{Y} = \sum_{h=1}^H \mathbf{P}_h \mathbf{X} \mathbf{W}_v^{(h)} \mathbf{W}_\text{out}^{(h)}$$

对每头应用 Hadamard 矩阵 $\mathbf{H}_{d_h}$：
$$\mathbf{W}_v^{(h)} \leftarrow \mathbf{W}_v^{(h)} \mathbf{H}_{d_h}, \quad \mathbf{W}_\text{out}^{(h)} \leftarrow \mathbf{H}_{d_h} \mathbf{W}_\text{out}^{(h)}$$

再通过 Kronecker 分解 $\mathbf{H}_{n_h \times d_h} = (\mathbf{I} \otimes \mathbf{H}_{d_h})(\mathbf{H}_{n_h} \otimes \mathbf{I})$，插入 "Hadamard heads" 块完成完整的注意力激活旋转。

#### 阶段 1d：Key 旋转

由于 RoPE 位置编码的存在，不能直接将 Hadamard 矩阵吸收进权重。因此采用**在线头级 Hadamard 旋转**（Post-RoPE Caching）：
$$\mathbf{Q} \leftarrow \text{Pos}(\mathbf{X}\mathbf{W}_q)(\mathbf{I} \otimes \mathbf{H}_{d_h}), \quad \mathbf{K} \leftarrow \text{Pos}(\mathbf{X}\mathbf{W}_k)(\mathbf{I} \otimes \mathbf{H}_{d_h})$$

由于 Query 和 Key 同时旋转，注意力分数 $\mathbf{P}_h$ 保持不变。这使得 KV 缓存也可以被量化。

### 2.2 阶段二：量化

#### 阶段 2a：权重量化

默认使用**GPTQ**对修改后的权重进行量化（也可使用简单的 RTN，代价是部分精度损失）。

#### 阶段 2b：在线激活性量化

对线性层输入使用**对称 per-token 量化**（每行一个 scale），clipping ratio 设为 0.9。反量化时将 INT32 的 GEMM 输出转为 FP16，乘以对应的 scale。

#### 阶段 2c：量化注意力

KV 缓存使用**非对称量化**（group size 128，clipping ratio 0.95）。Query 保持在 FP16，使用类似 Flash Attention 的在线 softmax 计算。

### 2.3 与其他方法的区别

| 对比维度 | QuaRot | QuIP# | SmoothQuant | QUIK |
|---------|--------|-------|-------------|------|
| 异常值处理 | 旋转消除 | 旋转 + 非相干处理 | 激活-权重平滑 | 保留高精度通道 |
| 需要保留高精度通道 | 否（0个） | 否 | 否 | 是（256个） |
| KV Cache 量化 | 支持 | - | - | - |
| 端到端 4-bit | 是 | 仅权重 | 否 | 否 |

## 3. 实验评估

### 3.1 模型与数据集

- **模型**: LLaMA2-7B, LLaMA2-13B, LLaMA2-70B（以及 LLaMA3 家族，见附录）
- **语言生成**: WikiText-2 困惑度
- **零样本任务**: PIQA, WinoGrande, HellaSwag, Arc-Easy, Arc-Challenge, LAMBADA
- **GPU**: NVIDIA RTX 3090（CUDA/CUTLASS 实现）

### 3.2 主要结果

#### 困惑度结果（A4W4KV4，GPTQ 权重量化）

| 方法 | 权重量化 | 异常特征数 | LLaMA2-7B↓ | LLaMA2-13B↓ | LLaMA2-70B↓ |
|------|---------|-----------|-----------|------------|------------|
| Baseline | - | - | 5.47 | 4.88 | 3.32 |
| SmoothQuant | RTN | 0 | 83.12 | 35.88 | - |
| OmniQuant | RTN | 0 | 14.26 | 12.30 | - |
| QUIK-4B | GPTQ | 256 | 8.87 | 7.78 | 6.91 |
| **QuaRot** | GPTQ | **0** | **6.10** | **5.40** | **3.79** |
| Atom-128G | GPTQ-128G | 128 | 6.03 | 5.26 | - |
| **QuaRot-128G** | GPTQ-128G | **0** | **5.93** | **5.26** | **3.61** |

- LLaMA2-70B 仅损失 **0.47** 困惑度，无需任何重训练
- 6/8-bit RTN 量化完全无损

#### 零样本任务准确率（A4W4KV4）

| 模型 | 方法 | Avg Acc↑ |
|------|------|----------|
| LLaMA2-7B | FP16 | 69.82 |
| LLaMA2-7B | QuaRot | 65.64 |
| LLaMA2-13B | FP16 | 72.59 |
| LLaMA2-13B | QuaRot | 69.79 |
| LLaMA2-70B | FP16 | 77.07 |
| LLaMA2-70B | QuaRot | **75.98** (保留 99%) |

### 3.3 性能评估

- **Prefill 加速**（RTX 3090, seq_len=2048）：LLaMA2-7B 达 1.97-2.16×，LLaMA2-70B 达 **3.33×**
- **Decoding 内存节省**：LLaMA2-7B 最高 3.75×，LLaMA2-70B 最高 **3.89×**

### 3.4 消融实验

- **RTN 权重量化 vs GPTQ**：8-bit RTN 完全无损；4-bit 下 GPTQ 优于 RTN，差距随模型增大而缩小
- **分组量化**：更小的 group size 带来更好的精度（64G > 128G > 256G > 无分组）
- **随机正交矩阵替代 Hadamard**：Hadamard 矩阵表现更优
- **FP16 Hadamard 变换**：精度足够，无需 FP32

## 4. 核心贡献

1. **首次实现端到端 4-bit LLM 推理**：包括所有权重、激活值和 KV 缓存，无需保留任何高精度异常特征通道
2. **计算不变性的新颖应用**：将 SliceGPT 的计算不变性思想从结构化剪枝扩展到量化领域
3. **系统性的旋转方案**：覆盖 FFN、注意力值投影、Key 旋转，形成完整的无异常值推理管道
4. **高效的 CUDA 内核**：基于 CUTLASS 和 FlashInfer 实现实用加速

## 5. 局限与展望

- 仅针对 Dense 模型，尚未扩展到 MoE 架构
- 残差连接的量化尚未探索
- 在线 Hadamard 变换引入额外计算开销（尽管已通过快速 Walsh-Hadamard 变换优化）
- 硬件友好性受限：当前 GPU 对 INT4 矩阵乘法的支持有限，未来硬件（如 B200 FP4）有望进一步提升性能

## 6. Q&A

### Q1: Hadamard 旋转是改变 activation 还是 parameters 的异常情况？

**正确的因果逻辑是：主动旋转 activation 消除 outlier，再将逆矩阵 Qᵀ 补偿进 weight 来维持输出不变。**

- **第一步（目的）**：在 activation $\mathbf{X}$ 上乘 $\mathbf{Q}$，使其变成 $\mathbf{XQ}$。Hadamard 矩阵将单一通道的能量"打散"到所有通道，从而消除个别通道的极端值（outlier），激活值分布变得均匀，利于量化。
- **第二步（补偿）**：为了输出不变，在 weight 上乘 $\mathbf{Q}^\top$（正交矩阵的逆即转置）：
  $$\underbrace{\mathbf{XQ}}_{\text{旋转后的激活}} \cdot \underbrace{\mathbf{Q}^\top \mathbf{W}}_{\text{补偿后的权重}} = \mathbf{X} \cdot \underbrace{(\mathbf{Q}\mathbf{Q}^\top)}_{=\mathbf{I}} \cdot \mathbf{W} = \mathbf{XW}$$

**注意**：weight 本身也从旋转中间接受益——权重与 $\mathbf{Q}^\top$ 相乘后也变得更"incoherent"，分布更均匀，更易量化。但这不是旋转的 primary motivation，primary motivation 是消除 activation 的 outlier。

**跨层传递的视角**：
- Block i 的输出侧 weight 乘 $\mathbf{Q}$，所以 Block i 输出 $\mathbf{YQ}$
- Block i+1 的输入侧 weight 乘 $\mathbf{Q}^\top$，所以 Block i+1 计算 $(\mathbf{YQ}) \cdot (\mathbf{Q}^\top \mathbf{W}) = \mathbf{Y}\mathbf{W}$
- $\mathbf{Q}$ 在层间交界处精确抵消，网络整体输出不变

### Q2: 吸收进权重矩阵为何不会影响输出一致性？

如上所述，核心是**正交矩阵的逆等于其转置**：$\mathbf{Q}^{-1} = \mathbf{Q}^\top$。

在 activation 上乘 $\mathbf{Q}$，在 weight 上乘 $\mathbf{Q}^\top$，两者相乘时 $\mathbf{Q}\mathbf{Q}^\top = \mathbf{I}$，精确抵消。

考虑 RMSNorm 的情况：
- 原始：$\mathbf{Y} = \text{RMSNorm}(\mathbf{X}) \cdot \text{diag}(\alpha) \cdot \mathbf{W}$
- 修改后激活为 $\mathbf{XQ}$，权重为 $\mathbf{Q}^\top \text{diag}(\alpha) \mathbf{W}$
- 利用 RMSNorm 旋转不变性 $\text{RMSNorm}(\mathbf{XQ}) \cdot \mathbf{Q}^\top = \text{RMSNorm}(\mathbf{X})$：
  $$\mathbf{Y}' = \text{RMSNorm}(\mathbf{XQ}) \cdot (\mathbf{Q}^\top \text{diag}(\alpha) \mathbf{W}) = \text{RMSNorm}(\mathbf{X}) \cdot \text{diag}(\alpha) \cdot \mathbf{W} = \mathbf{Y}$$
- $\checkmark$ 输出不变

### Q3: Hadamard 矩阵是什么？在 QuaRot 中的作用？

**Hadamard 矩阵**：$d \times d$ 方阵，所有元素为 ±1，各行向量正交（$\mathbf{H}_d \mathbf{H}_d^\top = d\mathbf{I}$）。归一化后（$\frac{1}{\sqrt{d}}\mathbf{H}_d$）为正交矩阵。可通过 Walsh-Hadamard 变换（WHT）在 $O(d \log d)$ 完成矩阵乘法，远快于普通 $O(d^2)$。Sylvester 构造：$\mathbf{H}_{2d} = \begin{bmatrix} \mathbf{H}_d & \mathbf{H}_d \\ \mathbf{H}_d & -\mathbf{H}_d \end{bmatrix}$。

**在 QuaRot 中的核心作用**：消除激活值异常通道。Hadamard 像一个均匀混合器——每行 ±1 各半，将少数通道集中的极端值均匀打散到所有通道，使分布平滑利于量化。因其正交性，可在权重中补偿逆矩阵 $\mathbf{Q}^\top$，维持数学等价。

### Q4: 所有阶段的旋转本质是否相同？

**是的，底层模式完全一致**：在矩阵乘法**输入前的瞬间**做 Hadamard 旋转 + 量化，逆矩阵吸收进下游权重。

| Stage | 被旋转的激活 | 逆矩阵吸收位置 | 在线步骤 | 说明 |
|-------|-------------|---------------|---------|------|
| 1a | Hidden state $\mathbf{X}$ | 输入侧 weight（$\mathbf{Q}^\top$） | 无 | 跨层传递，$\mathbf{Q}$ 在层间抵消 |
| 1b | FFN 中间激活 | $\mathbf{W}_\text{down}$ | 插入 $\mathbf{H}$ | 消除块内激活 outlier |
| 1c | V 值 | $\mathbf{W}_v$, $\mathbf{W}_\text{out}$（头内）+ $\mathbf{W}_\text{out}$（跨头） | 插入 "Hadamard heads" 块 | 多头结构 + 跨头混合补全 |
| 1d | K, Q（Post-RoPE） | Q/K 同时旋转，点积抵消 | 在线 Hadamard | 下游是 QK 点积，无权重可吸收 |

前三个 Stage 都是 **"旋转激活 + 逆矩阵吸收进权重"**。Stage 1d 的不同仅在于：K 的下游操作是 $\mathbf{Q} \cdot \mathbf{K}^\top$（点积），没有可学习的权重矩阵来吸收 $\mathbf{H}^\top$，所以改为**同时旋转 Q 和 K**，让两个 $\mathbf{H}$ 在点积中相互抵消。

### Q5: RoPE 如何影响 Stage 1d（Key 旋转）？

**RoPE 本身并不改变旋转的本质逻辑**。无论中间经历了什么操作（RoPE、LayerNorm、激活函数），只需在进入下一个矩阵乘法前做 Hadamard 旋转即可。

RoPE 的唯一"影响"是：Hadamard 与 RoPE 不可交换（两者都是旋转，顺序不同结果不同），因此 H 必须放在 RoPE **之后**，而不能预先吸收进 $\mathbf{W}_k$。

Stage 1d 的做法：
$$\mathbf{K} \xleftarrow{\text{归一化+量化存储}} \text{RoPE}(\mathbf{XW}_k) \cdot (\mathbf{I} \otimes \mathbf{H}_{d_h})$$
$$\mathbf{Q} \xleftarrow{\text{在线}} \text{RoPE}(\mathbf{XW}_q) \cdot (\mathbf{I} \otimes \mathbf{H}_{d_h})$$

Q 和 K 同时旋转，点积中 $\mathbf{HH}^\top = \mathbf{I}$，Attention Score 不变。如果位置编码是其他类型（如原始的绝对编码），处理逻辑完全相同。

### Q6: Stage 1c（Attention Value Projection）与跨头混合的关系？

Stage 1c 包含两部分，通过 Kronecker 恒等式联系起来：
$$\mathbf{H}_{d_\text{model}} = (\mathbf{I} \otimes \mathbf{H}_{d_h})(\mathbf{H}_{n_h} \otimes \mathbf{I})$$

| 部分 | 操作 | 作用 | 实现 |
|------|------|------|------|
| $(\mathbf{I} \otimes \mathbf{H}_{d_h})$ | **头内旋转** | 各头内通道均匀化 | 吸收进 $\mathbf{W}_v$, $\mathbf{W}_\text{out}$ |
| $(\mathbf{H}_{n_h} \otimes \mathbf{I})$ | **跨头混合** | 跨 Head 维混合，升级为完整的 $d_\text{model}$ 维旋转 | 在线 "Hadamard heads" 块 + 部分吸收进 $\mathbf{W}_\text{out}$ |

- **头内旋转**是核心：保证每个头内数学等价，消除 V 值 outlier
- **跨头混合**是优化：让旋转覆盖完整 $d_\text{model}$ 维度，使注意力输出 $\mathbf{Z}$ 在量化前分布更均匀。去掉它不影响计算一致性，仅影响量化精度

### Q7: QuaRot 的统一框架

**QuaRot = 在每次矩阵乘法前，对 Activation 输入做 Hadamard 旋转再量化，逆矩阵补偿到 Parameters 侧保证计算一致性，再对 Parameters 侧量化。**

| 步骤 | Activation 侧 | Parameters 侧 |
|:----:|--------------|---------------|
| 旋转 | $\mathbf{X} \to \mathbf{XQ}$ 消除 outlier | $\mathbf{W} \to \mathbf{Q}^\top \mathbf{W}$ 吸收逆矩阵 |
| 量化 | $\mathbf{XQ} \to \text{Quant}(\mathbf{XQ})$ | $\mathbf{Q}^\top \mathbf{W} \to \text{Quant}(\mathbf{Q}^\top \mathbf{W})$ |
| 计算 | $\text{Quant}(\mathbf{XQ}) \cdot \text{Quant}(\mathbf{Q}^\top \mathbf{W}) \approx \mathbf{XQ} \cdot \mathbf{Q}^\top \mathbf{W} = \mathbf{XW}$ | |

这适用于四个 Stage 中的前三个：

| Stage | Activation 旋转 | Parameters 补偿 |
|-------|----------------|-----------------|
| 1a | $\mathbf{X} \to \mathbf{XQ}$ | $\mathbf{W} \to \mathbf{Q}^\top \mathbf{W}$ |
| 1b | FFN 中间激活 $\to \mathbf{H}$ | $\mathbf{W}_\text{down} \to \mathbf{H} \mathbf{W}_\text{down}$ |
| 1c | V/Z $\to$ 头内 + 跨头 $\mathbf{H}$ | $\mathbf{W}_v, \mathbf{W}_\text{out} \to$ 吸收 $\mathbf{H}$ |
| 1d | K $\to \mathbf{H}$ | 无权重可吸收 → Q 同步旋转，点积中 $\mathbf{HH}^\top = \mathbf{I}$ 抵消 |

Stage 1d 的唯一例外：K 的下游是 $\mathbf{QK}^\top$ 点积而非可学习权重，因此"补偿"转移到 Q 侧——**Q 和 K 同时旋转，在点积中抵消**。思想一致，实现适配。

### Q8: 旋转矩阵 R 全部折叠到参数上的表达（以两层 MLP 为例）

以两层 MLP $\mathbf{Y} = \mathbf{X} \cdot \mathbf{W}_1 \cdot \mathbf{W}_2$ 为例，展示所有旋转矩阵如何折叠进参数。

**原始计算**：

$$\mathbf{Z} = \mathbf{X} \cdot \mathbf{W}_1, \quad \mathbf{Y} = \mathbf{Z} \cdot \mathbf{W}_2$$

其中 $\mathbf{X} \in \mathbb{R}^{1 \times d}$，$\mathbf{W}_1 \in \mathbb{R}^{d \times d_h}$，$\mathbf{W}_2 \in \mathbb{R}^{d_h \times d_o}$。

**Step 1：对每次矩阵乘法插入旋转**（QuaRot 的原始思路）：

在 $\mathbf{X} \cdot \mathbf{W}_1$ 之前旋转 $\mathbf{X}$，补偿进 $\mathbf{W}_1$；在 $\mathbf{Z} \cdot \mathbf{W}_2$ 之前旋转 $\mathbf{Z}$，补偿进 $\mathbf{W}_2$：

$$\begin{aligned} \mathbf{X}' &= \mathbf{X} \cdot \mathbf{R}_1 \\ \mathbf{W}_1' &= \mathbf{R}_1^\top \cdot \mathbf{W}_1 \\ \mathbf{Z} &= \mathbf{X}' \cdot \mathbf{W}_1' = \mathbf{X} \cdot \mathbf{R}_1 \cdot \mathbf{R}_1^\top \cdot \mathbf{W}_1 = \mathbf{X}\mathbf{W}_1 \quad \checkmark \end{aligned}$$

$$\begin{aligned} \mathbf{Z}' &= \mathbf{Z} \cdot \mathbf{R}_2 \\ \mathbf{W}_2' &= \mathbf{R}_2^\top \cdot \mathbf{W}_2 \\ \mathbf{Y} &= \mathbf{Z}' \cdot \mathbf{W}_2' = \mathbf{Z} \cdot \mathbf{R}_2 \cdot \mathbf{R}_2^\top \cdot \mathbf{W}_2 = \mathbf{Z} \cdot \mathbf{W}_2 \quad \checkmark \end{aligned}$$

**Step 2：把 $\mathbf{R}_2$ 折叠进 $\mathbf{W}_1$**：

$$\mathbf{Z}' = \mathbf{Z} \cdot \mathbf{R}_2 = (\mathbf{X} \cdot \mathbf{W}_1) \cdot \mathbf{R}_2 = \mathbf{X} \cdot (\mathbf{W}_1 \cdot \mathbf{R}_2)$$

即 $\mathbf{W}_1$ 右乘 $\mathbf{R}_2$ 等效于在中间激活 $\mathbf{Z}$ 上做旋转。

**折叠后的完整表达式**：

| 参数 | 折叠前 | 折叠后 |
|:---:|--------|--------|
| $\mathbf{W}_1$ | $\mathbf{R}_1^\top \cdot \mathbf{W}_1$ | $\boxed{\mathbf{W}_1^{\text{fold}} = \mathbf{R}_1^\top \cdot \mathbf{W}_1 \cdot \mathbf{R}_2}$ |
| $\mathbf{W}_2$ | $\mathbf{R}_2^\top \cdot \mathbf{W}_2$ | $\boxed{\mathbf{W}_2^{\text{fold}} = \mathbf{R}_2^\top \cdot \mathbf{W}_2}$ |

**Runtime 只需旋转初始输入**：

$$\boxed{\begin{aligned} \mathbf{X}_{\text{rot}} &= \mathbf{X} \cdot \mathbf{R}_1 \\[4pt] \mathbf{Z} &= \mathbf{X}_{\text{rot}} \cdot \mathbf{W}_1^{\text{fold}} = (\mathbf{X} \mathbf{R}_1) \cdot (\mathbf{R}_1^\top \mathbf{W}_1 \mathbf{R}_2) = \mathbf{X} \cdot \mathbf{W}_1 \cdot \mathbf{R}_2 \\[4pt] \mathbf{Y} &= \mathbf{Z} \cdot \mathbf{W}_2^{\text{fold}} = (\mathbf{X} \mathbf{W}_1 \mathbf{R}_2) \cdot (\mathbf{R}_2^\top \mathbf{W}_2) = \mathbf{X} \cdot \mathbf{W}_1 \cdot \mathbf{W}_2 \quad \checkmark \end{aligned}}$$

**核心规律**：

| 层 | $\mathbf{W}_i^{\text{fold}}$ | 左侧 $\mathbf{R}^\top$ 的作用 | 右侧 $\mathbf{R}$ 的作用 |
|:---:|------|------|------|
| 第一层 | $\mathbf{R}_1^\top \cdot \mathbf{W}_1 \cdot \mathbf{R}_2$ | 抵消输入旋转 $\mathbf{R}_1$ | 预先施加输出旋转 $\mathbf{R}_2$ |
| 最后一层 | $\mathbf{R}_2^\top \cdot \mathbf{W}_2$ | 抵消输入旋转 $\mathbf{R}_2$ | （无需，后无矩阵乘法） |
| 中间层（推广） | $\mathbf{R}_i^\top \cdot \mathbf{W}_i \cdot \mathbf{R}_{i+1}$ | 抵消上一层输出的旋转 | 预先施加传向下一层的旋转 |

**在 QuaRot 实际实现中**，跨层使用同一个全局 $\mathbf{Q}$（即 $\mathbf{R}_1 = \mathbf{R}_2 = \mathbf{Q}$），因此：

$$\boxed{\mathbf{W}_1^{\text{fold}} = \mathbf{Q}^\top \cdot \mathbf{W}_1 \cdot \mathbf{Q}, \quad \mathbf{W}_2^{\text{fold}} = \mathbf{Q}^\top \cdot \mathbf{W}_2}$$

**总结**：
- **中间层权重吸收两个方向的旋转**：左侧 $\mathbf{Q}^\top$ 抵消传入的旋转，右侧 $\mathbf{Q}$ 预施加传出的旋转
- **最后一层仅吸收左侧 $\mathbf{Q}^\top$**，因为后续没有矩阵乘法需要旋转
- **Runtime 开销极小**：仅初始输入做一次 Walsh-Hadamard 变换（$O(d\log d)$），中间所有旋转均被参数完全吸收
