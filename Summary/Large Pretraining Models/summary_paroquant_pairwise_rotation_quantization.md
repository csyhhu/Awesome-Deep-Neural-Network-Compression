# ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

**论文信息**
- 标题: ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference
- 作者: Yesheng Liang¹, Haisheng Chen¹, Zihan Zhang¹, Song Han²³, Zhijian Liu¹
- 机构: ¹UC San Diego, ²NVIDIA, ³MIT
- 链接: https://arxiv.org/abs/2511.10645
- 发表: ICLR 2026
- 项目页面: https://paroquant.z-lab.ai

---

## 1. 问题背景

LLM 的后训练量化 (PTQ) 面临严重的 outlier 问题——权重和激活值中存在极端值，这些值占据了低位宽表示的有限动态范围，导致大量精度损失。这一问题在**推理型 LLM (reasoning LLM)** 中尤为严重，因为量化误差会在长链式思维 (chain-of-thought) 的每次解码步骤中**累积**，最终导致显著的精度退化。

现有方法存在两难困境：
- **AWQ**（通道级缩放）：推理快，几乎无额外开销，但在推理任务上会导致 **2.8% 精度下降**（Qwen3-4B, MMLU-Pro）
- **QTIP**（Hadamard 变换+向量量化）：精度高，但推理速度比 AWQ **慢约30%**

随着推理型 LLM（Qwen3、DeepSeek-R1 等）的兴起，**精度和效率必须兼顾**，这对量化方法提出了新挑战。

---

## 2. 核心洞察

作者提出三个关键观察：

1. **旋转有效抑制 Outlier**：正交旋转能通过跨通道交互将异常值分散，比单纯的通道级缩放更有效
2. **旋转参数高度冗余**：实验表明，只优化幅度差异最大的 **前10%** 通道对的旋转，几乎可以达到全旋转的量化误差降低效果
3. **独立旋转可充分并行化**：若约束每次旋转中每个通道只参与一个对（独立对），则所有 Givens 旋转可完全并行执行

---

## 3. 方法：Scaled Pairwise Rotation（缩放成对旋转）

### 3.1 整体流程

ParoQuant 的核心是一种**硬件友好的、可优化的等价权重变换**，由以下部分组成：

$$\mathbf{T}_{\mathcal P, \Theta, \boldsymbol\alpha}(\mathbf W) = \left(\prod_{t=1}^{K} R(\mathcal P_t, \Theta_t)\right) \cdot \mathrm{diag}(\boldsymbol\alpha) \cdot \mathbf{W}$$

其中：
- $\boldsymbol\alpha$：通道级缩放因子（对角矩阵）
- $R(\mathcal P_t, \Theta_t)$：第 $t$ 个独立旋转（由多组独立的 Givens 旋转组成）
- $K$：独立旋转的数量（通常为 8）

### 3.2 Givens 旋转

用选择的小量通道对 $(i_k, j_k)$ 上的 Givens 旋转替代完整的矩阵乘法。每个 Givens 旋转仅需两次向量化的乘加操作：

$$\begin{aligned} \mathbf{W}[i,:] &= \cos\theta \cdot \mathbf{W}[i,:] - \sin\theta \cdot \mathbf{W}[j,:] \\ \mathbf{W}[j,:] &= \sin\theta \cdot \mathbf{W}[i,:] + \cos\theta \cdot \mathbf{W}[j,:] \end{aligned}$$

### 3.3 独立旋转 (Independent Rotation)

**关键设计**：约束每个通道在每个旋转中最多出现在一个对中（独立对），使得：
- 所有 Givens 旋转**完全并行化**，无依赖关系
- 天然的 GPU 友好设计
- 与分块量化（block-wise quantization）天然兼容——每个量化组使用独立的旋转

### 3.4 串联独立旋转

单个独立旋转只能容纳 $n/2$ 对（$n$ 为通道数），表达能力有限。因此作者串联多个（如 8 个）独立旋转，并在不同旋转间**避免重复使用通道对**以增加多样性。

### 3.5 通道级缩放

在旋转之后叠加通道级缩放，以**直接均衡全矩阵的幅度**。这对于抑制孤立异常值特别有效，弥补了独立旋转只能处理有限对的不足。

### 3.6 逐层优化

采用两阶段优化：
- **Stage 1**：优化旋转角度 $\theta$ 和缩放因子 $\boldsymbol\alpha$
- **Stage 2**：类似 EfficientQAT 的 QAT 方式，微调权重和量化参数 $(s, z)$

使用已量化前置层的输出作为当前层的校准输入，使后续层能**补偿前层的量化误差**。

### 3.7 推理 Kernel 协同设计

实现了一个融合的 CUDA kernel，三级并行：
1. **Token 级**：在 batch/token 维度并行
2. **通道组级**：不同 CUDA block 处理不同的量化组
3. **对级**：每个 CUDA 线程处理一个旋转对

通道组大小较小（128），激活张量可放入共享内存，旋转参数可放入寄存器，大幅降低内存访问延迟。当通道维度增大时，相比 Hadamard 变换的加速比更高（Hadamard 变换有跨所有通道的内在依赖）。

---

## 4. 实验结果

### 4.1 实验设置
- **量化位宽**：W4A16（权重量化到4bit，激活保持FP16）
- **量化方式**：分块线性量化，组大小 (group size) = 128
- **模型**：Llama-2 7B, Llama-3 8B/70B, Llama-3.1 8B, DeepSeek-R1-distill-Llama 8B, Qwen3 1.7B/4B/8B/14B
- **校准集**：2048 样本（均匀来自 WikiText2、C4、RedPajama）
- **GPU**：H200（训练），RTX A6000（推理测速）

### 4.2 困惑度 (Perplexity)

| 方法 | 类型 | L3-8B | Q3-4B | Q3-8B | 加速比 |
|------|------|-------|-------|-------|--------|
| FP16 | - | 5.54 | 7.01 | 6.24 | 1.0× |
| QTIP | Vector | 5.69 | 7.09 | 6.28 | 1.7× |
| AWQ | Linear | 5.92 | 7.36 | 6.45 | **2.4×** |
| **ParoQ** | **Linear** | **5.73** | **7.10** | **6.29** | 2.2× |

- ParoQ 在所有线性量化方法中达到 SOTA，匹敌 QTIP（向量量化）但推理更快

### 4.3 推理任务精度

在 MMLU-Pro、GPQA Diamond、AIME-24、AIME-25 四个推理基准上：
- ParoQ 平均精度退化仅 **0.9%**（相比 FP16）
- 相比 AWQ：平均提升 **2.4%**
- 相比 EfficientQAT：平均提升 **6.3%**
- 相比 QTIP：平均提升 **0.9%**

以 Qwen3-4B (MMLU-Pro) 为例：
- FP16: 71.0 → AWQ: 68.2（下降 2.8）→ **ParoQ: 70.1（仅降 0.9）**

### 4.4 非推理任务精度

在 BoolQ、ARC、HellaSwag 等常识推理基准上：
- ParoQ 精度退化几乎为零
- 相比 AWQ 提升 0.9%，相比 QTIP 提升 0.2%

### 4.5 推理效率

| 方法 | Qwen3-4B | L3-8B | Q3-14B |
|------|----------|-------|--------|
| AWQ | 176 tok/s | 120 tok/s | 70 tok/s |
| QTIP | 117 tok/s | 95 tok/s | 55 tok/s |
| **ParoQ** | **160 tok/s** | **112 tok/s** | **65 tok/s** |

- ParoQ 比 AWQ 慢约 10%，但**准确率大幅提升**
- ParoQ 比 QTIP 快 **15%-30%**，且**准确率相当或更优**

### 4.6 消融实验

- **通道级缩放 + 独立旋转**：两者互补，组合使用效果最佳
- **旋转数量**：精度随旋转数增加（1→8）单调提升
- **校准集大小**：仅 128 样本即可达强性能，2048 样本最佳
- **校准集多样性**：混合数据集优于单一 RedPajama

### 4.7 W4A4 扩展

ParoQ 可扩展到 4-bit 权重+激活量化：
- INT4：超越 SpinQuant，与 FlatQuant 相当
- MXFP4：超越 MR-GPTQ

---

## 5. 关键贡献

1. **Scaled Pairwise Rotation**：一种新的等价权重变换，将硬件友好的独立 Givens 旋转与通道级缩放结合，有效抑制异常值
2. **独立旋转设计**：通过约束通道对互不重叠，实现完全的 GPU 并行化，推理开销极小
3. **算法-系统协同设计**：融合 CUDA kernel 实现三级并行，使旋转变换在推理时几乎不增加延迟
4. **推理型 LLM 量化**：首次系统性地展示量化误差在长链式思维中的累积问题，并给出有效解决方案

---

## 6. 局限性与展望

- W4A4 扩展为初步探索，未专门调优
- 仅支持线性量化，向量量化方向未深入
- 仅测试了 4-bit 量化，更低位宽（如 3-bit、2-bit）的性能有待验证
- 旋转对的固定选择策略可以进一步改进（如动态选择）

---

## 7. 讨论与问答

### Q1: Givens 旋转是什么？

**Givens 旋转**是数值线性代数中的经典正交变换，由 Wallace Givens（1958）提出。它本质上是一个**只在一个二维平面内进行旋转**的操作：

**数学定义**：$G(i, j, \theta)$ 几乎是一个单位矩阵，仅 4 个位置不同：
- $G[i,i] = \cos\theta$，$G[i,j] = -\sin\theta$
- $G[j,i] = \sin\theta$，$G[j,j] = \cos\theta$

作用于矩阵 $\mathbf{W}$ 时，**只改变第 $i$ 行和第 $j$ 行**：
$$\begin{aligned} \mathbf{W}[i,:] &\leftarrow \cos\theta \cdot \mathbf{W}[i,:] - \sin\theta \cdot \mathbf{W}[j,:] \\ \mathbf{W}[j,:] &\leftarrow \sin\theta \cdot \mathbf{W}[i,:] + \cos\theta \cdot \mathbf{W}[j,:] \end{aligned}$$

核心性质：
- **正交矩阵**：$G^T G = I$，保范保距
- **极简计算**：每次只需 4 次乘法和 2 次加法（可在两个向量上并行执行）

**在这篇论文中的角色**：

ParoQuant 利用 Givens 旋转来**消除权重矩阵中的异常值**。核心思想是配对"异常值通道"和"正常值通道"，通过旋转将异常能量分散到两个通道，使权重分布更均匀、更易量化。

直观类比：两杯水，一杯很满（outlier）、一杯很空（normal），Givens 旋转就像倾斜杯子让水重新分配。

**Givens vs 完整正交旋转 vs Hadamard 变换**：

| 维度 | 完整正交旋转 | Hadamard 变换 | 独立 Givens 旋转 |
|------|-------------|---------------|------------------|
| **参数量** | $n^2$ | 固定/随机 | $n/2$ 对 × 1 角度 |
| **复杂度** | $\mathcal{O}(n^2)$ | $\mathcal{O}(n\log n)$ | $\mathcal{O}(n)$ |
| **并行性** | 差 | 中等（蝴蝶运算有依赖） | **极佳**（无依赖） |
| **可优化性** | 可学习 | 固定/随机种子 | **可学习** |
| **误差方差** | 低 | 高（不感知分布） | **低** |

**论文的关键设计**：
1. **独立对约束**：每个 channel 在一次旋转中只参与一个对 → 全部并行
2. **串联多次**：单次独立旋转仅 $n/2$ 对，表达力不够 → 串联 8 次，且每次选**不同的配对组合**，用低计算代价逼近完整正交旋转
3. **配对策略**：优先选择幅度差异最大的通道对进行旋转（论文发现仅需 top 10% 的配对即可接近全旋转效果）

换句话说，论文证明了一个重要洞察：**正交旋转中大部分参数是冗余的，精心选择少量关键通道对进行旋转，就能达到几乎相同的异常值抑制效果**。

---

### Q2: 用一个例子说明整个流程，并注明新增的可学习参数

下面用一个具体的 8×4 权重矩阵为例，分组大小 $g=4$，独立旋转数 $K=2$，逐步演示 ParoQuant 的完整流程。

**初始权重矩阵**（channel 1 和 5 是 outlier）：

$$\mathbf{W} = \begin{bmatrix}
0.02 & 0.01 & 0.03 & 0.01 \\  % ch0 normal
\mathbf{\color{red}1.50} & \mathbf{\color{red}1.30} & \mathbf{\color{red}1.70} & \mathbf{\color{red}1.40} \\  % ch1 OUTLIER
0.01 & 0.02 & 0.01 & 0.03 \\  % ch2 normal
0.03 & 0.01 & 0.02 & 0.01 \\  % ch3 normal
\hline
0.04 & 0.03 & 0.05 & 0.02 \\  % ch4 normal
\mathbf{\color{blue}1.20} & \mathbf{\color{blue}1.10} & \mathbf{\color{blue}1.40} & \mathbf{\color{blue}1.30} \\  % ch5 OUTLIER
0.02 & 0.04 & 0.01 & 0.03 \\  % ch6 normal
0.03 & 0.02 & 0.04 & 0.01 \\  % ch7 normal
\end{bmatrix}$$

如果不做任何处理直接量化到 INT4，Group 0 的量化范围被 channel 1 的 1.70 撑大，导致 channel 0/2/3 的正常值全部坍塌为零。

---

#### 新增可学习参数一览

| 参数 | 符号 | 维度 | 本例数量 | 说明 |
|------|------|------|:---:|------|
| **通道缩放因子** | $\boldsymbol\alpha$ | $C_{in}$ | **8 个** | 每通道一个标量 |
| **旋转角度** | $\theta$ | $K \times g/2$ 每组 | **8 个** | 2组 × 2轮 × 2对 |
| **量化 scale** | $s$ | $C_{in}/g$ 组 | **2 个** | 每组一个 |
| **量化 zero-point** | $z$ | $C_{in}/g$ 组 | **2 个** | 每组一个 |
| **权重微调** | $\mathbf{W}$ | $C_{in} \times C_{out}$ | **32 个** | Stage 2 |
| **通道配对（固定）** | $\mathcal{P}$ | 算法确定 | — | 非可学习 |

> **Stage 1 新增 20 个标量参数**，远少于完整 8×8 正交旋转的 64 个参数。

---

#### Step 1: 通道对选择（预处理，不可学习）

对 Group 0（ch0-3）随机洗牌后贪婪选择不重叠通道对：
- **旋转 1**：配对 (ch1, ch2) 和 (ch0, ch3) — ch1（outlier）与 ch2（normal）配对！
- **旋转 2**：配对 (ch1, ch3) 和 (ch0, ch2) — 不同配对组合

---

#### Step 2: 通道级缩放

每个通道乘以可学习的缩放因子 $\alpha_i$：

| channel | $\alpha_i$（可学习） | 缩放前均值 | 缩放后均值 |
|:---:|:---:|:---:|:---:|
| ch1 (outlier) | **0.02** | ~1.475 | ~**0.0295** |
| ch0/2/3 | 1.0 | ~0.017 | ~0.017 |

> 缩放后 outlier 通道被大幅压缩，量级与正常通道趋于一致。

---

#### Step 3: 第一轮独立旋转

对配对 (ch1, ch2) 执行 Givens 旋转，角度 $\theta_1 = 0.3$（可学习）：

$$\begin{aligned}
\mathbf{W}[1,:] &\leftarrow \cos(0.3) \cdot \mathbf{W}[1,:] - \sin(0.3) \cdot \mathbf{W}[2,:] \\
&\approx 0.955 \cdot [0.030, 0.026, 0.034, 0.028] - 0.296 \cdot [0.01, 0.02, 0.01, 0.03] \\
&\approx [0.026, 0.019, 0.030, 0.018] \\
\\
\mathbf{W}[2,:] &\leftarrow \sin(0.3) \cdot \mathbf{W}[1,:] + \cos(0.3) \cdot \mathbf{W}[2,:] \\
&\approx [0.018, 0.027, 0.020, 0.037]
\end{aligned}$$

> **关键效果**：outlier ch1 的能量被旋转到 ch2，现在 4 个通道的值都在 ~0.01-0.04 范围内。

同时配对 (ch0, ch3) 也执行旋转（角度 $\theta_2 = 0.1$，可学习）。

---

#### Step 4: 第二轮独立旋转

不同配对 (ch1, ch3) 和 (ch0, ch2)，进一步混合通道信息，丰富通道间交互。

---

#### Step 5: 量化

对变换后的权重做 INT4 量化。Group 0 的 scale $s_0$ 只需覆盖 ~[-0.04, 0.04]，相比原始 [-0.03, 1.70]，**动态范围缩小约 40 倍**。

---

#### Step 6: 推理时逆变换

将缩放和旋转的逆操作应用于激活值 $\mathbf{X}$：

$$\mathbf{X}' = \mathbf{X} \cdot \text{diag}(\boldsymbol\alpha)^{-1} \cdot R_1^{-1} \cdot R_2^{-1}$$

$R_t^{-1}$ 只需将角度取反（$\cos(-\theta)=\cos\theta$, $\sin(-\theta)=-\sin\theta$）。所有操作融合在单个 CUDA kernel 中完成。

---

#### 整体数据流总结

```
  原始 W (有outlier)
    │
    ▼ [可学习参数: α₁...α₈]
  diag(α) · W          ← 通道级缩放，压缩outlier
    │
    ▼ [可学习参数: θ₁,θ₂ 对 Group0; θ₃,θ₄ 对 Group1]
  R₁: 独立Givens旋转     ← 第1轮配对旋转，分散outlier能量
    │
    ▼ [可学习参数: θ₅,θ₆; θ₇,θ₈]
  R₂: 独立Givens旋转     ← 第2轮不同配对，丰富通道交互
    │
    ▼ [可学习参数: s₀,s₁, z₀,z₁]
  量化 Q(W_transformed)  ← INT4 量化，动态范围已大幅缩小
    │
    ▼ [可学习参数: W权重微调]
  Stage 2: 微调权重      ← QAT方式进一步减小误差
    │
    ▼
  部署推理 ← 逆变换在激活值上在线执行
```

**核心洞察**：用仅 20 个可学习标量参数（vs 完整正交旋转的 64 个），通过「组内独立旋转 + 跨轮不同配对」的巧妙设计，实现了接近完整正交旋转的异常值抑制效果。

---

```

> **Total learnable params per layer**: `alpha` (n scalars) + `theta` (K·n/2 scalars) + `scale/zp` (~2·n/g scalars) ≈ O(K·n), versus O(n²) for a full rotation matrix.

---

### Q4: 如何理解整个旋转的过程？和直接使用一个 rotation 矩阵相比有什么优劣？

#### 一、从四个视角理解旋转过程

**视角 1：几何直觉 —— "倾斜量杯，均分水面"**

把权重矩阵的每一行（channel）想象成一个量杯，杯中的"水量"是该 channel 权重的幅度。Outlier channel 的水面极高，正常 channel 几乎空着。

一个 Givens 旋转就是**同时倾斜两个杯子**，让水从满的流向空的。倾斜角度 $\theta$ 控制分配比例——$\theta=45°$ 对半均分，$\theta$ 很小则只转移少量。

```
旋转前:  ch1 ████████████  (outlier, 1.50)
         ch2 ▏             (normal, 0.01)

旋转后:  ch1 ██████        (0.80, ← 能量被分散)
         ch2 ██████        (0.71, ← 接收了部分能量)
```

**视角 2：矩阵分解 —— "用积木搭出任意旋转"**

线性代数告诉我们：**任何一个 $n \times n$ 正交矩阵都可以分解为若干 Givens 旋转的乘积**。ParoQuant 反过来做——不是先学习完整矩阵再分解，而是直接学习一组 Givens 旋转参数：

$$R_{\text{full}} \approx \prod_{t=1}^{K} G(\mathcal{P}_t, \Theta_t)$$

每一层的 $G(\mathcal{P}_t, \Theta_t)$ 由 $n/2$ 个**互不重叠的** Givens 旋转组成。$K=8$ 层串联后，表达能力以指数级增长，足以逼近完整正交旋转的效果。

类比：用少数几根标准长度的积木（Givens 旋转）搭建出任意形状（任意旋转矩阵），而不用专门铸造一个整体模具（完整 $n\times n$ 矩阵）。

**视角 3：信息流 —— "逐步混合通道信息"**

单层独立旋转 = 通道两两配对混合，信息只在配对的 2 个 channel 间流动。

但串联 $K$ 层、每层用**不同的配对组合**后，信息流变得非常丰富：

```
Round 1:  ch0↔ch1  ch2↔ch3  ch4↔ch5  ch6↔ch7
Round 2:  ch0↔ch2  ch1↔ch3  ch4↔ch6  ch5↔ch7
Round 3:  ch0↔ch4  ch1↔ch5  ch2↔ch6  ch3↔ch7
  ...
```

经过多层不同配对后，ch0 的信息已经被**间接**传播到了 ch1, ch2, ch3, ch4, ch5, ch6, ch7——覆盖了所有通道。这正是论文实验 $K=8$ 效果饱和的原因：8 层足以让信息在全通道间充分混合。

**视角 4：计算图 —— "训练向左，推理向右"**

```
                          训 练 阶 段（离线）
 ═══════════════════════════════════════════════════════════
 W ──→ [diag(α)] ──→ [Givens R1] ──→ ... ──→ [Givens RK] ──→ [Quantize] ──→ W_q (INT4)
       ↑可学习        ↑可学习                 ↑可学习           ↑可学习
       8 scalars     8 angles                 8 angles          s, z

                          推 理 阶 段（在线）
 ═══════════════════════════════════════════════════════════
 X ──→ [diag(α)⁻¹] ──→ [Givens R1⁻¹] ──→ ... ──→ [Givens RK⁻¹] ──→ X'
                                                                      ↓
                                                        Y ←── [INT4 GEMM] ←── W_q
```

关键点：训练阶段变换的是**权重 W**（离线），推理阶段把逆变换施加到**激活 X** 上（在线，融合 CUDA kernel），从而保证 INT4 GEMM 的正确性：

$$Y = X \cdot R^{-1} \cdot \mathrm{diag}(\alpha)^{-1} \cdot W_q = X \cdot(\mathrm{diag}(\alpha) \cdot R)^{-1} \cdot W_q$$

#### 二、与直接使用完整旋转矩阵的对比

| 维度 | 完整正交旋转矩阵 $R_{n\times n}$ | ParoQuant 独立 Givens 旋转 |
|:---|:---|:---|
| **参数量** | $n^2$（$n$=4096 时为 ~16.8M） | $K \cdot n/2$ 个角度（$K$=8 时为 ~16K） |
| **每层计算量** | $\mathcal{O}(n^2)$ | $\mathcal{O}(Kn)$ |
| **并行性** | 差（稠密矩阵乘法有 $n^2$ 依赖） | **极佳**（对间无依赖，全部并行） |
| **优化难度** | 高（需满足正交约束 $R^T R = I$） | 低（只需优化标量 $\theta$，无约束） |
| **与分组量化兼容** | 困难（$R$ 作用于全通道，破坏分组边界） | **天然兼容**（每组内部独立执行） |
| **推理开销** | 高（在线做 $\mathcal{O}(n^2)$ 激活变换） | 低（$\mathcal{O}(Kn)$ + CUDA 融合） |
| **表达能力** | **完备**（可表示任意正交变换） | **近似**（$K$ 层逼近，实验中 $K=8$ 饱和） |
| **存储开销** | $\mathcal{O}(n^2)$ 浮点数 | $\mathcal{O}(Kn)$ 浮点数 |

**优势（为什么 ParoQuant 选择 Givens 而非完整矩阵）：**

1. **参数效率**：$K \cdot n/2$ vs $n^2$，对于 $n=4096$，差距达 **~1000 倍**。这直接决定了训练和存储的可行性。
2. **优化更简单**：学习标量 $\theta$ 天然定义在 $(-\pi, \pi]$ 上，优化器无需处理正交约束。而学习完整 $n\times n$ 矩阵需要 Cayley 参数化或 SVD 投影，梯度不稳定且计算昂贵。
3. **推理延迟极低**：逆变换在激活上以元素级操作完成（只需 `c*xi - s*xj`），融合在单个 CUDA kernel 中，多个线程同时处理不同 token 和对。而完整矩阵逆变换是一个稠密 GEMM，无法融入 INT4 算子。
4. **与 block-wise 量化天然兼容**：每个量化组（gs=128）内独立执行旋转，组间无交互。完整矩阵按组分块会破坏正交性（分块矩阵的乘积不等于正交）。

**劣势（Givens 旋转的代价）：**

1. **表达能力有限**：$K$ 层独立 Givens 只是所有正交矩阵的一个**子集**。论文实验显示 $K=8$ 已接近饱和，但对于某些极端分布可能不够。
2. **需要设计配对策略**：通道配对的质量直接影响效果。论文用幅度差异启发式，但这是一个额外的人工设计步骤，完整矩阵不存在此问题。
3. **训练需要两阶段**：Stage 1 学 $\theta$ 和 $\alpha$，Stage 2 微调权重。而理论上，如果有一个端到端的完整矩阵学习方案，可能更简洁（尽管数学上困难得多）。

#### 三、一句话总结

> ParoQuant 的旋转本质是用 **$K$ 组独立、可并行的 2D 旋转** 逐步混合通道信息，以 $\mathcal{O}(Kn)$ 的极低成本逼近完整正交矩阵的 $\mathcal{O}(n^2)$ 效果。这种做法牺牲了完备性，换来了训练可行性、推理高效性和硬件友好性——恰恰是做量化时最重要的三个维度。

---

### Q5: 每个 Givens 旋转矩阵的维度是多少？

每个 Givens 旋转矩阵的维度是 **$C_{in} \times C_{in}$**（即权重矩阵的输入通道数 × 输入通道数）。

但它是一个**极端稀疏**的矩阵——几乎就是单位矩阵 $I$，仅有 4 个位置不同：

$$G(i, j, \theta) = \begin{bmatrix} 
1 & & & & & \\ 
& \ddots & & & & \\ 
& & \cos\theta & \cdots & -\sin\theta & \\ 
& & \vdots & \ddots & \vdots & \\ 
& & \sin\theta & \cdots & \cos\theta & \\ 
& & & & & \ddots & \\ 
& & & & & & 1 
\end{bmatrix} \begin{matrix} \\ \\ \leftarrow i \\ \\ \leftarrow j \\ \\ \\ \end{matrix}$$

- $G[i,i] = \cos\theta$，$G[i,j] = -\sin\theta$
- $G[j,i] = \sin\theta$，$G[j,j] = \cos\theta$
- 其余对角线 = 1，其余位置 = 0

**关键：这个矩阵从不被显式构造出来。** 实际计算时只做两行向量运算：

```python
W[i], W[j] = c*W[i] - s*W[j],  s*W[i] + c*W[j]
```

所以虽然数学上是 $O(n^2)$ 大小的矩阵，但实际存储和计算成本都是 $O(1)$——仅存一个标量 $\theta$，仅操作两行。这正是 Givens 旋转作为矩阵分解基元的优雅之处：**形式上是 $n\times n$ 的正交矩阵，实际只是一个标量参数 + 两次向量化乘加**。

**具体例子**：若 $C_{in}=768$，配对 $i=100$，$j=300$，$\theta=0.3$：

- 矩阵总元素：$768 \times 768 = 589,824$
- 非零位置：仅 **770 个**（766 个对角线 1 + 4 个 $\cos\theta/\pm\sin\theta$），其余 589,054 个全为 0
- **99.87% 为空**，计算时只操作第 100 行和第 300 行，$O(768)$ 而非 $O(768^2)$

$$G[100,100] = \cos(0.3) \approx 0.955,\quad G[100,300] = -\sin(0.3) \approx -0.296$$
$$G[300,100] = \sin(0.3) \approx 0.296,\quad G[300,300] = \cos(0.3) \approx 0.955$$

---

### Q6: 旋转角度 $\theta$ 是如何计算/学习的？

角度 $\theta$ **不是解析计算的，而是通过梯度下降学出来的**。这与传统的 Givens QR 分解（通过 $\tan\theta = w_j/w_i$ 消去特定元素）有本质区别。

#### 学习流程

```
前向: θ → Givens(W, θ) → Quantize → Y_q = X @ W_q  →  loss = ||Y_q - Y_fp16||
                                                          ↓
反向:  ∇θ = ∂loss/∂θ ← ∂loss/∂W_q ← ∂W_q/∂W_rot ← ∂W_rot/∂θ
       (STE 近似量化梯度)                     (cos/sin 导数)
```

#### 关键步骤

1. **初始化**：$\theta$ 初始化为 $0$（此时 $G=I$，即不做任何旋转，等价于原始权重）
2. **前向**：$W_{rot}[i], W_{rot}[j] = \cos\theta \cdot W[i] - \sin\theta \cdot W[j],\; \sin\theta \cdot W[i] + \cos\theta \cdot W[j]$
3. **量化**：对 $W_{rot}$ 做 RTN 量化得到 $W_q$，梯度通过 **STE (Straight-Through Estimator)** 近似传递
4. **损失**：$\mathcal{L} = \|X_{calib} \cdot W_q - X_{calib} \cdot W\|_2^2$（输出重建误差）
5. **反向**：
   $$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial W_{rot}[i]} \cdot \frac{\partial W_{rot}[i]}{\partial \theta} + \frac{\partial \mathcal{L}}{\partial W_{rot}[j]} \cdot \frac{\partial W_{rot}[j]}{\partial \theta}$$
   其中 $\frac{\partial W_{rot}[i]}{\partial \theta} = -\sin\theta \cdot W[i] - \cos\theta \cdot W[j]$，$\frac{\partial W_{rot}[j]}{\partial \theta} = \cos\theta \cdot W[i] - \sin\theta \cdot W[j]$
6. **更新**：$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$（Adam 优化器，逐层迭代）

#### 为什么能学到有效角度？

- 优化目标是最小化**量化后的输出重建误差**，而非单纯的旋转正交性
- 梯度会自然地将 $\theta$ 推向"让 outlier 能量分散到配对 channel"的方向——因为这样做能降低量化误差
- 由于 $\theta$ 是标量且无约束，优化极其稳定，收敛很快（论文中 Stage 1 仅需少量迭代）
- 逐层补偿：已量化前置层的输出作为当前层校准输入，后续层的 $\theta$ 会自动补偿前层量化带来的分布偏移

**一句话**：$\theta$ 是通过反向传播 + STE 近似，以最小化量化输出误差为目标，端到端学习得到的标量参数——而非通过解析公式计算。

---

### Q7: 角度 $\theta$ 是学出来的，如何保证更新后 Givens 矩阵仍保持正交形式？

**答案是：不需要刻意维护，正交性由参数化形式天然保证。**

Givens 矩阵的结构是**硬编码**的——它不是一个自由矩阵，而是由标量 $\theta$ 通过固定公式"生成"的：

$$G = \begin{cases} G[i,i]=\cos\theta,\;G[i,j]=-\sin\theta \\ G[j,i]=\sin\theta,\;G[j,j]=\cos\theta \\ G[k,k]=1 \;(k \neq i,j) \\ \text{其余}=0 \end{cases}$$

无论 $\theta$ 取什么值，$\cos^2\theta + \sin^2\theta = 1$ **恒成立**（三角函数恒等式），因此 $G^T G = I$ **永远满足**。

**与直接学习完整 R 矩阵的对比**：

```
学习完整 R_{n×n}:                    学习 Givens θ:
                              
  R = randn(n,n)                       θ = 0  (标量)
       ↓ 优化                              ↓ 优化
  R' = R - η·∇L    ← 不再正交！          θ' = θ - η·∇L
       ↓ 需要额外步骤                        ↓
  正交投影/重参数化:                    cos(θ'), sin(θ')  ← 自动正交
  Cayley: R ← (I+A)(I-A)⁻¹
  或 SVD: R ← U V^T                    
  或 Riemannian GD
```

- **完整矩阵问题**：梯度更新 $R' = R - \eta \nabla\mathcal{L}$ 后，$R'$ 不再满足 $R'^T R' = I$，必须额外做正交投影，计算昂贵且可能影响梯度方向
- **ParoQuant 做法**：只存 $\theta$，不存矩阵。应用时计算 $\cos\theta/\sin\theta$。正交性由三角函数恒等式保证，**与 $\theta$ 取何值无关**

**核心洞察**：这是**参数化保证 (parameterization guarantee)**，不是优化约束。通过把 $n\times n$ 矩阵的 $n^2$ 个自由参数 + 正交约束，降维为一个**无约束标量优化问题**，既简化了优化又保证了数学正确性。

---

### Q8: 理解点评

> 你的理解：*本文通过添加旋转聚合操作，把参数矩阵中 outliers 加载到普通信息中，从而使得待量化的单元在相似的区间，因而能较好实行量化。旋转矩阵使用多个 Givens 矩阵相乘进行，并通过学习得到每个 Givens 矩阵参数。*

**总体评价：方向正确，抓住了论文的核心脉络。** 以下几点可以让理解更精确：

**✅ 准确的部分**：
- "outlier 分散到普通通道中" ← 这是 Givens 旋转的核心目的
- "使待量化单元处于相似区间" ← 正确，旋转后各通道量级趋于均匀，减小量化误差
- "多个 Givens 矩阵相乘" ← 正确，$K$ 轮独立旋转串联
- "通过学出来的" ← 正确，$\theta$ 是梯度下降学习而非解析计算

**🔧 可以更精确的部分**：

| 你的表述 | 更精确的说法 |
|:---|:---|
| "旋转聚合操作" | 更准确的表述是**旋转分散/混合**——目的是把 outlier 的极端值"打散"到其他通道，而非"聚集" |
| "outlier 加载到普通信息中" | outlier 的能量（幅度）被**重新分配**到配对的两个通道，通过 $\cos\theta/\sin\theta$ 做加权混合 |
| "待量化的单元" | 量化单元是**分块 (block-wise)** 的，每个 group（128 通道）内部独立旋转和量化，而非全局统一 |
| 缺少的关键设计 | 1) **独立对约束**：每个 channel 每轮只参与一个 pair，保证并行性；2) **通道级缩放 $\alpha$**：旋转前先用 learnable scalar 压缩 outlier；3) **推理时逆变换**：逆旋转施加在激活 X 上而非权重上 |

**🎯 建议的精炼表述**：

> 本文通过**可学习的通道缩放 + 多轮独立 Givens 旋转**，将权重矩阵中的 outlier 能量分散到配对通道，使每个量化组内通道幅度趋于均匀，从而大幅降低 INT4 量化误差。旋转部分用多个独立旋转层（每层由 $n/2$ 个互不重叠的 Givens 对组成）串联逼近完整正交变换的效果，每对只需学习一个标量角度 $\theta$，正交性由 $\cos/\sin$ 构造天然保证。推理时将逆变换融合在单个 CUDA kernel 中应用于激活值，几乎不增加延迟。

---

### Q9: 本文是 Post-Training Quantization 还是 Quantization-Aware Training？

ParoQuant 定位为 **Post-Training Quantization (PTQ)**，但在实现中借用了轻量级 QAT 技巧。准确地说，它是 **"PTQ 方法 + 轻量逐层微调"** 的混合范式。

#### 两阶段对比

| 维度 | Stage 1 | Stage 2 |
|:---|:---|:---|
| 优化对象 | $\theta$（旋转角度）、$\alpha$（缩放因子） | $W$（权重）、$s, z$（量化参数） |
| 数据量 | 2048 校准样本 | 同校准集 |
| 训练方式 | 逐层（layer-wise） | 逐层 |
| 计算开销 | 轻量（~分钟级） | 稍重但仍远小于全量微调 |
| 归类 | 纯 PTQ | 轻量 QAT |

**为什么整体归为 PTQ：**

1. **数据需求小**：仅需 2048 个校准样本，无需完整训练集
2. **无需完整训练流水线**：无 optimizer state、无多 epoch、无 learning rate schedule
3. **逐层而非端到端**：每层独立优化，层间无梯度传递（除前向量化补偿）
4. **总计算量远小于一次 full fine-tuning**

**为什么 Stage 2 像 QAT：** 它确实更新了权重 $W$ 本身（通过 STE 近似量化梯度），类似 EfficientQAT 的做法。但真正的 QAT 通常需要完整数据集 + 多轮 epoch + 端到端训练，而 Stage 2 只是用极少量校准数据做轻量微调。

**边界逐渐模糊的趋势**：现代量化方法（如 GPTQ 的 Cholesky + 权重更新、EfficientQAT、ParoQuant）越来越多地在 PTQ 流程中引入轻量训练步骤。这个分类界限本身也在演化——核心区分标准已从"是否更新权重"变为**"数据与计算开销"**。ParoQuant 在这条轴上明显属于轻量端。

---