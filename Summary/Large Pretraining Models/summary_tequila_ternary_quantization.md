# Tequila: Trapping-free Ternary Quantization for Large Language Models

> **作者**: Shihao Zhang, Haoyu Zhang (UCSD), Ian Colbert (AMD), Rayan Saab (UCSD)
> **会议**: ICLR 2026
> **代码**: https://github.com/Tencent/AngelSlim
> **arXiv**: https://arxiv.org/abs/2505.11695

---

## 1. 核心问题

三元量化（Ternary Quantization）将权重限制在 `{-1, 0, +1}` 范围内，将计算昂贵的矩阵乘法转化为硬件友好的加法操作，非常适合在边缘设备上部署 LLM。然而，这种激进压缩会导致严重的精度下降，即使在使用大量数据进行 QAT（Quantization-Aware Training）之后也难以恢复。

论文识别出核心问题在于 **Deadzone Trapping（死区陷阱）**：三元量化创建了一个大的死区（deadzone，范围 `(-Δ, Δ)`），大量权重被量化为零。在训练过程中，这些"死亡"权重在前向传播中对输出没有任何贡献，因此反向传播时通过 STE（Straight-Through Estimator）只能获得噪声大、无信息量的梯度。由于缺乏一致的方向性信号，这些权重始终无法稳定地逃离死区，最终在死区边界附近做无效的振荡，导致模型容量和优化能力严重受损。

## 2. 方法：Tequila

Tequila（**Te**rnary **qu**antization for **la**rge language models）是一种无死区陷阱的三元量化方法，核心思想是将被困在死区中的权重**重新激活为动态偏置（dynamic biases）**。

### 2.1 Minima Reactivation（初步方案）

论文首先提出了 Minima Reactivation：利用死区权重的符号信息（正/负），将它们重新激活为有符号的极小值 `0+` 和 `0-`，从而创建一个有效的四元权重表示 `{-1, 0-, 0+, +1}`。

- **前向传播**：`Y = α∑sign(w_i)x_i + ε∑sign(x_i)sign(w_i)`（死区权重贡献一个类似偏置的项）
- **反向传播**：死区权重获得梯度 `∂L/∂w_i = ε · sign(x_i) · ∂L/∂Y`

**局限性**：
1. 梯度中仍依赖 STE 处理 `sign()` 操作，梯度噪声大
2. 额外项依赖输入，产生不可忽略的推理开销

### 2.2 Tequila 最终方案

Tequila 通过三个关键设计克服了上述局限：

1. **可微分重激活（Differentiable Reactivation）**：引入重激活参数 λ，为死区权重计算自适应最小值 `λ·w_i`，这是可微分的，绕过了 STE，提供直接的、有信息量的梯度信号。

2. **将死区权重重新用作偏置（Weights as Biases）**：观察发现输入在 Transformer 架构中大致对称分布，因此将输入相关的偏置项近似为输入无关的偏置 `∑λ·w_i`。该偏置可以离线预计算，推理时几乎零开销。

3. **混合角色（Hybrid Roles）**：重新激活的权重同时承担两种角色——参与三元矩阵乘法 + 充当自适应偏置。这样既保留了输入的依赖信息，又获得了直接清晰的梯度信号。

**最终前向传播**：
```
Y = X·Q(W) + C(W) = X·Ŵα + ∑λ·w_i    (i ∈ D)
```
其中 `C(W)` 为偏置项。

**最终梯度**（死区权重）：
```
∂L/∂w_i = x_i·∂L/∂Y + λ·∂L/∂Y    (混合梯度)
```

## 3. 实验

- **模型**：LLaMA-3.2-1B 和 LLaMA-3.2-3B
- **训练数据**：10B tokens（UltraFineWeb 数据集）
- **基础设施**：16 GPU 训练，Intel 8263C CPU 推理
- **基准评估**：PIQA、ARC-Easy/Challenge、HellaSwag、GPQA-Diamond、WinoGrande

### 3.1 性能对比

Tequila 在所有基准上均超越 SOTA 三元量化方法：
- **ARC 基准**：超过 SOTA 基线 >4%，与全精度性能差距 <1%
- **1B 模型平均准确率**：Tequila 0.471 vs Absmean 0.445（基准提升 2.6%）
- **3B 模型平均准确率**：Tequila 0.530 vs 最佳基线 0.510

### 3.2 与其他三元 LLM 对比

TequilaLLM 仅用 10B tokens 训练即超越 100B tokens 训练的 BitNet 和 Spectra：
- TequilaLLM-3B 在 6 项基准上平均准确率 0.576，超过 Spectra-3.9B（0.567，100B tokens）

### 3.3 收敛速度

Tequila 的收敛速度显著快于所有基线方法。

### 3.4 推理效率

在 Intel 8263C CPU 上：
- TequilaLLM 相比 BF16 LLaMA 实现 **3× 推理加速**
- 与 BitNet 推理速度几乎相同，额外开销 <0.1%

### 3.5 消融实验

验证了三个关键设计的逐步贡献：
- Minima Reactivation > Absmean（验证重激活效果）
- Tequila w/o Mixed Gradients > Minima Reactivation（验证可微分重激活优于 STE）
- Tequila > Tequila w/o Mixed Gradients（验证混合梯度优于纯偏置梯度）

## 4. 核心贡献

| 贡献 | 说明 |
|------|------|
| **问题识别** | 首次将三元量化的性能瓶颈归因为 Deadzone Trapping |
| **Tequila 方法** | 将死区权重重新用作动态偏置，提供无陷阱的三元量化 |
| **无推理开销** | 偏置项离线预计算，推理几乎零额外开销 |
| **即插即用** | 可集成到现有三元量化方法中 |
| **显著性能提升** | 仅 10B tokens 训练即达到 SOTA，推理加速 3× |

## 5. 关键优势

1. **增强模型容量**：重新激活死区权重有效扩展了模型参数空间
2. **无陷阱优化**：直接、有信息量的梯度使死区权重能够稳定逃逸
3. **训练稳定性**：可微分重激活保证稳定优化的同时保持量化约束
4. **即插即用设计**：简单的模块，可轻松集成到现有三元量化方法
5. **几乎零推理开销**：输入无关偏置可离线预计算并融合到计算核中

## 6. 推理设计

Tequila 采用查表法（Lookup Table-Based）实现无乘法推理：
- **离线阶段**：将三元权重和偏置打包为索引权重、符号权重和通道偏置（每三个权重打包为 4-bit 索引 + 1-bit 符号）
- **在线阶段**：输入值按段预处理为查表，通过权重索引检索结果，完全替换乘法为高效的表查找操作

---

## 7. 讨论与问答

### Q1: 以一个例子说明整个流程？

假设有一个简单的全精度权重向量和输入向量：

$$W = [0.6, -0.1, -0.8, 0.05], \quad X = [2.0, 1.5, -1.0, 3.0]$$

#### Step 1：传统三元量化（Absmean）

计算缩放因子和阈值：

$$\alpha = \text{mean}(|W|) = \frac{0.6 + 0.1 + 0.8 + 0.05}{4} = 0.3875, \quad \Delta = \frac{\alpha}{2} = 0.1938$$

| 权重 | 值 | \|wᵢ\| vs Δ | 量化结果 ŵᵢ | 状态 |
|------|-----|-------------|------------|------|
| w₁ | +0.6 | 0.6 ≥ 0.1938 | **+1** | ✅ 活跃 |
| w₂ | -0.1 | 0.1 < 0.1938 | **0** | ❌ 死区陷阱 |
| w₃ | -0.8 | 0.8 ≥ 0.1938 | **-1** | ✅ 活跃 |
| w₄ | +0.05 | 0.05 < 0.1938 | **0** | ❌ 死区陷阱 |

传统前向传播（w₂ 和 w₄ 贡献为零）：
$$Y = X^T \hat{W} \alpha = [2·1 + 1.5·0 + (-1)·(-1) + 3·0] · 0.3875 = 3 · 0.3875 = 1.1625$$

传统反向传播（STE 梯度，假设 ∂L/∂Y = 0.1）：
$$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial Y} · x_2 = 0.1 · 1.5 = 0.15$$
$$\frac{\partial L}{\partial w_4} = \frac{\partial L}{\partial Y} · x_4 = 0.1 · 3.0 = 0.30$$

> ⚠️ w₂ 和 w₄ 的梯度通过 STE 近似，充满噪声。梯度信号不直接告诉权重"该往哪里走"才能逃离死区，导致它们在做无效的振荡。

#### Step 2：Tequila 重激活

死区权重索引 D = {2, 4}（w₂ = -0.1, w₄ = +0.05），设定 λ = 10⁻³。

**偏置项**（输入无关，可离线预计算）：
$$C(W) = \sum_{i \in D} \lambda w_i = 10^{-3} · (-0.1 + 0.05) = -5 \times 10^{-5}$$

**Tequila 前向传播**：
$$Y = X^T \hat{W} \alpha + C(W) = 1.1625 - 5 \times 10^{-5} \approx 1.16245$$

死区权重通过偏置项重新贡献了信号！在 LLM 成百上千的线性层中累积效果显著。

**Tequila 反向传播——关键差异**：
$$\frac{\partial L}{\partial w_2} = \underbrace{x_2 · \frac{\partial L}{\partial Y}}_{\text{三元路径（STE）}} + \underbrace{\lambda · \frac{\partial L}{\partial Y}}_{\text{偏置路径（直达！）}} = 1.5·0.1 + 10^{-3}·0.1 = 0.1501$$

$$\frac{\partial L}{\partial w_4} = 3.0·0.1 + 10^{-3}·0.1 = 0.3001$$

> 🎯 等式第二项 λ·∂L/∂Y 通过**可微分的偏置路径直接回传**，完全不依赖 STE！这是一个**稳定、一致的方向信号**，告诉权重"继续朝当前方向移动"，帮助它们稳定逃离死区。

#### 对比总结

| 维度 | 传统三元量化 | Tequila |
|------|------------|---------|
| w₂, w₄ 对输出的贡献 | **0**（完全浪费） | 通过偏置 C(W) 贡献信号 |
| w₂ 梯度来源 | 仅 x₂ 路径（STE 噪声） | x₂ 路径 + **λ 直达路径** |
| 梯度方向性 | 随机、不稳定 | 清晰、一致 |
| 推理开销 | 无 | 偏置离线预计算，**<0.1%** |
| 死区权重命运 | 永久被困，无效振荡 | 稳定逃逸，恢复活性 |

### Q2: 之前做量化也会加入 stochastic noise 来进行死区逃脱，跟 Tequila 的做法有什么区别？

**传统 stochastic noise 方法的本质**：在 QAT 中注入随机噪声（如高斯噪声），试图将死区权重"震"出边界。但论文明确指出：

> *"Even when stochastic noise temporarily pushes a weight outside the deadzone, it is quickly pulled back in subsequent iterations because it is continually quantized to zero and lacks a consistent optimization signal."*

Stochastic noise 有三大缺陷：

| 缺陷 | 说明 |
|------|------|
| **方向随机** | 噪声是零均值的，正负方向各半，没有"哪个方向对 loss 有利"的信息 |
| **即使逃出去也会被拉回来** | 出去后仍通过 STE 接收梯度，信号质量没有改善；上一轮随机推出的，下一轮可能随机拉回 |
| **不改变问题结构** | 权重在死区内对输出贡献始终为零，始终接收不到有意义的梯度——恶性循环没有打破，随机扰动只是让循环多抖了两下 |

**Tequila 的根本不同**：Tequila 不是把权重"推出去"，而是**让权重在死区内就有用**。

- 死区权重通过偏置路径获得的梯度 `λ·∂L/∂Y` **完全不经过 STE**，是直接从 loss 回传的
- 如果 loss 梯度为正 → 一致地推动 wᵢ 增大；loss 梯度为负 → 一致地推动 wᵢ 减小
- 这是一个**有方向的、一致的优化信号**，权重知道自己每走一步是否在降低 loss

**直观类比**：

| | Stochastic Noise | Tequila |
|------|------|------|
| 类比 | 在漆黑的房间里，随机踢你一脚，踢到哪算哪 | 给你一根绳子，绳子上挂着 loss，拉着它就知道方向 |
| 信号性质 | 零均值随机噪声 | 有方向的梯度信号 |
| 是否改变问题结构 | 否，死区权重仍然"无用" | 是，死区权重被重新用途为偏置 |
| 收敛性 | 随机游走，不保证收敛 | 沿梯度下降，理论上可收敛 |
| 能否稳定逃逸 | 否，来回振荡 | 是，每一步都知道该往哪走 |

**一句话总结**：stochastic noise 是用"蛮力撞大运"，Tequila 是**改变了游戏规则**——不是在死区外找答案，而是让死区本身变得有价值。

### Q3: 在量化乘法后加 stochastic noise 可以吗？

方案：$$Y = X^T \hat{W} \alpha + \varepsilon$$（ε 为随机噪声）

**结论：不行。** 关键在于链式法则中，噪声 ε 不依赖死区权重 wᵢ，所以 **∂ε/∂wᵢ = 0**，新的梯度通道根本不存在：

$$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial w_2} = \frac{\partial L}{\partial Y} \cdot \left( x_2 + \frac{\partial \varepsilon}{\partial w_2} \right) = x_2 \cdot \frac{\partial L}{\partial Y}$$

> 死区权重的梯度仍然只是 STE 路径，雅可比矩阵结构完全没有改变。噪声只是让 loss 数值浮动，但 wᵢ **感知不到噪声的存在**。Tequila 的偏置与之相反：∂(λ·wᵢ)/∂wᵢ = λ ≠ 0，确实在雅可比中多了一条通路。

**一句话总结**：关键不在于"加了什么"，而在于"加的东西是否可微地依赖死区权重"。

### Q4: 用 noise 乘以 W 呢？即 ε·wᵢ 取代 λ·wᵢ？

方案：$$Y = X^T \hat{W} \alpha + \sum_{i \in D} \varepsilon_i \cdot w_i$$（εᵢ 从分布采样）

**结论：数学上打通了，统计上白费了。** 梯度确实变了：
$$\frac{\partial L}{\partial w_i} = x_i \cdot \frac{\partial L}{\partial Y} + \varepsilon_i \cdot \frac{\partial L}{\partial Y}$$

但连续迭代中 εᵢ 随机正负：
```
t:   ε₂=+0.5 → 正方向推
t+1: ε₂=-0.7 → 负方向拉
t+2: ε₂=+0.1 → 正方向推
t+3: ε₂=-0.3 → 负方向拉
```

因为 **E[εᵢ] = 0**，所以 **E[εᵢ·∂L/∂Y] = 0**——平均来看梯度通道给的信号为零，权重仍在原点附近随机游走。

**Tequila 三步递进**（用固定 λ 而非抽样噪声）：

| 步骤 | 说明 | ε·wᵢ 方案 | Tequila |
|------|------|-----------|---------|
| 1. 重激活 | 死区权重参与输出计算 | ✅ | ✅ |
| 2. 可微性 | 打通直达梯度通道 | ✅ | ✅ |
| 3. 一致性 | 信号方向稳定，不随机翻转 | ❌ | ✅（λ 为正常数） |

因为 λ > 0 固定，第二条通道方向完全取决于 ∂L/∂Y。loss 连续告诉模型"输出应该更大"时，每个死区权重都**持续收到正向推力**，累积推动下稳定逃逸。而 εᵢ 方案每一步都在自己打自己——往前推一步又往后拉一步，永远走不远。

### Q5: 综合来说，Tequila 就是在前向中加入 λ·wᵢ 项，λ 和 wᵢ 都是 learnable 的？

理解方向正确，但细节需修正：**λ 是固定超参数，不是 learnable 的**。

论文明确写道：*"We set λ = 10⁻³ for Tequila by default."* λ 在整个训练过程中不变。论文还做了敏感性分析（λ ∈ {10⁻⁵, ..., 10⁻¹}），性能在宽范围内鲁棒：*"Performance is robust across a wide range of values."*

**实际上 learnable 的只有 wᵢ**：

| 项 | learnable？ | 说明 |
|----|:---:|------|
| λ | ❌ 固定 | 超参数，默认 10⁻³ |
| wᵢ | ✅ | QAT 中维护的全精度副本，随梯度更新 |
| 偏置 C(W)=∑λ·wᵢ | ✅ 间接学习 | 随 wᵢ 更新而自动变化 |

Tequila 前向传播：
$$Y = X^T \hat{W} \alpha + \sum_{i \in D} \lambda \cdot w_i$$

wᵢ 通过 λ·wᵢ 路径获得额外梯度 **λ·∂L/∂Y**。λ 只是充当了一个固定的"信号放大器"——把 ∂L/∂Y 按恒定比例传回给 wᵢ，帮助它自救。λ 固定而非可学恰是巧妙之处：避免了引入额外可训练参数带来的优化复杂度，用一个固定小常数就解决了问题——"够用即好"。
