# Trajectory Consistency Distillation (TCD)

> **论文标题**: Trajectory Consistency Distillation: Improved Latent Consistency Distillation by Semi-Linear Consistency Function with Trajectory Mapping
>
> **作者**: Jianbin Zheng, Minghui Hu, Zhongyi Fan, Chaoyue Wang, Changxing Ding, Dacheng Tao, Tat-Jen Cham
>
> **机构**: South China University of Technology, Nanyang Technological University, The University of Sydney, Beijing Institute of Technology
>
> **会议**: ICML 2024
>
> **论文链接**: https://arxiv.org/abs/2402.19159
>
> **项目页面**: https://mhh0318.github.io/tcd

---

## 1. 研究背景与动机

Latent Consistency Model (LCM) 将 Consistency Model 扩展到潜在空间，通过引导一致性蒸馏实现了文本到图像合成的显著加速。然而，LCM 存在两个关键问题：

1. **低 NFE 下图像质量不佳**：LCM 在单步或少步（4~8步）采样时生成质量远不如教师模型。
2. **多步采样质量退化**：随着推理步数增加，LCM 生成的图像细节逐渐丢失、质量下降。

作者分析发现，这些问题源于：

- **参数化过程中的离散化误差**：LCM 使用 DDIM 形式对一致性函数进行参数化，其中包含由有限差分近似带来的离散化误差。
- **蒸馏误差上界较大**：一致性蒸馏的误差上界与时间跨度 t→0 成正比。
- **多步采样中的累积误差**：多步一致性采样中的随机噪声累积导致细节丢失。

## 2. 核心方法

TCD 包含两个核心组件：

### 2.1 Trajectory Consistency Function (TCF) — 轨迹一致性函数

**核心思想**：将一致性函数的边界条件从原点 (t→0) 扩展为任意中间点 (t→s)，利用 PF-ODE 的半线性（semi-linear）结构结合指数积分器（exponential integrator）进行参数化。

**定义**：f(x_t, t) → x_s，即将轨迹上的任意点 x_t 映射到轨迹上的任意中间点 x_s，而非仅限于原点。

**关键改进**：
- **参数化形式**基于指数积分器的半线性结构，使用 x-prediction 或 epsilon-prediction
- **TCF(1)**（一阶，默认使用）：利用一阶泰勒展开实现高效参数化
- **TCF(2)**（二阶）：引入中间点 u，实现更高阶近似
- **TCF(S+)**：引入额外网络参数 F_theta(x_t, t, s) 直接估计积分项

**误差分析**：
- 参数化离散化误差：O(h^{k+1})，其中 k 为 TCF 的阶数，h = lambda_s - lambda_t
- 蒸馏误差：从原始的 O((Δt)^p) · (t_n - t_0) 降至 O((Δt)^p) · (t_n - t_m)
- 缩短时间间隔不仅降低参数化误差，也降低了蒸馏误差上界

**边界条件**：
- 经典边界条件：f(x_0, 0, 0) = x_0
- 扩展边界条件：f(x_s, s, s) = x_s

**训练目标**：
```
L_TCD = E[ || f_theta(x_{t_{n+k}}, t_{n+k}, t_m) - f_{theta^-}(hat_x_{t_n}, t_n, t_m) ||^2 ]
```
其中 n~U[1, N-1], m~U[1, n]，同时采用 skipping-step 技术（跳 k 步）加速收敛。

### 2.2 Strategic Stochastic Sampling (SSS) — 策略性随机采样

**核心思想**：在采样过程中显式引入噪声控制参数 γ，灵活调节每步采样的随机强度。

**SSS 采样过程**：
1. 降噪子步：x_{t→s'} ← f_theta*(x_t, t)   （s' = (1-γ)·s）
2. 扩散子步：hat_x_s ← (α_s/α_{s'})·x_{t→s'} + sqrt(1-α_s²/α_{s'}²)·z

**关键洞察**：
- 原 LCM 的多步一致性采样本质上是具有固定随机强度的 DDIM 采样
- SSS 通过调整 γ，使得每次降噪的目标点不再是原点 0，而是更近的中间点 s'
- γ 越大 → 降噪目标越近 → 累积误差越小 → 细节越丰富
- 误差界：O(√(τ_n - (1-γ)·τ_{n+1}))

**与 CTM 的关系**：CTM 的 γ-sampling 与 SSS 在理论上具有相似的误差特性。TCD 的 SSS 专为 LCM 的多步一致性采样框架设计。

## 3. 实验

### 3.1 实验设置
- **骨干模型**：SDXL
- **数据集**：LAION5B High-Res（美学评分 > 5.8）
- **训练**：TCF(1) + LoRA（rank=64），3000 次迭代，8×A800 GPU，15 小时
- **默认设置**：TCF(1)，γ = 0.2
- **ODE Solver**：DDIM，skipping step k = 20
- **Classifier-free guidance**：ω ∈ [2, 14]

### 3.2 主要结果（COCO 验证集 5K 零样本生成）

**4步采样对比**：

| 方法 | FID↓ | IC Score↑ | ImageReward↑ | PickScore↑ |
|------|------|-----------|--------------|------------|
| Euler | 44.31 | 0.3639 | -189.41 | 18.71 |
| DDIM | 44.86 | 0.3633 | -189.96 | 18.68 |
| DPM++(2S) | 18.50 | 0.4496 | -1.27 | 20.68 |
| LCM | 15.03 | 0.4364 | 52.72 | 22.20 |
| **TCD (Ours)** | **12.68** | **0.5095** | **68.49** | **22.31** |

**关键发现**：
- TCD 在所有步数（2/4/8/20）和所有指标上均超越 LCM
- 在 20 步时，TCD 的 FID（13.56）接近 DPM++(2S)（12.15），但在 IC Score、ImageReward 和 PickScore 上均超越
- **TCD 在高 NFE 下能超越教师模型**（SDXL + DPM-Solver++）
- 传统数值方法（Euler/DDIM）在低步数下 FID 极差，LCM 改善显著，TCD 进一步提升

### 3.3 消融实验

**随机参数 γ 的影响**：
- γ 增大 → 图像视觉复杂度和精细度逐步提升
- γ = 0（完全确定性）→ 模型固有估计误差更明显，FID 变差
- 存在最优 γ 值，需根据经验确定

**参数化类型对比（4步）**：

| 类型 | FID↓ | IC Score↑ | ImageReward↑ | 训练迭代 |
|------|------|-----------|--------------|----------|
| TCF(1) | 12.68 | 0.5095 | 68.49 | 3,000 |
| TCF(2) | 13.35 | 0.5037 | 58.13 | 3,000 |
| TCF(S+) | 13.03 | 0.4176 | 57.96 | 43,000 |

- TCF(1) 表现最佳，训练效率最高
- TCF(2) 存在高阶求解器的不稳定性
- TCF(S+) 引入额外参数导致蒸馏效率低、收敛慢

### 3.4 通用性测试

TCD LoRA 可直接应用于多种社区模型，仅需 2-8 步即可高质量加速：
- Animagine XL V3（社区模型）
- Papercut XL（风格 LoRA）
- Depth ControlNet / Canny ControlNet
- IP-Adapter

**所有模型共享同一套 TCD LoRA 参数**，无需额外训练。

## 4. 与 CTM (Consistency Trajectory Models) 的对比

| 方面 | CTM | TCD |
|------|-----|-----|
| 核心思路 | 统一 CM 和 DM 的 "anytime-to-anytime" 框架 | 改进 LCM 的一致性函数参数化 |
| 参数化方法 | Euler Solver 启发的隐式方法 | 半线性结构 + 指数积分器的显式方法 |
| 训练损失 | 软一致性匹配 + DSM + GAN 损失 | 仅使用一致性蒸馏损失 + skipping-steps |
| 采样方式 | γ-sampling（需专用采样器） | SSS（显式，兼容现有 solver） |
| 训练代价 | 需要辅助损失 | 蒸馏损失即可，避免模式坍塌 |
| 目标 | 特定任务单步 SOTA | 通用任务少步高质量生成 |

## 5. 局限性与未来工作

1. 高阶 TCF（TCF(2)）存在不稳定性，需进一步分析稳定条件。
2. TCF(S+) 收敛速度慢，需要更好的设计来提高蒸馏效率。
3. 尚未实现单步生成的高质量效果。

## 6. 核心贡献总结

1. **TCF**：利用 PF-ODE 的半线性结构和指数积分器实现更好的参数化，缩小一致性边界条件的时间间隔，同时降低参数化误差和蒸馏误差。
2. **SSS**：引入 γ 参数显式控制随机强度，减少多步采样中的累积误差。
3. **实验验证**：TCD 在所有推理步数上超越 LCM，在高 NFE 下超越教师模型，同时保持通用性（一套 LoRA 适配多种模型）。

## 7. Q&A：Consistency Model 概述

> **问题**：介绍一下 Consistency Model 是什么？

### 什么是 Consistency Model？

**Consistency Model (CM)** 是由 Song et al. 在 2023 年（ICML 2023）提出的一类新型生成模型，核心思想是利用扩散过程的 PF-ODE（概率流常微分方程）轨迹，学习一个**一致性函数**，将轨迹上任意噪声点直接映射回原点（干净数据），从而实现单步或少量步骤的高质量生成。

### 核心定义

**一致性函数** `f(x_t, t)` 满足两个关键性质：

1. **自一致性（边界条件）**：`f(x_0, 0) = x_0`（原点处为恒等映射）
2. **轨迹一致性**：`f(x_t, t) = f(x_s, s)`（同一轨迹上任意两点映射到同一目标）

直观理解：在扩散的前向过程中，数据 `x_0` 通过添加噪声变为 `x_t`，形成一条从数据到纯噪声的轨迹。CM 学习的是这条轨迹的**逆过程**——无论从轨迹上的哪个点出发，一步就能跳回原点。

### 训练方式

CM 通过以下两种方式训练：

| 方式 | 说明 | 特点 |
|------|------|------|
| **Consistency Distillation (CD)** | 使用预训练扩散模型作为教师，蒸馏其 PF-ODE 轨迹 | 依赖高质量教师模型，收敛快 |
| **Consistency Training (CT)** | 从头训练，无需教师模型 | 独立训练，但高质量生成需要大量数据 |

训练目标为最小化相邻时间步一致性映射的距离：
```
L = E[ d(f_θ(x_{t+n}, t+n), f_θ-(x_t, t)) ]
```
其中 `f_θ-` 是 EMA（指数移动平均）版本的目标网络。

### 参数化形式

为保证边界条件 `f(x_0, 0) = x_0`，采用跳连接参数化：
```
f_θ(x, t) = c_skip(t) · x + c_out(t) · F_θ(x, t)
```
其中 `c_skip(0)=1`, `c_out(0)=0`，`F_θ` 为可学习的神经网络。

### 采样方式

- **单步生成**：`x_0 = f_θ(z, T)`，一步从纯噪声生成干净图像
- **多步采样**：交替进行降噪 + 扩散，逐步逼近目标，质量更高

### 与扩散模型的关系

扩散模型通过迭代求解 PF-ODE（如 DDIM、DPM-Solver），每步需要模型评估。CM 通过直接学习 PF-ODE 的解函数 `f(x_t, t) → x_0`，**将迭代求解转化为一步映射**，速度提升可达数十到数百倍。

### 后续发展

| 工作 | 改进方向 |
|------|----------|
| **LCM** (Latent CM) | 将 CM 扩展到 Stable Diffusion 的潜在空间，实现文生图加速 |
| **CTM** (Consistency Trajectory Models) | 将映射 x_t→x_0 扩展为 x_t→x_s（anytime-to-anytime） |
| **TCD** (Trajectory CD) | 改进 LCM 的参数化误差和多步采样累积误差 |

TCD 正是在 LCM 基础上，针对一致性函数参数化中的离散化误差和多步采样中的细节丢失问题，提出了半线性结构 + 指数积分器的 TCF 和策略性随机采样 SSS。

## 8. Q&A：Consistency Model 与 Flow Matching 的关系

> **问题**：Consistency Model 和 Flow Matching 有什么关系？

### 什么是 Flow Matching？

**Flow Matching (FM)** 由 Lipman et al. (ICLR 2023) 提出，同时期 Liu et al. 也提出了思想类似的 **Rectified Flow**。FM 通过直接回归一个时间依赖的**向量场** `v(x, t)` 来学习连续归一化流（Continuous Normalizing Flow, CNF），将简单噪声分布（如高斯）连续变换为数据分布。

训练目标极其简洁：
```
L = E[ || v_θ(x_t, t) - u_t(x_t | x_0, x_1) ||^2 ]
```
其中 `u_t` 为条件向量场（由插值路径 `x_t = (1-t)·x_0 + t·x_1` 决定），无需像扩散模型那样设计复杂的 noise schedule 或进行 score matching。

### 两者的核心关系

| 维度 | Consistency Model | Flow Matching |
|------|-------------------|---------------|
| **数学原型** | 扩散模型的 PF-ODE | 连续归一化流 (CNF) |
| **学习目标** | 一致性函数 f(x_t, t) → x_0（端点映射） | 向量场 v(x_t, t)（速度场） |
| **ODE 结构** | **半线性**（可拆分线性项 + 非线性项） | 一般非线性（无固定结构） |
| **推理速度** | 支持 1~4 步生成 | 通常需 20~50 步（有蒸馏变体） |
| **训练复杂度** | 需自举（EMA 目标网络） | 直接回归，无需自举 |
| **路径自由度** | 由扩散 SDE 决定（固定 noising 路径） | 可自由设计概率路径（直线/曲线） |

### 关键分析

**1. 扩散模型是 Flow Matching 的特例**

扩散模型的 PF-ODE 可以改写为 Flow Matching 的形式——只需将扩散的 noise schedule 对应为特定的条件概率路径。换句话说，FM 是比扩散模型更一般的 ODE 生成框架。Stable Diffusion 3 和 Flux 等最新模型已转向 Flow Matching 范式。

**2. CM 能否做在 Flow Matching 上？——可以，但有代价**

- CM 的核心思想（一致性映射）**不依赖扩散假设**，理论上可以蒸馏任何 ODE 轨迹。
- 已有工作在 Rectified Flow 上做一致性蒸馏，实现 flow-based 的少步生成。
- **但**：Flow Matching 的 ODE **不具备半线性结构**，因此 TCD 的核心创新——利用半线性结构 + 指数积分器进行高精度参数化——在 FM 上无法直接应用。

**3. 为什么 TCD 依赖半线性结构？**

扩散 PF-ODE 的独特形式：
```
dx_t/dt = f(t)·x_t + g²(t)/(2σ_t)·ε_θ(x_t, t)
```
其中 `f(t)·x_t` 为线性项（可解析求解），`ε_θ` 为非线性项。这种拆分使得指数积分器可以**精确处理线性项**，仅对非线性项做近似，从而获得 O(h^{k+1}) 的参数化误差。

Flow Matching 的 ODE 为：
```
dx_t/dt = v_θ(x_t, t)
```
没有天然的线性/非线性拆分，因此无法应用指数积分器的技巧。

**4. 总结对比**

```
Flow Matching　　　　→ 灵活、简洁、训练稳定，成为新一代主流
Consistency Model　 → 极致推理速度（1~4步），基于扩散 PF-ODE
TCD　　　　　　　　　→ 在 CM 框架内，利用扩散 ODE 的半线性结构实现高精度蒸馏
```

Flow Matching 胜在灵活性和训练简洁性；CM/TCD 胜在推理效率。未来方向可能是：**用 Flow Matching 训练教师模型 → 通过一致性蒸馏（如 TCD）加速**，但这需要解决非半线性 ODE 的参数化精度问题。

## 9. Q&A：CM 的训练模式与 Teacher 模型的关系

> **问题**：CM 一般是用来做蒸馏，而 teacher 模型还是用 FM 来训练？CM 可以直接用来训练 teacher 模型吗？

### CM 的两种训练模式

CM 从诞生之初就定义了**两种独立的训练方式**，不依赖外部的 FM 或 Diffusion teacher：

| 模式 | 全称 | 说明 | 是否需要外部 Teacher |
|------|------|------|---------------------|
| **CD** | Consistency Distillation | 从预训练扩散模型（或 FM）蒸馏 PF-ODE 轨迹 | ✅ 需要 |
| **CT** | Consistency Training | 从头训练，模型自举（self-bootstrapping） | ❌ 不需要 |

### CT 模式：CM 可以直接当 Teacher

CT 模式下，CM **不需要任何外部 teacher**，训练流程为：

```
1. 对数据 x_0 采样噪声 → x_t
2. 用模型 f_θ(x_t, t) 预测 x_0
3. 用 EMA 网络 f_θ-(x_s, s) 作为"伪 teacher"（自举）
4. 最小化 ||f_θ(x_t, t) - f_θ-(x_s, s)||²
```

其中 `t > s`（相邻时间步），`f_θ-` 是模型自身的 EMA 副本。**没有任何外部预训练模型参与**。

### 那为什么大家更常用 CD？

| 对比维度 | CD（蒸馏） | CT（从头训练） |
|----------|-----------|---------------|
| **生成质量** | 高（继承 teacher 的知识） | 中等（需大量数据弥补） |
| **训练收敛** | 快（有 teacher 指引轨迹方向） | 慢（纯自举，早期信号弱） |
| **适用场景** | 文生图、高清图像（LCM/TCD 均用 CD） | 小数据集、低分辨率（CIFAR-10/ImageNet 64×64） |
| **依赖** | 需要高质量预训练模型 | 完全独立 |

CT 理论上可行，CM 原始论文在 CIFAR-10 上验证了 CT 的有效性（FID 约 3.5，1 步）。但在**高分辨率文生图**（如 SDXL 级别），CT 从头收敛极其困难，因此 LCM/TCD 等工作都采用 CD 路线。

### Teacher 模型可以用什么训练？

Teacher 的训练方式与 CM **完全解耦**：

```
Teacher 训练方式：
├── Diffusion (Score Matching)  →  SD1.5, SDXL, DALL·E 2
├── Flow Matching               →  SD3, Flux, Lumina-Next
└── 任何能提供 ODE 轨迹的模型即可
```

CM 蒸馏只需要 teacher 能提供一条从噪声到数据的**可靠 ODE 轨迹**，不关心 teacher 是用什么范式训练的。

### 结论

- CM **不是只能做蒸馏**——CT 模式可以独立训练，但当前工程上高质量文生图仍依赖 CD + 预训练 teacher
- Teacher **可以用 FM 训练**（SD3/Flux），**也可以用 Diffusion 训练**（SDXL），二者均可
- 当前主流路线：**Diffusion/FM 训练 teacher → CD 蒸馏得到 CM → 1~4 步快速推理**，CM 在其中扮演"加速器"角色

## 10. 技术细节补充

### 半线性结构的重要性

与 LCM 使用的 DDIM 参数化不同，TCD 采用基于指数积分器的半线性结构。DDIM 本质上是 PF-ODE 的近似解，当时间间隔较大时离散化误差显著。而指数积分器通过分离线性和非线性部分，精确求解线性项，仅对非线性项进行近似，从而大幅降低参数化误差。

### 与 LCM 采样过程的联系

LCM 的多步一致性采样可重写为：
```
x_s = α_s · denoise(x_t) + σ_s · z
```
这等价于 η = sqrt(1-α_s²) 的 DDIM 采样。SSS 通过引入额外参数 γ 和中间目标 s'，将固定随机性的采样转化为可控随机性的采样。

### 引导蒸馏

TCD 支持 classifier-free guidance 的引导蒸馏，与 LCM 类似地通过求解增强 PF-ODE 实现：
```
hat_z_{t_n}^{φ,ω} = (1+ω)·Φ(z_{t_{n+k}}, t_{n+k}, t_n, c; φ) - ω·Φ(z_{t_{n+k}}, t_{n+k}, t_n, ∅; φ)
```
