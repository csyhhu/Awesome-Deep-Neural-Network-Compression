# WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching

> **论文信息**
> - **标题**: WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching
> - **作者**: Weilun Feng, Guoxin Fan, Haotong Qin, Mingqiang Wu, Yuqi Li, Xiangqi Li, Zhulin An, Libo Huang, Dingrui Wang, Longlong Liao, Michele Magno, Yongjun Xu, Chuanguang Yang
> - **机构**: 中国科学院计算技术研究所、UCAS、ETH Zurich、CUNY、慕尼黑工业大学、福州大学、厦门数据智能研究院
> - **会议**: ICML 2026
> - **链接**: https://arxiv.org/abs/2603.06331
> - **代码**: https://github.com/FofGofx/WorldCache

---

## 1. 研究动机

基于扩散模型的世界模型（World Models）在统一世界模拟方面展现出巨大潜力，但其迭代去噪推理成本极高，严重制约了交互式应用和长时域推演（long-horizon rollouts）的实用性。

现有的特征缓存（Feature Caching）方法在**单模态**图像/视频扩散中取得了显著加速效果，但直接迁移到世界模型时会出现严重的误差累积和推演不稳定。核心原因在于世界模型具有两个独特挑战：

1. **Token 异质性（Token Heterogeneity）**：世界模型需要联合建模多模态信息（如 RGB + Depth），不同模态、不同空间位置 token 的动态变化程度差异极大（长尾分布）。大多数 token 变化平滑，但少数"困难 token"（如运动边界、深度不连续区域）表现出剧烈非线性变化。统一缓存策略无法兼顾。
2. **非均匀时序动态（Non-uniform Temporal Dynamics）**：去噪过程中，模型可能长时间处于平滑状态，然后突然进入高度非线性阶段。缓存失败通常由少数瓶颈 token 导致，而非全局平均行为。

---

## 2. 方法

### 2.1 核心思想

WorldCache 是一个**免训练**的缓存加速框架，包含两个核心技术：

### 2.2 曲率引导的异质 Token 预测（CHTP: Curvature-guided Heterogeneous Token Prediction）

**曲率评分**：利用物理启发的曲率度量来估计每个 token 的可预测性。给定最近三次完整推理的输出 $\mathbf{y}_{t_0}, \mathbf{y}_{t_1}, \mathbf{y}_{t_2}$，计算：

$$\kappa_i = \frac{\|\mathbf{a}_{t_0,i}\|_2}{\|\mathbf{v}_{t_0,i}\|_2^2 + \varepsilon}$$

其中 $\mathbf{v}$ 为离散速度，$\mathbf{a}$ 为离散加速度。曲率 $\kappa_i$ 作为归一化的"转弯率"，$\kappa$ 小的 token 变化接近线性，适合复用或外推；$\kappa$ 大的 token 方向变化剧烈，直接缓存容易漂移。

**异质预测策略**：根据曲率百分位数将 token 分为三组：
- **稳定组**（$\mathcal{I}_{\text{stable}}$）：0阶复用（直接复用最近完整输出）
- **线性组**（$\mathcal{I}_{\text{linear}}$）：1阶外推（$\mathbf{y}_{t^*,i} + k\cdot \mathbf{v}_{t^*,i}$）
- **混沌组**（$\mathcal{I}_{\text{chaotic}}$）：带阻尼的 Hermite 插值预测

对于混沌 token，使用基于 Hermite 平滑步进函数的阻尼更新：

$$\mathbf{v}^{\text{adapt}}_i(k) = (1-\alpha_k)\mathbf{v}_{t^*,i} + \alpha_k\mathbf{v}_{t^*-1,i}, \quad \alpha_k = 3x_k^2 - 2x_k^3$$

随着缓存步数 $k$ 增加，预测更保守，有效抑制高曲率下的漂移。

### 2.3 混沌优先的自适应跳过（CAS: Chaotic-prioritized Adaptive Skipping）

**无量纲归一化漂移**：利用曲率构造无量纲漂移指示器，解决不同模态/时间步下特征尺度不同的问题。

> **定理**：对于 $\kappa_i$ 与特征偏差 $\Delta\mathbf{y}_{t,i}$，$\kappa_i\cdot\|\Delta\mathbf{y}_{t,i}\|_2$ 在全局特征重缩放下保持不变（即无量纲）。

仅在混沌 token 集合上计算每步归一化漂移：

$$e_i(t) = \kappa_i\cdot\|\tilde{\mathbf{y}}_{t,i}-\tilde{\mathbf{y}}_{t+1,i}\|_2, \quad i\in\mathcal{I}_{\text{chaotic}}$$

$$E(t) = \frac{1}{|\mathcal{I}_{\text{chaotic}}|}\sum_{i\in\mathcal{I}_{\text{chaotic}}} e_i(t)$$

累积不确定性 $E_{\text{acc}} \leftarrow E_{\text{acc}} + E(t)$，超过阈值 $\eta$ 时触发完整计算。

### 2.4 整体流程

去噪过程中交替进行 FULL（完整骨干网络评估，刷新曲率与分组）和 CACHE（异质 token 预测 + 漂移累积），当混沌 token 累积归一化漂移超限时切换回 FULL。

---

## 3. 实验结果

### 3.1 世界生成（World Generation）

在 HunyuanVoyager-13B 和 Aether-5B 两个世界模型上评估：

| 模型 | 方法 | PSNR ↑ | 加速比 |
|------|------|--------|--------|
| Voyager-13B | WorldCache | **23.49** | **3.65×** |
| Voyager-13B | EasyCache | 21.76 | — |
| Voyager-13B | Baseline (无缓存) | — | 1.0× |
| Aether-5B | WorldCache | 最高 | **1.68×** |

- **Voyager-13B**：3.65× 端到端加速，WorldScore 接近无损（45.43 vs. baseline 46.40），几乎无额外显存开销（50.58GB vs. 50.44GB）
- 相比之下，层级缓存方法（ToCa、DuCa、TaylorSeer）显存超过 100GB，单卡无法运行
- **Aether-5B**：最强保真度 + 最高加速比

### 3.2 3D 重建

在 Aether 上评估深度和相机姿态重建：
- **深度**：Abs Rel 0.341（baseline 0.340），$\delta$ 精度最高
- **姿态**：RPE trans 0.068（无损），旋转误差最低（0.796 vs. HERO 0.861）
- 重建延迟降至 **21.20s（2.61× 加速）**

### 3.3 消融实验

- **Token 预测策略**：曲率引导的异构分组显著优于统一复用/线性外推/随机混合
- **分组百分位数**：广泛范围均有效（PSNR 22.77–23.52），方法鲁棒
- **跳过策略**：CAS 优于固定间隔、全局差异阈值、仅曲率触发等替代方案
- **阈值敏感度**：$\eta=0.20$ 为推荐平衡点

### 3.4 控制开销

曲率估计、分组、触发评估仅占端到端延迟的 **≈0.05%**，加速收益来自更廉价的缓存预测。

---

## 4. 核心贡献

1. **识别问题**：首次系统分析了将单模态扩散缓存方法迁移到世界模型时的两大挑战——token 异质性和非均匀时序动态
2. **曲率引导的异构 token 预测**：基于物理启发的曲率评分分配不同的缓存策略（复用/外推/阻尼），首次实现 token 级别的异构缓存
3. **混沌优先的自适应跳过**：提出无量纲漂移指示器，仅监控瓶颈 token，实现统一的跨尺度/跨时间步触发决策
4. **SOTA 性能**：两个主流世界模型上实现最高 3.7× 加速，保持 98% 生成质量

---

## 5. 对模型压缩领域的启示

WorldCache 为 diffusion world model 提出了一种**免训练、零额外显存开销**的推理加速方案。其核心思想具有较好的泛化潜力：

- **物理启发的可预测性度量**：曲率作为 token 可缓存性的通用指示器，可推广到其他时序模型
- **异构资源分配**：根据 token 难度动态分配计算资源，比统一策略更高效
- **瓶颈驱动的调度**：聚焦最难处理的子集决定更新时机，避免平均行为误导决策

---

## 6. 讨论与问答

### Q1: 三类 token 都没有走完整计算吗？

**问题**：High level 来看，本文把 Token 区分成三类——稳定组直接复用、线性组加权求和（1阶外推）、混沌组插值预测（Hermite 阻尼）。那有没有按一般方法（走骨干网络完整前向）计算的 Token？

**答案**：

**在 CACHE 步骤中，没有任何 token 走完整计算**。三组 token 全部使用近似策略：

| Token 组 | CACHE 步骤的计算方式 | 公式 |
|---------|-------------------|------|
| 稳定组 | 0阶复用 | $\tilde{\mathbf{y}}_{t,i} = \mathbf{y}_{t^*,i}$ |
| 线性组 | 1阶外推 | $\tilde{\mathbf{y}}_{t,i} = \mathbf{y}_{t^*,i} + k\cdot\mathbf{v}_{t^*,i}$ |
| 混沌组 | Hermite 阻尼预测 | $\tilde{\mathbf{y}}_{t,i} = \mathbf{y}_{t^*,i} + k\cdot\mathbf{v}^{\text{adapt}}_i(k)$ |

**完整计算只发生在 FULL 步骤**：当 CAS 机制检测到混沌 token 累积漂移 $E_{\text{acc}}$ 超过阈值 $\eta$ 时，才会执行一次 FULL 步骤，此时**所有 token** 都通过骨干网络 $\mathcal{F}_\theta$ 进行完整前向计算，得到 $\mathbf{y}_t = \mathcal{F}_\theta(\mathbf{z}_t, t)$，同时刷新曲率 $\kappa$ 和分组信息，为后续 CACHE 步骤提供准确的"锚点"。

**设计逻辑**：
- 世界模型在大部分去噪时间步中，token 轨迹相对平滑，三种近似策略足够
- 只在瓶颈 token（混沌组）开始漂移的关键时刻才触发完整计算
- 本质上是将计算资源从"每个时间步均匀分配"变为"集中用于关键时间步"
- 这就是 CAS（Chaotic-prioritized Adaptive Skipping）的核心——不是每步选哪些 token 要完整算，而是选哪些时间步要完整算

### Q2: CHTP 中的速度 v 和加速度 a 是怎么得到的？

**问题**：2.2 节中曲率公式里的速度 $\mathbf{v}$ 和加速度 $\mathbf{a}$ 如何计算？

**答案**：

$\mathbf{v}$ 和 $\mathbf{a}$ 由最近三次 FULL 步骤的骨干网络输出对去噪时间轴做**有限差分**得到。

设最近三次 FULL 步骤的时间步为 $t_2 > t_1 > t_0$（$t_0$ 是最新一次），对应的骨干网络输出为 $\mathbf{y}_{t_2}, \mathbf{y}_{t_1}, \mathbf{y}_{t_0}$：

- **速度（一阶差分）**：
  $$\mathbf{v}_{t_0,i} = \frac{\mathbf{y}_{t_0,i} - \mathbf{y}_{t_1,i}}{t_0 - t_1}, \quad \mathbf{v}_{t_1,i} = \frac{\mathbf{y}_{t_1,i} - \mathbf{y}_{t_2,i}}{t_1 - t_2}$$

- **加速度（二阶差分）**：
  $$\mathbf{a}_{t_0,i} = \frac{\mathbf{v}_{t_0,i} - \mathbf{v}_{t_1,i}}{t_0 - t_1}$$

- **曲率**：
  $$\kappa_i = \frac{\|\mathbf{a}_{t_0,i}\|_2}{\|\mathbf{v}_{t_0,i}\|_2^2 + \varepsilon}$$

**关键要点**：
- $\mathbf{v}$ 和 $\mathbf{a}$ 刻画的不是物理空间的速度/加速度，而是 token 特征在**去噪时间轴**上的变化速率和变化加速率
- 需要至少 3 次 FULL 输出才能计算曲率（Algorithm 1 中 `if |H| < 3 → FULL` 的原因）
- 每次触发 FULL 后，用最新的三次 FULL 输出刷新曲率和 token 分组

### Q3: 怎么理解基于 Hermite 平滑步进函数的阻尼更新？

**问题**：混沌 token 的 Hermite 阻尼更新公式 $\mathbf{v}^{\text{adapt}}_i(k) = (1-\alpha_k)\mathbf{v}_{t^*,i} + \alpha_k\mathbf{v}_{t^*-1,i}$ 如何理解？

**答案**：

**1. 直觉：用两个历史速度做加权融合来抑制发散**

混沌 token 曲率高（方向变化剧烈），单独信任最新速度 $\mathbf{v}_{t^*}$ 做外推很危险——它可能刚经历急转弯，沿此方向继续外推会快速偏离真实轨迹。而 $\mathbf{v}_{t^*-1}$ 代表更早、更平缓的运动方向，混合它起到**阻尼/惯性**作用，抑制预测发散。

**2. Hermite 平滑步进函数的三个关键性质**

$$\alpha_k = 3x_k^2 - 2x_k^3, \quad x_k = \min(k/n_{\max}, 1)$$

| 性质 | 数学表达 | 实际意义 |
|------|---------|---------|
| $\alpha(0)=0$ | 缓存刚开始 | 完全信任最新速度，等价于线性外推 |
| $\alpha(1)=1$ | 缓存到极限 | 完全使用历史速度，极度保守 |
| $\alpha'(0)=\alpha'(1)=0$ | 两端导数归零 | 过渡过程平滑无突变，避免相邻步之间预测行为跳跃 |

**3. 为什么不能简单用线性 $\alpha_k = x_k$？**

线性过渡在 $x=0$ 和 $x=1$ 处权重变化率非零，相邻 CACHE 步之间的预测速度会突然跳变，容易在生成结果中引入视觉伪影。Hermite 曲线保证 C¹ 连续性，预测行为平滑演化。

**一句话**：Hermite 平滑步进让混沌 token 的预测从"激进外推"到"保守阻尼"平滑过渡——缓存越久越保守，且过渡无突变。

### Q4: 详细介绍 CAS（Chaotic-prioritized Adaptive Skipping）

**问题**：CAS 机制的完整设计原理是什么？

**答案**：

CAS 是 WorldCache 的调度核心，负责决定**何时触发 FULL 计算**，包含以下设计层次：

---

**1. 动机：为什么需要 CAS？**

世界模型去噪的时间难度分布不均匀——长段平滑期夹杂短段突变期。两种已有策略均有致命缺陷：
- **固定间隔**：可能跳过突变区的关键步骤，又在平滑期浪费计算
- **全局阈值**（监控所有 token 的平均变化）：瓶颈 token 已漂移但被稳定 token 稀释信号 → 触发太晚；稳定 token 的正常波动也累积 → 触发过频

**核心洞察**：缓存失败由少数混沌 token 主导，而非全局平均。调度器应聚焦这些"瓶颈 token"。

---

**2. 关键创新：曲率诱导的无量纲化**

直接用原始特征差 $\|\tilde{\mathbf{y}}_{t,i} - \tilde{\mathbf{y}}_{t+1,i}\|_2$ 做阈值判断不可行——RGB token 和 Depth token 的特征范数可能差数量级，不同时间步分布也不同，无法用统一阈值。

**定理 1** 证明：$\kappa_i \cdot \|\Delta\mathbf{y}_{t,i}\|_2$ 是无量纲的——对特征全局放缩 $s\cdot\mathbf{y}$ 保持不变。因为 $\kappa_i$ 的分母 $\|\mathbf{v}\|^2$ 自带尺度信息，乘积抵消了量纲依赖。这使得同一个阈值 $\eta$ 可以跨模态、跨时间步通用。

---

**3. CAS 三步计算流程**

仅对混沌组 $\mathcal{I}_{\text{chaotic}}$ 执行以下计算：

**(1) 每步归一化漂移**：
$$e_i(t) = \kappa_i \cdot \|\tilde{\mathbf{y}}_{t,i} - \tilde{\mathbf{y}}_{t+1,i}\|_2$$

**(2) 聚合为统一分数**：
$$E(t) = \frac{1}{|\mathcal{I}_{\text{chaotic}}|} \sum_{i \in \mathcal{I}_{\text{chaotic}}} e_i(t)$$

**(3) 累积并触发**：
$$E_{\text{acc}} \leftarrow E_{\text{acc}} + E(t)$$
当 $E_{\text{acc}} > \eta$ 时触发 FULL，随后重置 $E_{\text{acc}} \leftarrow 0$。

---

**4. 设计的三个精妙之处**

| 设计要点 | 为什么重要 |
|---------|-----------|
| **只监控混沌 token** | 避免稳定 token 噪声稀释危险信号（触发太晚），也避免稳定 token 正常波动引发误触发（触发过频） |
| **累积而非逐步判断** | 单步漂移可能很小但累积后不可忽视；突发剧烈漂移也能立刻推高 $E_{\text{acc}}$，同时覆盖渐进和突发两种情况 |
| **无量纲化** | 同一个 $\eta=0.20$ 在 Voyager-13B 和 Aether-5B 上直接可用，无需逐模型/逐模态调参 |

---

**5. 消融实验验证**

CAS 与替代策略的对比：固定间隔最差；全局 L2 阈值因尺度不一致难以校准；仅曲率触发（无实际位移）导致不必要重算过多。CAS 通过 **(1) 无量纲化 + (2) 混沌优先** 的组合取得最优速度-质量权衡。

---

## 7. 整体总结（一句话版）

WorldCache 将世界模型去噪过程分为 **FULL** 和 **CACHE** 两种步骤：

- **CAS** 负责调度——通过监控混沌 token 的无量纲累积漂移，判断哪些时间步可以走 CACHE、哪些必须走 FULL
- **CHTP** 负责 CACHE 步骤内的计算——将全部 token 按曲率分为三类，分别用复用（稳定组）、1阶外推（线性组）、Hermite 阻尼融合（混沌组）进行近似预测

三类 token 对历史信息的依赖程度不同：稳定组仅复用最近 1 次 FULL 输出，线性组利用 2 次 FULL 输出计算速度做外推，混沌组利用 3 次 FULL 输出（两个历史速度的平滑融合）。三次 FULL 输出同时为曲率估计和 token 分组提供基础。
