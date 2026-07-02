# PTQD: Accurate Post-Training Quantization for Diffusion Models

- **论文地址**: [https://arxiv.org/abs/2305.10657](https://arxiv.org/abs/2305.10657)
- **发表会议**: NeurIPS 2023
- **作者机构**: Zhejiang University, Monash University (ZIP Lab)
- **代码地址**: [https://github.com/ziplab/PTQD](https://github.com/ziplab/PTQD)

---

## 1. 研究动机

扩散模型在图像生成等任务中表现出色，但其推理阶段的**迭代去噪过程计算开销巨大**，限制了低延迟、大规模实际应用部署。现有后训练量化（PTQ）方法直接应用于低位扩散模型时会严重损害生成样本质量，主要面临两个挑战：

1. **逐步骤偏差**：量化噪声导致每步去噪过程中估计均值的偏差，并与预定义的方差调度不匹配。
2. **噪声累积效应**：随着采样推进，量化噪声逐渐累积，导致后期去噪步骤中信噪比（SNR）极低，严重影响去噪能力。

## 2. 核心方法

### 完整量化流程概览

PTQD 的整个量化过程分为**三个大阶段**：

```
┌─────────────────────────┐     ┌──────────────────────────┐     ┌──────────────────────┐
│  Phase 1: 模型量化       │ ──> │  Phase 2: 统计量收集     │ ──> │  Phase 3: 推理采样    │
│  使用 BRECQ/AdaRound     │     │  收集 k, μ_q, σ_q²       │     │  CNC + BC + VSC + MP  │
│  量化权重和激活           │     │  (1024 张样本)           │     │  逐步去噪生成图像      │
└─────────────────────────┘     └──────────────────────────┘     └──────────────────────┘
```

PTQD 提出了一个统一的框架，将量化噪声与扩散扰动噪声统一建模。核心包含三个关键技术：量化噪声解耦、量化噪声校正、步骤感知混合精度。下面逐步展开整个量化过程的数学推导和实现细节。

---

### 2.1 基础背景：扩散模型的反向过程

在介绍量化过程之前，先回顾 DDPM 的反向采样过程。给定当前带噪数据 $\mathbf{x}_t$，去噪一步得到 $\mathbf{x}_{t-1}$：

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z},\quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

其中：
- $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 是**噪声预测网络**（通常是 U-Net）的输出，预测当前步骤中添加的噪声
- $\alpha_t, \beta_t$ 是扩散过程的超参数，$\beta_t = 1-\alpha_t$，$\bar\alpha_t = \prod_{s=1}^t \alpha_s$
- $\sigma_t$ 是预定义的方差调度（variance schedule）
- $\frac{1}{\sqrt{\alpha_t}}\left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)$ 是估计的后验均值 $\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)$

---

### 2.2 Phase 1：模型量化 —— 量化噪声的引入

#### 2.2.1 均匀量化

PTQD 使用均匀量化将浮点权值和激活值映射到低位整数。给定浮点向量 $\mathbf{x}$ 和目标位宽 $b$：

$$\hat{\mathbf{x}} = \Delta \cdot \left(\text{clip}\big(\lfloor\frac{\mathbf{x}}{\Delta}\rceil+Z,\; 0,\; 2^{b}-1\big)-Z\right)$$

其中：
- $\Delta = \frac{\max(\mathbf{x}) - \min(\mathbf{x})}{2^b - 1}$ 为量化步长
- $Z = -\lfloor \frac{\min(\mathbf{x})}{\Delta}\rceil$ 为零点
- $\lfloor \cdot \rceil$ 为四舍五入操作

量化后的张量记为 $\hat{X}$，量化噪声定义为 $\Delta_X = \hat{X} - X$。

#### 2.2.2 具体量化配置

- **基础 PTQ 方法**：使用 BRECQ（主要）或 AdaRound 进行逐层校准
- **8-bit 实验**：在部分实验中使用 TensorRT 的朴素 PTQ（更简单快速）
- **量化范围**：模型的输入层和输出层固定为 8-bit，其他所有卷积层和线性层量化为目标位宽
- **混合精度配置**：权重固定为 4-bit，在不同步骤间共享（避免多次加载模型）；激活位宽根据步骤动态选择

#### 2.2.3 量化后的反向过程

量化后噪声预测网络的输出变为 $\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)$，反向采样变成：

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \big( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \Delta_{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)} \big)\right) + \sigma_t \mathbf{z}$$

量化噪声 $\Delta_{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}$ 会改变 $\mathbf{x}_{t-1}$ 的均值和方差，降低信噪比。**因此，必须在校正均值和方差后，才能在每个去噪步骤恢复 SNR。**

---

### 2.3 Phase 2：统计量收集

在推理之前，需要收集三个关键统计量。具体做法：**生成 1024 张样本**，同时运行全精度模型和量化模型，在每一步保存量化噪声 $\Delta_{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}$，然后计算：

| 统计量 | 符号 | 含义 | 计算方法 |
|--------|------|------|---------|
| 相关系数 | $k$ | 量化噪声中与全精度输出线性相关的比例 | 对 $\Delta_{\epsilon_\theta}$ 与 $\epsilon_\theta$ 做**线性回归** |
| 不相关噪声均值 | $\boldsymbol{\mu}_q$ | 残差不相关噪声的逐通道均值 | 统计 $\Delta'_{\epsilon_\theta}$ 的通道均值 |
| 不相关噪声方差 | $\sigma_q^2$ | 残差不相关噪声的逐通道方差 | 统计 $\Delta'_{\epsilon_\theta}$ 的通道方差 |

> **注意**：统计量收集是一次性的离线操作，不影响推理速度。收集完成后，$k$、$\boldsymbol{\mu}_q$、$\sigma_q^2$ 在整个推理过程中保持不变。

---

### 2.4 量化噪声解耦 (Correlation Disentanglement)

#### 2.4.1 为什么量化噪声会与输出相关？

**核心洞察**：虽然最初的量化噪声（在权值和激活被量化时）可能与原始信号不相关，但经过网络中的**归一化层**（BatchNorm、GroupNorm 等）后，相关性就会产生。

**命题1（数学证明）**：设 $Y$ 和 $\hat{Y}$ 分别是全精度模型和量化模型中某归一化层的输入，初始量化噪声 $\Delta_Y = \hat{Y} - Y$ 与 $Y$ 无关。经过归一化后：

$$\Delta_{\overline{Y}} = \frac{\hat{Y}-\mu_{\hat{Y}}}{\sigma_{\hat{Y}}} - \frac{{Y}-\mu_{Y}}{\sigma_{{Y}}} = \frac{\sigma_{{Y}}\Delta_Y - (\sigma_{\hat{Y}}-\sigma_Y){Y} + \sigma_{\hat{Y}}\mu_{Y} - \sigma_{Y}\mu_{\hat{Y}}}{\sigma_{\hat{Y}}\sigma_{{Y}}}$$

关键观察：分子中的第二项 $-(\sigma_{\hat{Y}}-\sigma_Y){Y}$ 与 $Y$ 直接相关（因为量化改变了方差 $\sigma_{\hat{Y}} \neq \sigma_Y$）。因此归一化后的量化噪声 $\Delta_{\overline{Y}}$ 获得了与 $Y$ 相关的分量。

论文通过实验收集了 4-bit LDM-8 在 LSUN-Churches 上 200 步的量化噪声数据，绘制了量化噪声 vs 全精度输出的散点图，验证了两者之间存在显著的线性相关性（尤其在低位宽时更明显）。

#### 2.4.2 解耦公式

基于以上理论分析和实验验证，将噪声预测网络输出端的量化噪声解耦为两部分：

$$\Delta_{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)} = k \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \Delta'_{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}$$

| 部分 | 表达式 | 含义 | 如何消除 |
|------|--------|------|---------|
| **相关部分** | $k \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ | 与全精度输出线性相关，$k$ 通过线性回归估计 | CNN：除以 $(1+k)$ |
| **不相关部分** | $\Delta'_{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}$ | 残差分量，假设与 $\boldsymbol{\epsilon}_\theta$ 不相关 | BC + VSC |

> 实践中强制 $k \geq 0$，若线性回归得到的 $k$ 为负则置零。当 $k \geq 0$ 时，相关噪声校正后，不相关部分的幅度也会缩小 $\frac{1}{1+k}$。

#### 2.4.3 相关系数 $k$ 的线性回归过程

对每个去噪步骤 $t$，我们收集了大量量化噪声 $\Delta_{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}$ 和全精度输出 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 的样本对，然后通过最小二乘拟合 $k$：

$$k = \arg\min_k \left\| \Delta_{\boldsymbol{\epsilon}_\theta} - k \cdot \boldsymbol{\epsilon}_\theta \right\|_2^2$$

一元线性回归的闭合解为：

$$k = \frac{\text{Cov}(\Delta_{\boldsymbol{\epsilon}_\theta},\; \boldsymbol{\epsilon}_\theta)}{\text{Var}(\boldsymbol{\epsilon}_\theta)}$$

论文附图显示，对于 W4A4 位宽的模型，Pearson 相关系数 $R$ 非常高，说明量化噪声主要由相关分量构成，这验证了 CNC 在该场景下的有效性。

---

### 2.5 Phase 3 推理：量化噪声校正（每步执行）

在实际推理时，每步去噪都需要执行以下校正操作：

#### 2.5.1 第一步：相关噪声校正 (CNC)

将解耦公式代入量化后的反向采样：

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \big( (1+k)\boldsymbol{\epsilon}_\theta + \Delta'_{\boldsymbol{\epsilon}_\theta} \big)\right) + \sigma_t \mathbf{z}$$

将量化网络输出 $\hat{\boldsymbol{\epsilon}}_\theta$ **除以 $(1+k)$**，即可消除相关部分：

$$\frac{\hat{\boldsymbol{\epsilon}}_\theta}{1+k} = \frac{(1+k)\boldsymbol{\epsilon}_\theta + \Delta'}{1+k} = \boldsymbol{\epsilon}_\theta + \frac{\Delta'}{1+k}$$

效果：
- ✅ 相关部分被完全消除
- ✅ 不相关部分的幅度从 $\Delta'$ 缩小为 $\frac{\Delta'}{1+k}$（当 $k>0$ 时）

校正后 $\mathbf{x}_{t-1}$ 的表达式变为：

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \boldsymbol{\epsilon}_\theta \right) + \sigma_t \mathbf{z} - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}(1+k)}\Delta'_{\boldsymbol{\epsilon}_\theta}$$

此时只剩下不相关量化噪声 $\Delta'$ 需要处理，且其幅度已被缩小。

---

#### 2.5.2 第二步：偏差校正 (BC)

采用与 DFQ 类似的逐通道偏差校正方法，从不相关量化噪声中减去其逐通道均值 $\boldsymbol{\mu}_q$：

$$\Delta'_{\text{corrected}} = \Delta'_{\boldsymbol{\epsilon}_\theta} - \boldsymbol{\mu}_q$$

其中 $\boldsymbol{\mu}_q$ 是 Phase 2 中收集的逐通道残差噪声均值。论文的实验显示，不同通道的偏差差异显著，因此必须进行**逐通道**校正。

---

#### 2.5.3 第三步：方差调度校准 (VSC)

经过 BC 后，残差不相关量化噪声均值被修正，但额外方差仍然存在。VSC 的核心思路是**将量化噪声的额外方差吸收到扩散噪声中**。

**关键假设**：残差不相关量化噪声 $\Delta'_{\boldsymbol{\epsilon}_\theta}$ 近似服从高斯分布 $\mathcal{N}(\boldsymbol{\mu}_q, \sigma_q^2)$。

> 论文使用了 SciPy 的 `normaltest`（基于 D'Agostino 和 Pearson 检验）进行正态性检验，在显著性水平 0.01 下，所有步骤均不能拒绝正态分布假设。

**VSC 的数学推导**：

校准后方差 $\sigma_t^{'2}$ 应该使总方差等于原始方差 $\sigma_t^2$：

$$\sigma_t^{'2} + \text{(量化噪声引入的额外方差)} = \sigma_t^2$$

量化噪声 $\Delta'$ 通过前面的系数 $\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}(1+k)}$ 传播到 $\mathbf{x}_{t-1}$，其贡献的方差为：

$$\left(\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}(1+k)}\right)^2 \cdot \sigma_q^2 = \frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)(1+k)^2}\sigma_q^2$$

因此校准后的方差调度为：

$${\sigma_t^{'2}} = \begin{cases} \sigma_t^2 - \frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)(1+k)^2}\sigma_q^2, & \text{if } \sigma_t^2 \ge \frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)(1+k)^2}\sigma_q^2 \\ 0, & \text{otherwise} \end{cases}$$

**关键分析**：系数 $\frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)(1+k)^2}$ 通常**非常小**，因此量化噪声的方差在绝大多数情况下可以被完全吸收。唯一的例外是确定性采样（$\sigma_t = 0$，即 $eta = 0$），此时 $\sigma_t^{'2} = 0$ 是最优解，无法使用 VSC——这正是 LSUN-Churches 实验中的一个场景。

---

#### 2.5.4 每步校正算法总结

```
Algorithm: 量化噪声校正（每步执行）

输入: x_t, t, 量化模型, k, μ_q, σ_q², 原始方差调度 σ_t²
输出: x_{t-1}

1. 前向传播量化模型，得到噪声预测:
   ϵ̂_θ(x_t, t) ← QuantizedModel.forward(x_t, t)

2. 相关噪声校正 (CNC):
   ϵ̂_θ_cnc ← ϵ̂_θ(x_t, t) / (1+k)

3. 偏差校正 (BC):
   ϵ̂_θ_corrected ← ϵ̂_θ_cnc - μ_q / (1+k)

4. 方差调度校准 (VSC):
   σ_t'² ← max( σ_t² - β_t²·σ_q² / [α_t(1-ᾱ_t)(1+k)²], 0 )

5. 计算估计均值:
   μ_θ ← (x_t - β_t/√(1-ᾱ_t) · ϵ̂_θ_corrected) / √α_t

6. 采样（关键！VSC 的结果 σ_t' 在此处使用）:
   x_{t-1} ← μ_θ + σ_t' · z,  其中 z ~ N(0, I)
```

---

#### 2.5.5 VSC 结果的使用方式详解

VSC 计算出的 $\sigma_t^{'2}$ 在**步骤 6 的采样阶段**起作用，它**直接替换了原始的方差调度** $\sigma_t^2$。具体来说：

**原始（无量化）的 DDPM 采样**：

$$x_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}\left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \boldsymbol{\epsilon}_\theta \right)}_{\text{估计均值 } \mu_\theta} + \underbrace{\sigma_t \cdot \mathbf{z}}_{\text{人为添加的高斯噪声}}$$

**PTQD 量化后的采样**：

$$x_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}\left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \boldsymbol{\epsilon}_\theta^{\text{corrected}} \right)}_{\text{校正后的估计均值 } \mu_\theta} + \underbrace{\sigma_t' \cdot \mathbf{z}}_{\text{缩小后的高斯噪声}}$$

**直观理解**：

```
总方差 = 扩散注入噪声的方差 + 量化引入的额外方差

         σ_t²                    σ_t'²              [β_t² · σ_q²] / [α_t(1-ᾱ_t)(1+k)²]
         ┌──────────┐            ┌──────────┐       ┌─────────────────────────────┐
原始:    │██████████│     =>     │████████··│   +   │············████··············│
         └──────────┘            └──────────┘       └─────────────────────────────┘
          人为添加                  人为减少               量化"白送"的噪声
          (太多)                   (补偿)                 (无法消除的部分)
```

- 量化本身已经给 $\mathbf{x}_{t-1}$ **额外注入了方差** $\frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)(1+k)^2}\sigma_q^2$
- 因此 VSC 将人为添加的高斯噪声方差从 $\sigma_t^2$ **缩小**为 $\sigma_t'^2$，使得"量化注入的 + 人为注入的 = 原始的人为注入的"
- 结果：$\mathbf{x}_{t-1}$ 的**总方差保持不变**，与全精度模型一致

**具体数值示例**（以 LDM-4 在 LSUN-Bedrooms 上 W4A8、某中间步骤为例）：

| 量 | 全精度值 | PTQD 量化后值 |
|---|---------|------------|
| 原始方差调度 $\sigma_t^2$ | $0.012$ | — |
| 量化噪声方差 $\sigma_q^2$ | — | $0.0003$ |
| 传播系数 $\frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)(1+k)^2}$ | — | $\approx 0.008$ |
| 量化引入的额外方差 | — | $0.008 \times 0.0003 = 2.4\times10^{-6}$ |
| 校准后方差 $\sigma_t'^2$ | — | $0.012 - 2.4\times10^{-6} \approx 0.012$ |

> 由于系数通常非常小，$\sigma_t'^2 \approx \sigma_t^2$，VSC 的调整幅度很小。但当量化噪声较大（如 W4A4）或系数较大时，VSC 的补偿作用就变得显著。

---

**总结三个校正的分工**：

| 校正技术 | 修正的对象 | 如何融入采样公式 |
|---------|----------|--------------|
| **CNC** | 估计均值 $\mu_\theta$ 中由相关量化噪声引入的系统性偏差 | 将噪声预测 $\hat{\epsilon}_\theta$ 除以 $(1+k)$ 后再计算均值 |
| **BC** | 估计均值 $\mu_\theta$ 中由不相关量化噪声的均值引入的偏差 | 减去 $\mu_q/(1+k)$ 后再计算均值 |
| **VSC** | $\mathbf{x}_{t-1}$ 中由不相关量化噪声的方差引入的额外方差 | 将采样时添加的高斯噪声方差从 $\sigma_t$ 缩小为 $\sigma_t'$ |

CNC 和 BC 修正的是 **均值路径**（步骤 2-3-5），VSC 修正的是 **方差路径**（步骤 4-6），两者在采样公式中作用的位置不同，但最终目标一致：让量化采样的分布尽可能逼近全精度采样的分布。

---

### 2.6 步骤感知混合精度 (Step-aware Mixed Precision)

#### 2.6.1 问题分析

即使校正了均值和方差，低比特量化模型生成满意样本仍然困难，原因是**量化噪声预测网络的 SNR 随去噪步数下降**。

定义量化模型的信噪比：

$$\text{SNR}^Q(t) = \frac{\|\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|_2}{\|\Delta_{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}\|_2}$$

定义前向过程的信噪比（衡量数据含噪程度）：

$$\text{SNR}^F(t) = \alpha_t^2 / \sigma_t^2$$

实验观察（论文 Figure 4）：
1. **$\text{SNR}^Q(t)$ 随 $t$ 减小而急剧下降**，在 $t \to 0$ 时，W4A4 模型的 $\text{SNR}^Q \approx 1$，意味着量化噪声幅度与原始信号相当
2. 高位宽模型具有更高的 $\text{SNR}^Q$
3. 提出的校正方法显著提高了 $\text{SNR}^Q$，尤其在大步数时

#### 2.6.2 比特宽度分配策略

为每个去噪步骤 $t$ 选择最优激活位宽：

**步骤1**：预定义可用位宽集合 $B = \{b_1, b_2, \ldots, b_n\}$（如 $\{4, 8\}$），分别评估各 $b_i$ 下的 $\text{SNR}^Q_{b_i}(t)$

**步骤2**：选择满足 $\text{SNR}^Q_{b}(t) > \text{SNR}^F(t)$ 的**最小位宽** $b_{\min}$

**步骤3**：若所有位宽都不满足，使用最大位宽保底

**设计考量**：
- **权重位宽固定**：权重位宽在所有步骤间共享，无需存储和加载多个模型状态文件
- **只调整激活位宽**：不同步骤使用不同位宽的激活值，需要分别用对应步骤范围的数据进行校准

#### 2.6.3 实际位宽分配结果

| 数据集 | 总步数 | W4A4 步骤范围 | W4A8 步骤范围 |
|--------|--------|--------------|--------------|
| ImageNet | 250 | 249 → 202 | 201 → 0 |
| ImageNet | 20 | 19 → 15 | 14 → 0 |
| LSUN-Bedrooms | 200 | 199 → 155 | 154 → 0 |
| LSUN-Churches | 200 | 199 → 146 | 145 → 0 |

> 规律：早期步骤（大 $t$）使用 W4A4（激进的加速），后期步骤（小 $t$）切换为 W4A8（保证生成质量）。

---

### 2.7 扩展到 DDIM 采样器

PTQD 同样适用于 DDIM 等快速采样器。量化后的 DDIM 采样公式为：

$$\mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}}\left(\frac{\mathbf{x}_t - \sqrt{1-\alpha_t}\hat{\boldsymbol{\epsilon}}_\theta}{\sqrt{\alpha_t}}\right) + \sqrt{1-\alpha_{t-1}-\sigma_t^2}\hat{\boldsymbol{\epsilon}}_\theta + \sigma_t\mathbf{z}$$

解耦和校正流程与 DDPM 一致，唯一区别在于 VSC 的系数变为：

$$\lambda_t = \frac{\sqrt{1 - \alpha_{t-1} - \sigma_t^2}}{1+k} - \frac{\sqrt{\alpha_{t-1}} \sqrt{1 - \alpha_t}}{(1+k)\sqrt{\alpha_t}}$$

校准后方差调度：${\sigma_t^{'2}} = \max(\sigma_t^2 - \lambda_t^2\sigma_q^2,\; 0)$

## 3. 实验结果

### 3.1 主要结果

| 数据集 | 模型配置 | 方法 | Bitwidth | FID↓ | BOPs压缩比 |
|--------|---------|------|----------|------|-----------|
| ImageNet 256×256 | LDM-4 (250步) | FP | 32/32 | 5.05 | - |
| ImageNet 256×256 | LDM-4 (250步) | PTQD | 4/8 | **5.11** | 19.96× |
| ImageNet 256×256 | LDM-4 (250步) | Q-Diffusion | MP | 9.97 | 21.25× |
| ImageNet 256×256 | LDM-4 (250步) | PTQD | MP | **6.44** | 21.25× |
| ImageNet 256×256 | LDM-4 (20步) | Q-Diffusion | MP | 116.61 | 21.61× |
| ImageNet 256×256 | LDM-4 (20步) | PTQD | MP | **7.75** | 21.61× |
| LSUN-Bedrooms | LDM-4 (200步) | FP | 32/32 | 3.00 | - |
| LSUN-Bedrooms | LDM-4 (200步) | PTQD | 4/8 | **5.94** | - |
| LSUN-Churches | LDM-8 (200步) | Q-Diffusion | MP | 218.59 | - |
| LSUN-Churches | LDM-8 (200步) | PTQD | MP | **17.99** | - |

### 3.2 消融实验

- CNC（相关噪声校正）：FID降低0.48，sFID降低6.55
- VSC（方差调度校准）：进一步降低FID 0.2
- BC（偏差校正）：完整PTQD达到FID 6.44，sFID 8.43

### 3.3 部署效率

- W8A8：推理速度提升 **2.03×**
- W4A4：推理速度提升 **3.34×**
- 混合精度：在速度和性能间取得良好平衡

## 4. 关键贡献

1. **首次统一建模**：将量化噪声和扩散噪声统一到同一框架下分析
2. **量化噪声解耦**：将量化噪声分解为相关部分和不相关部分，分别校正
3. **方差调度校准**：通过校准扩散方差调度来吸收量化引入的额外方差
4. **步骤感知混合精度**：动态为不同去噪步骤分配不同位宽，保持全过程的SNR
5. **显著的性能提升**：W4A8下FID仅比全精度增加0.06，节省19.9×比特运算

## 5. 局限性

- 仅量化了噪声预测网络，文本编码器和图像解码器等组件尚未量化
- 与其他深度生成模型类似，可能被滥用于制造虚假图像
- 方法基于DDPM推导，但可扩展到DDIM等快速采样器（附录中有详细推导）

---

## 6. 方法回顾与常见问题

### 6.1 方法逻辑链总结

PTQD 的核心思路可以总结为以下逻辑链：

```
观察 → 建模 → 解耦 → 校正 → 优化

│                │          │           │            │
│ 多步采样中      │ Δ =      │ CNC: ÷(1+k)│ 激活对量化  │ 
│ 量化噪声可分解  │ k·ε + Δ' │ BC: -μ_q   │ 影响最大    │
│                │          │ VSC: σ_t↓  │            │
▼                ▼          ▼           ▼            ▼
量化引入额外噪声   统一公式   分别校正     步骤感知     先低后高
                            均值/方差   混合精度     比特分配
```

### 6.2 常见问题

**Q1：噪声分解公式中的系数 $k$ 是每步都一样吗？**

**不是。** $k$、$\mu_q$、$\sigma_q^2$ 是**逐步独立统计**的，不同去噪步骤 $t$ 下的值可能不同。尤其是 $k$ 随位宽和步数变化明显 —— 低位宽（如 W4A4）且大 $t$ 时相关噪声显著，$k$ 较大；高位宽（如 W4A8）或小 $t$ 时 $k$ 较小。但统计量只在 Phase 2 收集一次，之后推理时直接查表使用。

> 公式 $\Delta_{\epsilon_\theta} = k \cdot \epsilon_\theta + \Delta'$ 的形式本身是**每步通用**的，只是系数 $k$ 的值随步数变化。

**Q2：统计量是如何收集的？**

在 Phase 2 中，使用 **1024 张校准样本**，同时运行全精度模型和量化模型，在**每一步**记录量化噪声 $\Delta_{\epsilon_\theta(t)}$，然后：

| 步骤 | 操作 |
|------|------|
| 1 | 对每步的 $\Delta_{\epsilon_\theta}$ 与全精度输出 $\epsilon_\theta$ 做线性回归，得 $k^{(t)}$ |
| 2 | 计算残差 $\Delta' = \Delta_{\epsilon_\theta} - k \cdot \epsilon_\theta$ |
| 3 | 统计 $\Delta'$ 的逐通道均值 $\mu_q^{(t)}$ 和方差 $\sigma_q^{2(t)}$ |

**这是一次性离线操作**，不影响推理速度。

**Q3：为什么 activation 对量化的影响更大？**

通过 SNR 分析发现：

- **SNR$^Q(t)$**（量化模型的信噪比）随 $t$ 减小而**急剧下降**
- 在后期步骤（$t$ 小），信号本身趋于干净（幅度小），量化噪声相对占比急剧放大
- W4A4 模型在 $t \to 0$ 时 SNR$^Q \approx 1$，意味着量化噪声幅度与原始信号相当

因此权重可以固定为 4-bit（所有步骤共享），而**激活位宽需要根据步骤动态调整**来维持足够的 SNR。

**Q4：比特分配是"循序渐进减少"吗？**

**不是，正相反 —— 是先低后高：**

| 步骤阶段 | 数据含噪程度 | 量化噪声影响 | 位宽策略 |
|---------|------------|------------|---------|
| 早期（$t$ 大） | 高，信号本身噪声大 | 小，量化噪声"淹没"在扩散噪声中 | **W4A4** 激进加速 |
| 后期（$t$ 小） | 低，信号趋于干净 | 大，量化噪声成为主要误差源 | **W4A8** 保证质量 |

选择依据：每一步选满足 SNR$^Q_b(t)$ > SNR$^F(t)$ 的**最小位宽**，即"刚好够用"原则。

**Q5：三个校正技术分别在采样公式的哪个位置起作用？**

| 校正技术 | 修正路径 | 在采样公式中的作用位置 | 具体操作 |
|---------|---------|-------------------|---------|
| **CNC** | 均值 | 噪声预测 $\hat{\epsilon}_\theta$ | 除以 $(1+k)$ |
| **BC** | 均值 | 噪声预测 $\hat{\epsilon}_\theta$ | 减去 $\mu_q/(1+k)$ |
| **VSC** | 方差 | 采样噪声 $\sigma_t \cdot \mathbf{z}$ | $\sigma_t \to \sigma_t'$ |

CNC + BC 修正估计均值的准确性，VSC 修正注入方差的准确性 —— 二者作用位置不同，最终目标一致：使量化采样分布逼近全精度采样分布。
