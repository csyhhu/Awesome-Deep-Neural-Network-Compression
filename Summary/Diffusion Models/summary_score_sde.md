# Score-Based Generative Modeling through Stochastic Differential Equations

- **论文标题**: Score-Based Generative Modeling through Stochastic Differential Equations
- **作者**: Yang Song (Stanford), Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Ben Poole (Google Brain), Stefano Ermon (Stanford)
- **发表**: ICLR 2021 (Oral)
- **arXiv**: https://arxiv.org/abs/2011.13456

---

## 一、核心思想

本文提出了一种**基于随机微分方程（SDE）的 Score-Based 生成模型统一框架**。其核心思路是将数据分布通过一个连续时间的扩散过程（前向 SDE）逐渐转化为噪声分布，然后通过估计该过程中每一时刻的**分数函数**（score function，即对数概率密度的梯度）来构建**逆向 SDE**，从而从噪声中生成数据。

关键创新在于：将此前两类主流方法——**Score Matching with Langevin Dynamics (SMLD)** 和 **Denoising Diffusion Probabilistic Models (DDPM)**——统一为两个特定 SDE 的离散化形式。

---

## 二、方法详述

### 2.1 前向 SDE（数据 → 噪声）

构造一个扩散过程 \(\{\mathbf{x}(t)\}_{t=0}^T\)，使得：
- \(\mathbf{x}(0) \sim p_0\)（数据分布）
- \(\mathbf{x}(T) \sim p_T\)（先验分布，如高斯噪声）

该过程建模为 Itô SDE：

\[
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}
\]

其中 \(\mathbf{f}\) 为漂移系数，\(g\) 为扩散系数，\(\mathbf{w}\) 为标准维纳过程。

### 2.2 逆向 SDE（噪声 → 数据）

根据 Anderson (1982) 的理论，扩散过程的逆过程也是一个扩散过程，由逆向 SDE 描述：

\[
d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})] dt + g(t) d\bar{\mathbf{w}}
\]

核心在于只需知道每个时刻边际分布的分数函数 \(\nabla_{\mathbf{x}} \log p_t(\mathbf{x})\) 即可完成逆向采样。

### 2.3 分数估计（训练）

通过连续化的去噪分数匹配目标训练时间相关的分数模型 \(\mathbf{s}_\theta(\mathbf{x}, t)\)：

\[
\theta^* = \arg\min_\theta \mathbb{E}_t \left\{ \lambda(t) \mathbb{E}_{\mathbf{x}(0)} \mathbb{E}_{\mathbf{x}(t)|\mathbf{x}(0)} \left[ \|\mathbf{s}_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t) | \mathbf{x}(0))\|_2^2 \right] \right\}
\]

### 2.4 三种具体的 SDE 设计

| SDE 类型 | 公式 | 特点 | 对应方法 |
|---------|------|------|---------|
| **VE SDE** (Variance Exploding) | \(d\mathbf{x} = \sqrt{\frac{d[\sigma^2(t)]}{dt}} d\mathbf{w}\) | 方差随时间趋于无穷 | SMLD 的连续推广 |
| **VP SDE** (Variance Preserving) | \(d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x} dt + \sqrt{\beta(t)} d\mathbf{w}\) | 方差保持有界（趋向1） | DDPM 的连续推广 |
| **sub-VP SDE** | \(d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x} dt + \sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s)ds})} d\mathbf{w}\) | 方差始终小于 VP SDE | 本文新提出，在似然度上表现最好 |

---

## 三、关键技术贡献

### 3.1 Predictor-Corrector (PC) 采样器

将数值 SDE 求解器与基于分数的 MCMC 方法结合：
- **Predictor**：数值 SDE 求解器（如 Euler-Maruyama）给出下一时刻的样本估计
- **Corrector**：利用分数信息，通过 Langevin MCMC 或 HMC 修正该估计的边际分布

PC 采样器统一并改进了 SMLD 和 DDPM 原有的采样方法。

### 3.2 概率流 ODE

证明存在一个与 SDE 共享相同边际分布的**确定性过程**，满足概率流 ODE：

\[
d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - \frac{1}{2} g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt
\]

该 ODE 带来的重要能力：
- **精确似然计算**：通过 neural ODE 的瞬时变量变换公式计算 exact likelihood
- **可逆编码**：将数据编码为潜在表示，支持插值、温度缩放等操作
- **唯一可识别编码**：编码由数据分布唯一确定（前向 SDE 无训练参数）
- **高效采样**：可使用黑盒 ODE 求解器（如 dopri5），以自适应步长大幅减少函数评估次数（可减少 90%+ 而不影响视觉质量）

### 3.3 可控生成

利用无条件分数模型实现条件生成，通过条件逆向 SDE：

\[
d\mathbf{x} = \{\mathbf{f}(\mathbf{x}, t) - g(t)^2[\nabla_{\mathbf{x}} \log p_t(\mathbf{x}) + \nabla_{\mathbf{x}} \log p_t(\mathbf{y} | \mathbf{x})]\} dt + g(t) d\bar{\mathbf{w}}
\]

应用场景：
- **类别条件生成**：训练时间相关分类器 \(p_t(\mathbf{y} | \mathbf{x}(t))\)
- **图像修复（Inpainting）**
- **图像上色（Colorization）**

---

## 四、架构改进

提出两种改进架构：
- **NCSN++**：针对 VE SDE 优化，CIFAR-10 FID = 2.45
- **DDPM++**：针对 VP/sub-VP SDE 优化，CIFAR-10 FID = 2.78

进一步使用连续训练目标并增加网络深度（deep 版本）后继续提升。

---

## 五、实验结果

### 5.1 CIFAR-10 无条件生成（样本质量）

| 模型 | FID↓ | IS↑ |
|------|------|-----|
| NCSN++ cont. (deep, VE) | **2.20** | **9.89** |
| DDPM++ cont. (deep, VP) | 2.41 | 9.68 |
| StyleGAN2-ADA (无条件) | 2.92 | 9.83 |
| DDPM (Ho et al.) | 3.17 | 9.46 |

> NCSN++ 在 CIFAR-10 上创造了无条件生成的**新纪录**，FID 2.20 甚至超越了当时最佳的条件生成模型 StyleGAN2-ADA（FID 2.42）。

### 5.2 CIFAR-10 似然度（bits/dim）

| 模型 | NLL↓ |
|------|------|
| DDPM++ cont. (deep, sub-VP) | **2.99** |
| DDPM++ cont. (sub-VP) | 3.02 |
| DDPM cont. (sub-VP) | 3.05 |
| DDPM (exact likelihood) | 3.28 |
| Glow | 3.35 |

> sub-VP SDE 在均匀去量化 CIFAR-10 上取得 **2.99 bits/dim** 的新纪录。

### 5.3 高分辨率生成

首次从 Score-Based 生成模型实现 **1024×1024** CelebA-HQ 高保真图像生成。

### 5.4 采样方法对比

- PC 采样器在所有情况下均优于纯 predictor 或纯 corrector 方法
- Reverse diffusion sampler 优于 ancestral sampling
- 概率流 ODE + black-box solver 在保证质量的同时大幅提升采样效率

---

## 六、关键发现与结论

1. **统一框架**：SMLD 和 DDPM 是同一框架下不同 SDE 的特例（VE SDE 和 VP SDE）
2. **VE vs VP**：VE SDE 在样本质量上更优，VP/sub-VP SDE 在似然度上更优——实际应用中需根据场景选择
3. **PC 采样**显著优于纯采样器，仅需增加少量计算
4. **概率流 ODE** 提供了精确似然计算和高效自适应采样的新能力
5. **可控生成**：单个无条件模型即可完成多种条件生成任务，无需重新训练

---

## 七、局限性与未来方向

- 采样速度仍慢于 GAN
- 采样器选择引入了大量超参数
- 未来方向：结合 Score-Based 模型的稳定训练与 GAN 的快速采样；自动选择和调节超参数

---

## 八、与模型压缩/轻量化的关联

虽然本文本身不直接涉及模型压缩，但其贡献对压缩领域有重要启示：
- **概率流 ODE 的高效采样**：通过自适应步长大幅减少函数评估次数（90%+），等价于推理加速
- **不同 SDE 的灵活选择**：sub-VP SDE 在保持质量的同时提高似然度，为精度-效率权衡提供新维度
- **单一模型多任务能力**：可控生成使一个模型完成多个任务，减少部署模型数量
