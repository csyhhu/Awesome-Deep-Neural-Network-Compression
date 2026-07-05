# Stable Velocity: A Variance Perspective on Flow Matching

**论文信息**:
- **标题**: Stable Velocity: A Variance Perspective on Flow Matching
- **作者**: Donglin Yang, Yongxing Zhang, Xin Yu, Liang Hou, Xin Tao, Pengfei Wan, Xiaojuan Qi, Renjie Liao
- **单位**: 香港大学、UBC、快手Kling团队、Vector Institute
- **会议**: ICML 2026
- **链接**: https://arxiv.org/abs/2602.05435
- **代码**: https://github.com/linYDTHU/StableVelocity

---

## TL;DR

本文提出 **Stable Velocity** 框架，核心洞察是流匹配的生成轨迹天然分为**高方差区**（$t$ 大，速度场噪声大）和**低方差区**（$t$ 小，后验坍缩为单峰、轨迹变为直线）。基于此提出三个互补组件：

| 组件 | 阶段 | 做了什么 |
|------|------|---------|
| **StableVM** | 训练 | 用 n 个参考样本的条件速度做**距离加权平均**，构造更稳定的训练**目标**（label），降低高方差区的优化噪声 |
| **VA-REPA** | 训练 | 用冻结的 DINOv2 做教师，对齐 DiT 中间层特征，并通过 $w(t)$ 只在**低方差区**激活这个辅助损失 |
| **StableVS** | 推理 | 在低方差区**大步跳跃积分**（轨迹是直线，$v_t$ 近似恒定），无需微调即可 ~2× 加速 |

> ⚠️ 常见误解：StableVM 的加权综合结果是**训练目标（label）**，不是"下一次解码的输入"。$x_t$ 的构造方式和模型的前向传播路径完全不变。

---

## 1. 核心动机

Conditional Flow Matching (CFM) 虽然优雅，但其训练目标存在一个根本性问题：**条件速度场 $\mathbf{v}_t(\mathbf{x}_t \mid \mathbf{x}_0)$ 是对真实边际速度场 $\mathbf{v}_t(\mathbf{x}_t)$ 的单样本蒙特卡洛估计**，在靠近先验分布的时间步上呈现出高方差，导致优化不稳定、收敛缓慢。

论文从方差角度系统分析了随机插值（stochastic interpolants），揭示了生成轨迹中天然存在的**双区制结构**（two-regime structure）。

---

## 2. 关键发现：双区制结构

通过对 GMM、CIFAR-10、ImageNet latent 的方差曲线分析（Fig. 1），发现：

- **低方差距（Low-Variance Regime）**：$t \in [0, \xi]$，后验分布 $p_t(\mathbf{x}_0 \mid \mathbf{x}_t)$ 集中在单个参考样本上，条件速度与真实速度几乎一致，$\mathcal{V}_{\text{CFM}}(t) \approx 0$。
- **高方差距（High-Variance Regime）**：$t \in (\xi, 1]$，后验分布在多个参考样本之间扩散，条件速度波动大，训练噪声大。
- **维度效应**：数据维度越高，分割点 $\xi$ 越接近 1，低方差区越大。

---

## 3. 方法：Stable Velocity 框架

### 3.1 StableVM（Stable Velocity Matching）

**核心思想**：用多参考样本的自归一化重要性加权平均替代单样本条件速度目标，降低训练方差同时保持无偏性。

- 定义复合条件概率路径 $p_t^{\text{GMM}}(\mathbf{x}_t \mid \{\mathbf{x}_0^i\}_{i=1}^n) = \sum_{i=1}^n \frac{1}{n} p_t(\mathbf{x}_t \mid \mathbf{x}_0^i)$（高斯混合模型路径）
- StableVM 目标：
  $$\widehat{\mathbf{v}}_{\text{StableVM}}(\mathbf{x}_t; \{\mathbf{x}_0^i\}) := \frac{\sum_{k=1}^n p_t(\mathbf{x}_t \mid \mathbf{x}_0^k) \mathbf{v}_t(\mathbf{x}_t \mid \mathbf{x}_0^k)}{\sum_{j=1}^n p_t(\mathbf{x}_t \mid \mathbf{x}_0^j)}$$

- **理论保证**：
  - **无偏性（Theorem 1）**：$\mathbb{E}[\widehat{\mathbf{v}}_{\text{StableVM}}] = \mathbf{v}_t(\mathbf{x}_t)$，全局最优解与 CFM 相同
  - **方差界（Theorem 2 & 3）**：$\mathcal{V}_{\text{StableVM}}(t) < \mathcal{V}_{\text{CFM}}(t)$ 严格成立，且方差以 $O(1/n)$ 速率衰减

- **扩展**：通过维护每类 FIFO memory bank（容量 $K=256$），扩展到 class-conditional + CFG 场景，保持无偏性。

### 3.2 VA-REPA（Variance-Aware Representation Alignment）

**核心思想**：表示对齐（REPA）的优势主要集中在**低方差区**，高方差区对齐信号差。VA-REPA 自适应地在低方差区增强辅助监督。

- 权重函数 $w(t)$ 仅在低方差区 $[0, \xi]$ 激活表示对齐损失：
  - **硬阈值**: $w_{\text{hard}}(t) = \mathbb{I}[t < \xi]$
  - **Sigmoid**: $w_{\text{sigmoid}}(t) = \sigma(k(\xi - t))$
  - **SNR-based**: $w_{\text{SNR}}(t) = \frac{\text{SNR}(t)}{\text{SNR}(t) + \text{SNR}(\xi)}$

- 损失：$\mathcal{L} = \mathcal{L}_{\text{StableVM}} + \lambda_{\text{RA}} \frac{\mathbb{E}[w(t) \ell_{\text{RA}}]}{\mathbb{E}[w(t)]}$（归一化避免梯度消失）

### 3.3 StableVS（Stable Velocity Sampling）

**核心思想**：在低方差区，后验坍缩为单点，采样轨迹变为确定性直线，可用闭式解进行大步长积分，无需微调即可加速。

- **SDE 版本**：DDIM-style posterior $p_\tau(\mathbf{x}_\tau \mid \mathbf{x}_t, \mathbf{v}_t) = \mathcal{N}(\boldsymbol{\mu}_{\tau \mid t}, \beta_t^2 \mathbf{I})$
- **ODE 版本**：PF-ODE 的闭式解：
  $$\mathbf{x}_{\tau} = \sigma_{\tau} \left[ \left( \frac{1}{\sigma_{t}} - \frac{\sigma_{t}'}{\sigma_{t}} \Psi_{t, \tau} \right) \mathbf{x}_{t} + \Psi_{t, \tau} \mathbf{v}_{t}(\mathbf{x}_{t}) \right]$$

- 在线性插值（$\alpha_t=1-t, \sigma_t=t$）下退化为简单的 Euler 步：
  $$\mathbf{x}_{\tau} = \mathbf{x}_{t} + (\tau - t) \mathbf{v}_{t}(\mathbf{x}_{t})$$

---

## 4. 实验

### 4.1 StableVM + VA-REPA 训练

**ImageNet 256×256 类别条件生成（SiT-XL）**:
- 80 epochs with CFG → FID=1.80, IS=272.4，超越同期的 REPA、iREPA、REG 等方法
- 多个模型规模（SiT-B/L/XL）一致提升 FID、IS、Precision、Recall
- 与 REPA 变体（REPA/REG/iREPA）的结合均一致提升性能
- 分割点 $\xi=0.7$ 最优，sigmoid 权重优于硬阈值

### 4.2 StableVS 采样加速

| 模型 | Baseline 步数 | StableVS 步数 | 效果 |
|------|-------------|-------------|------|
| SD3.5-Large | 30步 | 20步（低方差区9步） | GenEval Overall 保持 0.723，PSNR 从 16.93→36.92 |
| Flux-dev | 30步 | 20步 | Overall 从 0.659→0.666 |
| Qwen-Image | 30步 | 17步 | Overall 从 0.721→0.731 |
| Wan2.2 | 30步（UniPC） | 20步 | 参考指标显著提升，感知质量持平 |

**关键结论**：低方差区采用 StableVS 可减少约 2 倍采样步数，且质量不降。

---

## 5. 与 STF 的对比

| 维度 | STF | StableVM |
|------|-----|----------|
| 框架 | VP Diffusion | 通用随机插值（含 Flow Matching） |
| 无偏性 | 有偏（bias 随 n→∞ 消失） | **严格无偏** |
| 输入构造 | 单参考样本 $\mathbf{x}_t \sim p_t(\cdot\mid\mathbf{x}_0^1)$ | 复合 GMM 路径 |
| 条件扩展 | 不支持 | 支持 class-conditional + CFG |

---

## 6. 总结与启示

1. **方差视角**是理解流匹配训练与采样动态的统一框架
2. **双区制结构**自然导出了训练与采样的差异化策略
3. 三个组件（StableVM/VA-REPA/StableVS）互补且正交，可独立或组合使用
4. StableVS 是**免微调**的推理加速，对已部署的大模型（如 SD3.5、Flux、Wan2.2）即插即用

---

## 7. 讨论：训练与推理流程示例

**Q: 用一个例子说明本文在训练、推理时的流程？**

**A:** 以 **ImageNet 类别条件生成**（目标类别 "Golden Retriever"，类别 207，SiT-XL backbone）为例：

### 训练流程

1. **数据预处理**：256×256 图片 → SD VAE → latent `z ∈ R^(32×32×4)`。每类维护 FIFO memory bank（容量 K=256）。

2. **采样时间步与参考样本**：假设 `t = 0.85`（高方差区，$t > \xi=0.7$），从类别 207 的 memory bank 取 n 个参考 latent $\{z_0^1,...,z_0^n\}$。

3. **StableVM 构造训练目标**（与 CFM 的核心区别）：
   - **CFM**：仅用单个 $z_0^1$，$x_t = \alpha_t z_0^1 + \sigma_t \varepsilon$，目标 = $\mathbf{v}_t(x_t|z_0^1)$
   - **StableVM**：从 GMM 复合路径采样 $x_t \sim \frac{1}{n}\sum_i p_t(x_t|z_0^i)$，目标 = 自归一化加权平均 $\frac{\sum_k p_t(x_t|z_0^k)\mathbf{v}_t(x_t|z_0^k)}{\sum_j p_t(x_t|z_0^j)}$
   - 本质：用多参考样本的重要性加权平均替代单样本 Monte Carlo 估计，**无偏且方差以 O(1/n) 衰减**

4. **VA-REPA 自适应监督**：
   - `t = 0.3`（低方差区）：$x_t$ 保留金毛犬语义信息 → sigmoid 权重 $w(t) \approx 0.999$ → 激活 REPA 表示对齐
   - `t = 0.85`（高方差区）：$x_t$ 接近纯噪声 → $w(t) \approx 0.047$ → 几乎不激活对齐

5. **总损失**：$\mathcal{L} = \mathcal{L}_{\text{StableVM}} + \lambda_{RA} \cdot \frac{\mathbb{E}[w(t)\ell_{RA}]}{\mathbb{E}[w(t)]}$，归一化分母避免梯度消失。更新参数后，将新 latent 推入 memory bank。

### 推理流程（StableVS，以 SD3.5-Large 为例）

**输入**：文本 prompt = "a golden retriever playing in the park"，总计 **20 步**（基线 30 步），分割点 $\xi=0.85$：

| 时间区间 | 步数 | 求解器 | 原理 |
|---------|------|--------|------|
| $t \in [1, 0.85]$ | 11 步 | 标准 Euler | 高方差区，后验多模态分布，需精细数值积分 |
| $t \in [0.85, 0]$ | **9 步** | **StableVS** | 低方差区，后验坍缩为单峰，轨迹为确定性直线 |

在低方差区，线性插值下的 PF-ODE 退化为：$x_{\tau} = x_t + (\tau - t) \cdot \mathbf{v}_t(x_t)$，即**大步长 Euler 一次跳过多个时间步**，因为速度场几乎恒定。

**效果**：GenEval Overall 从 20 步的 0.710 恢复到 0.723（与 30 步持平），PSNR 从 16.93 → 36.92，实现 **~1.5× 推理加速**且质量无损。

---

## 8. 讨论：GMM、StableVM加权机制与训练目标

**Q1: GMM 是什么？**

**A:** 本文中的 GMM（Gaussian Mixture Model）指 **StableVM 中构造的复合条件概率路径**，即 n 个高斯分布的等权重混合：

$$p_t^{\text{GMM}}(x_t \mid \{x_0^i\}_{i=1}^n) = \frac{1}{n} \sum_{i=1}^n \mathcal{N}(x_t \mid \alpha_t x_0^i, \sigma_t^2 \mathbf{I})$$

与 CFM 的对比如下：

| | CFM 条件路径 | StableVM GMM 路径 |
|---|---|---|
| 参考样本数 | 1 个 | n 个 |
| 分布形式 | 单个高斯 | n 个高斯的等权重混合 |
| 覆盖范围 | 数据分布的一个极小局部 | 更好地覆盖真实边际分布 |

**为什么用 GMM**：单个条件路径只能覆盖数据分布中单个样本的"势力范围"，而 GMM 路径用 n 个参考样本的混合更忠实地逼近真实前向扩散的边际分布 $p_t(x_t)$，从而降低训练目标中来自"参考样本选择随机性"的方差。

**Q2: StableVM 是拿随机一个参考样本的采样结果对当前样本进行加权？**

**A:** 更准确的描述是三步走的 **自归一化重要性采样（SNIS）** 过程：

1. **采样 $x_t$**：先从 n 个参考中等概率随机选一个 $x_0^k$，再从该参考的条件分布中采样 $x_t \sim \mathcal{N}(\alpha_t x_0^k, \sigma_t^2 \mathbf{I})$。等价于从 GMM 路径直接采样。
2. **计算似然权重**：拿到 $x_t$ 后，将其代入**所有 n 个参考样本**的条件分布，计算每个参考的似然 $p_t(x_t \mid x_0^j)$。这相当于"回溯"问：$x_t$ 有多大可能来自参考 j？
3. **加权平均**：$\widehat{v}_{\text{StableVM}} = \frac{\sum_{k=1}^n p_t(x_t|x_0^k) \cdot v_t(x_t|x_0^k)}{\sum_{j=1}^n p_t(x_t|x_0^j)}$。似然越大的参考，其条件速度权重越大。

**直觉**：如果 $x_t$ 恰好落在某个参考的"领地"内（即 $p_t(x_t|x_0^A) \gg$ 其他），说明 $x_t$ 大概率是从参考 A 的内插生成的，此时加权平均自动让参考 A 的条件速度占主导——这是一种**数据驱动的自适应加权**，无需任何超参数。

**Q3: 训练目标的 label 如何获得？**

**A:** **完全不需要 label！** 这是流匹配（Flow Matching）的根本优势——训练目标是一个**纯解析的闭式解**。

给定参考样本 $x_0$（即一张图片的 VAE latent）、构造的噪声输入 $x_t$、时间步 $t$ 和插值参数 $\alpha_t, \sigma_t$，条件速度场可直接计算：

$$v_t(x_t \mid x_0) = \frac{\sigma_t'}{\sigma_t}(x_t - \alpha_t x_0) + \alpha_t' x_0$$

以最常用的线性插值 $\alpha_t = 1-t, \sigma_t = t$ 为例：

$$v_t(x_t \mid x_0) = \frac{1}{t}(x_t - (1-t)x_0) - x_0 = \frac{x_t - x_0}{t}$$

这就是训练时模型要学习的"label"——它描述了把噪声 $x_t$"推"回干净数据 $x_0$ 的方向和速率。

对于 StableVM，目标稍有不同但同样闭式可算：

$$\widehat{v}_{\text{StableVM}} = \frac{\sum_k \exp\!\left(-\frac{\|x_t - \alpha_t x_0^k\|^2}{2\sigma_t^2}\right) \cdot \left[\frac{\sigma_t'}{\sigma_t}(x_t - \alpha_t x_0^k) + \alpha_t' x_0^k\right]}{\sum_j \exp\!\left(-\frac{\|x_t - \alpha_t x_0^j\|^2}{2\sigma_t^2}\right)}$$

分子分母中的所有量——$x_t$、$x_0^k$、$t$、$\alpha_t$、$\sigma_t$——均为已知，可以**完全解析地计算**出训练目标，无需任何外部标注或预训练模型。

**总结**：流匹配本质是让神经网络学习一个由数学公式定义的确定性映射。这不同于传统监督学习需要人工标注，也不同于 GAN 需要判别器——训练信号天然来自于随机插值的数学结构。

---

## 9. 讨论：如何计算 $x_t$ 在每个参考样本下的似然？

**Q: 如何计算 $x_t$ 在每个参考样本下的似然？**

**A:** 对于参考样本 $x_0^i$，其条件路径分布是 d 维各向同性高斯：

$$p_t(x_t \mid x_0^i) = \mathcal{N}(x_t \mid \alpha_t x_0^i, \; \sigma_t^2 \mathbf{I})$$

**完整 PDF 公式**：

$$p_t(x_t \mid x_0^i) = \frac{1}{(2\pi\sigma_t^2)^{d/2}} \exp\!\left(-\frac{\|x_t - \alpha_t x_0^i\|^2}{2\sigma_t^2}\right)$$

但在 StableVM 的自归一化加权中，归一化常数 $\frac{1}{(2\pi\sigma_t^2)^{d/2}}$ 在分子分母中**完全抵消**，实际只需计算：

$$w_i = \exp\!\left(-\frac{\|x_t - \alpha_t x_0^i\|^2}{2\sigma_t^2}\right)$$

**伪代码**（线性插值 $\alpha_t=1-t, \sigma_t=t$）：

```python
# x_t:       [d]      当前噪声样本
# x0_bank:   [n, d]   n个参考样本

sq_dist = ((x_t - (1-t) * x0_bank) ** 2).sum(dim=-1)  # [n] 欧氏距离平方
log_weights = -sq_dist / (2 * t ** 2)                  # [n] 对数权重
weights = exp(log_weights)                              # [n] 未归一化权重

v_cond = (x_t.unsqueeze(0) - x0_bank) / t              # [n, d] 条件速度
target = (weights.unsqueeze(-1) * v_cond).sum(0) / weights.sum()
```

**类 Softmax 注意力视角**：

| 概念 | 注意力机制 | StableVM |
|------|-----------|----------|
| Query | $Q$ | $x_t$ |
| Key | $K_i$ | $\alpha_t x_0^i$（参考样本的插值中心） |
| 注意力分数 | $\exp(Q \cdot K_i)$ | $\exp\!\left(-\frac{\|x_t - \alpha_t x_0^i\|^2}{2\sigma_t^2}\right)$ |
| Value | $V_i$ | $v_t(x_t \mid x_0^i)$（条件速度） |
| 温度参数 | $\sqrt{d_k}$ | $\sigma_t^2 = t^2$ |

**$\sigma_t^2$ 的双区制含义**：
- **高方差区**（$t$ 大，如 $t=0.9$）：$\sigma_t^2=0.81$，温度高 → 权重分布均匀，多个参考共同参与 → 方差大，需要多样本加权来稳定
- **低方差区**（$t$ 小，如 $t=0.1$）：$\sigma_t^2=0.01$，温度低 → 权重高度集中在最近的参考上（后验坍缩为单峰）→ 方差小，单样本 CFM 就足够

---

## 10. 讨论：VA-REPA 的权重函数作用对象

**Q: VA-REPA 的权重函数 $w(t)$ 是直接作用在这一步解码的表达上吗？**

**A: 不是。** $w(t)$ 是一个仅依赖时间步 $t$ 的**标量权重**，作用在对齐**损失值**上，而非直接修改特征表示。VA-REPA 的损失公式为：

$$\mathcal{L} = \mathcal{L}_{\text{StableVM}} + \lambda_{\text{RA}} \cdot \frac{\mathbb{E}_{t, x_t}\!\left[w(t)\,\ell_{\text{RA}}(x_t)\right]}{\mathbb{E}_{t}\!\left[w(t)\right]}$$

可以拆解为三层：

| 层次 | 内容 | 说明 |
|------|------|------|
| 特征层 | $f_\theta(x_t, t)$ → 投影头 → 与 $f_{\text{DINO}}(x_0)$ 对比 | 这是 REPA 的核心，提取 DiT 中间层特征与预训练表示对齐 |
| 损失层 | $\ell_{\text{RA}}(x_t)$，如余弦相似度损失 | 衡量特征对齐的好坏 |
| 权重层 | $w(t) \in [0,1]$，乘在 $\ell_{\text{RA}}$ 上 | **仅控制这个损失的贡献幅度** |

**$w(t)$ 的角色类比**：像一个"音量旋钮"——不改变音乐本身（特征表示），只控制辅助信号在总损失中响多大：
- $t=0.2$（低方差区）：$w(t) \approx 1$，对齐信号"音量全开"
- $t=0.9$（高方差区）：$w(t) \approx 0$，对齐信号"几乎静音"

**分母 $\mathbb{E}[w(t)]$ 的作用**：归一化。如果没有它，当大部分训练样本落在高方差区（$w(t) \approx 0$）时，对齐项的有效梯度会消失。除以 $\mathbb{E}[w(t)]$ 保证对齐项的梯度幅值不因采样时间分布而衰减——本质是**按有效样本数归一化**，而非按总样本数归一化。

**与特征层的关系**：VA-REPA 的权重选择（低方差区激活 vs 高方差区抑制）的**依据**来自 $x_t$ 的语义信息量（即 $x_t$ 中保留了多少 $x_0$ 的信息），但权重本身并不改变 $x_t$ 或中间特征的值。它只在损失层面做有选择性的放大/抑制。

---

## 11. 讨论：训练时如何激活 REPA 表示对齐？

**Q: 训练时如何激活 REPA 表示对齐？是在 loss 上加一项吗？**

**A: 是的，本质就是在流匹配损失上加一个辅助对齐损失项。** 完整的训练前向传播如下：

### 一次训练迭代的完整流程

```
输入: x_0 (VAE latent), t (时间步), ε (噪声)

┌─ Step 1: 构造噪声输入 ──────────────────────────
│  x_t = α_t · x_0 + σ_t · ε
│
├─ Step 2: DiT 前向传播（同时服务两个目标）──────────
│                          ┌→ v_θ(x_t, t) → 用于流匹配损失
│  x_t, t → DiT backbone ─┤
│                          └→ h (第 l 层中间特征) → 用于 REPA
│
├─ Step 3: 流匹配损失（主损失）────────────────────
│  v_target = σ'_t/σ_t · (x_t - α_t·x_0) + α'_t · x_0   ← 闭式解,无需 label
│  L_vm = ||v_θ(x_t, t) - v_target||²
│                                                      
├─ Step 4: 表示对齐损失（辅助损失）← 这就是"激活 REPA"──
│  h_proj = MLP_proj(h)            ← 可训练投影头，映射到 DINOv2 维度
│  f_dino = DINOv2(x_0)             ← 冻结的预训练编码器，提供语义监督
│  ℓ_RA = cos_sim_loss(h_proj, f_dino)  或  ||h_proj - f_dino||²
│                                                         
├─ Step 5: 总损失（VA-REPA 版本）───────────────────
│  L = L_vm + λ_RA · [w(t) · ℓ_RA] / E[w(t)]
│       ↑              ↑
│    主损失      辅助对齐损失（低方差区才有效）
│
└─ Step 6: 反向传播 ────────────────────────────────
   更新: DiT backbone ✅  +  MLP投影头 ✅  (DINOv2 冻结 ❌)
```

### 与普通 REPA 的唯一区别

| 组件 | 普通 REPA | VA-REPA |
|------|----------|---------|
| 主损失 | $\mathcal{L}_{\text{CFM}}$ | $\mathcal{L}_{\text{StableVM}}$ |
| 对齐损失 | $\lambda \cdot \ell_{\text{RA}}(x_t)$ | $\lambda \cdot \frac{w(t) \cdot \ell_{\text{RA}}(x_t)}{\mathbb{E}[w(t)]}$ |
| 投影头 | ✅ 相同 | ✅ 相同 |
| DINOv2 编码器 | ✅ 相同（冻结） | ✅ 相同（冻结） |

唯一的改动就是加了一个 $w(t)$ 权重和归一化分母，其余完全一致。这也是为什么 VA-REPA 可以**无缝插入**到 REPA、REG、iREPA 等所有变体。

### 为什么这样做有效？

**直觉**：在低方差区（$t < 0.7$），$x_t$ 仍保留 $x_0$ 的语义结构（如"这是一只金毛犬"），DINOv2 提取的特征是有意义的。强制 DiT 中间层向这个语义表示对齐，相当于一个**辅助教材**——告诉模型"你在去噪时要始终记住目标是什么"。

到了高方差区（$t > 0.7$），$x_t$ 几乎是纯噪声，DINOv2 的特征和 $x_t$ 之间没有语义关联，强制对齐反而引入噪声——所以 $w(t) \to 0$ 关掉这个信号。

---

## 12. 讨论：为什么需要 MLP_proj？DINOv2 是什么？

**Q1: Step 4 中为什么需要 MLP_proj？**

**A:** 三个原因——**维度对齐、空间转换、梯度隔离**：

### 1. 维度不匹配

```
DiT 隐藏层维度 (如 SiT-XL: 1152)  ≠  DINOv2 输出维度 (如 dino-v2-g: 1536)
```

直接算不了损失，MLP 做一个 $1152 \to 1536$ 的线性映射把维度对齐。通常是 2 层 MLP + GELU 激活。

### 2. 表示空间不同

DiT 的特征形成于"去噪"任务，DINOv2 的特征来自"自监督视觉理解"任务，两者的内在结构不同。MLP 充当**可训练适配器**，学习从去噪表示空间翻译到语义表示空间。

### 3. 梯度隔离

对齐损失的梯度流向是单向的：
```
ℓ_RA → MLP_proj (训练 ✅) → DiT backbone (训练 ✅)
                            DINOv2 (冻结 ❌, 不受影响)
```
保证 DiT 向 DINOv2 的语义标准对齐，而不是反过来。

---

**Q2: DINOv2 是什么？**

**A:** DINOv2（Oquab et al., 2023, Meta AI）是一个基于 ViT 的**自监督预训练**视觉模型。

| 特性 | 说明 |
|------|------|
| 训练方式 | 自蒸馏（student-teacher），无需人工标注 |
| 训练数据 | LVD-142M（142M 张精选图片） |
| 架构 | ViT-g（1.1B 参数）或 ViT-l（300M） |
| 核心优势 | 特征在语义分割、深度估计、图像检索上表现极强，**线性可分** |
| 空间感知 | patch-level 特征保留精细空间位置信息 |

### 为什么选 DINOv2 而不是 CLIP？

- **空间结构好**：DINOv2 的 patch 级特征保留精确空间信息，与 DiT 处理 latent patch 的方式天然匹配
- **无需文本**：不依赖 prompt，适合类别条件生成场景
- **语义干净**：同类特征高度聚集，异类清晰分离 → 对齐信号确定、无歧义

### 在 REPA 中的角色

```
x_0 (干净图片) ──→ [冻结 DINOv2] ──→ f_dino  ← "语义参考答案，永不改变"
x_t (噪声图片) ──→ [DiT] → h → [MLP] → h_proj ← "去噪中间表示，不断学习"
                                         ↓
                                   让 h_proj ≈ f_dino
```

DINOv2 像一个**不说话的助教**——始终用固定的语义标准给出"参考答案"，帮助 DiT 的中间层更快地学会有用的视觉表示。这比纯靠最终像素级重建损失（逐像素 MSE）来驱动特征学习**高效得多**，因为语义级别监督信号更早、更稳定。

---

## 13. 讨论：用更好的 ViT 会更好吗？这算蒸馏吗？

**Q1: 用更好的 ViT 模型（如 SD3 的编码器）替代 DINOv2 会更好吗？**

**A: 不一定。** DINOv2 有独特的优势使其特别适合做 REPA 的教师，而非单纯"能力更强"就能替代：

| 维度 | DINOv2 | SD3 / Flux 编码器 |
|------|--------|-------------------|
| 训练目标 | 自监督视觉表示学习 | 服务于文生图流水线 |
| 空间结构 | patch级精细空间位置信息 | 被跨注意力与文本耦合，空间被稀释 |
| 特征性质 | **线性可分**（零样本分类/分割） | 为生成任务优化，非显式判别特征 |
| 文本依赖 | 无，纯视觉 | 深度依赖文本嵌入 |

**DINOv2 为何最优的四个原因**：

1. **空间保真度**：DINOv2 是纯 ViT，每个 patch 的特征对应原图的一个精确区域。DiT 处理 latent patch 的方式天然要求这种空间对齐。
2. **线性可分性**：同类物体特征高度聚集，异类清晰分离。对齐到这个"语义度量空间"能直接提升学生特征的判别力——这正是加速训练的关键。
3. **无文本依赖**：ImageNet 类别条件生成不需要文本 prompt，SD3 编码器的文本耦合特征反而不匹配。
4. **已验证**：REPA 原文消融了 CLIP、DINO、MAE 等多种编码器，DINOv2 效果最好。这是社区共识。

**结论**："更好的 ViT"的关键不是更大/更强的参数，而是特征空间的**语义结构**。DINOv2 的自监督训练目标（DINO + iBOT）恰好让它的特征空间天然适合作为视觉语义参考。

---

**Q2: 这算不算一种蒸馏（distillation）？**

**A: 绝对算！** REPA/VA-REPA 是典型的**特征级知识蒸馏（Feature-level Knowledge Distillation）**。

| 维度 | 经典 KD (Hinton 2015) | REPA / VA-REPA |
|------|----------------------|----------------|
| 教师 | 大模型（同任务） | DINOv2（冻结 ❌，不同任务） |
| 学生 | 小模型 | DiT / SiT |
| 蒸馏位置 | 输出 logits | **中间层 hidden states** |
| 蒸馏信号 | soft labels | **特征向量对齐** |
| 分类 | 响应蒸馏 / 输出蒸馏 | **特征蒸馏 / 表示蒸馏** |

更精确的分类：**跨架构、跨任务、冻结教师的特征级蒸馏**。

### 与传统蒸馏的本质区别

REPA 蒸馏传递的不是"答案"而是**"表示能力"**：

| | 经典 KD | REPA |
|---|---|---|
| 教师做什么 | 分类 | 提取视觉特征 |
| 学生做什么 | 分类 | 去噪生成（完全不同的任务！） |
| 传递什么 | 类别概率分布 | **如何构造好的中间表示** |
| 类比 | 数学老师教做数学题 | 语文老师教组织语言来写数学证明 |

---

## 14. 讨论：StableVS 有什么作用？

**Q: StableVS 有什么作用？**

**A: 免微调的推理加速。** 利用低方差区轨迹为直线的性质，用大步跳跃替代小步积分，实现约 2× 加速且质量不降。

### 核心原理

回顾双区制结构在推理阶段的应用：

| 区间 | 后验 $p(x_0\|x_t)$ | 轨迹形状 | StableVS 策略 |
|------|-------------------|----------|-------------|
| $t \in [0.85, 1]$ 高方差区 | 分散在多参考上 | 弯曲，需精细积分 | 保持标准 Euler（不减步） |
| $t \in [0, 0.85]$ **低方差区** | **坍缩为单峰** | **确定性直线** | **大步跳跃积分** |

在低方差区，条件速度 ≈ 真实边际速度，PF-ODE 退化为：

$$x_\tau = x_t + (\tau - t) \cdot v_t(x_t)$$

轨迹是直线 → 速度场几乎恒定 → **10 小步和 1 大步结果一样**。StableVS 正是利用这一点做大步跳步。

### 实际效果（SD3.5-Large, $\xi=0.85$）

```
30步 Euler (baseline):  Overall = 0.723
  高方差区 11步 + 低方差区 19步 ← 大量小碎步

20步 Euler (naive):     Overall = 0.710  PSNR = 16.93
  高方差区 7步  + 低方差区 13步 ← 精度丢失

20步 StableVS:          Overall = 0.723  PSNR = 36.92 ✓
  高方差区 11步 + 低方差区 9步  ← 只在直线区大步跳
```

### 关键特性

- **免微调**：不改变模型参数，只改积分步长策略，对已部署模型（SD3.5、Flux、Wan2.2）即插即用
- **不降质**：精度损失只发生在高方差区（轨迹弯曲），StableVS 恰好不在那里减步
- **求解器无关**：Euler、DPM-Solver++、UniPC 均适用，只需在低方差区替换为 StableVS
- **模态无关**：T2I（SD3.5、Flux、Qwen-Image）和 T2V（Wan2.2）均有效

### 与一致性模型/蒸馏的区别

| 方法 | 需要训练 | 原理 |
|------|---------|------|
| 一致性模型 | ✅ 蒸馏或一致性训练 | 学习一步映射 |
| 渐进蒸馏 | ✅ 多轮蒸馏 | 逐步减少步数 |
| **StableVS** | **❌ 免训练** | 利用直线轨迹做大步积分 |

StableVS 的独特之处在于：不是让模型学会少步生成，而是**发现**在低方差区步数本来就多余。

---

## 15. 讨论：StableVM、VA-REPA 与 StableVS 如何交互？

**Q: StableVS 怎么和 StableVM、VA-REPA 交互？**

**A:** 三者构成一个 **训练→推理的完整流水线**，共享同一个"双区制"理论基础，分工明确：

### 全局视角：统一的流水线

```
┌──────────── 训练阶段 ────────────┐     ┌── 推理阶段 ──┐
│                                  │     │              │
│  StableVM ──→ 主损失 (L_vm)     │     │  StableVS    │
│     +                    → 反向传播 →  训练好的模型  →  大步跳跃积分
│  VA-REPA ──→ 辅助损失 (L_rep)   │     │  ⚡ ~2× 加速  │
│                                  │     │              │
└──────────────────────────────────┘     └──────────────┘
```

### 三者在双区制中的分工

| | $t \in [0.85, 1]$ 高方差区 | $t \in [0, 0.85]$ 低方差区 |
|---|---|---|
| **StableVM** | 🔥 **核心作用区**：GMM 多参考加权降低训练方差 | 后验坍缩为单峰 → GMM ≈ 单样本 CFM（自动退化） |
| **VA-REPA** | $w(t) \to 0$：**关闭对齐**，避免噪声特征污染语义监督 | 💡 **核心作用区**：$w(t) \to 1$，DINOv2 语义对齐全量激活 |
| **StableVS** | 保持标准小步 Euler（轨迹弯曲，不能跳） | ⚡ **核心作用区**：轨迹为直线，大步跳跃，加速推理 |

**关键洞察**：三者的激活区域形成完美互补——

- **高方差区**：StableVM 全力降低训练方差，VA-REPA 主动关闭（此时 $x_t$ 无语义），StableVS 不加速（轨迹弯曲）
- **低方差区**：StableVM 自动退化（无需多参考），VA-REPA 全量激活（$x_t$ 有语义），StableVS 大幅加速（轨迹是直线）

### 训练时的交互：同一前向传播，两条损失路径

```
一次训练迭代：

x_t → [DiT backbone] ─┬→ 预测速度 v_θ(x_t, t) ──→ L_vm (StableVM，GMM 加权目标)
                      │
                      └→ 中间层特征 h → [MLP_proj] → h_proj
                                                      ↓
                                          f_dino = DINOv2(x_0) ← 冻结
                                                      ↓
                                          ℓ_RA = ||h_proj - f_dino||²
                                                      ↓
                                          L_rep = λ · w(t) · ℓ_RA / E[w(t)]  (VA-REPA)

总损失: L = L_vm + L_rep
```

- **StableVM** 和 **VA-REPA** 在同一前向传播中并行计算，梯度一起回传
- DiT backbone 同时接收两个信号的更新：去噪能力（来自 StableVM）+ 语义表示能力（来自 VA-REPA）
- 两者是**正交互补**的——StableVM 负责"预测更准"，VA-REPA 负责"表示更好"

### 从训练到推理的因果链

```
StableVM → 更稳定的速度场预测 → 低方差区轨迹更接近理想直线
                                        ↓
VA-REPA → 更好的中间表示     → 速度场预测质量进一步改善
                                        ↓
                                   StableVS 的大步跳跃更加可靠
                                   （直线偏离更小 → 加速后质量更高）
```

这不是简单的"训练完再加速"，而是：**训练阶段让低方差区的轨迹更接近完美直线，推理阶段利用这个直线性质做大步跳跃**。两者相互增强——训练越好，加速越可靠。

### 论文中的实验验证

论文在 ImageNet 256×256、SD3.5、Flux、Wan2.2 等模型上验证了完整的流水线：

1. **StableVM + VA-REPA 联合训练**：在 SiT-XL 上以仅 80 epoch 达到与 REPA-E 相当的 FID（2.26 vs 2.16），训练成本大幅降低
2. **StableVS 作用于训练好的模型**：在 SD3.5-Large 上从 30 步减到 20 步，GenEval Overall 从 baseline 的 0.723 仅略微波动到 0.714（~1.5× 加速）
3. **三者可拆分使用**：StableVM/VA-REPA 可无缝插入 REPA、REG、iREPA 等变体；StableVS 对任意 flow-based 预训练模型即插即用

### 一句话总结

**StableVM 在训练时"减小噪声" → VA-REPA 在训练时"注入语义" → StableVS 在推理时"利用确定性加速"**。三者基于同一个双区制理论，在时间轴上错位激活、彼此增强。

---

## 16. 讨论：StableVS 仅用于推理？SDE/ODE 与普通采样有何区别？

**Q1: StableVS 只在推理时使用？**

**A: 是的。** StableVS 是一种**纯推理时**的采样加速策略，不涉及任何训练或微调。它只改变数值积分的步长策略，对已部署的 flow-based 模型（SD3.5、Flux、Wan2.2 等）即插即用。这与 StableVM 和 VA-REPA（训练时技术）形成鲜明对比。

**Q2: SDE 和 ODE 的公式与普通采样有什么区别？**

**A:** 区别源于"低方差区轨迹为直线"这一核心洞察。先回顾基本定义，再对比 StableVS 如何利用它做大步跳跃。

### 2.1 基本定义：PF-ODE vs Reverse SDE

在流匹配框架中，存在两种等价的逆向动力学：

**PF-ODE（Probability Flow ODE）— 确定性轨迹**：
$$\dd \vx_t = \vv_t(\vx_t)\,\dd t$$

**Reverse SDE — 随机轨迹**：
$$\dd \vx_t = \vv_t(\vx_t)\,\dd t - \tfrac{1}{2}w_t \vs_t(\vx_t)\,\dd t + \sqrt{w_t}\,\dd \overline{\mathbf{W}}_t$$

其中 $\sqrt{w_t}$ 是扩散系数，$\overline{\mathbf{W}}_t$ 是逆向维纳过程。两者的**边际分布 $p_t(\vx)$ 在任意时刻 $t$ 完全相同**，但路径性质不同：
- ODE：确定性的，同一输入 → 同一输出
- SDE：随机的，每次运行得到不同结果（可增加多样性或修正误差）

### 2.2 StableVS 对两者的闭式解

在**低方差区**（$t \in [0, \xi]$），后验 $p(\vx_0|\vx_t)$ 坍缩为单峰 → $\vv_t(\vx_t) \approx \vv_t(\vx_t|\vx_0)$ → 速度场近似恒定。代入 ODE/SDE 可求得精确解：

| | StableVS-SDE（DDIM-style 后验） | StableVS-ODE（PF-ODE 精确解） |
|---|---|---|
| **公式** | $p_\tau(\vx_\tau \mid \vx_t, \vv_t) = \mathcal{N}(\boldsymbol{\mu}_{\tau \mid t},\ \beta_t^2 \mathbf{I})$ | $\vx_{\tau} = \sigma_{\tau} \big[ \big( \frac{1}{\sigma_{t}} - \frac{\sigma_{t}'}{\sigma_{t}} \Psi_{t, \tau} \big) \vx_{t} + \Psi_{t, \tau} \vv_{t} \big]$ |
| **关键参数** | $\beta_t = f_\beta \sigma_\tau$，$f_\beta \in [0,1]$ 控制随机性 | $\Psi_{t, \tau} = \frac{1}{C_t}\int_{t}^{\tau} \frac{C(s)}{\sigma_s} \dd s$，$C(s)=\alpha_s' - \alpha_s \sigma_s'/\sigma_s$ |
| **线性插值特例** | $\beta_t=0$ 时：$\vx_{\tau} = \vx_{t} + (\tau - t) \vv_{t}(\vx_{t})$ | $\vx_{\tau} = \vx_{t} + (\tau - t) \vv_{t}(\vx_{t})$ |
| **性质** | $f_\beta=0$ 退化为确定性与 ODE 等价；$f_\beta>0$ 添加可控噪声 | 完全确定性，在低方差区是精确解而非近似 |

**线性插值（$\alpha_t=1-t, \sigma_t=t$）下两者完全统一**：就是简单的 $\vx_{\tau} = \vx_{t} + (\tau - t) \vv_{t}(\vx_{t})$。

### 2.3 与普通采样的本质区别

```
普通 Euler 采样（高方差区 + 低方差区统一用小步）：
  t=1 ─Δt─→ t=0.95 ─Δt─→ ... ─Δt─→ t=0.85 ─Δt─→ ... ─Δt─→ t=0
  每一步: x_{t-Δt} = x_t - Δt · v_t(x_t)           Δt 很小
  原因: v_t(x_t) 沿轨迹不断变化，大步会累积误差

StableVS 采样（高方差区用小步，低方差区用大步）：
  t=1 ─Δt─→ ... ─Δt─→ t=0.85 ────大步────→ t=0    ← 关键！
  高方差区: 标准小步         低方差区: 一大步跳过去
  原因: v_t(x_t) ≈ CONSTANT, 10小步 = 1大步
```

**核心差异**不只在步长大小，而在**为什么可以这么做**：

| | 普通 Euler/高阶求解器 | StableVS |
|---|---|---|
| **依赖假设** | $v_t$ 连续可微（泰勒展开） | $v_t$ **恒定**（后验坍缩） |
| **步长限制** | 受截断误差约束，步长有限 | **任意**步长（理论上） |
| **误差来源** | 局部截断误差 $O(\Delta t^p)$ | 模型预测误差 $\|v_\theta - v_{\text{true}}\|$ |
| **高方差区** | 可用 | 不可用（轨迹弯曲，$v_t$ 不恒定） |
| **低方差区** | 可以但低效（小步浪费计算） | **最优**（1 大步 = N 小步） |

### 2.4 直观类比

想象你在高速公路上开车，GPS 告诉你当前位置和目的地方向：

- **普通采样**：每隔 100 米重新算一次方向 → 即使是笔直的高速也要频繁调整 → 慢
- **StableVS**：在弯曲的山路（高方差区）保持频繁调整 → 上了"确定性直线高速"（低方差区）后看一眼方向直接踩油门冲到终点

### 2.5 与高阶 ODE 求解器的兼容性

StableVS 并非要替代 Euler/DPM-Solver++/UniPC 等求解器，而是与它们**互补**：
- 在高方差区，继续使用原有求解器（DPM-Solver++、UniPC 等）
- 在低方差区，切换到 StableVS 的大步跳跃
- 论文实验表明这种混合策略在 SD3.5、Flux 等模型上均稳定有效

---

## 17. 讨论：StableVS 与 StableVM 是耦合的吗？

**Q: 必须使用 StableVM 训练的模型才能用 StableVS 做推理吗？**

**A: 不耦合，不必须。** StableVS 和 StableVM 是**独立可拆分的**，任何 flow-based 模型都可以用 StableVS 加速，无论它是用什么方法训练的。

### 为什么可以独立？

StableVS 依赖的核心前提是**低方差区轨迹为直线**——这是流匹配本身的数学性质，不是 StableVM 训练出来的：

```
低方差区 v_t(x_t) ≈ constant（直线轨迹）
    ↑
来自: 后验 p(x_0|x_t) 坍缩为单峰
    ↑
来自: 这是流匹配的固有性质，任何 flow-based 模型都满足
```

论文的 StableVS 实验直接跑在**未经过 StableVM 训练的预训练模型**上：
- SD3.5（用原始 CFM 训练的）
- Flux（用原始 CFM 训练的）
- Qwen-Image
- Wan2.2

这些模型都没有经过 StableVM 训练，StableVS 照样有效。

### 那 StableVM 训练对 StableVS 有什么帮助？

是**增强关系**，不是**依赖关系**：

| | 不用 StableVM 训练 | 用 StableVM 训练 |
|---|---|---|
| StableVS 能用吗？ | ✅ 能用 | ✅ 能用 |
| 低方差区 $v_\theta$ 预测精度 | 正常 | **更准**（训练方差更低） |
| 大步跳跃的可靠性 | 正常 | **更可靠**（偏离理想直线更小） |
| 加速效果 | ~2× | ~2×（同等质量），或相同步数下质量更高 |

因果关系：**StableVM 让模型的 $v_\theta(x_t, t)$ 更接近真实 $v_t(x_t)$ → 低方差区真实轨迹更接近理论直线 → StableVS 的大步跳跃误差更小。** 但即使没有 StableVM，低方差区的直线性质本身就存在，只是模型预测精度稍低一点。

### 类比

想象一列高铁：
- **铁轨是直的**（这是流匹配的数学性质决定的） → **任何列车**在这段直轨上都可以加速
- **StableVM** 把列车的轮子打磨得更圆 → 加速时更平稳、振动更小
- 但即使轮子不够圆（没用 StableVM），直线铁轨上加速依然比弯曲铁轨安全得多

### 三者的独立性与可组合性

```
        训练阶段                      推理阶段
    ┌──────────────┐            ┌──────────────┐
    │  StableVM    │            │              │
    │  (可选 ✅)    │──训练出──→│   模型权重    │
    │              │            │              │
    │  VA-REPA     │            │  StableVS    │
    │  (可选 ✅)    │            │  (可选 ✅)    │
    └──────────────┘            └──────────────┘
    
    任意组合均有效：
    ✓ CFM 训练 + Euler 推理（baseline）
    ✓ CFM 训练 + StableVS 推理（只加速，不重训）
    ✓ StableVM 训练 + Euler 推理（只提升质量）
    ✓ StableVM+VA-REPA 训练 + StableVS 推理（最强组合）
```

---

## 18. 讨论：StableVS 如何实现采样加速？

**Q: StableVS 如何实现采样加速？**

**A:** 分三个层次理解：**为什么能加速 → 数学怎么推导 → 工程怎么落地。**

### 层次一：为什么能加速？— 核心前提

在低方差区（$t \in [0, \xi]$），公式（21）表明 $\mathcal{V}_{\text{CFM}}(t) \approx 0$，这意味着：

$$p(\vx_0 \mid \vx_t) \approx \delta(\vx_0 - \vx_0^*) \quad \Rightarrow \quad \vv_t(\vx_t) \approx \vv_t(\vx_t \mid \vx_0^*)$$

后验坍缩为单峰 → 速度场近似恒定（由唯一的 $\vx_0^*$ 决定）→ **PF-ODE 轨迹退化为直线**，即 $\vv_t(\vx_t)$ 沿轨迹几乎不变。

**类比**：普通 Euler 采样像是蒙着眼在山路上走，每一步都要重新摸一摸方向 → 必须小步走。StableVS 发现低方差区是"GPS 信号完美区"——看一眼就知道方向，且方向全程不变 → 直接大步冲过去。关键不是"步子更大"而是**"方向不变，不需要反复确认"**。

### 层次二：数学怎么推导？— 闭式解

**2.1 Reverse SDE 的 DDIM-style 后验**

Reverse SDE（附录公式 15-16）：
$$\dd \vx_t = \vv_t(\vx_t)\,\dd t - \tfrac{1}{2}w_t \vs_t(\vx_t)\,\dd t + \sqrt{w_t}\,\dd \overline{\mathbf{W}}_t$$

在低方差区，利用 $p(\vx_0|\vx_t) \approx \delta(\vx_0 - \vx_0^*)$，后验坍缩为：

$$p_\tau(\vx_\tau \mid \vx_t) \approx p_\tau(\vx_\tau \mid \vx_0^*, \vx_t)$$

利用 $\vx_t = \alpha_t \vx_0^* + \sigma_t \varepsilon$ 和 $\vx_\tau = \alpha_\tau \vx_0^* + \sigma_\tau \varepsilon'$ 的联合高斯性质，推导出 DDIM-style 后验：

$$p_\tau(\vx_\tau \mid \vx_t, \vv_t) = \mathcal{N}\!\left(\boldsymbol{\mu}_{\tau \mid t},\ \beta_t^2 \mathbf{I}\right)$$

其中：
- $\beta_t = f_\beta \sigma_\tau$，$f_\beta \in [0,1]$ 控制随机性
- $\rho_t = \sqrt{(\sigma_\tau^2 - \beta_t^2) / \sigma_t^2}$
- $\lambda_t = (\alpha_\tau - \alpha_t \rho_t) / (\alpha_t' - \alpha_t \sigma_t'/\sigma_t)$
- $\boldsymbol{\mu}_{\tau \mid t} = (\rho_t - \lambda_t \sigma_t'/\sigma_t)\,\vx_t + \lambda_t\,\vv_t(\vx_t)$

**$f_\beta$ 的作用**：$f_\beta = 0$ → 完全确定性（等价于 ODE），$f_\beta = 1$ → 全噪声。实际使用 $f_\beta=0$。

**2.2 PF-ODE 的精确解**（Appendix C.2）

PF-ODE（公式 9）：$\dd\vx_t = \vv_t(\vx_t)\,\dd t$

在低方差区代入 $\vv_t(\vx_t) \approx \frac{\sigma_t'}{\sigma_t}(\vx_t - \alpha_t\vx_0) + \alpha_t'\vx_0$，重组为：

$$\frac{\dd\vx_t}{\dd t} + a(t)\vx_t = b(t), \quad a(t) = -\frac{\sigma_t'}{\sigma_t}, \quad b(t) = C_t \vx_0$$

这是一阶线性非齐次 ODE。用积分因子 $\mu(t) = 1/\sigma_t$ 求精确解：

$$\frac{\vx_{\tau}}{\sigma_{\tau}} = \frac{\vx_{t}}{\sigma_{t}} + \int_{t}^{\tau} \frac{C(s)}{\sigma_s} \dd s \cdot \vx_0$$

再代入 $\vx_0$ 的闭式表达式 $\vx_0 = \frac{\vv_t - (\sigma_t'/\sigma_t)\vx_t}{C_t}$，得最终公式：

$$\vx_{\tau} = \sigma_{\tau}\!\left[ \left(\frac{1}{\sigma_{t}} - \frac{\sigma_{t}'}{\sigma_{t}} \Psi_{t,\tau}\right)\vx_{t} + \Psi_{t,\tau}\,\vv_{t}(\vx_{t}) \right]$$

其中 $\Psi_{t,\tau} = \frac{1}{C_t} \int_{t}^{\tau} \frac{C(s)}{\sigma_s} \dd s$ 是预计算积分因子。

**2.3 线性插值特例：极简形式**

当 $\alpha_t = 1-t, \sigma_t = t$，且设 $\beta_t = 0$（$f_\beta = 0$），SDE 和 ODE 的公式**统一为**：

$$\vx_{\tau} = \vx_{t} + (\tau - t)\,\vv_{t}(\vx_{t})$$

形式上就是一个普通的 Euler 步，但关键区别在于：普通 Euler 要求步长极小以保证精度，而这里 **$(\tau - t)$ 可以任意大**——因为速度场在低方差区恒定。

### 层次三：工程怎么落地？— 两阶段混合策略

```
总步数 30 的 StableVS 推理流程：

高方差区 [1 → ξ]（如 [1 → 0.85]）:
  ├─ 使用标准求解器（Euler / DPM-Solver++ / UniPC）
  ├─ 11 小步，步长 Δt = (1-0.85)/11 = 0.0136
  └─ 原因：轨迹弯曲，v_t 不断变化，必须精细积分

低方差区 [ξ → 0]（如 [0.85 → 0]）:
  ├─ 切换到 StableVS
  ├─ 9 大步，步长 (0.85-0)/9 = 0.0944（是普通 Euler 的 ~7 倍！）
  └─ 每步公式：x_τ = x_t + (τ-t) v_t(x_t)
```

**关键参数**：
- $\xi$（分割点）: 通常取 0.85（论文基于方差曲线确定，对所有模型一致有效）
- 总步数: 30 → 20（低方差区 19 步 → 9 步），实现 ~1.5× 总加速
- 低方差区加速倍率: 19/9 ≈ 2.1×（真正的加速来源）

### 数值结果（SD3.5-Large, GenEval）

| 配置 | 总步数 | Overall | PSNR | SSIM | LPIPS |
|------|--------|---------|------|------|-------|
| Euler baseline | 30 | 0.723 | — | — | — |
| Euler naive | 20 | 0.710 ↓ | 16.93 | 0.753 | 0.333 |
| **StableVS** | **20** | **0.723** ✓ | **36.92** | **0.980** | **0.021** |

**解读**：单纯减少步数（30→20）质量显著下降（Overall 0.723→0.710，PSNR 只有 16.93）。StableVS 在相同 20 步下完全恢复质量（Overall 回到 0.723，PSNR 飙升至 36.92），因为它只在"本可以大步跳"的低方差区减步数，高方差区保持精细积分。

### 与求解器的兼容性

StableVS 不是要**替代** Euler/DPM-Solver++/UniPC，而是**组合使用**：

| 基础求解器 | 30 步 baseline | 20 步 naive | 20 步 + StableVS |
|-----------|---------------|------------|------------------|
| Euler | 0.723 | 0.710 | **0.723** |
| DPM-Solver++ | 0.724 | 0.717 | **0.719** |

两种求解器下 StableVS 均能恢复大部分质量，且 DPM-Solver++ 搭配 StableVS 效果最佳。论文在 Wan2.2（T2V）上也验证了 UniPC + StableVS 的组合（PSNR 从 15.61 → 31.10）。

这也是论文强调的"即插即用"——三者设计为**正交模块**，每个都解决独立的问题，组合使用有叠加收益但不强制依赖。

---

## 19. 讨论：StableVS 与一般 SDE/ODE 采样公式的区别

**Q: StableVS 和一般的 SDE/ODE 采样，公式上到底有什么区别？**

**A:** 分两种情况——**线性插值**（$\alpha_t=1-t, \sigma_t=t$，大多数 flow-based 模型使用）和**一般插值**。

### 线性插值：公式长得一样，但数学地位完全不同

这是最反直觉的一点：**StableVS 的递推公式和普通 Euler 步在字母上一模一样。**

|| 公式 | 成立条件 | 数学地位 |
|---|---|---|---|
| **标准 Euler（ODE）** | $\vx_{t_{k+1}} = \vx_{t_k} + \underbrace{\Delta t}_{\text{必须} \ll 1} \cdot \vv_{t_k}(\vx_{t_k})$ | 若 $\Delta t \to 0$ | **一阶近似**（泰勒展开截断） |
| **标准 Euler-Maruyama（SDE）** | $\vx_{t_{k+1}} = \vx_{t_k} + \Delta t \cdot \vv_{t_k} - \frac{1}{2} w_{t_k} \Delta t \cdot \vs_{t_k} + \sqrt{w_{t_k} \Delta t} \cdot \boldsymbol{\varepsilon}$ | 若 $\Delta t \to 0$ | **一阶近似** + 布朗运动离散化 |
| **StableVS** | $\vx_{\tau} = \vx_{t} + \underbrace{(\tau - t)}_{\text{可任意大！}} \cdot \vv_{t}(\vx_{t})$ | 若 $t \in [0,\xi]$（低方差区） | **精确解**（ODE 在 $v_t$ 恒定时的解析解） |

**公式长得一样，但数学含义完全不同**：

```
标准 Euler:
  dx = v dt  ─一阶泰勒离散化─→  x_{t+Δt} = x_t + Δt·v_t  （近似，Δt→0）
  
StableVS:
  dx = v dt, v = CONSTANT  ─直接积分─→  x_τ = x_t + (τ-t)·v_t  （精确，τ-t 可任意）
```

**误差来源也不同**：

|| 标准 Euler | StableVS |
|---|---|---|
| **主要误差** | 局部截断误差 $O(\Delta t^2)$，累积 $O(\Delta t)$ | 模型误差 $\|\vv_\theta - \vv_{\text{true}}\|$（$v_t$ 不完全恒定的程度） |
| **步长如何影响误差** | 步长越大 → 截断误差越大 | 步长增大 → 误差不变（$v_t$ 是恒定的） |
| **失效条件** | 步长超过稳定域 → 数值发散 | 进入高方差区（$v_t$ 不恒定） → 公式不再精确 |

### 一般插值（$\alpha_t, \sigma_t$ 非平凡）：公式完全不一样

对于非线性的 schedule，StableVS 的公式与标准 Euler/高阶求解器**完全不同**：

| 方法 | 公式 | 依赖信息 |
|---|---|---|
| **标准 Euler** | $\vx_{t_{k+1}} = \vx_{t_k} + \Delta t \cdot \vv_{t_k}(\vx_{t_k})$ | 仅需 $v_{t_k}$ |
| **DPM-Solver++** | 复杂的线性多步公式，依赖多个历史 $v$ | $v_{t_k}, v_{t_{k-1}}, \ldots$ |
| **StableVS-ODE** | $\vx_{\tau} = \sigma_{\tau}\!\left[ \left(\frac{1}{\sigma_{t}} - \frac{\sigma_{t}'}{\sigma_{t}} \Psi_{t,\tau}\right)\vx_{t} + \Psi_{t,\tau}\,\vv_{t} \right]$ | $v_t$ + 预计算的 $\Psi_{t,\tau}$ |
| **StableVS-SDE** | $\vx_{\tau} \sim \mathcal{N}\!\left(\boldsymbol{\mu}_{\tau \mid t},\ \beta_t^2 \mathbf{I}\right)$，$\boldsymbol{\mu}$ 含复杂系数 | $v_t$ + $\rho_t, \lambda_t$（皆可预计算） |

**根本差异**：

1. **Euler 是对 ODE 的局部线性近似**：假设 $v_t$ 在 $\Delta t$ 内不变 → 用当前点的导数做线性外推 → 仅当 $\Delta t \to 0$ 时精确

2. **DPM-Solver++/UniPC 是对 ODE 的高阶多项式近似**：利用多个历史点的 $v_t$ 做高阶插值 → 步长可以更大但仍受收敛半径限制

3. **StableVS 是对 ODE 在 $v_t$ 恒定假设下的精确解**：不依赖泰勒展开 → 直接解析求解一阶线性 ODE → 步长**理论上无限**，唯一限制是 $v_t$ 必须近似恒定

### 核心区别总结

| 对比维度 | 一般 SDE/ODE 采样 | StableVS |
|---|---|---|
| **推导方式** | 数值分析（泰勒展开 / Runge-Kutta / 线性多步法） | 解析求解（$v_t$ 恒定假设下的闭式解） |
| **公式复杂度** | 依赖求解器阶数（Euler=线性, Heun=梯形, DPM=多项式） | 依赖插值 schedule（线性插值极简，一般插值含积分因子） |
| **$x_t$ 系数项** | Euler: $x_{k+1} = x_k + \Delta t \cdot v_k$（无 $x_k$ 缩放） | ODE 解: $\sigma_\tau/\sigma_t$ 缩放 + 额外 $x_t$ 修正项 |
| **$v_t$ 系数项** | Euler: 固定为 $\Delta t$ | 含预计算的积分因子 $\Psi_{t,\tau}$ |
| **噪声项** | SDE: $\sqrt{w_t \Delta t} \cdot \varepsilon$，方差正比于 $\Delta t$ | SDE: $\beta_t = f_\beta \sigma_\tau$，方差**不依赖** $\Delta t$ |
| **步长自由度** | 约束严格（稳定性+精度） | 低方差区内无约束（$v_t$ 恒定） |
| **对 schedule 的依赖** | 仅通过 $v_t$ 的评估点依赖 | 必须预计算 $\Psi_{t,\tau}$（与 $\alpha_t, \sigma_t$ 强耦合） |
| **能否独立使用** | ✅ 独立采样 | ❌ 必须与标准求解器**混合**（高方差区用小步求解器，低方差区切 StableVS） |

### 一句话

**StableVS 不是改写了采样公式，而是发现了一个"公式自动简化"的区域。** 在这个区域里，标准 SDE/ODE 退化为一阶线性常系数 ODE，有精确的闭式解。StableVS 做的就是用这个闭式解做大步跳跃，而不是用数值近似做小步推进。

---

## 20. 讨论：$\tau$ 和 $t$ 的关系——StableVS 为什么能"一步跨过去"？

**Q: $\tau$ 和 $t$ 是两个不同的步数吗？标准 Euler 从 $t$ 到 $\tau$ 需要多次采样，StableVS 用某个 $v_t$ 直接走 $\tau-t$ 次？**

**A: 理解基本正确，但需要澄清概念。**

### $\tau$ 和 $t$ 是时间步（timesteps），不是"步数"

- $t$：当前所在的时间步，例如 $t=0.85$
- $\tau$：目标时间步，例如 $\tau=0$（终点）
- $\tau - t$：时间差（不是步数差！），例如 $0 - 0.85 = -0.85$

### 标准 Euler 如何从 $t$ 走到 $\tau$？

需要**多个 Euler 步**：

```
t = 0.85 → 0.84 → 0.83 → ... → 0.01 → τ = 0
   ↑         ↑                      ↑
 评估 v    评估 v    ... (84次)    评估 v
```

每一步只走 $\Delta t = 0.01$，**每走一步都要重新跑一次 DiT 前向传播**来评估 $v_{t_k}(x_{t_k})$。如果 $\Delta t = 0.01$，从 0.85 到 0 需要 **85 次模型前向传播**。

为什么必须这样？因为轨迹是弯曲的——$x$ 每走一步就到了新的位置，$v_t(x)$ 也变了，必须重新评估。

### StableVS 如何从 $t$ 走到 $\tau$？

**一次模型前向传播就够了：**

```
t = 0.85 ──────────── 一次跳跃 ────────────→ τ = 0
        评估 v_{0.85}(x_{0.85})，仅此一次！
        x_0 = x_{0.85} + (0 - 0.85) · v_{0.85}(x_{0.85})
```

**只跑一次 DiT**，计算出 $v_{0.85}(x_{0.85})$，然后直接一步跳到 $\tau=0$。

为什么可以？因为在低方差区，$v_t(x_t)$ 沿轨迹**几乎不变**，所以从 $t$ 到 $\tau$ 之间的所有中间点，$v$ 都是同一个值——评估一次就知道全程。

### 直观对比

```
标准 Euler（低方差区，Δt=0.01，从 0.85 到 0）：
  [DiT 前向] [DiT 前向] [DiT 前向] ... (85次) ... [DiT 前向]
      ↓           ↓           ↓                        ↓
  v_{0.85}    v_{0.84}    v_{0.83}                 v_{0.01}
      ↓           ↓           ↓                        ↓
  x_{0.84} = x_{0.85} - 0.01·v_{0.85}
              x_{0.83} = x_{0.84} - 0.01·v_{0.84}
                          ...                    x_0 = x_{0.01} - 0.01·v_{0.01}

StableVS（低方差区，从 0.85 直接跳到 0）：
  [DiT 前向] → v_{0.85} → x_0 = x_{0.85} - 0.85·v_{0.85}
       ↑                        ↑
   只跑一次！              一步到位！
```

### 关键洞察

$\tau - t$ 不是"走了多少步"，而是**在一次跳跃中跨越的时间跨度**。StableVS 的核心是：

$$(\tau - t) \;\text{可以任意大，因为}\; v_t(x_t) \approx \text{const}$$

而标准 Euler 中 $\Delta t$ 会称为 "步长"，受到截断误差的严格约束。

### 两阶段混合的实际流程

重新用具体数字看一下 StableVS 的完整推理（总步数=20）：

```
阶段一：高方差区 [1 → 0.85]
  11 次标准 Euler 步，每步 Δt ≈ 0.0136
  每步都需要跑 DiT → 共 11 次前向传播
  原因：轨迹弯曲，v_t 不断变化

阶段二：低方差区 [0.85 → 0]  ← StableVS 区域
  9 次 StableVS 大步，每步跳跃 ≈ 0.0944
  每步只需要跑一次 DiT → 共 9 次前向传播
  原因：v_t 恒定，大步 = 精确积分
  
总计：20 次 DiT 前向传播（vs. 基线 30 次）
低方差区内加速比：19/9 ≈ 2.1×
```

**StableVS 不是"一步走完 19 小步"，而是把 19 小步合并为 9 大步，每大步等价于约 2 小步的 Euler，但精度不损失。** 因为大步和小步的公式在 $v_t$ 恒定假设下是**数学等价的**：

$$\underbrace{x_{\tau} = x_t + \sum_{k=0}^{N-1} \Delta t \cdot v_{t_k}}_{\text{N 小步 Euler}} = \underbrace{x_t + (\tau - t) \cdot v_t}_{\text{1 大步 StableVS}} \quad \text{因为 } v_{t_0} = v_{t_1} = \cdots = v_{t_{N-1}}$$

---

## 21. 讨论：具体从哪一步开始可以用 StableVS？

**Q: 具体哪一步开始可以直接使用大步跳跃的解码？**

**A:** 分割点 $\xi = 0.85$（推理时），即 $t \in [0, 0.85]$ 为 StableVS 大步跳跃区。但这不是一个绝对的"开关"，而是基于方差曲线的工程选择。

### 不是"某一步突然可以"，而是一个渐变

关键理解：$\mathcal{V}_{\text{CFM}}(t)$ 是从 $t=0$ 到 $t=1$ **连续上升**的，不存在一个"魔法时间点"让方差瞬间归零。

```
t:  0 ──────── 方差逐渐增大 ────────→ 1
    ← 低方差区 →|← 高方差区 →
    
    ξ 是人类划的分界线，不是物理定律
```

方差在 $t$ 很小时几乎为 0，随着 $t$ 增大缓慢上升，接近 $t=1$ 时急剧膨胀。**$\xi$ 的选择是一个 trade-off：**

- $\xi$ 越小 → 低方差区保守 → 加速少但更安全
- $\xi$ 越大 → 低方差区激进 → 加速多但有精度损失风险

### 论文的实际取值

论文在实验中使用两个不同的 $\xi$：

| 用途 | $\xi$ | 原因 |
|---|---|---|
| **训练时的 StableVM / VA-REPA** | 0.7 | 保守值，确保低方差区内的 GMM 退化和 REPA 对齐是精确的 |
| **推理时的 StableVS** | **0.85** | 平衡值，在 ImageNet latent（$32\times32\times4 = 4096$ 维）空间上低速方差区足够大 |

### 为什么推理用 0.85 而训练用 0.7？

两者目的不同：

- **训练**：VA-REPA 在低方差区用 DINOv2 做语义对齐 → 必须严格保证 $x_t$ 有清晰的语义；$\xi=0.7$ 更保守，确保 DINOv2 特征有意义
- **推理**：StableVS 只要求 $v_t$ 近似恒定 → 条件更宽松；$\xi=0.85$ 可以覆盖更大的低方差区，获得更多加速

### 消融实验：不同 $\xi$ 的影响

论文在 **SD3.5-Large ($1024\times1024$)** 上做了 $\xi$ 的消融（附录 Table）：

| $\xi$ | 总步数 | 低方差区步数 | GenEval Overall | PSNR | 观察 |
|---|---|---|---|---|---|
| 0.70 | 26 | 9 | **0.726** | **43.65** | 最安全，但总步数多（加速少） |
| 0.80 | 22 | 9 | **0.724** | 39.15 | 加速提升，质量微降 |
| **0.85** | **20** | **9** | **0.723** | 36.92 | ✅ 默认配置，加速与质量的最佳平衡 |
| 0.90 | 17 | 9 | **0.728** | 31.99 ⚠️ | 质量指标出现塌方！PSNR 暴跌 |

**关键教训**：$\xi=0.90$ 的 Overall 分数虽然更高，但 PSNR 从 36.92 暴跌到 31.99，说明 $x_t$ 的逐像素重建质量已经受损——$\xi$ 太大意味着把部分"还不够直"的轨迹也当直线处理了。

### 维度依赖性：更大的模型 → 更大的 $\xi$

论文强调 $\xi$ **不是固定常数**，而是随数据维度 $d$ 增大而向 1 靠近：

$$d \uparrow \;\Rightarrow\; \xi \uparrow$$

直观理解：维度越高，后验分布越难"混合"——$p(x_0|x_t)$ 不容易同时覆盖多个不同样本，因此单峰假设在更大的 $t$ 下仍然成立。

实际指导：
- ImageNet 256×256（latent $32\times32\times4$）：$\xi=0.85$
- 更大模型（SD3.5 1024×1024）：$\xi=0.85$ 仍然合适
- 视频模型（Wan2.2 T2V）：更高维 latent → $\xi$ 可能更大，但 0.85 是安全的默认值

### 一句话总结

**$t \in [0, 0.85]$ 是 StableVS 的大步跳跃区**（推理时）。这不是一个硬件开关，而是基于方差曲线分析的工程阈值——权衡了保真度（$\xi$ 不能太大）和加速比（$\xi$ 不能太小）。对于绝大多数 flow-based 模型，$\xi=0.85$ 是可即插即用的默认选择。
