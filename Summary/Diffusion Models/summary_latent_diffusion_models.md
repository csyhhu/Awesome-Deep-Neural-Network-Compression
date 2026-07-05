# High-Resolution Image Synthesis with Latent Diffusion Models

> **作者**: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer  
> **机构**: Ludwig Maximilian University of Munich & IWR, Heidelberg University / Runway ML  
> **发表**: CVPR 2022  
> **链接**: [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)  
> **标签**: Diffusion Models, Image Generation, Text-to-Image, Latent Space, Perceptual Compression

---

## 1. 核心动机

扩散模型 (Diffusion Models, DMs) 在图像合成上达到了 SOTA，但存在两大瓶颈：

1. **训练成本极高**：强大的 DM 需要数百 GPU 天（如 150-1000 V100 天）
2. **推理速度慢**：生成 50k 样本约需 5 天（单 A100），因为需要在像素空间进行多步序列化评估

核心观察：基于似然的模型学习过程可分解为两个阶段：
- **感知压缩 (Perceptual Compression)**：去除高频不可感知细节
- **语义压缩 (Semantic Compression)**：学习数据的语义和概念组成

**核心思想**：将扩散模型从像素空间迁移到预训练自编码器的低维潜在空间中进行训练和推理。

---

## 2. 方法

### 2.1 感知压缩（第一阶段）

训练一个自编码器，将图像从像素空间映射到低维潜在空间：

- **编码器** $\mathcal{E}$：将 $x \in \mathbb{R}^{H \times W \times 3}$ 映射为 $z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$，下采样因子 $f = H/h = 2^m$
- **解码器** $\mathcal{D}$：从潜在空间重建图像 $\tilde{x} = \mathcal{D}(z)$
- **损失函数**：感知损失 + 基于 Patch 的对抗损失，保证重建保真度和局部真实感
- **正则化**：
  - *KL-reg.*：施加轻微 KL 散度惩罚（权重约 $10^{-6}$），使潜在变量接近标准正态分布
  - *VQ-reg.*：在解码器中使用向量量化层（类似 VQGAN）

### 2.2 潜在扩散模型（第二阶段）

在学到的潜在空间中训练扩散模型：

$$\mathcal{L}_{LDM} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t}\left[ \|\epsilon - \epsilon_\theta(z_t, t)\|_2^2 \right]$$

- 神经网络骨干 $\epsilon_\theta$ 为时间条件的 UNet
- 主要基于 2D 卷积层构建，充分利用图像的空间归纳偏置
- 相比之前基于自回归 Transformer 的方法，LDMs 对更高维度的潜在空间缩放更友好

### 2.3 条件机制：Cross-Attention

通过将 **交叉注意力 (Cross-Attention)** 引入 UNet 架构，实现通用的条件控制：

$$Q = W_Q^{(i)} \cdot \varphi_i(z_t), \quad K = W_K^{(i)} \cdot \tau_\theta(y), \quad V = W_V^{(i)} \cdot \tau_\theta(y)$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

- $\tau_\theta$ 为领域专用编码器（如文本用 Transformer、语义布局用空间编码）
- $\varphi_i(z_t)$ 为 UNet 中间层表示
- 支持多模态训练：文本、语义布局、类别标签等

---

## 3. 实验与结果

### 3.1 压缩率分析

| 下采样因子 $f$ | 结论 |
|---|---|
| $f=1$ (像素级) | 训练慢，计算量大 |
| $f=2$ | 训练仍较慢 |
| **$f=4\sim16$** | **最优平衡点**：效率高且质量好 |
| $f=32$ | 信息损失过大，质量瓶颈 |

**LDM-8** 在 2M 训练步后 FID 比像素级 DM 低 38，同时采样速度显著提升。

### 3.2 无条件生成

| 数据集 | FID | 对比 |
|---|---|---|
| CelebA-HQ | **5.11** (SOTA) | 超越 GAN 和 LSGM |
| FFHQ | 4.31 | 接近 SOTA |
| LSUN-Churches | 4.02 | 超越之前方法 |
| LSUN-Bedrooms | 2.95 | 接近 ADM |

### 3.3 文本到图像生成

- 在 LAION-400M 上用 **1.45B** 参数 KL-regularized LDM 训练
- BERT tokenizer + Transformer 编码文本
- 在 MS-COCO 上评估，结合 **classifier-free guidance** 达到与 SOTA AR 和 DM 模型相当的性能

### 3.4 ImageNet 类别条件生成

- LDM-8 超越 ADM，同时参数量和计算需求大幅降低
- 可结合 latent space 中的 classifier guidance 进一步提升

### 3.5 超分辨率 (Super-Resolution)

- 通过 concatenation 方式条件于低分辨率图像
- FID 优于 SR3，IS 略低于 SR3
- 用户研究表明人类偏好 LDM-SR 结果
- 提出 **LDM-BSR**：使用多样化退化管道，对真实世界图像泛化更好

### 3.6 图像修复 (Inpainting)

- 相比像素级 DM：训练速度提升 **2.7×**，FID 改进 **1.6×**
- 大模型 (387M 参数) 在 Places 数据集上达到 **SOTA FID**
- 用户研究确认人类偏好 LDM 生成结果

### 3.7 卷积式高分辨率合成

- 通过卷积方式，模型可泛化到训练分辨率之外（如 $256^2$ → $1024^2$ 甚至 megapixel 级别）
- 潜在空间的信噪比 (SNR) 显著影响效果

---

## 4. 关键贡献

1. **两阶段分离训练**：感知压缩与语义生成解耦，自编码器只需训练一次，可复用于多个 DM 训练
2. **灵活的压缩率**：基于卷积 UNet 的架构使得 LDM 可以优雅地使用更高维的潜在空间（$f=4\sim16$），无需像之前 AR 方法那样过度压缩
3. **通用条件机制**：Cross-attention 实现多模态条件生成（文本、布局、类别），无需任务特定架构
4. **计算效率大幅提升**：显著降低训练和推理成本，使高分辨率图像合成更加"民主化"
5. **开源**：发布了预训练的 LDM 和自编码模型

---

## 5. 局限性与社会影响

### 局限性
- 序列化采样仍慢于 GAN
- 对像素级别的高精度重建任务存在瓶颈（$f=4$ 仍有轻微质量损失）

### 社会影响
- 生成模型是双刃剑：既促进创意应用，也可能被滥用于制造虚假信息
- 训练数据偏见可能被模型放大
- 模型可能泄露训练数据中的敏感信息

---

## 6. 技术要点总结

| 组件 | 细节 |
|---|---|
| 第一阶自编码器 | 感知损失 + PatchGAN 对抗损失 + KL/VQ 正则化 |
| 潜在空间压缩率 | $f \in \{4, 8, 16\}$ 最优 |
| DM 骨干 | 时间条件 UNet，基于 2D 卷积 |
| 条件机制 | Cross-Attention + 领域专用编码器 $\tau_\theta$ |
| 文本编码器 | BERT tokenizer + Transformer |
| 采样加速 | DDIM 采样器 |
| 质量提升 | Classifier-free guidance |
| 高分辨率合成 | 卷积方式泛化 + 潜在空间 SNR 调整 |

---

## 7. 影响力

这篇论文奠定了 **Stable Diffusion** 的理论基础，是文本到图像生成领域最具影响力的工作之一。其提出的"在潜在空间训练扩散模型"的范式已成为后续扩散模型研究的标准做法，其代码和预训练模型也被广泛应用于各类生成任务中。
