# FLUX.1 Kontext: 基于流匹配的潜在空间上下文图像生成与编辑

> **论文**: [FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space](https://arxiv.org/abs/2506.15742)
> **作者**: Black Forest Labs (Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, Luke Smith)
> **机构**: Black Forest Labs

---

## 一、核心贡献

本文提出了 **FLUX.1 Kontext**，一个统一的流匹配（Flow Matching）生成模型，能够同时处理**图像生成**（text-to-image）和**图像编辑**（image-to-image）任务。其核心贡献包括：

1. **统一架构**：通过简单的序列拼接（Sequence Concatenation），在单一模型中支持局部编辑、全局编辑、角色参考、风格参考和文本编辑等多种任务
2. **角色一致性**：在多轮迭代编辑中显著优于现有方案，AuraFace 相似度平均保持率达 0.908（对比 Runway Gen-4 的 0.774 和 GPT-4o High 的 0.416）
3. **交互速度**：在 1024×1024 分辨率下，文本生图和图生图均只需 3–5 秒，比竞品快一个数量级
4. **KontextBench 基准**：包含 1026 组真实世界图像-提示对，覆盖 5 大类任务

---

## 二、背景与动机

### 2.1 图像生成与编辑的两大范式

- **局部编辑（Local Editing）**：保持周围上下文不变的局部修改，如改变汽车颜色、替换背景
- **生成式编辑（Generative Editing）**：提取视觉概念并在新环境中重现，类似大语言模型的上下文学习

### 2.2 现有方法的局限

| 问题 | 描述 |
|------|------|
| 合成数据偏差 | 基于合成指令-响应对训练的方法继承了生成管线的缺陷 |
| 角色漂移 | 跨多次编辑保持角色/物体外观仍然是开放问题 |
| 速度瓶颈 | 自回归编辑模型延迟高，无法交互式使用 |

---

## 三、FLUX.1 架构基础

### 3.1 编码器（VAE）

- 从零训练卷积自编码器，使用对抗目标
- 16 个潜在通道，重建能力超越 SD3-VAE、SDXL-VAE 等
- VAE 重建质量对比（4096 张 ImageNet 图像）：

| 模型 | PDist ↓ | SSIM ↑ | PSNR ↑ |
|------|---------|--------|--------|
| **FLUX-VAE** | **0.332** | **0.896** | **31.1** |
| SD3-VAE | 0.452 | 0.858 | 29.6 |
| SDXL-VAE | 0.890 | 0.748 | 25.9 |
| SD-VAE | 0.949 | 0.720 | 25.0 |

### 3.2 Transformer 结构

- **双流块（Double Stream Blocks）**：图像 token 和文本 token 使用独立权重，通过拼接注意力实现混合
- **单流块（Single Stream Blocks）**：38 个块处理拼接后的图像和文本 token，之后丢弃文本 token 仅解码图像
- **融合前馈块（Fused Feed-Forward Blocks）**：受 [Dehghani et al. 2023] 启发，将注意力输入/输出线性层与 MLP 融合，减少调制参数并增大矩阵乘法
- **3D RoPE**：因式分解的三维旋转位置编码，每个潜在 token 用时空坐标 $(t, h, w)$ 索引

---

## 四、FLUX.1 Kontext 方法

### 4.1 条件分布建模

目标学习条件分布：

$$p(x \mid y, c)$$

其中 $x$ 是目标图像，$y$ 是可选的上下文图像（或空集），$c$ 是自然语言指令。

- 当 $y \neq \varnothing$ 时：执行图像驱动的编辑
- 当 $y = \varnothing$ 时：从零创建新内容（纯文本生图）

### 4.2 Token 序列构建

1. 图像由冻结的 FLUX 自编码器编码为潜在 token
2. 上下文图像 token $y$ 追加到目标图像 token $x$ 后面
3. **3D RoPE 位置编码**为上下文 token 设置恒定偏移：
   - 目标 token：$\mathbf{u}_x = (0, h, w)$
   - 第 $i$ 个上下文图像的 token：$\mathbf{u}_{y_i} = (i, h, w)$

这个偏移作为**虚拟时间步**，干净地分离上下文和目标块，同时保留各自的空间结构。

> **设计选择**：也测试了通道拼接（channel-wise concatenation），但初始实验中序列拼接表现更好。

### 4.3 Rectified Flow 训练目标

$$\mathcal{L}_\theta = \mathbb{E}_{t \sim p(t), x, y, c} \left[ \lVert v_\theta(z_t, t, y, c) - (\varepsilon - x) \rVert_2^2 \right]$$

其中：
- $z_t = (1-t) x + t \varepsilon$：$x$ 和噪声 $\varepsilon \sim \mathcal{N}(0, 1)$ 的线性插值
- $p(t)$：Logit-Normal 移位分布，mode $\mu$ 根据训练数据分辨率调整
- 当采样纯文本-图像对（$y = \varnothing$）时，省略所有 $y$ token

### 4.4 对抗扩散蒸馏（LADD）

为解决流匹配模型采样慢（50–250 次网络评估）和可能引入视觉伪影的问题，采用潜在对抗扩散蒸馏（LADD），减少采样步数同时提升质量。

### 4.5 模型变体

| 变体 | 描述 |
|------|------|
| **Kontext [pro]** | 流目标训练后接 LADD 蒸馏 |
| **Kontext [dev]** | 引导蒸馏到 12B 扩散 Transformer，仅训练图生图任务 |
| **Kontext [max]** | 使用更多计算资源以提升生成性能 |

### 4.6 实现细节

- 从纯文本生图检查点开始，联合微调图生图和文生图任务
- **FSDP2** 混合精度：all-gather 使用 bfloat16，reduce-scatter 使用 float32
- **选择性激活检查点**：降低最大显存占用
- **Flash Attention 3** + 区域编译 Transformer 块以提升吞吐

---

## 五、实验评估

### 5.1 KontextBench 基准

| 任务类型 | 示例数 |
|----------|--------|
| 局部指令编辑 | 416 |
| 全局指令编辑 | 262 |
| 文本编辑 | 92 |
| 风格参考 | 63 |
| 角色参考 | 193 |
| **总计** | **1026** |

数据来源：108 张基础图像（个人照片、CC 授权艺术、公共领域图像、AI 生成内容）。

### 5.2 图生图（I2I）结果

**人工评估（KontextBench）**：
- **Kontext [pro]** 在文本编辑和角色保持（CREF）类别中排名第一
- 全局编辑和风格参考（SREF）仅次于 gpt-image-1 和 Gen-4 References

**定量评估（AuraFace 人脸识别）**：
- FLUX.1 Kontext：0.908（平均相似度）
- Runway Gen-4：0.774
- GPT-4o High：0.416

**延迟对比**：Kontext 在 I2I 任务上比竞品快一个数量级

### 5.3 文生图（T2I）结果

引入五个评估维度（将"AI 美学"泛化现象称为 **bakeyness**）：
- Prompt Following（指令遵循）
- Aesthetics（美学）
- Realism（真实感）
- Typography Accuracy（排版准确性）
- Inference Speed（推理速度）

在 Internal-T2I-Bench 和 GenAI-bench 上，FLUX.1 Kontext 在各维度上均衡表现，且持续优于前一代 FLUX1.1 [pro]。

### 5.4 迭代编辑

在多轮编辑中，Kontext 展现出显著更强的角色身份保持能力：
- 5 步编辑后 AuraFace 相似度：Kontext 0.7427，Gen-4 0.4986，GPT-4o High 0.2915
- 应用场景：故事板生成、品牌角色维护、产品电商编辑

### 5.5 特色应用

- **风格参考（SREF）**：从参考图像提取艺术风格并应用到新场景
- **视觉线索编辑**：通过几何标记（如红色椭圆）引导目标修改
- **文本编辑**：Logo 优化、拼写纠正、风格适配
- **产品摄影**：提取商品、展示面料细节

---

## 六、流匹配理论基础（附录）

### 6.1 Rectified Flow Matching 入门

前向加噪过程：
$$z_t = a_t x_0 + b_t \varepsilon$$

条件流匹配损失：
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, \varepsilon} \lVert v_\Theta(z_t, t) - \frac{a_t'}{a_t} z_t + \frac{b_t}{2} \lambda_t' \varepsilon \rVert_2^2$$

对于 rectified flow，$a_t = 1-t$，$b_t = t$，损失简化为：
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, \varepsilon, x_0} \lVert v_\Theta(z_t, t) + x_0 - \varepsilon \rVert_2^2$$

### 6.2 Logit-Normal 时间步调度

$$p(t) = \frac{\exp(-0.5 \cdot (\text{logit}(t) - \mu)^2 / \sigma^2)}{\sigma \sqrt{2\pi} \cdot (1-t) \cdot t}$$

其中 $\text{logit}(t) = \log\frac{t}{1-t}$，$Y = \text{logit}(t) \sim \mathcal{N}(\mu, \sigma)$。

### 6.3 调度移位

$\alpha$-移位的 log-SNR 可以通过 Logit-Normal 分布的 $\mu$ 参数表达：
- $\mu = \log \alpha$，$\sigma = 1.0$
- 对于 $\alpha = 3.0$，对应 $\mu = 1.0986$

移位后的时间步重分配公式：
$$t' = \frac{e^\mu}{e^\mu + (1/t - 1)^\sigma}$$

---

## 七、局限与展望

### 局限
1. **过度多轮编辑**会引入视觉伪影，降低图像质量
2. 偶尔不能准确遵循指令，忽略特定 prompt 需求
3. 蒸馏过程可能引入视觉伪影，影响输出保真度

### 未来方向
- 扩展到多图像输入
- 进一步扩大模型规模
- 降低推理延迟实现实时应用
- 扩展到视频域
- 解决多轮编辑中的质量退化问题

---

## 八、关键数据总结

| 维度 | 指标 | FLUX.1 Kontext | 对比方法 |
|------|------|----------------|----------|
| 角色一致性 | AuraFace 5 步平均 | **0.908** | Gen-4: 0.774, GPT-4o: 0.416 |
| 推理速度 | 1024×1024 I2I | **3–5s** | 比竞品快一个数量级 |
| VAE 重建 | PDist | **0.332** | SD3: 0.452, SDXL: 0.890 |
| 基准规模 | KontextBench | **1026 对** | 5 大任务类别 |

---

## 九、架构要点速记

```
输入: 文本指令 c + 可选上下文图像 y
  ↓
FLUX VAE 编码 → 潜在 token 序列
  ↓
序列拼接: [目标 token x | 上下文 token y]
  ↓
3D RoPE (上下文带时间步偏移)
  ↓
双流块 (图像/文本独立权重 + 拼接注意力)
  ↓
38× 融合单流块
  ↓
丢弃文本 token → 解码图像 token
  ↓
LADD 蒸馏 → 快速采样
```