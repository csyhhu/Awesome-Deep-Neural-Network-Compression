# DMD2: Improved Distribution Matching Distillation for Fast Image Synthesis

> **作者**: Tianwei Yin$^{1}$, Michaël Gharbi$^{2}$, Taesung Park$^{2}$, Richard Zhang$^{2}$, Eli Shechtman$^{2}$, Frédo Durand$^{1}$, William T. Freeman$^{1}$  
> **机构**: $^{1}$MIT, $^{2}$Adobe Research  
> **链接**: [arXiv 2405.14867](https://arxiv.org/abs/2405.14867) | [项目主页](https://tianweiy.github.io/dmd2/)  
> **会议**: NeurIPS 2024

---

## 一、研究背景与动机

扩散模型（Diffusion Models）在图像生成领域取得了极高质量，但其采样过程需要数十步迭代去噪，推理成本高昂。已有的蒸馏方法（如 DMD）尝试将 teacher 模型蒸馏为少步或单步 student 生成器，但存在以下问题：

1. **DMD 依赖回归损失（Regression Loss）**：原版 DMD 需要额外构建大规模的噪声-图像配对数据集（通过 teacher 多步确定性采样生成），在文本到图像的大规模场景下计算成本极高（如 SDXL 需要约 700 A100 天）。
2. **回归损失与分布匹配理念冲突**：回归损失强制 student 学习 teacher 的特定采样路径，而非纯粹的分布匹配，导致 student 的生成质量被 teacher 模型上界限制。
3. **无法利用真实数据**：原版 DMD 仅使用 teacher 的分数估计来监督，teacher 模型的近似误差会传播给 student。
4. **仅支持单步生成**：原版 DMD 只支持单步 student，对 SDXL 等大模型难以在一步内学习复杂的噪声到图像映射。

---

## 二、核心贡献

DMD2 提出四项关键改进，解决了上述问题：

### 1. 移除回归损失（Removing the Regression Loss）
- 完全移除 DMD 中昂贵的回归损失和配对数据集构建，实现真正的纯分布匹配蒸馏。
- 大幅降低大规模 text-to-image 训练的计算开销。

### 2. 双时间尺度更新规则（Two Time-scale Update Rule, TTUR）
- **问题诊断**：直接移除回归损失后训练不稳定，原因是 fake diffusion critic（$\mu_\text{fake}$）无法准确估计生成器动态变化的输出分布。
- **解决方案**：采用 TTUR——每 1 次生成器更新对应 5 次 fake critic 更新，确保 $\mu_\text{fake}$ 准确跟踪生成器分布。
- 分析表明 5 次更新在稳定性和收敛速度之间取得最佳平衡，比 10 次更新更快收敛，比 1 次更新更稳定。

### 3. 引入 GAN 损失（GAN Loss with Real Data）
- 在 fake diffusion denoiser 的 bottleneck 上附加分类分支作为判别器，区分**真实图像**与生成图像。
- **关键优势**：GAN 判别器使用真实数据训练，不受 teacher 模型近似误差影响，使 student 有可能**超越** teacher。
- GAN 损失与分布匹配目标一致（均不需要配对数据，独立于 teacher 采样路径）。
- GAN 分类器训练使用带噪声注入的扩散样本（遵循 DiffusionGAN 思路）。

### 4. 多步生成器 + 反向模拟（Multi-step Generator with Backward Simulation）
- **多步采样协议**：固定时间步调度（如 999, 749, 499, 249），每步交替执行去噪和噪声注入（类似 Consistency Model）。
- **反向模拟（Backward Simulation）**：
  - **问题**：传统多步蒸馏方法在训练时使用前向扩散过程的加噪真实图像，但在推理时，除第一步外，后续步骤输入来自 student 自身的采样输出，存在训练-推理域偏移（domain mismatch）。
  - **解决**：训练时模拟推理过程——使用 student 生成器自身的多步输出作为训练输入，消除域偏移。
  - 与并发工作 Imagine Flash 的区别：Imagine Flash 的回归损失依赖 teacher，teacher 本身也面临训练-测试域偏移，而 DMD2 的分布匹配损失**不依赖 student 的输入**，避免了此问题。

---

## 三、整体训练流程

从预训练扩散模型出发，交替进行：

1. **优化生成器 $G_\theta$**：最小化 DMD 分布匹配目标 + GAN 目标
2. **优化 fake score estimator $\mu_\text{fake}$**：使用 denoising score matching + GAN 分类损失，以更高频率（5:1）更新

---

## 四、实验结果

### ImageNet-64×64（类别条件生成）
| 方法 | 前向步数 ↓ | FID ↓ |
|------|-----------|-------|
| BigGAN-deep | 1 | 4.06 |
| StyleGAN-XL | 1 | 1.52 |
| DMD | 1 | 2.62 |
| **DMD2 (Ours)** | 1 | **1.51** |
| **DMD2 (longer training)** | 1 | **1.28** |
| EDM Teacher (ODE, 511步) | 511 | 2.32 |
| EDM Teacher (SDE, 511步) | 511 | 1.36 |

- **单步 FID 1.28**，超越 511 步 SDE teacher 和所有蒸馏/GAN 方法。

### COCO 2014 零样本文本到图像（SDXL 蒸馏）
| 方法 | 步数 | FID ↓ | Patch FID ↓ | CLIP ↑ |
|------|------|-------|-------------|--------|
| SDXL Teacher (cfg=6) | 100 | 19.36 | 21.38 | 0.332 |
| **DMD2 1-step** | 1 | 19.01 | 26.98 | 0.336 |
| **DMD2 4-step** | 4 | **19.32** | **20.86** | 0.332 |
| SDXL-Turbo 4-step | 4 | 23.19 | 23.27 | 0.334 |
| SDXL-Lightning 4-step | 4 | 24.46 | 24.56 | 0.323 |

- **4步 FID 19.32**，与 100 步 teacher 持平，推理速度提升 **25×**。
- 用户偏好研究：DMD2 在图像质量上在 24% 的样本中超越 teacher，文本对齐能力相当。

### SD v1.5 蒸馏（COCO 零样本）
| 方法 | 步数 | FID ↓ |
|------|------|-------|
| DMD | 1 | 11.49 |
| **DMD2 (Ours)** | 1 | **8.35** |
| SDv1.5 Teacher (ODE, 50步) | 50 | 8.59 |

- 单步 FID 8.35，超越 50 步 teacher 采样器（3.14 点提升），推理加速 **500×**。

### 消融实验关键结论
| 实验 | 发现 |
|------|------|
| 移除回归损失 → | FID 从 2.62 恶化到 3.48（不稳定） |
| + TTUR → | FID 恢复到 2.61（无需配对数据） |
| + GAN → | FID 提升到 1.51 |
| 纯 GAN（无分布匹配） → | SDXL 上 FID 最低但文本对齐显著下降 |
| 去除反向模拟 → | Patch FID 显著变差（24.21 vs 20.86） |

---

## 五、方法局限

1. **多样性轻微下降**：蒸馏模型相比 teacher 的多样性分数略有降低（可通过调整 GAN 权重缓解）。
2. **SDXL 仍需 4 步**：最大 SDXL 模型仍需 4 步才能匹配 teacher 质量。
3. **固定引导尺度**：训练时使用固定的 classifier-free guidance scale，用户推理时无法灵活调整。
4. **计算资源需求高**：大模型训练仍需要大量 GPU 资源（SDXL 蒸馏需 64 块 A100）。

---

## 六、关键洞察总结

| 洞察 | 说明 |
|------|------|
| DMD 不稳定的根源 | fake critic 未能准确跟踪生成器分布，而非分布匹配目标本身的问题 |
| TTUR 比异步学习率更有效 | 增加 fake critic 更新频率优于单纯增大其学习率 |
| 分布匹配 + GAN 互补 | 分布匹配保证文本对齐和模式覆盖，GAN 利用真实数据提升视觉质量 |
| 反向模拟消除域偏移 | 推理时用 student 自身输出，训练时也应如此，避免 teacher 依赖 |
| 分布式匹配天然适配多步 | 分布式匹配损失不依赖输入来源，比回归损失更自然地支持反向模拟 |

---

## 七、讨论与问答

### Q1: 以一个例子说明本文的训练和推理

我们以 **SDXL 4 步蒸馏** 为例，用文本 prompt `"a cat reading a newspaper"` 来说明。

#### 🟢 推理阶段（Inference）

推理使用 4 个固定时间步：`{t₁=999, t₂=749, t₃=499, t₄=249}`（teacher 训练共 1000 步）。每步交替执行「去噪 → 加噪」：

```
Step 1 (t=999):
  输入: x_999 = 纯高斯噪声 z ~ N(0, I)
  输出: x̂_999 = G(x_999, t=999, prompt="a cat...")
  → 生成一个粗糙的轮廓
  → 加噪准备下一步: x_749 = α_749 · x̂_999 + σ_749 · ε,  ε ~ N(0, I)

Step 2 (t=749):
  输入: x_749（上一步 student 输出的加噪版本）
  输出: x̂_749 = G(x_749, t=749, prompt="a cat...")
  → 去噪，细节增多
  → 加噪: x_499 = α_499 · x̂_749 + σ_499 · ε

Step 3 (t=499):
  输入: x_499
  输出: x̂_499 = G(x_499, t=499, prompt="a cat...")
  → 猫和报纸的轮廓逐渐清晰
  → 加噪: x_249 = α_249 · x̂_499 + σ_249 · ε

Step 4 (t=249):
  输入: x_249
  输出: x̂_249 = G(x_249, t=249, prompt="a cat...")
  → 最终输出：一只正在看报纸的猫 🎉
```

#### 🔵 训练阶段（Training）

训练交替进行两个步骤。**核心前置操作：反向模拟（Backward Simulation）**——训练时模拟推理过程，使用 student 自身多步输出作为中间步的输入，而非对真实图像加噪：

```
# 反向模拟：用 student 自己的输出来模拟推理过程
z ~ N(0,I)
x̂_999 = G(z, 999, prompt)              ← student step 1 输出
x_749  = α_749·x̂_999 + σ_749·ε          ← 模拟推理的加噪
x̂_749 = G(x_749, 749, prompt)          ← student step 2 输出
x_499  = α_499·x̂_749 + σ_499·ε
x̂_499 = G(x_499, 499, prompt)          ← student step 3 输出
x_249  = α_249·x̂_499 + σ_249·ε
x̂_249 = G(x_249, 249, prompt)          ← 最终 student 输出
```

**步骤 A：更新生成器 G（1 次）**，对最终输出 `x̂_249` 计算两个损失：

> ① **DMD 分布匹配损失**：对 `x̂_249` 做前向扩散（加噪），得到不同噪声级别的 fake 样本，然后用 teacher（冻结的 μ_real）和 fake critic（动态的 μ_fake）分别估计分数，梯度为 `-(s_real - s_fake) · dG/dθ`，拉近两个分布。
>
> ② **GAN 损失**：判别器 D 接收加噪后的 student 输出和加噪后的真实图像，生成器最小化 `-log(D(加噪学生输出))`，目标是骗过判别器。

**步骤 B：更新 fake critic μ_fake（5 次，比 G 频繁 5 倍）**：

> ① Denoising Score Matching：在 student 输出上训练去噪，让 μ_fake 准确建模 fake 分布。
>
> ② GAN 判别器训练：D 学习区分加噪真实图像 vs 加噪 student 输出。

#### 🟠 为什么反向模拟很关键？

| 传统多步蒸馏（如 SDXL-Lightning） | DMD2 的反向模拟 |
|---|---|
| 训练时中间输入：对**真实图像**加噪 | 训练时中间输入：student **自己的多步输出** |
| 推理时中间输入：student 自己的输出 | 推理时中间输入：student 自己的输出 |
| ❌ 训练 ≠ 推理 → **域偏移（domain mismatch）** | ✅ 训练 = 推理 → **一致** |
| 回归损失依赖 teacher，teacher 同样面临域偏移 | 分布匹配损失不依赖 student 输入来源 |

**直观理解**：假设 student 在 step 2 生成了一只「歪嘴的猫」。传统方法训练时从未见过这种不完美输入（训练时都是加噪的真实照片），推理时突然面对就会质量下降。DMD2 的反向模拟让 student 在训练时就习惯处理自己的不完美输出，推理时自然游刃有余。此外，DMD2 的分布匹配损失（Eq.1）只关心**最终输出**的分布，不关心中间步的输入是哪里来的——这个性质使得反向模拟天然可行，而回归损失做不到这一点。

### Q2: 模拟推理中 α_749 是什么？

`α_t` 是 teacher 扩散模型**噪声调度（noise schedule）的参数**，是预定义的固定常数，不需要学习。

在前向扩散过程中：`x_t = α_t · x + σ_t · ε`，其中 `α_t` 是信号缩放系数，`σ_t` 是噪声缩放系数。多步推理中，加噪步骤复用此公式：

```
Step 1: x̂_999 = G(纯噪声, t=999)       ← 去噪
→ 加噪: x_749 = α_749 · x̂_999 + σ_749 · ε  ← 给粗糙图像注入 t=749 级别的噪声
Step 2: x̂_749 = G(x_749, t=749)        ← 去噪
...
```

不同时间步的 α_t / σ_t 含义：

| 时间步 | α_t | σ_t | 含义 |
|--------|-----|-----|------|
| t=999 | 很小 | 很大 | 几乎全是噪声 |
| t=749 | 中等偏小 | 中等偏大 | 主要是噪声，少量信号 |
| t=499 | 中等 | 中等 | 信号和噪声各半 |
| t=249 | 很大 | 很小 | 几乎是干净图像 |

### Q3: Student 跑几步，teacher 跑全步然后计算 loss？

**不对。** Teacher 在整个训练过程中是**冻结的，不跑任何采样**。它不是一个"生成器"来跟 student 对比输出，而是一个**静态分数函数（score function）**：输入加噪图像 `x_t`，输出对干净图像的估计 `μ(x_t, t)`。

DMD2 中有**四个模型**，角色完全不同：

| 模型 | 是否训练 | 从何初始化 | 功能 |
|------|----------|-----------|------|
| **G（student 生成器）** | ✅ 训练 | teacher 权重 | 输入噪声 → 输出图像（1 或 4 步） |
| **μ_real（teacher）** | ❌ 冻结 | 原始预训练扩散模型 | 一次前向：输入加噪图 → 输出分数 s_real |
| **μ_fake（fake critic）** | ✅ 训练 | teacher 权重 | 一次前向：输入加噪图 → 输出分数 s_fake |
| **D（GAN 判别器）** | ✅ 训练 | 附加在 μ_fake 的 bottleneck | 一次前向：输入加噪图 → 输出真/假 |

**DMD loss 的工作方式**：对 student 输出 `x̂` 加噪后，分别送入 μ_real 和 μ_fake 获取两个分数，梯度 = `-(s_real - s_fake) · dG/dθ`，拉近两个分布。

**GAN loss 的工作方式**：D 区分**加噪的真实训练图像** vs **加噪的 student 生成图像**（不是 student vs teacher），这正是 DMD2 能**超越 teacher** 的关键。

### Q4: μ_fake 是额外训练的模型吗？它与 student 的关系？

是的。G（student 生成器）和 μ_fake（fake critic）是两个**独立模型**，但都从同一个 teacher 初始化：

```
Teacher (SDXL UNet, 冻结)
    │
    ├── 复制权重 → G（训练：最小化 DMD loss + GAN loss）
    │
    └── 复制权重 → μ_fake（训练：denoising score matching + 挂载 GAN 判别器 D）
                       │
                       └── bottleneck 上挂 D（分类头）
```

- **G** 继承 UNet，做一步或多步去噪生成图像
- **μ_fake** 也继承 UNet，但只做**单次前向评分**：输入加噪图像 `x_t` → 输出干净估计 `μ(x_t, t)`，不跑多步采样
- DMD 梯度需要两个分数（s_real 和 s_fake），所以必须有 μ_fake，且它必须准确——这就是 TTUR（5:1 更新频率）的动机
- μ_fake 不是「判别器」，它不区分 student/teacher，而是一个**分布建模器**，精确刻画 student 当前输出分布

### Q5: 本文可以作用于不同的采样方式吗（DDPM、FM）？

**可以，DMD2 与 teacher 的采样方式无关。** 原因：DMD2 根本不使用 teacher 做采样。

无论 teacher 是用 DDPM、DDIM 还是 EDM 训练的，最终都是同一个去噪网络 `μ(x_t, t)`。DMD2 只需要 teacher 的**单步分数估计能力**，与采样方式解耦。论文中验证了 EDM（ImageNet）和 DDPM-based 的 SDXL/SDv1.5，均表现一致。

对于 **Flow Matching（FM）**：
- FM 预测 velocity `v(x_t, t)` 而非 noise `ε` 或 data `x_0`
- 原理上存在确定性转换：`s(x_t, t) ≈ -v(x_t, t) / σ_t`（特定参数化下）
- 只要能提取出分数函数 `∇_x log p_t(x)`，DMD2 框架完全适用
- 理论上兼容，但论文未做实验验证

Student 的推理协议（去噪→加噪交替）是 DMD2 自己设计的，与 teacher 原始采样器无关。
