# Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency

- **会议/年份**: ICLR 2026
- **作者**: Kaiwen Zheng (Tsinghua & NVIDIA), Yuji Wang (Tsinghua), Qianli Ma (NVIDIA), Huayu Chen (Tsinghua & NVIDIA), Jintao Zhang (Tsinghua), Yogesh Balaji (NVIDIA), Jianfei Chen (Tsinghua), Ming-Yu Liu (NVIDIA), Jun Zhu (Tsinghua), Qinsheng Zhang (NVIDIA)
- **链接**: https://arxiv.org/abs/2510.08431

---

## 1. 核心贡献

本文首次将连续时间一致性模型（sCM）扩展到大规模文本到图像（T2I）和文本到视频（T2V）任务中，并提出 **Score-Regularized Continuous-Time Consistency Model (rCM)**，通过结合前向散度（sCM）和逆向散度（Score Distillation / DMD）实现高质量、高多样性的少步扩散模型蒸馏。

### 主要贡献：

1. **基础设施创新**：开发了兼容 FlashAttention-2 的 JVP（Jacobian-Vector Product）Triton kernel，支持 FSDP 和 Context Parallelism，使 sCM 可在 100 亿+参数模型和高维视频数据上训练。

2. **问题诊断**：揭示了 sCM 在大规模蒸馏中的质量缺陷——误差累积导致的细节失真，从理论上分析了 JVP 的自反馈机制在 BF16 精度下引入的数值误差。

3. **方法提出**：提出 rCM，将前向一致性目标（mode-covering，保障多样性）与逆向散度正则化（mode-seeking，保障质量）相结合，以 score distillation 作为长跳连接的正则项。

4. **大规模验证**：在 Cosmos-Predict2（14B）和 Wan2.1（14B）上验证，支持 5 秒视频生成，实现 15-50 倍加速。

---

## 2. 背景

### 2.1 连续时间一致性模型 (sCM)

sCM 将一致性训练推广到连续时间极限（Δt→0），损失函数为：

$$L_{sCM}(\theta) = \mathbb{E}_{x_0, \epsilon, t}\left[\left\|F_\theta(x_t, t) - F_{\theta^-}(x_t, t) - \frac{g}{\|g\|_2^2 + c}\right\|_2^2\right]$$

其中 $g = w(t)\frac{d f_{\theta^-}(x_t, t)}{d t}$ 通过 JVP 计算，tangent 包含两部分：
$$\frac{d f_{\theta^-}}{dt} = -\cos(t)(F_{\theta^-} - F_{teacher}) - \sin(t)(x_t + \underbrace{\frac{d F_{\theta^-}}{dt}}_{\text{自反馈 (JVP)}})$$

- **关键问题**：$\sin(t)$ 加权的 JVP 项引入了数值脆弱的一阶自反馈信号，误差从小时间步向大时间步传播并放大。当 $t$ 较大时，$\cos(t)/\sin(t) \to 0$，教师监督信号消失，学习动态完全由 JVP 主导。

### 2.2 Score Distillation (DMD)

Score distillation 通过扩散后匹配学生分布 $p_\theta$ 和教师分布 $p_{teacher}$，最小化逆向散度：

$$D_f(p_\theta^t \| p_{teacher}^t) = \mathbb{E}_{p_\theta^t(x_t)}[f(r(x_t))], \quad r = \frac{p_{teacher}^t}{p_{teacher}^t}$$

DMD 使用 Reverse KL（$f(r)=-\log r$），是"mode-seeking"的，擅长生成质量但易模式坍塌。

### 2.3 前向 vs 逆向散度

| 类型 | 代表方法 | 数据来源 | 特性 | 优势 | 劣势 |
|------|---------|---------|------|------|------|
| 前向散度 | CMs, sCM, MeanFlow | 离线数据（教师生成） | mode-covering | 高多样性 | 生成质量差 |
| 逆向散度 | DMD, VSD, SiD | 在线数据（学生自生成） | mode-seeking | 高质量 | 模式坍塌，低多样性 |

---

## 3. 方法：rCM

### 3.1 核心思路

rCM 将前向一致性目标与逆向散度正则化结合：

$$L_{rCM}(\theta) = L_{sCM}(\theta) + \lambda L_{DMD}(\theta)$$

- **$L_{sCM}$**：离线数据路径，保持多样性和 mode-covering 特性
- **$L_{DMD}$**：在线（学生自生成）数据路径，作为长跳正则项修复 sCM 的质量缺陷
- **$\lambda = 0.01$** 在不同模型和任务上具有普适性

### 3.2 工程简化

1. **任意噪声调度适配**：通过 SNR 匹配将 TrigFlow 包装器应用于任意预训练教师，无需重新训练。
2. **简化实现**：保留原始网络结构（位置时间嵌入、AdaLN、QK 归一化），移除 sCM 中不稳定的 Fourier 时间嵌入等组件。
3. **训练交替**：学生更新（rCM loss）与 fake score 更新（flow-matching loss）交替进行。

### 3.3 Rollout 策略

学生生成 $\x_0 \sim p_\theta$ 用于 DMD loss 和 fake score 训练：
- 随机选择采样步数 $N \in [1, N_{\max}]$
- 采用随机递减时间步方案（而非固定时间步）
- 仅对最后一步反向传播 DMD loss

### 3.4 稳定的时间导数计算

提出两种策略稳定 JVP 计算：

1. **半连续时间**（适用于 2B 规模 T2I）：空间部分用精确 JVP，时间导数用有限差分近似。
2. **高精度时间**（适用于 10B+ 模型和视频）：对时间嵌入层强制使用 FP32 精度。

### 3.5 基础设施

- **FlashAttention-2 JVP Kernel**：Triton 实现，支持 self 和 cross attention
- **FSDP 兼容**：逐层 JVP 实现，匹配 FSDP 粒度
- **Context Parallelism (Ulysses)**：通过 all-to-all 操作分发 QKV tangents，局部使用 FA2 JVP kernel

---

## 4. 实验

### 4.1 实验设置

| 项目 | 详情 |
|------|------|
| 模型 | Cosmos-Predict2 (0.6B/2B/14B T2I), Wan2.1 (1.3B/14B T2V) |
| 分辨率 | T2I: 1360×768; T2V: 832×480×81 (480p), 1280×704×93 (720p) |
| 评估 | GenEval (T2I), VBench (T2V) |
| 训练 | Full-parameter tuning（非 LoRA），FSDP2 + Ulysses CP + SAC |
| 推理 | 4-step 为主，1-4 step 消融 |

### 4.2 主要结果

**T2I (GenEval)**：
- Cosmos-Predict2 14B + rCM: 4-step 达到 Overall 0.83（接近 teacher 0.84）
- 远超同规模的蒸馏模型（SDXL-DMD2: 0.58, FLUX.1-schnell: 0.69）
- 与 SANA-Sprint 可比，但模型规模大 9 倍且分辨率更高

**T2V (VBench)**：
- Wan2.1 14B + rCM: 4-step 达到 Total Score 84.92，超越 teacher (83.58)
- Wan2.1 1.3B + rCM (4-step, 14.6 FPS) vs teacher (50×2 step, 0.72 FPS): 约 20 倍加速

**与 DMD2 对比**：
- rCM 在质量指标上与 DMD2 持平或超越
- rCM 在多样性上明显优于 DMD2（视频中尤为显著）
- DMD2 倾向于模式坍塌（物体位置/方向趋同），rCM 保留 sCM 的多样性

### 4.3 少步生成能力

| 任务 | 1-step | 2-step | 4-step |
|------|--------|--------|--------|
| T2I | 可接受（简单 prompt 接近 4-step，复杂 prompt 文字渲染差） | 良好 | 优秀（与 teacher 接近） |
| T2V | 模糊、质量下降明显 | 接近 teacher | 超越 teacher |

### 4.4 消融实验（λ 值）

| λ | 质量 | 多样性 |
|---|------|--------|
| 1.0 | 最好 | 最差 |
| 0.1 | 好 | 较差 |
| **0.01** | 良好 | 良好（**sweet spot**） |
| 0.001 | 较差（接近 sCM） | 最好 |

---

## 5. 局限与展望

- 蒸馏模型在物理一致性和极端多样性上仍不如 teacher
- 前向+逆向散度结合的范式可推广到自回归视频扩散等方向
- rCM 无需 GAN 调参或大量超参搜索，具有良好的实用性和可扩展性

---

## 6. 总结

rCM 通过将前向散度（sCM, mode-covering）与逆向散度（DMD, mode-seeking）融合，成功将连续时间一致性蒸馏扩展到 14B 参数级别的图像与视频扩散模型。方法简洁高效——仅需一个 λ 超参数（0.01），无需 GAN 训练或多阶段流水线，即可在保持高质量的同时获得显著优于纯逆向散度方法的多样性。这一"前向+逆向散度结合"的范式为大规模扩散模型蒸馏提供了统一的理论与实践框架。
