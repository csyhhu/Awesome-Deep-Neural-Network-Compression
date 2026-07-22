# TurboDiffusion: 将视频扩散模型加速 100-200 倍

## 论文信息

- **标题**: TurboDiffusion: Accelerating Video Diffusion Models by 100--200 Times
- **arXiv**: [2512.16093](https://arxiv.org/abs/2512.16093)
- **作者**: Jintao Zhang, Kaiwen Zheng, Kai Jiang, Haoxu Wang, Ion Stoica, Joseph E. Gonzalez, Jianfei Chen, Jun Zhu
- **机构**: 清华大学, 生数科技, UC Berkeley
- **代码**: [https://github.com/thu-ml/TurboDiffusion](https://github.com/thu-ml/TurboDiffusion)

## 摘要

本文提出了 TurboDiffusion，一个视频生成加速框架，能够在保持视频质量的同时将端到端扩散生成速度提升 **100-200 倍**。

## 核心技术

TurboDiffusion 主要依赖以下四个技术组件进行加速：

### 1. 注意力加速 (Attention Acceleration)

- **SageAttention**: 采用低比特量化的 SageAttention2++ 变体来加速注意力计算
- **Sparse-Linear Attention (SLA)**: 可训练的稀疏线性注意力，利用稀疏计算与低比特 Tensor Core 加速的正交性，在 SageAttention 基础上提供累积加速效果

### 2. 步数蒸馏 (Step Distillation)

- 采用 **rCM** (recent state-of-the-art diffusion distillation method) 进行高效的步数蒸馏
- 通过模型权重合并，rCM 自然继承了注意力级别的加速效果

### 3. W8A8 量化 (Linear Layer Quantization)

- 将模型参数和激活量化为 **INT8**
- 量化粒度为 **block-wise**，块大小为 **128 × 128**
- 使用 INT8 Tensor Cores 执行线性层计算
- 模型大小压缩约一半，线性层计算更快

### 4. 其他优化

- 使用 Triton 或 CUDA 重新实现 LayerNorm 和 RMSNorm 等操作以提高效率

## 训练流程

1. 将预训练模型的完整注意力替换为 SLA，并微调以适应稀疏性
2. 使用 rCM 将预训练模型蒸馏为具有更少采样步骤的学生模型
3. 将 SLA 微调与 rCM 训练的参数更新合并到单个模型中
4. 训练可使用真实数据或合成数据

## 推理流程

1. **注意力加速**: 将 SLA 替换为 SageSLA（基于 SageAttention 的 CUDA 实现）
2. **步数蒸馏**: 将采样步数从 100 减少到 3 或 4
3. **线性层量化**: 参数和激活都量化为 INT8，使用 INT8 Tensor Cores 加速
4. **其他优化**: LayerNorm/RMSNorm 等操作的 Triton/CUDA 优化

## 实验结果

### 测试模型

- `Wan2.2-I2V-A14B-720P` (图像到视频，14B 参数，720P)
- `Wan2.1-T2V-1.3B-480P` (文本到视频，1.3B 参数，480P)
- `Wan2.1-T2V-14B-720P` (文本到视频，14B 参数，720P)
- `Wan2.1-T2V-14B-480P` (文本到视频，14B 参数，480P)

### 加速效果（单 RTX 5090 GPU）

| 模型 | 原始延迟 | TurboDiffusion 延迟 | 加速比 |
|------|----------|---------------------|--------|
| Wan2.2-I2V-A14B-720P | 4549s | 38s | ~120× |
| Wan2.1-T2V-1.3B-480P | 184s | 1.9s | ~97× |
| Wan2.1-T2V-14B-720P | - | - | ~200× |

### 对比基线

- **Original**: Wan 官方实现
- **FastVideo**: 现有加速方案

实验表明，TurboDiffusion 不仅效率最高，而且保持了与原始模型相当的视频质量，明显优于 FastVideo。

### 超参数设置

- Top-K 比率: 0.1（对应 90% 注意力稀疏度）
- 采样步数: 3（推荐使用 4 步以获得最佳质量）
- Top-K 推荐范围: [0.1, 0.15]

## 关键亮点

1. **算法与系统协同优化**: 通过算法层面的稀疏化和蒸馏，结合系统层面的量化和底层算子优化，实现了约 200 倍的端到端加速
2. **质量保持**: 在极端加速的同时，视频质量与原始模型几乎一致
3. **硬件兼容性**: 主要在 RTX 5090 上测试，在 RTX 4090 和 H100 上也能获得显著加速

## 结论

TurboDiffusion 通过整合 SageAttention、SLA、rCM 和 W8A8 量化等多种技术，实现了视频扩散模型的突破性加速，为实时视频生成应用奠定了基础。