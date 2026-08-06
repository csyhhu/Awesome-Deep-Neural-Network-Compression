# DiTFastAttn: Attention Compression for Diffusion Transformer Models

> **Paper**: [arXiv:2406.08552](https://arxiv.org/abs/2406.08552)
> **Authors**: Zhihang Yuan, Hanling Zhang, Pu Lu, Xuefei Ning, Linfeng Zhang, Tianchen Zhao, Shengen Yan, Guohao Dai, Yu Wang
> **Venue**: NeurIPS 2024
> **Project**: http://nics-effalg.com/DiTFastAttn
> **Tag**: `ditfastattn_attention_compression`

---

## 1. 核心问题

DiT 的 self-attention 计算量随 token 数量呈平方级增长（O(N²)），在高分辨率生成时成为主要瓶颈。本文定位了 DiT 推理过程中**三种注意力冗余**，并提出对应的压缩技术。

## 2. 三类冗余与解决方案

### 冗余 1：空间冗余 → Window Attention with Residual Sharing (WA-RS)

**观察**：许多 attention head 的注意力值呈现**窗口模式**（集中在对角线附近），说明这些 head 主要捕获局部信息。

**问题**：直接使用 Window Attention 会丢失远距离依赖，需要较大的 window size 才能保持质量，限制了加速效果。

**解决方案**：**残差共享**
1. 在特定时间步 r，同时计算 full attention $O_r$ 和 window attention $W_r$，得到残差 $R_r = O_r - W_r$
2. 后续 k 步仅计算 window attention，输出为 $O_k = W_k + R_r$
3. 残差 $R_r$ 在多个相邻步间共享（因为残差变化远小于 attention 输出本身）

**效果**：可用更小的 window size 实现更大加速，同时保持性能

### 冗余 2：时间步冗余 → Attention Sharing across Timesteps (AST)

**观察**：相邻时间步的 attention 输出具有高相似度（余弦相似度分析证实）。

**解决方案**：
1. 将相似的连续时间步组成集合 K
2. 仅在集合的第一步计算 attention，后续步骤复用该输出
3. 跳过后续步骤的 attention 计算

### 冗余 3：CFG 冗余 → Attention Sharing across CFG (ASC)

**观察**：Classifier-Free Guidance（CFG）需要条件/无条件两次前向传播，但在很多层和步骤中，两者的 attention 输出高度相似。

**解决方案**：
1. 仅计算条件前向的 attention 输出
2. 无条件前向复用条件输出的 attention 结果
3. 节省约一半的 attention 计算量

## 3. 压缩计划选择

### Greedy 校准算法

为每个 (step, layer) 组合选择最优压缩策略：
1. 策略列表 S = [AST, WA+ASC, WA, ASC]
2. 按压缩比从大到小排序尝试
3. 对每个策略，计算压缩前后输出的损失 $L(O, O')$
4. 若损失低于阈值 $\delta_i$（与层索引相关），则采用该策略
5. 若无策略满足阈值，则使用 full attention

### 阈值设置
- $\delta_i = i/M \cdot \delta$，i 为层索引，M 为层数
- 深层（i 大）容忍度更高（因为深层处理局部细节，对压缩更敏感）

## 4. 实验结果

### 4.1 图像生成 (DiT, PixArt-Sigma)

| 分辨率 | Attention FLOPs 减少 | 端到端加速 | FID 变化 |
|---|---|---|---|
| 512×512 | ~50% | ~1.3× | 微小 |
| 1024×1024 | ~65% | ~1.5× | 可控 |
| 2048×2048 | **~76%** | **~1.8×** | 微小 |

### 4.2 视频生成 (OpenSora)

| 分辨率 | Attention FLOPs 减少 | 加速 |
|---|---|---|
| 240p, 16帧 | ~48% | ~1.5× |

### 4.3 与其他方法对比

| 方法 | 压缩维度 | 训练需求 | 加速效果 |
|---|---|---|---|
| Flash Attention | 实现优化 | 无需 | 中等 |
| GQA | 架构修改 | 需要 | 中等 |
| **DiTFastAttn** | 注意力冗余压缩 | **无需训练** | 高（2K 分辨率下 1.8×） |

## 5. 创新点总结

1. **首次系统分析 DiT 注意力计算中的三类冗余**
2. **WA-RS**：创造性地使用残差共享解决 Window Attention 的远距离依赖问题
3. **AST**：跨时间步共享 attention 输出，利用生成过程的时序相似性
4. **ASC**：在 CFG 双前向传播间共享 attention，直接减半计算
5. **Greedy 校准算法**：自动为每个 (step, layer) 选择最优压缩策略
6. **Post-training 方法**：无需重新训练，可直接应用于预训练 DiT 模型

## 6. 局限性

- 仅压缩 attention 计算，不涉及 MLP 和其他模块
- AST 带来额外 VRAM 开销（存储前步 attention hidden states）
- Greedy 算法不是全局最优
- 简单压缩方案可能无法找到最优压缩计划

## 7. 与其他方法的对比特点

| 维度 | ToMeSD | DyDiT | **DiTFastAttn** |
|---|---|---|---|
| 压缩对象 | Token 数量 | Head/Channel/Token | Attention 计算 |
| 核心机制 | Token Merge/Unmerge | 动态宽度 + Token 跳过 | Window + Step/CFG 共享 |
| 时间步动态 | 简单线性 | 动态宽度 | AST (步间共享) |
| CFG 优化 | 无 | 无 | ASC (条件/无条件共享) |
| 训练需求 | 无训练 | 微调 | **无训练** |
| 适用范围 | U-Net/DiT | DiT | DiT + MMDiT |
