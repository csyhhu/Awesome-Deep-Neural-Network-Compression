# TokenCache: Token Caching for Diffusion Transformer Acceleration

> **Paper**: [arXiv:2409.18523](https://arxiv.org/abs/2409.18523)
> **Authors**: Jinming Lou, Wenyang Luo, Yufan Liu, Bing Li, Xinmiao Ding, Weiming Hu, Yuming Li, Chenguang Ma
> **Venue**: IEEE Journal of Selected Topics in Signal Processing
> **Code**: https://github.com/
> **Tag**: `tokencache_token_caching_dit`

---

## 1. 核心问题

DiT 的多步去噪推理过程中，**相邻时间步的 token 更新具有高度相似性**，但每一步都执行完整计算，造成大量冗余。TokenCache 通过缓存复用机制跳过冗余计算。

## 2. 核心思想

在**三个层级**上自适应地决定缓存策略：
1. **Token 级别**：预测每个 token 的重要性分数，低重要性 token 的计算结果从缓存中复用
2. **Block 级别**：根据 block 重要性自适应分配缓存比例
3. **Timestep 级别**：将时间步划分为 I-step（全计算）和 P-step（缓存复用）

## 3. 方法详解

### 3.1 框架概览

TokenCache 框架包含三个层级的缓存决策：
- **Timestep Level**：Cache Predictor 决定当前步是 I-step（独立步，全计算）还是 P-step（预测步，使用缓存）
- **Block Level**：为每个 block 分配不同的缓存比例 $r_l$
- **Token Level**：在每个 block 内，根据重要性分数 $\alpha$ 选择需要更新的 token

### 3.2 Token 缓存机制

**核心公式**：
```
x_t^{l,n} = x_t^{l-1,n} + α_t^{l,n} · f_t^{l,n}(x_t^{l-1})
            + (1-α_t^{l,n}) · f_{t+1}^{l,n}(x_{t+1}^{l-1})
```
- $α_t^{l,n} = 1$：当前 token 正常计算
- $α_t^{l,n} = 0$：复用前一步缓存的 token 更新

**Token 重要性分析**（Figure 3）：
- Token 更新模式在相邻时间步间高度相似（horizontal strips）
- 不同 token 具有不同更新模式
- Token 更新长期不均匀分布

### 3.3 Cache Predictor

**设计**：
- 轻量级 DiT block（复用第一层权重初始化）
- 输入：$x_t$ 和 $t$
- 输出：所有 block × token 的重要性分数（总共 L×N 个标量）
- 参数量增加约 3.57%

**训练目标**：
```
L = ||x̂_t^L - x_t^L||² + λ·||α_θ(x_t, t)||_1
```
- Cache Loss：缓存输出与真实输出的 L2 距离
- Cost Loss：L1 正则化，鼓励稀疏缓存决策

**优化技巧**：
- α 松弛为 [0,1] 连续值，用 sigmoid 激活
- 最后 K 层固定为非缓存模式 + LoRA 适配

### 3.4 推理策略

**Token 级别调度**：
1. 全局 cache ratio $r$ 下，将 α 量化为二值决策
2. top-(1-r)·L·N 个 token 标记为"需更新"

**Block 级别调度**：
- 计算每个 block 的 cache ratio $r_l$（该 block 内被标记为"需更新"的 token 比例）
- 在 block 内部按 $r_l$ 重新量化选择需要更新的 token

**Timestep 调度**：
- 收集 256 个样本的 α，计算时间步重要性分布
- 均匀百分位划分确定 I-step 位置
- I-step：全计算；P-step：缓存复用

## 4. 实验结果

### 4.1 类别条件图像生成 (DiT-XL/2, 256×256)

| 方法 | NFE | FID | 延迟(s) | 速度提升 |
|---|---|---|---|---|
| DiT-XL/2 (baseline) | 250 | 2.09 | 21.57 | 1.00× |
| DiT-XL/2 (2×速度) | 125 | 2.13 | 10.77 | 2.00× |
| FORA | 250 | 2.82 | 8.60 | 2.96× |
| ToCa | 250 | 2.58 | 11.13 | 3.10× |
| **TokenCache (Ours)** | 250 | **2.07** | 10.51 | **3.01×** |

在 50 NFE 下：
| 方法 | FID | 速度提升 |
|---|---|---|
| DiT-XL/2 | 2.25 | 1.00× |
| FORA | 40.82 | 2.37× |
| Learning-to-Cache | 2.61 | 2.00× |
| ToCa | 3.31 | 2.25× |
| **TokenCache (Ours)** | **2.33** | **2.40×** |

### 4.2 文生图 (PixArt-α)

- 在保持 FID 几乎不变的前提下取得显著加速

### 4.3 文生视频 (OpenSora)

- 在视频生成任务上同样有效

## 5. 创新点总结

1. **首个将 token 级缓存机制引入 DiT 的工作**
2. **Cache Predictor**：统一预测所有 token 的重要性分数，实现自适应缓存决策
3. **三级调度**：Token → Block → Timestep 三级联调度，平衡速度与质量
4. **基于 α 的时间步调度**：用累积重要性分数确定 I-step 位置，理论上更优
5. **无需手动超参数调整**：所有缓存决策由 Cache Predictor 自动学习

## 6. 局限性

- 需要训练 Cache Predictor（约增加 3.57% 参数和计算量）
- 缓存开销带来额外 VRAM 使用
- 每个采样设置需要离线确定 I-step 位置

## 7. 与其他方法的对比特点

| 维度 | ToMeSD | DyDiT | **TokenCache** |
|---|---|---|---|
| 核心机制 | Token Merge/Unmerge | 动态宽度 + Token 跳过 | Token 缓存复用 |
| 压缩类型 | Token 数量减少 | Head/Channel/Token 跳过 | Token 结果复用 |
| 时间步动态 | 简单线性 | 动态宽度 (TDW) | I-step/P-step 调度 |
| 训练需求 | 无训练 | 微调 + Router | Cache Predictor 训练 |
| 计算节省 | 减少实际计算量 | 减少激活的计算 | 跳过重复计算 |
