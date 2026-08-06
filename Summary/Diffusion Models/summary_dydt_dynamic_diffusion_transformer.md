# DyDiT: Dynamic Diffusion Transformer

> **Paper**: [arXiv:2410.03456](https://arxiv.org/abs/2410.03456)
> **Authors**: Wangbo Zhao, Yizeng Han, Jiasheng Tang, Kai Wang, Yibing Song, Gao Huang, Fan Wang, Yang You
> **Venue**: ICLR 2025
> **Code**: https://github.com/alibaba-damo-academy/DyDiT
> **Tag**: `dydt_dynamic_diffusion_transformer`

---

## 1. 核心问题

DiT 使用**静态推理范式**——所有时间步和所有空间位置都使用相同的模型宽度和计算量。但实际上：
- **时间步维度**：早期步（接近噪声）预测任务简单，小模型就能处理；后期步（接近真实图像）需要大模型
- **空间维度**：不同 patch 的预测难度不同（天空、沙滩等简单区域 vs 人物、物体等复杂区域）

## 2. 核心思想

从**时间步**和**空间**两个维度动态调整 DiT 的计算量：
- **TDW (Timestep-wise Dynamic Width)**：根据时间步动态调整 MHSA 的 head 数和 MLP 的 channel 组数
- **SDT (Spatial-wise Dynamic Token)**：根据 token 预测难度动态跳过 MLP 计算

## 3. 方法详解

### 3.1 Timestep-wise Dynamic Width (TDW)

**核心**：根据时间步 embedding 动态激活不同的 attention heads 和 MLP channel groups

**Heads 和 Channels 分组**：
- 将 MHSA 的 H 个 head 分组，MLP 的隐藏通道也分成 H 组
- 每个 head/group 对应一个路由决策

**动态路由**：
- 时间步 embedding $E_t$ 输入 Router（线性层 + Sigmoid）
- 输出每个 head/group 的激活概率 $S_{head} \in [0,1]^H$, $S_{channel} \in [0,1]^H$
- 阈值 0.5 二值化为 mask $M_{head}, M_{channel}$

**推理**：只计算被激活的 head 和 channel group
- 预计算所有时间步的 mask，部署时直接使用，无需动态推理图
- 支持 batched 推理

### 3.2 Spatial-wise Dynamic Token (SDT)

**核心**：对"容易"的 token 跳过 MLP 计算

**Token 路由**：
- 每个 MLP 层前，输入 $X \in \mathbb{R}^{N \times C}$ 送入 token router
- 预测每个 token 被处理的概率 $S_{token} \in [0,1]^N$
- 阈值 0.5 二值化为 mask $M_{token}$

**推理**：
- 将被选中的 token gather 送入 MLP
- 处理后 scatter 回原位
- **仅应用于 MLP 块**（不在 MHSA 上使用，因为 token 交互对质量至关重要）
- 支持 batched 推理（MLP 内无 token 间交互）

### 3.3 FLOPs-aware 端到端训练

**训练策略**：
1. **Straight-Through Estimator + Gumbel-Sigmoid**：使路由决策可微分
2. **FLOPs 约束损失**：
   ```
   L_FLOPs = (1/B * Σ F_dynamic / F_static - λ)²
   ```
   λ 为目标 FLOPs 比例
3. **总损失**：L = L_DiT + L_FLOPs

**训练稳定化**：
- Warm-up 阶段：同时训练完整 DiT + DyDiT，避免训练不稳定
- 重要性排序：预训练时按 magnitude 排序 heads 和 channels，保证最关键的始终被激活

## 4. 实验结果

### 4.1 图像生成 (ImageNet)

| 模型 | 方法 | FLOPs | 速度提升 | FID |
|---|---|---|---|---|
| DiT-XL | 基线 | 118.64G | 1.00× | 2.27 |
| DiT-XL | Magnitude Pruning | ~95G | ~1.25× | ~2.4 |
| DiT-XL | ToMe (20% merge) | ~95G | ~1.20× | ~14.7 |
| DiT-XL | **DyDiT-S** | ~58G (**-51%**) | **1.73×** | 2.07 |
| DiT-XL | **DyDiT-L** | ~37G (**-69%**) | **2.73×** | 2.88 |

### 4.2 视频生成 (Latte)

- FLOPs 减少 50%+
- 速度提升 1.77×

### 4.3 Flow Matching 扩展

- DyDiT 可扩展到 Flow Matching 范式（SD3, FLUX）
- 同样取得显著加速效果

## 5. 创新点总结

1. **首次在 DiT 中引入时间步和空间维度的动态推理机制**
2. **TDW**：根据时间步动态调整模型宽度（head/channel 级别），而非静态结构
3. **SDT**：根据 token 难度跳过 MLP 计算，实现 token 级别的动态推理
4. **FLOPs-aware 训练**：通过可学习的 Router 实现端到端的计算量控制
5. **预计算 mask**：所有路由决策仅依赖时间步 embedding，可离线预计算，不影响推理效率

## 6. 局限性

- 动态路由增加了少量参数和计算开销（Router 约 3.57% 参数增量）
- 预训练的 DyDiT 需要针对不同推理步数重新训练
- SDT 仅应用于 MLP 块，MHSA 仍然是全 token 计算

## 7. 与其他方法的对比特点

| 维度 | ToMeSD | SparseDiT | **DyDiT** |
|---|---|---|---|
| 压缩粒度 | Token (merge) | Token (sparse-dense) | Head + Channel + Token |
| 时间步动态 | 静态/简单线性 | 动态剪枝率 | 动态宽度 (TDW) |
| 空间动态 | 静态 | 分层固定 | 动态 token 路由 (SDT) |
| 训练需求 | 无训练 | 微调 | 微调 + Router 训练 |
| 核心创新 | Token Merging | Sparse-Dense Token | 动态宽度 + Token 跳过 |
