# SparseDiT: Token Sparsification for Efficient Diffusion Transformer

> **Paper**: [arXiv:2412.06028v2](https://arxiv.org/abs/2412.06028v2)
> **Authors**: Shuning Chang, Pichao Wang, Jiasheng Tang, Fan Wang, Yi Yang (Zhejiang University & Alibaba Group)
> **Conference**: NeurIPS 2025
> **Tag**: `sparse_dit_token_sparsification`

---

## 1. 核心洞察 (Core Insights)

通过分析 DiT 模型中不同层的注意力图（attention map），作者发现了三个关键观察：

1. **底层 self-attention 趋于均匀分布**：浅层的注意力权重接近均匀分布，类似于全局平均池化，说明这些层主要捕获全局特征，复杂的 self-attention 计算在此处贡献有限。

2. **中间层交替捕获全局与局部信息**：DiT 架构在不同层之间交替进行全局特征提取和局部细节增强，这一模式在所有采样步骤中保持一致。

3. **去噪过程中局部信息需求递增**：随着去噪步骤推进，注意力方差增大，模型越来越关注局部细节，token 数量需求逐步增加。

## 2. SparseDiT 架构设计

SparseDiT 采用**三段式架构**，将 Transformer 层分为底部、中部、顶部三个部分：

### 2.1 底部层 — Poolingformer

- 用 **Poolingformer** 替代原始 self-attention
- 移除 Query 和 Key 的计算，仅对 Value 做全局平均池化：
  ```
  X = X + V̄  (V̄ 是 V 在 token 维度的均值，重复 N 次后与输入相加)
  ```
- 实验验证：将前两层 attention map 替换为全 1 矩阵，生成图像质量几乎不变
- 注：底层仍保留完整 token 数量，不做稀疏化（训练不稳定）

### 2.2 中间层 — Sparse-Dense Token Module (SDTM)

这是 SparseDiT 的核心模块，通过**交替使用稀疏 token 和稠密 token** 来平衡全局结构信息与局部细节：

**稀疏 Token 生成：**
1. 将稠密 token `X ∈ ℝ^(N×C)` reshape 为 `H × W × C`
2. 通过空间池化下采样到 `H' × W' × C`，得到 `M` 个稀疏 token（通常 `M ≪ N`）
3. 稀疏 token 与全量稠密 token 通过 cross-attention 交互：
   ```
   X_s = X_s + MHA(X_s, X, X)  # Q=稀疏token, K/V=稠密token
   ```

**稀疏 Token 处理：**
- 稀疏 token 经过多个 Transformer 层（Sparse Transformer）处理

**稠密 Token 恢复：**
1. 将稀疏 token 上采样回原始分辨率
2. 与原始稠密 token 通过线性层融合：
   ```
   X_merged = UpSample(X_s) · W1 + X · W2
   ```
3. 再通过 cross-attention 恢复稠密表示：
   ```
   X = X_merged + MHA(X_merged, X_s, X_s)  # Q=融合token, K/V=稀疏token
   ```
4. 稠密 Token 经过少量 Dense Transformer 层增强局部细节

**SDTM 级联：**多个 SDTM 级联，反复在稀疏/稠密 token 间转换，实现全局与局部信息的交替处理。

### 2.3 顶部层 — Dense Transformer

- 标准 Transformer 结构，处理全量稠密 token
- 专注于高频细节的最终精炼

### 2.4 配置示例 (DiT-XL)

| 段 | 层数 | 具体配置 |
|---|---|---|
| 底部 | 2 | Poolingformer |
| 中部 | 24 | 4 个 SDTM：(1 稀疏生成 + 3 稀疏 + 1 稠密恢复 + 1 稠密) × 4 |
| 顶部 | 2 | 标准 Transformer |

## 3. 时间步维度的动态剪枝策略 (Timestep-wise Pruning Rate)

观察发现：去噪早期主要生成低频全局结构，后期则需要更多高频细节。因此：

- **去噪早期**（前 T/4 步）：使用固定的高剪枝率 `r_min`，token 数量少
- **去噪后期**（后 3T/4 步）：剪枝率线性递减，逐步增加 token 数量

公式：
```
r = r_min                          (t_i < T/4)
r = f(r_min, r_max, t_i)           (T/4 ≤ t_i < T)
```

训练时采用分段函数，解决 batch 训练与随机采样的矛盾。

## 4. 实验结果

### 4.1 类别条件图像生成 (ImageNet)

| 模型 | 分辨率 | FLOPs | 速度提升 | FID |
|---|---|---|---|---|
| DiT-XL | 512×512 | 525G | baseline | 3.04 |
| **SparseDiT-XL** (r∈[0.61,0.86]) | 512×512 | 286G (**-46%**) | **+145%** | 2.96 |
| **SparseDiT-XL** (r∈[0.90,0.96]) | 512×512 | 235G (**-55%**) | **+175%** | 3.13 |

### 4.2 视频生成 (Latte-XL)

- FLOPs 减少 **56%**，速度提升 **111%**
- FVD 分数保持竞争力

### 4.3 文生图 (PixArt-α)

- FLOPs 减少 **38%**，速度提升 **69%**
- FID 从 4.53 降至 **4.29**（略有改善）

### 4.4 与现有方法对比

| 方法 | FLOPs 减少 | 速度提升 | FID 变化 |
|---|---|---|---|
| ToMeSD | - | +66% | +12.5 (严重退化) |
| DyDiT | -29% | +31% | +0.16 |
| TokenCache | -39% | +32% | +0.13 |
| Ditfastattn | - | +98% | +1.36 |
| **SparseDiT (Ours)** | **-46%** | **+145%** | **-0.08** |

### 4.5 与高效采样器的结合

SparseDiT 可与 DDIM、Rectified Flow 等高效采样器无缝结合：
- 25 步 DDIM：约 **18.7×** 推理速度提升
- 5 步 RFlow：约 **93.4×** 推理速度提升

## 5. 消融研究

- **SDTM 数量**：4 个 SDTM 最优（FID 2.38），1 个时训练崩溃（NAN）
- **Poolingformer 数量**：2 个最优；0 个训练不稳定；3 个性能下降
- **时间步剪枝策略**：动态调整优于固定 token 数量；后期 token 数量对最终结果影响最大
- **可视化**：SparseDiT 生成的图像在细节精度上甚至优于基线（如金毛猎犬的鼻子、金刚鹦鹉的眼睛）

## 6. 创新点总结

1. **首次针对 DiT 架构特点设计的 token 稀疏化方法**：不像 ToMeSD 那样通用地在每层做 token merging，而是基于对 DiT 内部全局/局部特征交替的深入分析来设计
2. **空间维度三段式自适应架构**：Poolingformer（底部）→ SDTM 交替（中部）→ Dense（顶部）
3. **时间步维度动态剪枝**：根据去噪阶段自适应调整 token 密度
4. **Sparse-Dense Token Module (SDTM)**：稀疏 token 捕获全局结构，稠密 token 保留局部细节，交替处理
5. **跨任务通用性**：在图像生成、视频生成、文生图三个任务上均验证有效

## 7. 局限性

- 模型结构需要**手动预定义**（每层的 token 数量、各模块的层数等）
- 每层中稀疏 token 的数量是固定预设的，缺乏自适应调整能力

## 8. 个人思考

### 与其他方法的对比

- **vs ToMeSD**：ToMeSD 在每层做 token merging/recovery，缺乏对 DiT 架构特性的理解，性能退化严重。SparseDiT 的分段设计更符合 DiT 的内在规律。
- **vs DyDiT**：DyDiT 在 token/head/channel 三个维度做动态剪枝，SparseDiT 仅关注 token 维度但取得了更好的效率-质量权衡。
- **vs VSA/USV**：这些方法在视频生成中做空间-时间维度的 token 稀疏化，SparseDiT 的思想可以扩展到视频域（论文中已在 Latte 上验证）。

### 可扩展方向

1. **自动搜索最优结构**：用 NAS 方法自动确定各段的层数和 token 数量
2. **与 KV Cache 结合**：在推理缓存场景下进一步优化
3. **扩展到更多 DiT 变体**：如 Flux、SVD 等最新架构
4. **统一框架**：将时间步剪枝率也作为可学习参数
