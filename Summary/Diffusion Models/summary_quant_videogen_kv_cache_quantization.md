# Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization

**论文信息**: Haocheng Xi, Shuo Yang, Yilong Zhao, Muyang Li, Han Cai, et al. (UC Berkeley, NVIDIA, MIT, Amazon, UT Austin), ICML 2026

**资源**: [Website](https://svg-project.github.io/qvg) | [GitHub](https://github.com/svg-project/Quant-VideoGen)

---

## 1. 研究动机与问题

自回归（Auto-Regressive, AR）视频扩散模型（如 CausVid、Self-Forcing、LongCat-Video、HY-WorldPlay）通过因果注意力机制实现了流式、交互式长视频生成。然而其核心瓶颈在于 **KV-Cache 内存**：

- KV-Cache 随历史帧线性增长，以 LongCat-Video 为例，生成 5 秒 480p 视频需约 38K tokens，KV-Cache 占约 **34 GB**，已超过单张 RTX 5090（32 GB）的内存容量。
- 内存限制不仅影响部署，更直接影响**长时一致性**：KV-Cache 截断（滑动窗口）会导致身份、布局、运动语义的累积漂移。

现有 LLM 领域的 KV-Cache 量化方法（KIVI、KVQuant、QuaRot）直接迁移到视频扩散模型会产生严重质量退化，原因在于视频模型的 **激活分布高度异质**：不同 token、不同 channel 的数值范围差异极大（Key cache 最大值 ~1e2，Value cache 最大值 ~1e3），且 channel 维度的异常值模式在 token 间不一致。

## 2. 核心方法

作者提出 **Quant VideoGen (QVG)**，一个无需训练的 KV-Cache 量化框架，包含两个核心组件：

### 2.1 Semantic-Aware Smoothing (SAS)

基于视频 token 的时空冗余性，SAS 通过以下步骤使 KV-Cache 分布更适合作量化：

1. **语义分组（Semantic-based Grouping）**：对每个 chunk 的 N 个 token，使用 k-means 算法按其隐空间表示划分为 C 个组，同组 token 具有相似的语义。
2. **残差计算（Centroid Subtraction）**：每组 token 减去其质心（centroid），得到残差张量 R。由于 k-means 将大值聚类到质心中，残差张量的数值范围大幅缩小，量化误差降低。

实验表明，SAS 将 Key Cache 量化误差降低约 **6.9 倍**，Value Cache 量化误差降低约 **2.6 倍**。

### 2.2 Progressive Residual Quantization (PRQ)

受视频的渐进式编码结构启发，PRQ 以**由粗到细**的方式多阶段量化残差：

- 第一阶段对原始 KV-Cache 做 SAS → 量化，得到残差 R^(1)。
- 第二阶段对 R^(1) 再次做 SAS → 量化，捕捉更细粒度的信息。
- 重复 T 轮（T=1 或 4），最终量化最后的残差 R^(T)。

重建时反向操作：从量化的 R^(T) 开始，逐阶段加回各层的质心。多阶段设计提供了灵活的质量-压缩率权衡。

### 2.3 算法-系统协同设计

- **流式质心缓存**：利用相邻 chunk 的时空局部性，用上一 chunk 的分配策略初始化当前 k-means，将 k-means 开销降低 **3 倍**。
- **融合去量化 Kernel**：在单个 CUDA kernel 中完成去量化 + 质心回加，中间结果存于寄存器减少显存访问。
- **Pre-RoPE Key Caching**：在 RoPE 之前存储 Key Cache，避免旋转操作破坏 token 间的语义相似性。
- 缩放因子用 FP8 E4M3 存储以降低元数据开销。

## 3. 实验与结果

### 3.1 实验设置

- **模型**: LongCat-Video-13B, HY-WorldPlay-8B, Self-Forcing-Wan-1.3B
- **分辨率**: 480p
- **硬件**: NVIDIA H100 GPU (CUDA 12.8)
- **配置**: QVG（S=1, B=64, 更激进压缩）; QVG-Pro（S=4, B=16, 更高质量）
- **Baselines**: RTN, KIVI, QuaRot
- **中心数**: K=256（索引用 uint8 存储）

### 3.2 主要结果

**LongCat-Video-13B (INT2, 480p)**:
| 方法 | 压缩比 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|--------|-------|-------|--------|
| RTN | 6.40× | 20.87 | 0.719 | 0.203 |
| KIVI | 6.40× | 20.32 | 0.719 | 0.208 |
| QuaRot | 6.40× | 21.57 | 0.759 | 0.171 |
| **QVG-Pro** | 4.97× | **30.38** | **0.935** | **0.048** |
| **QVG** | **6.94×** | 28.72 | 0.909 | 0.065 |

**HY-WorldPlay-8B (INT2, 480p)**:
| 方法 | 压缩比 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|--------|-------|-------|--------|
| RTN | 6.40× | 24.20 | 0.696 | 0.229 |
| KIVI | 6.40× | 24.27 | 0.701 | 0.230 |
| QuaRot | 6.40× | 25.21 | 0.738 | 0.205 |
| **QVG-Pro** | 5.20× | **31.56** | **0.923** | **0.069** |
| **QVG** | **7.05×** | 29.17 | 0.882 | 0.094 |

- QVG 在 INT4 设置下进一步达到 PSNR 35+，接近无损。
- Self-Forcing 模型上，QVG 在 700 帧的长时生成中保持近乎无损的画质，而其他 baseline 在 100 帧后急剧退化。

### 3.3 效率

- **端到端延迟开销**: LongCat +2.1%, HY-World +1.5%, Self-Forcing +4.3%
- **内存构成**: 量化值占 >65%，assignment vector 和 centroids 的开销较低
- **首次实现**: HY-WorldPlay-8B 可在单张 RTX 4090 上运行（PSNR > 29 vs BF16）

### 3.4 消融实验

- PRQ 阶段数：第一阶段贡献最显著的 MSE 降低（5.83×），后续阶段贡献递减但仍有增益
- 量化 block size：B=64 压缩比最优，B=16 质量最优

## 4. 核心洞察

1. **视频 KV-Cache 具有强时空冗余**：相邻帧的同一空间位置的 token 高度相似，相邻空间位置的 token 余弦相似度高。
2. **直接量化不可行**：视频模型的激活分布比 LLM 更异质，需要专门设计。
3. **SAS 是关键**：k-means 分组 + 质心减法将大值分离，使残差分布更集中、更适合作低比特量化。
4. **渐进式设计有效**：由粗到细的多阶段量化天然匹配视频的渐进式编码结构。

## 5. 局限性与展望

- 方法专门针对 AR 视频扩散模型，可能不适用于双向注意力模型。
- k-means 虽然经过优化，但在极低延迟场景下仍有一定开销。
- 与稀疏化方法（token eviction）正交，可组合使用。

---

> **总结**: QVG 通过利用视频的时空冗余性（SAS 语义感知平滑 + PRQ 渐进残差量化），实现了训练无关的 2-bit KV-Cache 量化，在 AR 视频生成模型中实现最高 7.05× 压缩比、<4% 延迟开销且近乎无损的画质，首次使 8B 参数的世界模型可在消费级 GPU 上运行。
