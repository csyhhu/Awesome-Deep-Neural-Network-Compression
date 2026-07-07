# PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis

- **论文链接**: [https://arxiv.org/abs/2310.00426](https://arxiv.org/abs/2310.00426)
- **发表会议**: ICLR 2024
- **作者单位**: 华为诺亚方舟实验室、大连理工大学、香港大学、香港科技大学
- **核心贡献**: 提出 PixArt-α，一个基于 Diffusion Transformer (DiT) 的文生图模型，以极低的训练成本（仅为 SDv1.5 的 12% 训练时长，RAPHAEL 的 1%）达到接近商业标准的图像生成质量。

---

## 1. 研究动机

当前最先进的文生图（T2I）模型（如 DALL·E 2、Imagen、Stable Diffusion、RAPHAEL）虽然生成质量出色，但训练成本极其高昂：
- **SDv1.5**: ~6,250 A100 GPU days，约 \$320,000
- **RAPHAEL**: ~60,000 A100 GPU days，约 \$3,080,000，产出 35 吨 CO₂ 排放

高昂的成本严重阻碍了 AIGC 领域的创新。本文提出核心问题：**能否以可负担的资源开发高质量的图像生成器？**

---

## 2. 核心方法

PixArt-α 通过三大核心设计实现高效训练：

### 2.1 训练策略分解（Training Strategy Decomposition）

将复杂的 T2I 任务解耦为三个递进阶段：

| 阶段 | 目标 | 数据/方法 |
|------|------|-----------|
| **Stage 1: 像素依赖学习** | 学习自然图像的像素分布 | 使用低成本类别条件模型（ImageNet 预训练的 DiT）初始化 |
| **Stage 2: 图文对齐学习** | 学习文本与图像的精确对齐 | 在高质量、高概念密度的图文对数据上预训练 |
| **Stage 3: 高分辨率美学优化** | 提升图像美学质量和高分辨率生成 | 在高质量美学数据上微调（JourneyDB + 内部数据集） |

这种分阶段策略使得每个子任务都能高效学习，显著降低了整体训练难度。

### 2.2 高效 T2I Transformer 架构

基于 DiT 架构，提出三项关键改进：

- **Cross-Attention 层**：在 self-attention 和 feed-forward 之间插入多头交叉注意力层，用于注入文本条件。输出投影层初始化为零，以保持与预训练权重的兼容性。

- **adaLN-single**：原始 DiT 中，每层都有独立的 MLP 将类别条件 + 时间嵌入映射为 adaLN 的 scale/shift 参数，这部分占了总参数的 27%。作者提出 adaLN-single，**仅在第一个 block 中计算全局的 scale/shift 参数** \(\overline{S} = f(t)\)，然后在每层加上一个可学习的层特定嵌入 \(E^{(i)}\)：\(S^{(i)} = g(\overline{S}, E^{(i)})\)。该设计减少了 **26% 参数量**（833M → 611M）和 **21% GPU 显存**（29GB → 23GB）。

- **Re-parameterization**：为了兼容 ImageNet 预训练权重，所有 \(E^{(i)}\) 被初始化为使模型等效于原始 DiT 的值。这使得 T2I 模型可以直接加载类别条件模型的参数，加速收敛。

### 2.3 高信息密度数据构建

研究发现 LAION 数据集存在严重问题：
- 文本描述信息不完整（平均每图仅 6.4 个名词）
- 长尾效应严重（大量名词出现频率极低）
- 图文不对齐、质量低

**解决方案：自动标注管线**
1. 使用 VLM 模型 **LLaVA** 对图像自动生成详细描述（Prompt: "Describe this image and its style in a very detailed manner"）
2. 选择 **SAM 数据集**（分割任务数据集，图像包含丰富的多样化物体）替代 LAION
3. 结果：**SAM-LLaVA 平均每图含 29.3 个名词**（LAION 的 4.6 倍），有效名词比例从 8.5% 提升到 18.6%

---

## 3. 实验结果

### 3.1 训练效率

| 指标 | PixArt-α | SDv1.5 | RAPHAEL |
|------|----------|--------|---------|
| 参数量 | **0.6B** | 0.9B | 3.0B |
| 训练图像数 | **25M** | 2000M | 5000M+ |
| GPU days | **753 A100** | 6,250 A100 | 60,000 A100 |
| 训练成本 | **\$28,400** | \$320,000 | \$3,080,000 |
| FID-30K (MSCOCO) | **7.32** | 9.62 | 6.61 |

PixArt-α 仅使用 **1.25%** 的训练数据和 **12%** 的训练时间，节省约 \$300K 并减少 90% CO₂ 排放。

### 3.2 图像质量评估

- **FID-30K**: 7.32（零样本 COCO），在同等资源量级下表现最优
- **用户研究**：在 300 个 prompt 的图像对比中，PixArt-α 相比 SDv2 在图像质量上提升 7.2%，在对齐精度上大幅提升 42.4%

### 3.3 Compositional Generation（T2I-CompBench）

PixArt-α 在 6 项评测指标中的 5 项达到最佳：

| 指标 | PixArt-α | SDXL |
|------|----------|------|
| Color | **0.6886** | 0.6369 |
| Shape | **0.5582** | 0.5408 |
| Texture | **0.7044** | 0.5637 |
| Spatial | **0.2082** | 0.2032 |
| Non-Spatial | 0.3179 | 0.3110 |
| Complex | **0.4117** | 0.4091 |

### 3.4 消融实验

- **w/o re-param**（从头训练）：生成图像严重失真，缺乏关键细节
- **adaLN**：FID 与 adaLN-single 相近，但参数多 26%，显存多 21%
- **adaLN-single-L**（更长训练）：最终选用方案，在参数效率和生成质量间取得最佳平衡

---

## 4. 扩展应用

- **DreamBooth**：无需修改即可应用到 PixArt-α，仅需 300 步微调即可实现高质量个性化生成
- **ControlNet**：冻结 DiT Block，创建可训练副本，支持 HED 边缘信号控制，20K 步训练即可生成高质量条件图像

---

## 5. 局限性与未来工作

- 对目标数量控制精度不足
- 细节处理（如人手特征）存在缺陷
- 文字生成能力较弱（训练数据中字体/字母相关图像较少）
- 计划未来探索 scaling up 进一步提升性能

---

## 6. 总结

PixArt-α 是首个以极低成本实现接近商业级文生图质量的 Diffusion Transformer 模型。其核心洞察在于：
1. **解耦训练**：将 T2I 拆分为三个阶段递进训练，大幅降低难度
2. **架构优化**：adaLN-single + Cross-Attention + Re-parameterization 实现参数高效
3. **数据质量优先于数量**：通过 VLM 自动标注构建高概念密度的图文对，远比盲目堆数据量更有效

该工作为个人研究者和小型创业团队训练自己的高质量文生图模型提供了可行的低成本方案。

---

## 7. 讨论：PixArt-α 与 Stable Diffusion 架构对比

> **Q: 本文的架构和 SD 的架构有什么区别？**

PixArt-α 和 Stable Diffusion 虽然共享 **Latent Diffusion 框架**（都使用预训练 VAE 在潜空间做扩散），但在网络骨架和条件注入机制上有本质区别。

### 核心区别一览

| 维度 | **Stable Diffusion** | **PixArt-α** |
|------|---------------------|--------------|
| **网络骨架** | 卷积 U-Net（CNN-based） | Diffusion Transformer（纯 Transformer，DiT-XL/2，28 层） |
| **文本编码器** | CLIP（~123M） | T5-XXL（4.3B Flan-T5） |
| **文本 Token 数** | 标准 77 tokens | 120 tokens（适配更密集的描述） |
| **文本注入方式** | U-Net 内部的 Cross-Attention 层 | Transformer block 中的 Cross-Attention（位于 Self-Attn 和 FFN 之间） |
| **时间步条件注入** | U-Net ResBlock 中的时间嵌入 | **adaLN-single**（全局 MLP + 层特定可学习嵌入） |
| **参数量** | ~0.9B（SDv1.5） | **~0.6B** |
| **训练数据量** | 2B 张图 | **25M 张图** |
| **训练时长** | 6,250 A100 GPU days | **753 A100 GPU days** |

### 深入分析

**1. U-Net vs Transformer：最本质的分野**

SD 使用卷积 U-Net，依赖局部卷积感受野逐层扩大来建模图像结构，这是 CNN 的固有范式。PixArt-α 使用 DiT，将图像 patchify 后作为 token 序列送入 Transformer，通过 Self-Attention 的全局感受野直接建模像素间的长程依赖。

论文附录中专门讨论了这一点（Section "Discussion on Transformer vs U-Net"），指出 Transformer 的 multi-head attention 在建模**组合语义信息**上天然优于 U-Net，这也是 PixArt-α 在 T2I-CompBench 的组合性指标上全面超越所有 U-Net 模型的原因。

**2. 文本编码器：T5-XXL vs CLIP**

PixArt-α 使用 4.3B 参数的 T5-XXL（跟随 Imagen 的选择），远大于 SD 的 CLIP。更强的文本编码器意味着更丰富的语义表示，这也是它能用更少数据达到更好对齐的原因之一。此外，由于 PixArt-α 的文本描述更加密集，token 长度也扩展到了 120（SD 为 77）。

**3. 条件注入机制的本质差异**

- **SD 的 U-Net**：Cross-Attention 嵌入在 U-Net 的多个分辨率层级中，text embedding 在不同尺度上与图像特征交互。时间嵌入则注入到每个 ResBlock 中。
- **PixArt-α 的 DiT**：每个 Transformer block 都包含 Cross-Attention（位于 Self-Attention 和 FFN 之间），在统一的 token 空间中持续交互文本和图像信息。时间步条件通过 adaLN-single 注入，而非 U-Net 中嵌入到 ResBlock 的方式。

**4. adaLN-single：PixArt-α 独有的效率设计**

这是 PixArt-α 最精巧的架构设计。原始 DiT 每层有独立 MLP 将 (class + time) 映射为 adaLN 的 scale/shift 参数，占 27% 参数量。PixArt-α 将其替换为：
- 仅在第一个 block 用全局 MLP 计算基础 scale/shift：\(\overline{S} = f(t)\)
- 每个 block 加一个可学习的小嵌入 \(E^{(i)}\) 做调整：\(S^{(i)} = \overline{S} + E^{(i)}\)

这节省了 **26% 参数量**（833M → 611M）和 **21% 显存**（29GB → 23GB），且生成质量几乎无损。配合 Re-parameterization 技巧，还能直接加载 ImageNet 预训练的 DiT 权重，加速 Stage 1 的收敛。

### 总结

两者的根本差异在于：**SD 是"在 U-Net 里嵌入 Cross-Attention 做条件控制"，PixArt-α 是"在 Transformer 里内置 Cross-Attention 做多模态融合"**。Transformer 天然的多模态融合能力和全局注意力机制，加上 T5-XXL 更强的语义理解以及 adaLN-single 的参数高效设计，使得 PixArt-α 能在远少于 SD 的数据和训练时间下达到更好的组合生成能力。
