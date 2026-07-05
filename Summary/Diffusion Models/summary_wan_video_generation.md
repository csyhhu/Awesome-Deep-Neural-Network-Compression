# Wan: Open and Advanced Large-Scale Video Generative Models

- **论文地址**: https://arxiv.org/abs/2503.20314
- **作者**: Wan Team, Alibaba Group（阿里巴巴通义万相团队）
- **发表时间**: 2025年3月
- **代码仓库**: https://github.com/Wan-Video/Wan2.1

---

## 1. 论文概述

本文介绍了 **Wan（通义万相）**，一个全面且开源的大规模视频基础模型套件。Wan 基于主流的 Diffusion Transformer (DiT) 范式构建，通过一系列技术创新在视频生成能力上取得了显著进步，包括：新颖的时空变分自编码器 (VAE)、可扩展的预训练策略、大规模数据整理以及自动化评估指标。

Wan 具有四大核心特点：
- **领先性能**：14B 模型在数十亿图像和视频上训练，展示了视频生成的 scaling laws，在多个内部和外部基准上持续超越现有开源模型和商业方案
- **全面性**：提供 1.3B 和 14B 两个模型，涵盖图像到视频、视频编辑、个性化视频生成等 **8 个下游任务**，且是首个支持中英文视觉文本生成的模型
- **消费级效率**：1.3B 模型仅需 8.19 GB 显存，可在消费级 GPU 上运行
- **开放性**：完全开源，包括源代码和所有模型权重

---

## 2. 核心方法

### 2.1 模型架构

Wan 的核心架构由三部分组成：

```
Wan-VAE → 视频压缩到潜在空间 → DiT (Diffusion Transformer) → 生成视频
                                ↑
                          umT5 文本编码器
```

#### 2.1.1 Wan-VAE（时空变分自编码器）
- 采用 **3D 因果 VAE** 架构，对视频进行 **4×8×8 倍**时空压缩（时间 4 倍，空间 8×8 倍）
- 潜在通道数 $C=16$，模型参数量仅 127M
- 将所有 GroupNorm 替换为 RMSNorm 以保持时间因果关系
- 空间上采样层将输入特征通道减半，推理内存减少 33%
- 三阶段训练：2D 图像 VAE → 3D 膨胀（低分辨率小帧数）→ 高质量微调（引入 3D GAN loss）
- **特征缓存机制**：支持任意长度视频的编解码，通过 chunk-wise 策略将每块帧数限制在 4 帧以内，维护历史帧的特征缓存保证时间连续性
- 视频重建速度是 HunYuan Video VAE 的 **2.5 倍**

#### 2.1.2 视频 Diffusion Transformer
- 采用 patchify 模块（3D 卷积核 $(1,2,2)$）将潜在特征展平为序列
- 使用**交叉注意力**嵌入文本条件，确保长上下文下的指令遵循能力
- **共享 AdaLN 设计**：MLP 在所有 Transformer 块之间共享，每块学习独立的偏置集，参数量减少约 25%，同时性能更优
- 消融实验表明：增加网络深度比增加 AdaLN 参数量更有效
- 采用全时空注意力机制，有效建模时空上下文关系

#### 2.1.3 文本编码器
- 选择 **umT5** 作为文本编码器，优势在于：
  - 多语言编码能力强，支持中英文
  - 双向注意力机制，优于单向 LLM
  - 在相同参数量下收敛更快
- 消融实验对比了 umT5、Qwen2.5-7B 和 GLM-4-9B，umT5 表现最优

### 2.2 训练策略

#### 2.2.1 Flow Matching 框架
- 基于 Rectified Flows，训练目标为预测速度场 $v_t = x_1 - x_0$
- 中间潜在变量：$x_t = t x_1 + (1-t)x_0$
- 损失函数为预测速度与真实速度的 MSE

#### 2.2.2 预训练流程
1. **图像预训练**：低分辨率（256px）文本到图像预训练，建立跨模态语义对齐
2. **图像-视频联合训练**：三阶段分辨率渐进课程
   - 阶段一：256px 图像 + 192px/5秒视频（16fps）
   - 阶段二：升级到 480px 分辨率
   - 阶段三：升级到 720px 分辨率

#### 2.2.3 后训练
- 使用高质量后训练数据（图像和视频），保持相同架构和优化器配置
- 专注于提升视觉保真度和运动质量

### 2.3 大规模训练优化

#### 并行策略
- **FSDP**（全分片数据并行）作为参数分片策略
- **2D Context Parallelism**：外层 Ring Attention + 内层 Ulysses，在 256K 序列长度、16 GPU 跨 2 台机器场景下，通信开销从 Ulysses 的超过 10% 降到 1% 以下
- VAE 和 Text Encoder 使用 DP，DiT 使用 DP + CP 组合

#### 内存优化
- **激活卸载（Activation Offloading）**：将部分激活卸载到 CPU，与计算重叠
- 结合梯度检查点策略，优先对 GPU 内存/计算比高的层使用检查点

### 2.4 推理优化

- **扩散缓存（Diffusion Cache）**：
  - 注意力缓存：利用相邻采样步之间注意力输出的相似性，每几步计算一次并缓存复用
  - CFG 缓存：在采样后期，条件与无条件输出存在相似性，减少无条件前向传播
  - 14B T2V 模型推理速度提升 **1.62×**

- **量化**：
  - FP8 GEMM：对所有权重使用 per-tensor 量化，对激活使用 per-token 量化，DiT 模块加速 1.13×
  - 8-Bit FlashAttention：混合 Int8/FP8 量化策略（S 用 Int8，O 用 FP8）+ FP32 跨块累积，在 H20 GPU 上达到 95% MFU，推理效率提升 1.27×

- **并行推理**：结合 2D Context Parallel 和 FSDP，14B 模型在多 GPU 上实现近线性加速

### 2.5 Prompt 对齐

- 为每个视频/图像生成多种风格和长度的标注，增加文本-视频映射多样性
- 使用 LLM（Qwen2.5-Plus）重写用户提示词，使其分布与训练标注对齐
- 重写原则：添加细节但不改变含义、加入自然运动属性、结构化为"风格+内容摘要+详细描述"

---

## 3. 数据工程

### 3.1 数据整理原则
高质量、高多样性、大规模

### 3.2 预训练数据处理
**四步清洗流程**：

1. **基础维度过滤**（过滤约 50% 数据）：
   - 文本检测（OCR 覆盖度）
   - 美学评估（LAION 分类器）
   - NSFW 安全评分
   - 水印和 Logo 检测
   - 黑边检测
   - 过曝检测
   - 合成图像检测（<10% 的污染即可显著降低性能）
   - 模糊检测
   - 时长和分辨率约束

2. **视觉质量评估**：聚类（100 个簇防止长尾数据丢失）+ 人工评分训练专家模型

3. **运动质量评估**（六个等级）：
   - 最优运动、中等质量运动、静态视频、相机驱动运动、低质量运动、抖动素材

4. **视觉文本数据**：
   - 合成分支：在白底上渲染中文字符
   - 真实分支：OCR 识别 + Qwen2-VL 生成自然描述
   - 融合合成和真实数据训练，实现中英文视觉文本生成

### 3.3 后训练数据
- 图像：专家模型 Top 20% + 人工筛选，补充缺失概念
- 视频：视觉质量 + 运动质量分类器筛选，涵盖 12 大类

### 3.4 密集视频标注（Dense Caption）
- 训练内部标注模型（LLaVA 式架构：ViT + Qwen LLM）
- 数据来源：开源视觉-语言数据集 + 自建数据集（名人/地标识别、物体计数、OCR、相机角度/运动、细粒度分类、空间关系、重标注、编辑指令等）
- 视频处理：3 帧/秒采样，上限 129 帧，slow-fast 编码策略
- 标注质量与 Google Gemini 1.5 Pro 整体相当

---

## 4. Wan-Bench 评估基准

提出了自动化、全面且与人类对齐的 **Wan-Bench** 评估框架，包括三大核心维度 14 个细粒度指标：

### 4.1 动态质量
- **大幅运动生成**：RAFT 光流法评估
- **人体伪影**：YOLOv3 检测 AI 生成伪影
- **物理合理性 & 平滑度**：Qwen2-VL 视频问答检测物理违规
- **像素级稳定性**：静态区域帧间差异
- **ID 一致性**：DINO 特征帧间相似度（人/动物/物体）

### 4.2 图像质量
- **综合图像质量**：MANIQA + LAION 美学预测器 + MUSIQ
- **场景生成质量**：CLIP 帧间一致性 + 文本对齐
- **风格化能力**：Qwen2-VL 帧级问答

### 4.3 指令遵循
- 单/多物体 & 空间位置准确性
- 相机控制（平移/升降/缩放/航拍/跟踪）
- 动作指令遵循

### 4.4 加权策略
收集 5000+ 人类偏好对比对，基于 Pearson 相关系数确定各维度权重

---

## 5. 实验结果

### 5.1 Wan-Bench 评测
14B 模型加权总分 **0.724**，超越 Sora（0.700）和所有商业/开源竞品

### 5.2 VBench 榜单
- Wan 14B：总分 **86.22%**（质量分 86.67%，语义分 84.44%），显著超越 Sora（84.28%）
- Wan 1.3B：总分 **83.96%**，超越 HunyuanVideo 和 Kling 1.0

### 5.3 人工评估
- T2V：Wan 14B 在视觉质量、运动质量、匹配度、总体排名上均大幅领先
- I2V：Wan 在视觉质量、运动质量、匹配度上全面领先

### 5.4 消融实验
- **AdaLN 共享**：全共享 + 增加深度优于部分共享 + 更多参数
- **文本编码器**：umT5 优于 Qwen2.5-7B 和 GLM-4-9B
- **VAE**：标准 VAE 优于 VAE-D（扩散损失替换重建损失）

---

## 6. 扩展应用

### 6.1 图像到视频（I2V）
- 通道拼接条件图像 + 二值掩码机制
- 统一框架支持 I2V、视频延续、首尾帧转换、随机帧插值
- CLIP 图像编码器 + MLP 提供全局上下文

### 6.2 视频编辑（VACE）
- 统一的视频条件单元（VCU）：文本 + 帧序列 + 掩码序列
- 概念解耦策略：将修改区域和保留区域分离处理
- 支持全量微调和 Context Adapter 两种模式
- 可实现长视频重渲染等复合任务

### 6.3 文本到图像（T2I）
- 图像-视频联合训练使模型同时具备出色的图像生成能力
- 图像数据量是视频数据的近 10 倍

### 6.4 视频个性化
- 直接在 Wan-VAE 潜在空间中条件化输入人脸图像（不使用特征提取器，避免信息损失）
- 自注意力范式：在时序轴上扩展 K 帧（人脸 + 掩码），以 inpainting 方式进行扩散
- 数据策略：人脸检测/分割 + ArcFace 相似度过滤 + Instant-ID 合成多样化人脸

### 6.5 相机运动控制
- 使用 Plücker 坐标编码外参和内参
- 相机姿态编码器 + 自适应归一化适配器注入 DiT
- 使用 VGG-SfM 提取训练数据的相机轨迹

### 6.6 实时视频生成（Streamer）
- **滑动时间窗口机制**：在去噪队列中维护固定长度窗口，逐步出队/入队实现无限长视频生成
- **一致性模型蒸馏（LCM/VideoLCM）**：将扩散过程蒸馏为 4 步一致性模型，推理加速 10-20×
- **量化部署**：int8（注意力层/线性头）+ TensorRT 量化，在 RTX 4090 上达到 8-20 FPS
- 使用 8 张 A100 可实时生成 15 分钟视频（8 FPS）

### 6.7 音频生成（V2A）
- 视频到音频生成（不含语音），包括环境音和背景音乐
- 1D-VAE 直接在原始波形上压缩，保留时间对齐
- CLIP 提取视觉特征 + umT5 文本编码，DiT 融合生成
- 支持文本提示控制声音风格

---

## 7. 局限性与未来工作

1. **大运动场景下的细节保真度**仍存在挑战
2. **计算成本**：14B 模型在单张高端 GPU 上推理需约 30 分钟
3. **领域专业性**不足：教育、医疗等特定场景表现待提升
4. 音频生成不支持人声（语音、笑声等）

---

## 8. 总结

Wan 是阿里巴巴推出的开源视频生成基础模型，核心贡献包括：
- 高效的 3D 因果 VAE（Wan-VAE），仅 127M 参数，支持任意长度视频
- 基于 DiT + Flow Matching 的架构设计，共享 AdaLN 降低参数
- 完整的数据工程管线（四步清洗 + 密集标注）
- 先进的分布式训练和推理优化（2D CP、扩散缓存、FP8/INT8 量化）
- 全面的 Wan-Bench 评估基准
- 涵盖 8 个下游任务的完整生态（T2V、I2V、V2V、个性化、相机控制、实时生成、音频生成）
- 在多个基准上超越 Sora 等商业模型，完全开源

---

## 9. 深入理解 Q&A

### Q1: 以一个具体的训练样本为例，描述整个训练过程

以一段 5 秒、720×720、16fps 的视频（"一位女性在樱花树下骑自行车"）为例：

#### 第一阶段：数据处理管线

**Step 1 — 基础维度过滤（九维检查）**：文本检测（OCR 无文字）、美学评分（4.3/5）、NSFW 安全、水印/Logo、黑边、过曝、合成图像检测、模糊检测、时长>4s 和分辨率 → 全部通过。此阶段淘汰约 50% 初始数据。

**Step 2 — 视觉质量评估**：数据聚类为 100 个簇（防止长尾数据丢失），人工标注训练专家模型打分 → 本样本 4.2/5。

**Step 3 — 运动质量评估**：分六个等级，本样本属"最优运动（Optimal motion）"：显著运动布局、透视、幅度，运动干净流畅。

**Step 4 — 密集描述生成**：原始标题 `"A woman riding a bicycle under cherry blossoms"` 过于简略，经过 LLaVA 式内部稠密描述模型（ViT + Qwen LLM）生成包含 10 个维度的详细描述（动作、相机角度、相机运动、物体类别/颜色/计数、OCR、场景、风格、事件），质量与 Gemini 1.5 Pro 相当。

#### 第二阶段：VAE 编码

采用 $1+T$ 帧格式（首帧单独处理，借鉴 MagViT-v2），81 帧 = 1 + 80：

$$V \in \mathbb{R}^{81 \times 720 \times 720 \times 3} \xrightarrow{\text{3D Causal VAE}} x_1 \in \mathbb{R}^{21 \times 90 \times 90 \times 16}$$

- 时间压缩 4×：$80/4=20$，加第一帧 → $1+20=21$
- 空间压缩 8×：$720/8=90$，通道扩展至 16
- VAE 仅 127M 参数，Feature Cache 分块处理避免显存溢出

#### 第三阶段：DiT 训练 — Flow Matching

**Step 5 — 文本编码**：稠密描述 → umT5 → $c_{txt} \in \mathbb{R}^{512 \times d}$

**Step 6 — Patchify**：VAE 潜变量 $x_1$ → 3D 卷积 (kernel=1×2×2) → Flatten → $L = 21 \times 45 \times 45 = 42{,}525$ tokens

**Step 7 — Flow Matching 核心步骤**：

1. 采样：$x_0 \sim \mathcal{N}(0,I)$，$t \sim \text{LogitNormal}$
2. 线性插值：$x_t = t \cdot x_1 + (1-t) \cdot x_0$
3. 真实速度：$v_t = x_1 - x_0$
4. 模型预测：$u(x_t, c_{txt}, t; \theta)$（每个 DiT Block 中 Cross-Attention 注入文本条件，Shared AdaLN 注入时间条件）
5. 损失：$\mathcal{L} = \|u(x_t, c_{txt}, t; \theta) - v_t\|^2$

#### 第四阶段：多阶段训练课程

| 训练阶段 | 图像 | 视频 | 本样本 |
|----------|------|------|--------|
| Image Pre-training | 256px | 无 | 不参与 |
| 阶段一 | 256px | 192px, 5s, 16fps | 降采样至 192px |
| 阶段二 | 480px | 480px, 5s | 升至 480px |
| 阶段三 | 720px | 720px, 5s | 全分辨率 |
| Post-training | 720px 精选 | 720px 精选 | 精选高质量版 |

**训练配置**：bf16 混合精度，AdamW（weight decay=$10^{-3}$），lr=$10^{-4}$（根据 FID/CLIP Score plateau 衰减），VAE 和 Text Encoder 冻结。

#### 第五阶段：分布式并行

128 GPU 配置下的 4D 并行：DP=4（外层），FSDP=32（参数分片），CP=16（2D Context Parallel：外层 Ring Attention + 内层 Ulysses）。VAE/Text Encoder 仅用 DP，DiT 用 DP+FSDP+CP。显存优化：Activation Offloading + Gradient Checkpointing。

---

### Q2: 以类似的例子描述推理阶段的完整流程

**用户 Prompt**：`"A woman riding a bicycle under cherry blossoms, cinematic shot"`

#### Step 1 — Prompt 改写（Qwen2.5-Plus）

用户 prompt 通常仅 10 个词，而训练 caption 是 200+ 词的稠密描述，存在分布不匹配。LLM 按三条原则改写：
1. 保持原意地添加细节
2. 注入自然运动属性
3. 与 post-training caption 结构对齐（风格→内容概要→细节）

改写为 ~200 词的纪实话语风格描述。

#### Step 2 — 文本编码

改写 prompt → umT5 → $c_{txt} \in \mathbb{R}^{512 \times d}$（条件嵌入）
空字符串 → umT5 → $c_{\emptyset}$（无条件嵌入，供 CFG 使用）

#### Step 3 — 初始化噪声

$\hat{z}_1 \sim \mathcal{N}(0,I) \in \mathbb{R}^{21 \times 90 \times 90 \times 16}$

#### Step 4 — 多步去噪（~50 步 ODE 求解）

每步执行：
1. **条件前向**：$\hat{z}_\tau + c_{txt} \to \text{DiT} \to u_{\text{cond}}$
2. **无条件前向**：$\hat{z}_\tau + c_{\emptyset} \to \text{DiT} \to u_{\text{uncond}}$
3. **CFG 融合**：$u_{\text{cfg}} = u_{\text{uncond}} + w \cdot (u_{\text{cond}} - u_{\text{uncond}})$，$w$ 通常 5-7
4. **ODE 步进**：$\hat{z}_{\tau-h} = \hat{z}_\tau - u_{\text{cfg}} \cdot h$，$h=1/50$

```
迭代示意（简化为 5 步）：
  Step 1:  τ=1.0    ẑ₁ = 纯噪声      → DiT → u → ẑ_{0.8}
  Step 2:  τ=0.8    ẑ_{0.8}          → DiT → u → ẑ_{0.6}
  Step 3:  τ=0.6    ẑ_{0.6}          → DiT → u → ẑ_{0.4}
  Step 4:  τ=0.4    ẑ_{0.4}          → DiT → u → ẑ_{0.2}
  Step 5:  τ=0.2    ẑ_{0.2}          → DiT → u → ẑ₀ ≈ 干净潜变量
```

#### Step 5 — 三项推理优化

| 优化 | 方法 | 加速比 |
|------|------|--------|
| Diffusion Cache | 注意力缓存（每隔几步计算一次注意力并复用）+ CFG 缓存（后期每隔几步计算无条件分支，加残差补偿） | **1.62×** |
| FP8 GEMM | 所有权重用 per-tensor 量化，激活用 per-token 量化 | **1.13×** |
| 8-Bit FlashAttention | $S=QK^T$ 用 INT8，$O=PV$ 用 FP8，块间 FP32 累加，H20 上达 95% MFU | **1.27×** |

累计加速：约 **2.32×**。多 GPU 2D Context Parallel + FSDP 实现近线性加速。

#### Step 6 — VAE 解码

$\hat{z}_0 \in \mathbb{R}^{21 \times 90 \times 90 \times 16} \xrightarrow{\text{3D Causal VAE Decoder + Feature Cache}} V \in \mathbb{R}^{81 \times 720 \times 720 \times 3}$

Feature Cache 每次只处理一个 latent chunk（≤4 帧），缓存历史帧特征保证时间连续性。

#### 推理成本

| 模型 | 条件 | 推理时间 |
|------|------|----------|
| 14B | 单张高端 GPU，无优化 | ~30 min |
| 14B | + Diffusion Cache + FP8 + 8-Bit FlashAttention | ~13 min |
| 14B | + 多 GPU 并行 | 近线性缩短 |
| 1.3B | 消费级 GPU（8.19GB） | 几分钟 |

---

### Q3: 3D Causal VAE 的 Encoder（训练用）和 Decoder（推理用）有什么区别？

两者是同一 VAE（127M 参数）的**对称两半**：

| 维度 | Encoder（训练用） | Decoder（推理用） |
|------|------------------|-------------------|
| **方向** | 像素空间 → 潜空间 | 潜空间 → 像素空间 |
| **维度变化** | $(1+T) \times H \times W \times 3 \to (1+T/4) \times H/8 \times W/8 \times 16$ | 反向 |
| **操作** | 下采样（strided causal conv） | 上采样（插值/转置卷积） |
| **何时运行** | 训练时每样本编码一次即冻结 | 推理时每个生成请求解码一次 |
| **因果依赖** | 未来帧不影响过去帧 | 生成帧只依赖历史帧 |

**架构差异**：
- Encoder 经过**两级压缩**：橙色（2× 时空联合压缩）→ 绿色（2 级 2× 纯空间压缩，合计 4×8×8 总压缩）
- Decoder 做完全对称的逆操作（先空间上采样 2 级，再时空上采样）
- Decoder 独有的优化：**空间上采样层输入通道减半**，推理显存减少 33%

**Feature Cache 两者均使用**：因果卷积 kernel=3，初始 chunk 用 2 个 dummy frame 零填充，后续 chunk 复用前一 chunk 最后 2 帧。

---

### Q4: DiT 为什么不直接在像素空间生成视频？

核心原因：**视频像素维度过大，注意力计算 O(s²) 不可承受**。

| 表示方式 | 维度 | Patchify 后 tokens |
|----------|------|-------------------|
| 像素空间 | $81 \times 720 \times 720 \times 3$ | ~**1.9M** tokens |
| VAE 潜空间 | $21 \times 90 \times 90 \times 16$ | ~**42K** tokens |

潜空间压缩 **46 倍**，token 数从 1.9M → 42K。

**三个致命问题**：
1. **计算成本**：注意力与 $s^2$ 成正比。论文指出序列长度达 1M 时注意力占 95% 训练时间。像素空间 token 数已达 1.9M。
2. **显存**：DiT 的 $\gamma > 60$，14B 模型在 1M token 下激活存储已达 8 TB，像素空间远超出任何硬件能力。
3. **训练稳定性**：长序列 → 低吞吐量 → 小 batch → 梯度方差尖峰 → 训练不稳定。这就是为什么 Wan 先用 256px 图像预训练建立语义对齐，再逐步引入视频。

---

### Q5: DiT 如何根据 prompt 生成视频？

DiT 不是"理解 prompt → 画出视频"，而是**学习了一个条件速度场**：

$$
\frac{dz}{d\tau} = u(z, c_{txt}, \tau; \theta)
$$

从 $\tau=1$（噪声）积分到 $\tau=0$（数据）。

**具体机制**：
- 每个 DiT Block 中，Cross-Attention 让**潜空间 token（Q）查询文本 token（K/V）**，找到每个空间位置对应的语义
- Shared AdaLN 根据时间 $\tau$ 自动切换行为模式：$\tau=1.0 \to 0.7$ 建立全局布局 → $0.7 \to 0.3$ 形成物体轮廓和运动方向 → $0.3 \to 0.0$ 细化纹理和细节
- CFG 将生成结果"强力推向"文本条件方向

**三阶段演化**：
- 前 10 步（高噪声）：潜空间中出现低频、粗粒度结构 — "粉色区域=樱花树"，"灰色区域=路面"
- 中间步（中噪声）：物体轮廓和运动轨迹成形 — 自行车运动模糊、女性人形轮廓
- 最后步（低噪声）：细节完善 — 面部表情、花瓣飘落路径、光影斑驳效果

---

### Q6: Flow Matching 框架能否不借助 VAE 直接在像素空间生成视频/图像？

**理论上可以，实际上不可行。**

Flow Matching 是一个通用的连续时间生成框架，数学上**不依赖 VAE**。训练和推理的公式在像素空间和潜空间完全一致：

- 像素空间：$x_1 \in \mathbb{R}^{81 \times 720 \times 720 \times 3} \approx$ 126M 维
- 潜空间：$x_1 \in \mathbb{R}^{21 \times 90 \times 90 \times 16} \approx$ 2.7M 维

**为什么像素空间不可行（三个致命问题）**：

1. **MSE Loss 导致模糊**：像素级 MSE 天然趋向回归均值。对于同一 prompt "一只猫"，训练集中有无数不同的猫 → 模型输出所有猫的平均像素 → 模糊。VAE 将像素的多模态分布映射为潜空间的近似单峰分布，解决此问题。

2. **维度灾难**：注意力计算 $O(s^2)$ 在 1.9M tokens 下不可行，模型也难以学习像素级变化规律（效率极低）。

3. **文本引导效率低**：像素空间中一个 token 代表 1-2 个像素，过于局部，难以通过 Cross-Attention 关联全局语义。潜空间中一个 token 代表 64 个像素的语义块，天然适合文本引导。

---

### Q7: DDPM 也需要 VAE 吗？

**取决于生成目标**：

| | 图像 256² | 图像 1024² | 视频 720p, 5s |
|--|----------|-----------|--------------|
| DDPM 像素空间 | ✅ 可行（原始论文证明） | ⚠️ 勉强 | ❌ 不可行 |
| DDPM 潜空间 | ✅ 更高效 | ✅ | ✅ |
| Flow Matching 潜空间 | ✅ 更高效 | ✅ | ✅ |

DDPM 的原始工作（Ho et al. 2020）确实在像素空间成功生成了 256×256 图像。但视频 126M 维 × 1000 步采样，计算完全不可行。

**关键差异**：DDPM 比 Flow Matching 更需要 VAE。因为 DDPM 的 1000 步采样意味着每一步的像素空间计算压力都被放大。即使 DDIM 加速到 50-100 步，每步的 token 数（1.9M）仍然是不可逾越的障碍。

---

### Q8: 所以结论是"视频生成需要 VAE，无论 DDPM 还是 FM"？

**完全正确**。核心逻辑链：

```
视频绝对维度太大（81帧×720×720 = 126M 像素值）
  → 所有扩散/流匹配框架在像素空间都无法承受 O(s²) 的注意力计算
  → 必须压缩到潜空间（Wan 压缩 46 倍，token 数从 ~1.9M → ~42K）
  → VAE 是实现这种压缩的唯一手段
  → 视频生成 → 需要 VAE，与框架选择无关
```

区别仅在于：DDPM 因步数更多（1000 vs 50），对 VAE 压缩的依赖比 Flow Matching **更强**。

---

### Q9: DiT 在视频生成中究竟起到什么作用？

**DiT 的作用：在潜空间中，根据 prompt 引导，从噪声生成出"好的"潜变量。**

三组件分工：

| 组件 | 做什么 | 在哪做 |
|------|--------|--------|
| **DiT** | 根据 prompt 从噪声生成潜变量（"导演"） | 潜空间（压缩 46×） |
| **VAE Decoder** | 把潜变量翻译成像素视频（"摄影师"） | 潜空间 → 像素空间 |
| **VAE Encoder** | 把像素视频压缩成潜变量（训练用） | 像素空间 → 潜空间 |

DiT 通过 Cross-Attention 在训练中自动学到文本语义到潜空间结构的映射：`"cherry blossoms" → 粉色色块分布`、`"riding bicycle" → 运动模糊模式`、`"tracking shot" → 全局平移运动`。推理时从纯噪声出发，每步根据当前状态+文本条件+时间步预测"往哪个方向走"，50 步后得到 VAE Decoder 可还原的干净潜变量。DiT 永远不碰像素，只管潜空间。

---

### Q10: 详细介绍 CFG（Classifier-Free Guidance）

#### 背景：为什么需要"引导"？

扩散模型学习的是数据分布 $p(\mathbf{x} \mid \mathbf{c})$，但直接用学到的分布采样，生成结果往往"不够好"——画面模糊、文本对齐弱。根本矛盾在于**高似然 vs 高保真**：模型如果能生成所有可能的"猫"，倾向于输出"各种猫的平均值"（safe but boring）；实际需要的是"最具猫特征的猫"。CFG 就是扩散模型的"截断"机制。

#### 前身：分类器引导（Classifier Guidance）

Dhariwal & Nichol (2021) 提出额外训练一个**噪声图像分类器** $p_\theta(\mathbf{c} \mid \mathbf{z}_\lambda)$，在采样每一步用分类器的梯度"推动"生成方向：

$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda, \mathbf{c}) - w \cdot \sigma_\lambda \nabla_{\mathbf{z}_\lambda} \log p_\theta(\mathbf{c} \mid \mathbf{z}_\lambda)$$

**问题**：需要额外训练分类器（成本高），噪声图像上训练分类器难，且文本等复杂条件无法用于分类器引导。

#### CFG 核心思想

Ho & Salimans (2022) 的关键洞察：**分类器引导中的 $\nabla \log p(\mathbf{c} \mid \mathbf{z})$ 可以从扩散模型自身推导出来，不需要额外分类器。**

根据贝叶斯公式：

$$\nabla_{\mathbf{z}_\lambda} \log p(\mathbf{c} \mid \mathbf{z}_\lambda) = \nabla_{\mathbf{z}_\lambda} \log p(\mathbf{z}_\lambda \mid \mathbf{c}) - \nabla_{\mathbf{z}_\lambda} \log p(\mathbf{z}_\lambda)$$

右边两项对应**条件得分** $\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda, \mathbf{c})$ 和**无条件得分** $\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda)$。代入得分与 $\epsilon$ 的关系 $\boldsymbol{\epsilon}_\theta \approx -\sigma_\lambda \nabla \log p$，得 CFG 核心公式：

$$\boxed{\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda, \mathbf{c}) = (1+w) \cdot \boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda, \mathbf{c}) - w \cdot \boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda)}$$

在 Wan 的 Flow Matching 框架中，因为预测的是速度场 $u$ 而非噪声 $\epsilon$，等价写为（参见 Q2 Step 4）：

$$u_{\text{cfg}} = u_{\text{uncond}} + w \cdot (u_{\text{cond}} - u_{\text{uncond}})$$

#### 训练：一个网络同时学会两份任务

**训练算法**（伪代码）：

```
输入: 无条件概率 p_uncond（通常 0.1~0.2）

循环每个训练步骤:
    (x, c) ← 从数据集采样（视频潜变量 + 文本）
    c ← ∅  以概率 p_uncond        # ← 随机丢弃文本条件
    t ← 采样时间步
    x_t = t·x + (1-t)·x_0          # 加噪
    计算损失: ||u(x_t, c, t; θ) - v_t||²
    梯度更新
```

- 当 `c` 是真实条件时，网络学习 $u(x_t, c_{txt}, t)$（条件模式）
- 当 `c = \varnothing$` 时，网络学习 $u(x_t, c_{\emptyset}, t)$（无条件模式）

一个网络，两种技能，训练时几乎零额外开销。在 Wan 中，虽然论文原文没有显式描述这一步，但推理时明确区分了 `conditional output` 和 `unconditional output`，说明训练阶段使用了这种 text dropout 策略。

#### 推理：双前向 + 外推

```
for t = 1...T:
    u_cond   = DiT(z_t, c_txt, t)       # 条件前向（有 prompt）
    u_uncond = DiT(z_t, c_∅, t)         # 无条件前向（空字符串）
    u_final  = u_uncond + w·(u_cond - u_uncond)  # CFG 融合
    z_{t+1} ← 用 u_final 更新 z_t        # ODE 步进
```

每一步需要**两次完整前向传播**，这是 CFG 的主要计算代价。

#### 直观理解

把公式写成差值形式最能揭示本质：

$$\tilde{u} = u_{\text{cond}} + w \cdot \underbrace{(u_{\text{cond}} - u_{\text{uncond}})}_{\text{"文本引导的方向"}}$$

| 分量 | 含义 |
|------|------|
| $u_{\text{cond}}$ | "有 prompt 时，模型认为该往哪个方向走" |
| $u_{\text{uncond}}$ | "没有任何条件时，模型自由发挥会走的方向" |
| $u_{\text{cond}} - u_{\text{uncond}}$ | **prompt 施加的"推力方向"** |
| $w$（通常 5-7） | 放大这个推力，使生成结果更强地偏离无条件分布 |

**$w$ 的效应**：
- $w=1$：标准条件采样，质量和多样性平衡
- $w=3\sim5$：显著增强细节、色彩饱和度、文本对齐
- $w=7\sim10$：极大引导强度，细节夸张但多样性骤降，可能产生伪影
- $w > 10$：过度引导，画面过饱和、失真、出现奇怪图案

#### CFG ≠ 对抗攻击

扩散模型的得分场 $\boldsymbol{\epsilon}_\theta$ 来自神经网络，而非保守势场（即 $\boldsymbol{\epsilon}_\theta$ 不对应于某个真实分布的精确梯度）。因此 CFG **不是**真正意义上的分类器梯度，也 **不是**对抗攻击。它更像一种**启发式外推**：从"有条件"和"无条件"两个已知点出发，沿连线方向向外延伸。

#### CFG 在 Wan 中的三个具体体现

**① 标准 CFG 推理**（对应 Q2 Step 4）：
- 空字符串 `""` → umT5 → $c_\emptyset$ 产生无条件嵌入
- 每步去噪：条件前向 + 无条件前向 → CFG 融合，$w$ 通常 5-7

**② CFG Cache**（Wan 的创新优化，论文 `content/4_4_inference.tex`）：
> *"In the later stages of sampling, there is a notable similarity between conditional and unconditional DiT outputs."*

采样后期，$u_{\text{cond}}$ 和 $u_{\text{uncond}}$ 越来越接近。原因：
- 早期（高噪声）：条件起决定性的布局作用，条件 vs 无条件差异很大
- 后期（低噪声）：画面大体结构已定，主要是纹理细化，两者输出趋同

Wan 利用这一点：**采样后期每隔几步才计算一次无条件前向**，中间步复用条件结果（加残差补偿防细节损失），避免接近一半的前向传播，是实现 **1.62× 加速**的重要组成部分。

**③ 一致性模型蒸馏**：Wan 的 Streamer 模块使用 LCM/VideoLCM 蒸馏时，把 CFG 也蒸馏进一致性模型（论文 `content/5_6_realtime_generation.tex`），蒸馏后的模型只需 4 步采样，每步仅一次前向（CFG 效果已"固化"在权重中），实现 10-20× 加速。

---

### Q11: Wan 如何完成 Prompt 信息的注入？（条件注入体系）

Wan 没有使用 ControlNet，但有一套完整的、分层的条件注入体系。Wan 的 DiT 架构使用 **Transformer 原生的注入方式**，分为四个层面：

#### 第 1 层：文本编码

```
用户 Prompt → Qwen2.5-Plus 改写 → umT5 编码 → c_txt ∈ ℝ^(512×d)
空字符串 "" → umT5 编码 → c_∅ ∈ ℝ^(512×d)（为 CFG 准备）
```

Wan 选择 umT5 作为文本编码器，因为：
- 多语言编码能力强，支持中英文
- **双向注意力**机制（论文消融实验证明优于单向 LLM：umT5 > Qwen2.5-7B > GLM-4-9B）
- 同参数规模下收敛更快

#### 第 2 层：架构内注入（DiT 每个 Block 内部）

这是 Wan 条件注入的**主通道**。每个 DiT Block 内部结构为：

```
[Self-Attention] → [Cross-Attention] → [FFN + AdaLN]
```

**① Cross-Attention —— 文本条件注入**（论文 `4_2_model_training.tex` 第 41 行）：

> *"We employ the cross-attention mechanism to embed the input text conditions, which can ensure the model's ability to follow instructions even under long-context modeling."*

```
Q = 潜变量 token（42,525 个，每个代表 8×8 像素区域）
K = 文本 token（512 个，来自 umT5）
V = 文本 token（512 个，来自 umT5）

每个空间 token 通过 softmax(QKᵀ/√d)·V 查询自己对应的语义
```

这是 Wan 文本注入的**唯一主通道**，实现了空间-语义细粒度对应：潜空间中"粉色区域"的 token 关注文本中 "cherry blossoms"，"灰色区域"的 token 关注 "road"。

**② Shared AdaLN —— 时间步条件注入**（论文第 43-47 行）：

> *"We employ an MLP with a Linear layer and a SiLU layer to process the input time embeddings and predict six modulation parameters individually. This MLP is shared across all transformer blocks, with each block learning a distinct set of biases."*

```
时间步 t → Sinusoidal Embedding → Shared MLP → (γ, β, α₁...α₆)
                                                     │
                                            scale/shift 注入 FFN + Attention 层
```

时间步告诉模型"当前在去噪的哪个阶段"——早期建立全局布局，中期形成物体轮廓和运动方向，后期细化纹理和细节。

**Wan 的创新——Shared AdaLN**：MLP 在所有 Block 间共享，每 Block 只学独立偏置。参数减少约 25%，性能反而更好（因为鼓励了更深网络而非更宽 AdaLN）。

#### 第 3 层：下游任务特定的条件注入

Wan 为每个下游任务定制了轻量注入方式：

**① I2V（图像到视频）：通道拼接 + Decoupled Cross-Attention**（论文 `5_1_i2v.tex`）：

```
条件图像 → VAE Encoder → z_c（条件 latent）
噪声     → z_t（噪声 latent）
掩码     → M

[z_t | z_c | M] → 通道拼接 → DiT（输入通道从 c 变为 2c+s）

另外：
条件图像 → CLIP Encoder → MLP → Decoupled Cross-Attention → DiT
                                   （全局上下文注入，类似 IP-Adapter）
```

关键细节：**新增的投影层用零初始化**，保护预训练权重不被破坏。

**② Camera Motion（相机控制）**（论文 `5_5_camera_motion.tex`）：

```
相机参数 [R,t], K → Plücker 坐标 P ∈ ℝ^(6×F×H×W)
                          ↓
                   PixelUnshuffle（降分辨率）
                          ↓
                   CNN 编码器（多层，每层对应一个 DiT Block 层级）
                          ↓
                   Zero-init Conv → (γ_i, β_i)
                          ↓
                   注入 DiT Block：f_i = (γ_i+1)·f_{i-1} + β_i
```

使用 zero-initialized convolution，与 ControlNet 的零卷积理念一致。

**③ Video Editing（VACE）：统一条件单元**（论文 `5_2_video_eidting.tex`）：

```
VCU（Video Condition Unit）= [文本嵌入 + 帧序列 + 掩码序列]
                                    ↓
                           概念解耦策略（修改区域 vs 保留区域 → 分离处理）
                                    ↓
                          Cross-Attention → DiT
```

**④ 视频个性化：潜在空间直接条件 + 自注意力扩展**（论文 `5_4_video_personlization.tex`）：

```
参考人脸 → VAE Encoder → 人脸 latent（不放特征提取器，避免信息损失）
                              ↓
              在时序轴上扩展 K 帧（人脸 + 掩码）
                              ↓
              自注意力 inpainting 范式 → 扩散生成
```

#### 第 4 层：Prompt Rewriting（输入预处理对齐）

```
用户 Prompt（10词）→ Qwen2.5-Plus 改写 → 训练分布对齐（200词稠密描述）
```

改写遵循三条原则：
1. 保持原意地添加细节
2. 注入自然运动属性
3. 与 post-training caption 结构对齐（风格→内容概要→细节）

为什么需要？训练时用的是 200+ 词的高质量标注，推理时用户只输入 10 个词 → 分布严重不匹配 → 改写弥合差距。

#### 总结：Wan 的完整条件注入栈

```
用户: "a woman riding a bicycle under cherry blossoms"
  │
  ├─ 第1层 Qwen2.5-Plus 改写（对齐训练分布）
  │     ↓
  ├─ 第1层 umT5 编码 → 512×d 文本嵌入
  │     ↓
  ├─ 第1层 umT5 编码空字符串 → 512×d 无条件嵌入（为 CFG 准备）
  │
  ▼
  ├─ 第2层 Cross-Attention ─┐
  ├─ 第2层 Shared AdaLN  ──┤ 在每个 DiT Block 中注入
  │                         │
  ▼  50步 ODE 求解 ────────┘
  │
  ├─ 第3层 如有 I2V/Camera Adapter/VACE → 任务特定条件注入
  │
  ├─ CFG (w=5~7)，每步两次前向 → 外推融合
  │
  ▼
最终潜变量 → VAE Decoder → 视频
```

**核心要点**：Wan 的 Prompt 注入 = Cross-Attention（文本主通道）+ Shared AdaLN（时间条件）+ 任务特定 Adapter（图像/相机/编辑等辅助条件）+ Prompt Rewriting（输入预处理对齐）+ CFG（推理时放大所有条件的效果）。

---

### Q12: 训练时输入 DiT 的是 latent + prompt，推理时只有 prompt，latent 从哪来？

#### 核心问题

训练时 DiT 接收**真实视频加噪后的潜变量** + **文本嵌入**，但推理时没有真实视频，DiT 如何产生可供 VAE 解码的 latent？

#### 关键回答：latent 不是"没有"，而是从**纯随机噪声初始化**，由 DiT 逐步塑造出来的

#### 训练 vs 推理对比

**训练时（已知目标）**：

```
真实视频 → VAE Encoder → z₀（干净潜变量，真实值）
                            │
                    加噪：z_t = t·z₀ + (1-t)·ε（Flow Matching 前向过程）
                            │
              DiT(z_t, t, text_embed) → 预测速度场 u
                            │
              损失 = ||u - v_t||²（v_t = z₀ - ε 是真实速度）
```

训练时 DiT 看到的是已知真实视频加噪后的潜变量，它学习的是"给定当前噪声状态和文本，预测该往哪个方向变"。

**推理时（无目标，从零生成）**：

```
Step 0: z_T ~ N(0, I)          ← 初始潜变量 = 纯高斯噪声！
                ↓
Step 1: DiT(z_T, t=T, c_txt)   ← prompt 通过 Cross-Attn 指导方向
        预测速度 u
        z_{T-1} = z_T + u·Δt    ← ODE 步进
                ↓
Step 2: DiT(z_{T-1}, t=T-1, c_txt)
        z_{T-2} = z_{T-1} + u·Δt
                ↓
        ...（重复共 50 步）...
                ↓
Step 50: z₀                    ← 最终干净潜变量
                ↓
        VAE Decoder(z₀)        ← 潜变量 → 像素空间视频
```

**关键点**：推理的初始 latent 就是随机的——DiT 在每一步根据 prompt 预测"该往哪个方向变"，50 步后随机噪声就被塑造成了有意义的潜变量。

#### 类比理解

| | 训练 | 推理 |
|------|------|------|
| **类比** | 学生看着标准答案的草稿，学习如何描摹 | 学生面对一张白纸，凭记忆画出来 |
| **Latent 来源** | 真实视频加噪 → 已知 | 纯随机初始化 |
| **DiT 的任务** | 学习去噪/速度场映射 | 从噪声出发，逐步塑造结构 |
| **Prompt 的作用** | 告诉 DiT "这个噪声画面应该变成什么" | 告诉 DiT "你现在往哪个方向塑造" |

#### 为什么这能工作？

扩散模型（包括 Flow Matching）的核心能力是**学到的"去噪"方向在推理时被反过来用于"从零创造"**。

训练时，模型学会了：给定任意噪声状态 $z_t$ 和文本条件 $c$，预测该往哪个方向走能到达真实视频 $z_0$。推理时，虽然起点是纯噪声（而非真实视频加噪），但模型已经学会了**整个潜空间中"向真实数据分布移动"的方向场**——只要沿着这个方向场一步一步走，就能从任意噪声状态走到一个符合文本描述的干净潜变量。

```
训练：真实数据 → 加噪 → 学"去噪方向"
推理：随机噪声 → 沿学到的方向走 50 步 → 到达"看起来像真实数据"的区域
```

#### 每一步中 Prompt 的具体作用

```
DiT 的输入：z_t（当前噪声潜变量，纯数据）+ c_txt（文本嵌入，纯语义）

z_t 通过 Self-Attention  → 让各个空间位置互相协调
c_txt 通过 Cross-Attention → 让每个空间位置知道自己"应该变成什么"
时间步 t 通过 AdaLN      → 告诉网络"当前在去噪的哪个阶段"
```

Prompt 不是一次性"生成"整个 latent，而是在**每一步都参与**：通过 Cross-Attention 告诉 DiT"在当前去噪阶段，每个空间区域应该往什么语义方向变化"。

#### 结合 Wan 的完整流程

```
Step 0: z_T ~ N(0,I), shape [C, F, H, W]     ← 纯噪声初始化
Step 1: 改写 prompt → umT5 → c_txt
        空字符串 → umT5 → c_∅（为 CFG 准备）
Step 2: for t = T down to 1:
          u_cond   = DiT(z_t, t, c_txt)       ← 条件前向
          u_uncond = DiT(z_t, t, c_∅)         ← 无条件前向
          u_cfg    = u_uncond + w·(u_cond - u_uncond)  ← CFG 融合
          z_{t-1}  = z_t + u_cfg·Δt            ← ODE 步进
Step 3: z₀ → VAE Decoder → 视频

整个过程：纯噪声 ──50步 DiT 引导──→ 有意义的潜变量 ──VAE──→ 视频
```

**一句话总结**：推理的 latent 不是"没有"，而是从纯噪声初始化，由 DiT 在 prompt 引导下逐步塑造出来的。训练时 DiT 学会的是"去噪"能力，推理时这个能力被反过来用于"从零创造"——这正是扩散模型的本质。

---

### Q13: 本文在训练和推理使用了哪种量化策略？

Wan 在**训练**和**推理**阶段采用了完全不同的量化/精度策略：

#### 训练阶段

训练阶段**不使用量化**，仅采用 **BF16 混合精度训练（bf16-mixed precision）**：

- 论文原文（`4_2_model_training.tex` 第 104 行）：*"We employ efficient training at bf16-mixed precision combined with the AdamW optimizer."*
- 训练时 VAE 和 Text Encoder 冻结，仅 DiT 参与优化，DiT 占整体训练计算量的 **85% 以上**
- 训练不需要量化，因为主要瓶颈是注意力计算的 $O(s^2)$ 复杂度和显存（激活存储可达 8 TB），而非数值精度

#### 推理阶段

推理阶段针对**两种不同场景**使用了不同的量化策略：

##### 场景一：标准推理（服务端，14B 模型）

两种量化技术协同工作：

| 量化技术 | 策略 | 加速比 |
|----------|------|--------|
| **FP8 GEMM** | 权重：per-tensor 量化；激活：per-token 量化。DiT Block 中所有 GEMM 操作使用 FP8 精度 | **1.13×** |
| **8-Bit FlashAttention** | 混合 INT8/FP8：$S=QK^T$ 用 **INT8**，$O=PV$ 用 **FP8**；块间累积用 **FP32**（借鉴 DeepSeek-V3 的 FP8 GEMM 方法），H20 GPU 上达到 **95% MFU** | **1.27×** |

FP8 GEMM + 8-Bit FlashAttention + Diffusion Cache（1.62×）累计加速约 **2.32×**。

**8-Bit FlashAttention 的设计动机**：
- 原生 FlashAttention3 的 FP8 实现在视频生成中质量下降严重
- SageAttention 使用 INT8+FP16 混合精度减少精度损失，但未针对 Hopper GPU 优化
- Wan 的自研方案：混合 INT8/FP8 + FP32 跨块累积，解决了长序列下 14-bit 累加器溢出问题

##### 场景二：消费级部署（Streamer 实时视频生成）

针对 RTX 4090 等消费级 GPU，使用两种量化策略（论文 `5_6_realtime_generation.tex`）：

| 量化技术 | 目标 | 效果 |
|----------|------|------|
| **INT8 量化**（torchao） | 注意力层 + 线性头 | 显著减少显存占用，保持生成质量，但加速有限 |
| **TensorRT 量化** | 整体模型 | 大幅加速，单张 RTX 4090 达 8-20 FPS，但可能引入轻微伪影和不稳定性 |

#### 总结对比

| 阶段 | 量化策略 | 核心目的 |
|------|----------|----------|
| **训练** | BF16 混合精度（非量化） | 保证训练稳定性和数值精度 |
| **推理（服务端）** | FP8 GEMM + 混合INT8/FP8 FlashAttention | 降低延迟，加速单步去噪 |
| **推理（消费级）** | INT8 (torchao) + TensorRT 量化 | 降低显存，实现实时生成（8-20 FPS） |

**关键要点**：Wan 的训练阶段不涉及量化，仅使用混合精度训练；推理阶段根据部署场景分层使用 FP8（服务端 GPU）、INT8（消费级 GPU）和 TensorRT 量化，其中 8-Bit FlashAttention 的混合 INT8/FP8 策略（$S$ 用 INT8、$O$ 用 FP8、跨块 FP32 累积）是本文在量化方面的核心创新。
