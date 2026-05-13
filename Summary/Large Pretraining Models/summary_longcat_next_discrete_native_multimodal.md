# LongCat-Next: 将多模态词汇化为离散 Token

**论文来源**：arXiv:2603.27538  
**发布时间**：2026 年 3 月 29 日  
**作者团队**：美团 LongCat Team（88 位作者）  
**GitHub**：https://github.com/meituan-longcat/LongCat-Next  
**HuggingFace**：https://huggingface.co/meituan-longcat/LongCat-Next

---

## 一、核心问题与动机

当前主流多模态系统仍然是"**以语言为中心**"的范式（language-plus-auxiliary），将视觉、音频等非语言模态作为外挂附件处理，导致：
1. 架构碎片化，理解与生成需要分开建模；
2. 不同模态无法在同一优化目标下统一学习；
3. 离散视觉建模长期被认为有性能上限（information loss）。

**核心洞察**：如果能把所有模态像"语言 token"一样表示成离散序列，就可以用同一个 Next-Token Prediction（NTP）目标统一训练所有模态——这正是本文的核心主张。

---

## 二、核心框架：Discrete Native Autoregression (DiNA)

**DiNA** 是一个将多模态信息全部表示在共享离散空间中的统一框架，其四大优势：

| 优势 | 说明 |
|------|------|
| 架构协同 | 多模态数据可直接复用 LLM 成熟的训练与部署基础设施 |
| 理解与生成统一 | 同一 NTP 目标同时覆盖判别性理解和高保真生成 |
| 无缝跨模态交互 | 无需特定任务设计即可处理视觉、语言、音频的交互 |
| 原生数据扩展 | 通用离散空间将多模态内容展平为统一 token 序列，NTP 作为自监督机制 |

DiNA 的核心挑战转化为：**设计各模态专属的 tokenizer-detokenizer 对**，将 LLM backbone 保持为模态无关的多任务学习器。

---

## 三、视觉 Tokenizer：dNaViT

### 3.1 语义完整性（Semantic Completeness）原则

离散视觉建模的双重瓶颈：
1. **视觉表示的容量**：编码器是否能提供足够丰富的语义
2. **离散化带来的信息损失**：量化不可避免地损失信息

**语义完整性**的形式化定义：对任意图像相关查询 Q，离散表示 z 应满足：
$$\mathcal{P}(A \mid z, \mathcal{Q}) \approx \mathcal{P}(A \mid I, \mathcal{Q})$$

即离散表示在下游任务上的条件分布应逼近原始图像条件下的分布，包含：
- **判别不变性**：离散化不应降低判别任务性能
- **生成充分性**：离散编码应足以用于高保真图像重建

### 3.2 Semantic-and-Aligned Encoder (SAE)

本文提出 **SAE**（语义对齐编码器）类别，指经过大规模视觉-语言对齐训练的编码器（如 QwenViT、MoonViT、AIMv2），具备：
- 语义丰富性：捕获高层语义与细粒度视觉细节（OCR 等）
- 与语言模型的亲和性：可无缝集成到统一离散空间

训练目标：
$$\mathcal{L}_{\text{SAE}} = \mathbb{E}_{(I,\mathcal{Q},A)}\left[-\log P(A \mid \mathbf{z_p}, \mathcal{Q})\right]$$

> **实践细节**：本文直接采用 Qwen2.5-ViT（28× 空间压缩比）作为 SAE，跳过了从头训练 SAE 的高昂代价。

### 3.3 Residual Vector Quantization (RVQ)

为弥补连续特征到离散 token 的保真损失，采用 **8 层级联 RVQ**，逐级编码残差误差：

$$\mathbf{r}_0 = f_{\text{proj}}(\mathbf{z}), \quad \hat{\mathbf{q}}_l = \operatorname{VQ}(\mathbf{r}_{l-1}), \quad \mathbf{r}_l = \mathbf{r}_{l-1} - \hat{\mathbf{q}}_l, \quad \hat{\mathbf{z}} = \sum_{l=1}^{L} \hat{\mathbf{q}}_l$$

- Codebook 更新使用 EMA（指数移动平均）而非梯度下降
- 量化目标：$\mathcal{L}_{\text{quant}} = \lambda_c \mathcal{L}_{\text{commit}} + \lambda_s \mathcal{L}_{\text{semantic}}$

**关键发现（信息恢复能力）**：即使随机初始化的 ViT-Base 也具有比预训练版本更强的图像重建能力（PSNR: 30.52 vs 21.86）。这源于残差架构天然保留了低层信号的传递路径：
$$\mathbf{z}_p = \mathbf{x}_0 + \sum_{l=1}^{L} \mathcal{F}_l(\mathbf{x}_{l-1})$$

### 3.4 dNaViT 的去 Tokenization（De-tokenization）

- **像素解码器（Pixel Decoder）**：400M 参数的 ViT，从离散 token 重建图像
- **图像精炼器（Image Refiner）**：基于 flow-matching 的轻量网络，增强高频细节
- 去 tokenization 损失：$\mathcal{L}_{\text{dec}} = \lambda_1 \mathcal{L}_{\text{pixel}} + \lambda_2 \mathcal{L}_{\text{percep}} + \lambda_3 \mathcal{L}_{\text{align}}$

### 3.5 任意分辨率支持

dNaViT 直接在图像原始分辨率上操作，使用 variable-length FlashAttention 处理不同长度的 token 序列，最大训练分辨率 1736×1736，最大序列长度 8192。

---

## 四、音频 Tokenizer

架构设计：
- **编码器**：Whisper-large-v3（初始化）+ 后续微调
- **RVQ**：8 层，codebook 大小分别为 8k/4k/2k/1k/1k/1k/1k/1k
- **下采样倍率**：4× → 12.5 Hz 的离散音频 token
- **解码器**：对称架构，重建粗粒度 Mel 频谱图
- **精炼网络**：flow-matching 模型，输出 24kHz Mel 频谱图，再经 vocoder（HiFi-GAN）还原波形

训练语料：约 250 万小时（网络爬取 + 合成语音 + 任务专项数据）

**内部语言引导（Internal Linguistic Guidance）**：
参考 Moshi 方案，引入文字引导音频模态，支持两种生成策略：
1. **并行生成**：文本和音频 token 同步生成（延迟起始），适合全双工场景
2. **串行生成**：先生成文本 token，再生成音频 token，语义质量更高

统一训练方式：随机延迟 delay 步数（1 至文本段长度），将并行与串行视为同一框架的两个极端。

---

## 五、LLM Backbone 与多模态组件

**Backbone**：LongCat-Flash-Lite A3B（MoE 架构，总参数 68.5B，激活参数 3B）

**End-to-End 多模态嵌入**：
- 视觉 codebook：8 层 × 16,384 大小，多层加法求和
- 音频 codebook：8 层递减大小，同样随机初始化端对端训练
- 预量化特征**仅用于建立 RVQ 的聚类分配**，嵌入值完全由训练决定

**DepthTransformer 解码头**：在生成阶段，LLM 的隐藏状态通过 DepthTransformer 并行解码出多层离散 token，单步自回归预测实现高效多层解码。

---

## 六、训练流程

### 6.1 四阶段训练

| 阶段 | 目的 | 可训练模块 | Batch/SeqLen |
|------|------|-----------|--------------|
| Stage 1: Pre-Align | 预对齐 codebook 嵌入 | Embedding + DepthTransformer | 8192 / 8K |
| Stage 2: Pre-training | 全模态预训练 | 全部模块 | 8192 / 8K |
| Stage 3: Mid-training | 引入合成数据、长 CoT、任意分辨率生成 | 全部模块 | 1024 / 32K |
| Stage 4: SFT | 指令遵循微调 | 全部模块 | 128 / 64K |

总训练 token 数：~2 万亿

### 6.2 Pre-Buffer 模块
为解决多层 codebook 嵌入求和后重编码不足的问题，引入单层 FFN 作为 **Pre-Buffer**，显著加速收敛。

---

## 七、实验结果

### 7.1 视觉理解

LongCat-Next（A3B 激活参数）的关键成绩：

| 基准 | LongCat-Next | Qwen3-Omni-A3B | Qwen3-VL-A3B（专精）|
|------|:---:|:---:|:---:|
| MMMU | **70.6** | 69.1 | 74.2 |
| MathVista | **83.1** | 75.9 | 80.1 |
| MathVision | **64.7** | 56.3 | 60.2 |
| OmniDocBench_en ↓ | **0.152** | 0.289 | 0.183 |
| CharXiv_RQ | **60.1** | 42.8 | 48.9 |
| OCRBench | 86.5 | 85.4 | 90.3 |
| VisuLogic | **29.4** | 20.0 | 23.0 |

> **结论**：LongCat-Next（统一模型）在多数视觉理解任务上超越 Qwen3-Omni，在推理和 OCR 任务上甚至超越专精视觉模型。

### 7.2 视觉生成

与专精 T2I 模型对比：

| 基准 | LongCat-Next | FLUX.1-dev | Emu-3.5 |
|------|:---:|:---:|:---:|
| GenEval | 84.44 | 66 | 72.67 |
| LongText-EN | 93.15 | 60.70 | 97.60 |
| TIFF | 82.85/84.38 | 71.10/71.80 | 89.48/88.18 |

### 7.3 音频

- **ASR**：LibriSpeech test-clean WER=1.63%，接近专精语音模型水平
- **TTS**：SeedTTS_zh WER=1.90%，远超 Kimi-Audio
- **音频理解**：MMAU=76.40，TUT2017=43.09（大幅超越 MiMo-Audio 的 15.06）
- **音频对话**：ReasoningQA=87.52，超越 Gemini 和 MiMo-Audio

### 7.4 纯文本能力

| 基准 | LongCat-Next | Kimi-Linear-48B |
|------|:---:|:---:|
| SWE-Bench | **43.0%** | 32.8% |
| Tau2-Telecom | **62.06** | 15.68 |
| MMLU | 83.95 | 79.91 |

> **关键成果**：LongCat-Next 有效避免了"多模态税"——扩展多模态能力没有损害基础语言性能。

---

## 八、重要实验发现

### 8.1 离散 vs. 连续建模对比

通过消融实验验证：
- **Pre-Buffer** 模块对于弥补离散与连续模型之间的差距至关重要
- **数据规模扩大后**，离散模型的性能接近连续模型（差距 <1%）
- 离散建模**没有内在的性能上限**，性能主要受数据量影响

### 8.2 理解与生成的协同关系

实验证明：**生成任务不妨碍理解；反之，理解任务积极增强生成**。在相同 token 预算下，联合训练模型与纯生成模型相比 loss 低 0.02，而与纯理解模型相比仅差 0.006。

### 8.3 MoE 内部涌现模态专化

在模态无关 MoE 架构中，通过多模态训练后：
- 部分专家逐渐对特定模态（视觉/音频）产生偏好
- 每个专家路由的平均 token 数从 507.1 增加到 584.6（容量利用率提升）

### 8.4 柏拉图表示假说验证

t-SNE 可视化显示，LongCat-Next 中视觉 token 和文本 token 的分布**相互交织**而非形成独立簇，而对比的 Qwen2.5-VL 则产生明显分离的模态簇。这支持了"同一底层现实的不同表达"的柏拉图假说。

---

## 九、强化学习（RL）

利用离散表示的优势，直接应用 **GRPO** 进行多模态 RL：

**图像生成 RL**：多维度奖励
1. 综合能力（对象计数、颜色准确度、空间位置）
2. OCR 能力（文字渲染准确度）
3. 语义对齐（VLM 评判）
4. 图像质量（HPS、美学分）

**图像理解 RL**：提出了解决熵爆炸的两种序列级过滤机制：
- **熵过滤**：丢弃批次内熵值超过均值 +n 个标准差的序列
- **训练-推理概率差异过滤**：任何 token 的采样策略与 actor 策略概率差超过阈值 δ 时丢弃整个序列

RL 后性能提升：GenEval +3.39%（counting +7.5%, position +6.75%），MathVision +4.24%。

---

## 十、基础设施：VHalf-based Pipeline Parallelism

针对多模态模型中异构计算负载设计的 **V形流水线并行**：

1. **V形调度**：将 Embedding 层和模态损失模块放在同一物理设备上，消除跨设备通信
2. **共享 Buffer**：允许模态损失模块直接访问嵌入层后的隐藏状态
3. **LLM Head 解耦**：防止锚定设备过载，单独分配 pipeline stage

效果：有效消除流水线 bubble，实现接近完美的负载均衡。

---

## 十一、关键结论与未来工作

**已验证**：
- 离散视觉建模不存在内在性能上限
- DiNA 范式可在统一架构下同时实现强视觉理解、高保真图像生成和高质量语音交互
- 多模态训练与语言能力之间不存在不可调和的冲突

**局限与未来方向**：
1. **Vision Tokenizer 优化**：当前版本重语义解码一致性，未完全追求像素保真度
2. **Any-to-Any 生成**：向任意模态组合的输入输出扩展
3. **数据扩展与表示学习**：共同设计数据、预训练目标和离散化策略
4. **更大规模验证**：当前受算力限制，许多结论需在更大模型上验证

---

## 附：关键技术点速查

| 技术 | 具体方案 |
|------|---------|
| 视觉编码器 | Qwen2.5-ViT（SAE 代理），28× 压缩 |
| 视觉量化 | 8 层 RVQ，codebook 大小 16,384 |
| 视觉解码器 | 400M ViT Pixel Decoder + Flow-matching Refiner |
| 音频编码器 | Whisper-large-v3，4× 下采样 |
| 音频量化 | 8 层 RVQ，codebook 8k/4k/.../1k |
| 语言 Backbone | LongCat-Flash-Lite A3B（MoE，68.5B total/3B active） |
| 多模态解码头 | DepthTransformer（单步自回归并行多层解码）|
| 训练规模 | ~2T tokens，4 个阶段 |
| RL 方法 | GRPO + 序列级熵/概率差过滤 |
