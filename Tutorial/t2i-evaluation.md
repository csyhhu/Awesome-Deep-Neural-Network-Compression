# Text-to-Image 评估方法教程

> 基于 FLUX.1 Kontext、Qwen-Image、Wan、Sana 四大工作的 T2I 评估体系综合整理
> 
> arXiv 参考: [2506.15742](https://arxiv.org/abs/2506.15742), [2508.02324](https://arxiv.org/abs/2508.02324), [2503.20314](https://arxiv.org/abs/2503.20314), [2410.10629](https://arxiv.org/abs/2410.10629)

---

## 目录

- [一、核心 Benchmark 与指标速查表](#一核心-benmark-与指标速查表)
- [二、为什么需要多维度评估？](#二为什么需要多维度评估)
- [三、各指标详细说明](#三各指标详细说明)
  - [3.1 FID (Fréchet Inception Distance)](#31-fid-fréchet-inception-distance)
  - [3.2 CLIP Score](#32-clip-score)
  - [3.3 GenEval](#33-geneval)
  - [3.4 GenEval 2](#34-geneval-2)
  - [3.5 DPG (Dense Prompt Grounding)](#35-dpg-dense-prompt-grounding)
  - [3.6 ImageReward](#36-imagereward)
  - [3.7 Aesthetic Score (LAION)](#37-aesthetic-score-laion)
  - [3.8 文本渲染指标 (TextCrafter)](#38-文本渲染指标-textcrafter)
- [四、各工作的评估方案对比](#四各工作的评估方案对比)
- [五、推荐的低成本自动化评估流程](#五推荐的低成本自动化评估流程)
- [六、参考资料](#六参考资料)

---

## 一、核心 Benchmark 与指标速查表

以下是从 FLUX.1 Kontext、Qwen-Image、Wan、Sana 四个工作中提取的最常用、可自动化、低成本的 T2I 评估指标。**推荐至少覆盖前 5 项**即可获得全面评估结果。

| # | Benchmark / Metric | 评估目标 | 含义 | 如何计算 | 数据源 / 下载 |
|---|-------------------|---------|------|----------|--------------|
| 1 | **FID** | 生成质量分布 | 衡量生成图像分布与真实图像分布的距离，越低越好 | 用 Inception v3 提取特征，计算两组图像特征的 Fréchet 距离 | 需真实图像数据集（如 COCO、LAION），代码: [pytorch-fid](https://github.com/mseitzer/pytorch-fid) |
| 2 | **CLIP Score** | 图文对齐 | 衡量生成图像与输入文本 prompt 的语义相似度，越高越好 | 用 CLIP 编码图像和文本，计算余弦相似度 | 无需下载，使用 [open_clip](https://github.com/mlfoundations/open_clip) 库 |
| 3 | **GenEval / GenEval 2** | 指令遵循 + 组合推理 | 评估模型对物体、属性、数量、颜色、位置、关系的准确还原率 | GenEval: ~500 prompts, 6 子任务; GenEval 2: 更多物体/属性/关系 + Soft-TIFA 评估方法 | [GenEval](https://github.com/djghosh13/geneval), [GenEval 2](https://github.com/facebookresearch/GenEval2) |
| 4 | **DPG (Dense Prompt Grounding)** | 细粒度指令遵循 | 评估模型处理密集 prompt 的能力，检查实体、属性、关系是否准确 | 对 1K 个 Dense Prompt 生成图像，用 MLLM（GPT-4V 等）进行结构化打分，输出 5 个子维度分数 | [GitHub](https://github.com/OpenDataLab/Ella) |
| 5 | **ImageReward** | 人类偏好近似 | 基于人类反馈训练的奖励模型分数，越高越好 | 将图像+prompt 输入预训练的 ImageReward 模型，输出 0-10 分 | [GitHub](https://github.com/THUDM/ImageReward) |
| 6 | **Aesthetic Score (LAION)** | 美学质量 | LAION 数据集上训练的美学打分模型，分越高越"美" | 用 [LAION/aesthetic-nsfw-v2](https://huggingface.co/LAION/aesthetic-nsfw-v2) 对每张图像打分（0-1 分） | HuggingFace 模型权重 |
| 7 | **CVTG-2K / TextCrafter** | 文本渲染（英文） | 评估图像中渲染的英文文字准确率 | 对 2K 个文字渲染 prompt 生成图像，用 OCR 提取文字，计算 Word Accuracy、NED、CLIPScore | [GitHub](https://github.com/textcrafter/TextCrafter) |
| 8 | **ChineseWord** | 文本渲染（中文） | 评估图像中渲染的中文字符准确率（分 3 级难度） | 对 8105 个汉字逐一生成图像，用 OCR（PaddleOCR）识别并计算字符准确率 | 参考 GB/T 28039-2011 自行构建 |

---

## 二、为什么需要多维度评估？

### "Bakeyness" 问题

FLUX.1 Kontext 论文指出了一个关键现象：**单一的"哪个图像更好"偏好评估会导致 AI 美学泛化（Bakeyness）**——模型倾向于生成过饱和色彩、中心焦点突出、强散景效果和同质化风格的图像。这是因为通用美学评判器（如 LAION Aesthetic Predictor）倾向于将高分给予这类"AI 味"图像。

**结论**：不能仅用单一美学指标评估 T2I 模型，必须覆盖多个独立维度。

### 评估维度分类

| 维度 | 对应指标 | 是否自动化 | 成本 |
|------|---------|-----------|------|
| 生成质量分布 | FID | ✅ 全自动 | 低 |
| 图文对齐 | CLIP Score | ✅ 全自动 | 低 |
| 指令遵循 | GenEval, DPG | ✅ 全自动 | 中（需 MLLM） |
| 人类偏好近似 | ImageReward | ✅ 全自动 | 低 |
| 美学质量 | LAION Aesthetic | ✅ 全自动 | 低 |
| 文本渲染 | TextCrafter, ChineseWord | ✅ 全自动 | 中（需 OCR） |

---

## 三、各指标详细说明

### 3.1 FID (Fréchet Inception Distance)

**含义**：衡量生成图像分布与真实图像分布在特征空间中的距离。FID 越低，说明生成图像的统计分布越接近真实图像。

**计算方式**：
1. 使用 Inception v3 模型对两组图像（生成集和真实集）分别提取 2048 维特征向量
2. 计算每组特征的均值 (μ) 和协方差矩阵 (Σ)
3. 计算 Fréchet 距离：
   $$FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

**所需数据**：
- 真实图像数据集（推荐 COCO 2017 val, 5000 张）
- 待测模型生成的图像（对同一组 prompt 生成）

**工具**：
- `pip install pytorch-fid`
- 运行: `python -m pytorch_fid path/to/generated path/to/real`

**注意**：FID 对图像数量敏感，建议至少 5000 张。

---

### 3.2 CLIP Score

**含义**：衡量生成图像与输入文本之间的语义相似度。CLIP Score 越高，说明生成图像越符合 prompt 的语义。

**计算方式**：
1. 使用 CLIP 模型分别编码图像和文本
2. 计算两者嵌入向量的余弦相似度
3. 通常对一批 prompt-image 对计算平均

**所需数据**：
- 一组 prompt 列表
- 待测模型对每个 prompt 生成的图像

**工具**：
- `pip install open_clip_torch`
- 使用 CLIPScore 函数计算

**注意**：CLIP Score 反映语义对齐，不反映美学质量或真实性。

---

### 3.3 GenEval

**含义**：面向对象的组合 prompt 评估，衡量模型对物体、属性、数量、颜色、空间位置的准确还原能力。

**规模**：~500 个精心设计的组合 prompt

**子指标**：
| 子任务 | 评估内容 |
|--------|---------|
| Single Object | 单物体识别率 |
| Two Object | 双物体识别率 |
| Counting | 物体数量准确率 |
| Colors | 颜色属性准确率 |
| Position | 空间位置准确率 |
| Attribute Binding | 属性绑定准确率 |
| Overall | 以上指标加权平均 |

**计算方式**：
1. 从 [GitHub](https://github.com/djghosh13/geneval) 下载基准
2. 对每个 prompt 使用模型生成 4 张图像
3. 使用 GenEval 提供的预训练评估模型自动计算各子任务准确率
4. 报告 Overall 分数

**所需数据**：GenEval prompt 列表（随代码提供）

> ⚠️ **注意**：GenEval 发布于 2023 年，FAIR 在 2025 年 12 月的研究表明它已存在 **benchmark drift** 问题——当前 SOTA 模型的评估结果与人类判断偏差高达 17.7%。建议对 2025 年后的模型使用 **GenEval 2**（见下文）。

---

### 3.4 GenEval 2

**发布时间**：2025 年 12 月 | **机构**：FAIR at Meta + University of Washington

**核心动机**：解决原 GenEval 的 **benchmark drift** 问题。随着 T2I 模型能力快速提升，原 GenEval 的自动评估（基于 COCO 训练的检测模型 + CLIP）与人类判断的偏差已达 17.7%，对当前模型不再可靠。

**改进内容**：

| 维度 | GenEval (2023) | GenEval 2 (2025) |
|------|---------------|-----------------|
| 物体 | ~20 个 COCO 类别 | **40 个**（20 COCO + 20 新物体），分 animate/inanimate |
| 属性 | 颜色 | **18 个**（颜色 + 材料 + 图案） |
| 关系 | 位置（2 个） | **9 个**（3D 空间介词 + 及物动词） |
| 计数 | 1-9 | **2-7**（更聚焦常用范围） |
| 评估方法 | 检测模型 + CLIP | **Soft-TIFA**（原子判断融合） |
| 组合复杂度 | 低 | **更高**（更多物体组合 + 更复杂关系） |

**评估方法：Soft-TIFA**

Soft-TIFA（Soft Text-Image Fidelity Assessment）是 GenEval 2 提出的新型评估方法：
- 将图像质量评估分解为对**原子视觉元素**（物体、属性、关系等）的独立判断
- 使用 VQA 模型对每个原子元素进行 yes/no 判定
- 通过加权融合所有原子判断得到最终分数
- **优势**：比 VQAScore 等整体性判别模型更符合人类判断，且更不容易发生 benchmark drift

**产出指标**：

| 指标 | 含义 |
|------|------|
| Object Accuracy | 物体出现准确率 |
| Counting Accuracy | 数量准确率 |
| Attribute Accuracy | 属性（颜色/材料/图案）准确率 |
| Relation Accuracy | 空间关系准确率 |
| Overall (Soft-TIFA) | 所有原子判断的加权融合分 |

**计算方式**：
1. 从 [GitHub](https://github.com/facebookresearch/GenEval2) 下载基准
2. 对每个 prompt 生成 4 张图像
3. 使用 GenEval 2 的 Soft-TIFA 评估框架自动计算各指标
4. 报告各子指标及 Overall

**推荐**：对于 2025 年后的 T2I 模型，**优先使用 GenEval 2** 替代原 GenEval。

---

### 3.5 DPG (Dense Prompt Grounding)

**含义**：评估模型处理密集、细致 prompt 的能力，检查生成图像中的实体、属性、关系是否都准确还原。

**规模**：1K 个 Dense Prompt

**子维度**：
| 维度 | 评估内容 |
|------|---------|
| Global | 全局一致性 |
| Entity | 实体准确性 |
| Attribute | 属性准确性 |
| Relation | 关系准确性 |
| Other | 其他 |

**计算方式**：
1. 从 [GitHub](https://github.com/OpenDataLab/Ella) 下载基准
2. 对每个 prompt 生成图像
3. 使用 MLLM（如 GPT-4V、Qwen-VL）对生成结果进行结构化打分
4. 报告各维度分数及 Overall

**所需数据**：DPG prompt 列表（随代码提供）

---

### 3.6 ImageReward

**含义**：基于大规模人类反馈训练的奖励模型，给定 prompt + 图像输出一个 0-10 的分数，近似人类偏好判断。

**计算方式**：
1. 从 [GitHub](https://github.com/THUDM/ImageReward) 安装
2. 加载预训练的 ImageReward 模型
3. 对每个 (prompt, image) 对计算奖励分数
4. 对所有分数取平均

**优点**：完全自动化，一次推理即可得到近似人类偏好的分数。

---

### 3.7 Aesthetic Score (LAION)

**含义**：在 LAION 大规模数据集上训练的美学打分模型，输出 0-1 分的美学评分。分越高，图像越符合"美学标准"。

**计算方式**：
1. 加载 [LAION/aesthetic-nsfw-v2](https://huggingface.co/LAION/aesthetic-nsfw-v2) 模型
2. 对每张生成图像进行推理
3. 收集所有分数计算平均值

**注意**：单独使用此指标会导致 Bakeyness 问题，应与其他指标结合使用。

---

### 3.8 文本渲染指标 (TextCrafter)

**含义**：专门评估 T2I 模型在图像中渲染文字的能力。

**三大基准**：

| 基准 | 语言 | 规模 | 核心指标 |
|------|------|------|---------|
| CVTG-2K | English | 2K prompts | Word Accuracy, NED, CLIPScore |
| ChineseWord | 中文 | 8105 字 | Character Accuracy (3 级难度) |
| LongText-Bench | EN + ZH | 160 prompts | Long text accuracy |

**CVTG-2K 指标计算**：
- **Word Accuracy**：OCR 正确识别的单词比例
- **NED (Normalized Edit Distance)**：编辑距离的归一化值（0 = 完美匹配，1 = 完全不同）
- **CLIPScore**：OCR 结果文本与目标文本的 CLIP 相似度

**ChineseWord 指标计算**：
- 参照《GB/T 28039-2011 现代汉语常用字表》分 3 级难度
- 使用 PaddleOCR 对生成图像进行字符识别
- 计算每级的字符识别准确率

---

## 四、各工作的评估方案对比

| 评估维度 | FLUX.1 Kontext | Qwen-Image | Wan | Sana |
|----------|---------------|------------|-----|------|
| **FID** | - | - | ✅ | ✅ |
| **CLIP Score** | - | ✅ | ✅ | ✅ |
| **GenEval** | ✅ | ✅ | - | ✅ |
| **GenEval 2** | - | - | - | - |
| **DPG** | - | ✅ | - | ✅ |
| **ImageReward** | - | - | - | ✅ |
| **Aesthetic (LAION)** | ✅ | ✅ | ✅ | ✅ |
| **Text Rendering** | ✅ | ✅ | - | - |
| **Human Preference (ELO)** | ✅ (核心) | ✅ (AI Arena) | ✅ | - |

**Sana 是自动化评估最彻底的工作**：FID + CLIP Score + GenEval + DPG + ImageReward + Aesthetic 全覆盖，无需人工参与。

---

## 五、推荐的低成本自动化评估流程

### Step 1：准备数据集

| 数据集 | 用途 | 获取方式 |
|--------|------|---------|
| GenEval (~500 prompts) | 指令遵循 + 组合推理 | `git clone https://github.com/djghosh13/geneval` |
| GenEval 2 (更新版) | 指令遵循 + 组合推理（推荐） | `git clone https://github.com/facebookresearch/GenEval2` |
| DPG (1K prompts) | 细粒度指令遵循 | `git clone https://github.com/OpenDataLab/Ella` |
| CVTG-2K (2K prompts) | 英文文本渲染 | `git clone https://github.com/textcrafter/TextCrafter` |
| COCO 2017 val (5K images) | FID 计算 | [COCO 官网](https://cocodataset.org) |
| Diverse Prompts (500+) | CLIP Score / Aesthetic / ImageReward | 从 GenEval + DPG 中选取 |

### Step 2：生成图像

对每个 benchmark 的 prompt，使用待测模型生成图像。建议设置：
- 分辨率：1024×1024
- 每个 prompt 生成 4 张（取最优或全部评估）
- 保存格式：`{benchmark_name}/{prompt_id}/{image_id}.png`

### Step 3：计算评估指标

按以下顺序依次计算：

| 顺序 | 指标 | 命令 / 工具 | 输出 |
|------|------|-----------|------|
| 1 | FID | `pytorch-fid gen_dir real_dir` | 单个数值 |
| 2 | CLIP Score | `open_clip` + 自定义脚本 | 平均相似度 |
| 3 | Aesthetic Score | `LAION/aesthetic-nsfw-v2` | 平均美学分 |
| 4 | ImageReward | `ImageReward` 库 | 平均奖励分 |
| 5 | GenEval / GenEval 2 | 官方代码 | 子任务分 + Overall |
| 6 | DPG | Ella 官方代码 + MLLM | 5 个子维度 + Overall |
| 7 | TextCrafter | TextCrafter 官方代码 | Word Acc, NED, CLIPScore |

### Step 4：汇总报告

生成一张结果表格，包含所有指标：

| 模型 | FID↓ | CLIP↑ | Aesthetic↑ | ImageReward↑ | GenEval↑ | DPG↑ | TextAcc↑ |
|------|------|-------|-----------|-------------|----------|------|----------|
| Model A | | | | | | | |
| Model B | | | | | | | |

### 最小可行评估方案

如果时间有限，推荐至少完成：
1. **CLIP Score** — 快速衡量图文对齐
2. **Aesthetic Score** — 快速衡量美学质量
3. **GenEval** — 衡量指令遵循核心能力
4. **ImageReward** — 近似人类偏好

以上 4 项均可自动化完成，无需人工参与。

---

## 六、参考资料

| 资源 | 链接 |
|------|------|
| FLUX.1 Kontext 论文 | https://arxiv.org/abs/2506.15742 |
| Qwen-Image 论文 | https://arxiv.org/abs/2508.02324 |
| Wan 论文 | https://arxiv.org/abs/2503.20314 |
| Sana 论文 | https://arxiv.org/abs/2410.10629 |
| pytorch-fid | https://github.com/mseitzer/pytorch-fid |
| GenEval 基准 | https://github.com/djghosh13/geneval |
| GenEval 2 基准 | https://github.com/facebookresearch/GenEval2 |
| GenEval 2 论文 | https://arxiv.org/abs/2512.16853 |
| DPG (Ella) | https://github.com/OpenDataLab/Ella |
| TextCrafter/CVTG-2K | https://github.com/textcrafter/TextCrafter |
| ImageReward | https://github.com/THUDM/ImageReward |
| LAION Aesthetic | https://huggingface.co/LAION/aesthetic-nsfw-v2 |
| OpenCLIP | https://github.com/mlfoundations/open_clip |