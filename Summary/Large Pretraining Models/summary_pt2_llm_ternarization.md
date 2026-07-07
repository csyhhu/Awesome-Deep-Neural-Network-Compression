# PT²-LLM: Post-Training Ternarization for Large Language Models

**论文信息**
- **标题**: PT²-LLM: Post-Training Ternarization for Large Language Models
- **作者**: Xianglong Yan, Chengzhu Bao, Zhiteng Li, Tianao Zhang, Kaicheng Yang (上海交通大学), Haotong Qin (ETH Zürich), Ruobing Xie, Xingwu Sun (腾讯混元), Yulun Zhang (上海交通大学)
- **会议**: ICLR 2026
- **链接**: https://arxiv.org/abs/2510.03267
- **代码**: https://github.com/XIANGLONGYAN/PT2-LLM

---

## 1. 研究动机与问题

大语言模型（LLM）在推理和部署时需要巨大的显存和计算资源。**三值化（Ternarization）** 将权重量化到 $\{-1, 0, +1\}$，相比 2-4 bit 量化，它消除了大部分浮点乘法（仅需加法），同时相比二值化具有更强的表达能力，是资源受限场景下非常有潜力的压缩方案。

然而，现有的三值化方法（如 TWN、BitNet b1.58、TernaryLLM 等）几乎都依赖 **量化感知训练（QAT）**，需要完整训练数据和大量计算资源对 LLM 重新训练，这在实践中非常不现实。相比之下，**后训练量化（PTQ）** 无需重新训练，更加高效实用。但直接对 LLM 做 PTQ 三值化会导致严重的性能下降，面临两大核心挑战：

1. **训练无关参数优化困难**：PTQ 无法像 QAT 那样通过梯度下降在大规模数据上优化三值参数；
2. **离群值和分散权重导致量化误差大**：三值化作为一种极端低比特方案，难以表示分散分布或含大量离群值的权重。

---

## 2. 方法

PT²-LLM 提出了一套完整的后训练三值化框架，包含两个核心组件：

### 2.1 非对称三值量化器（Asymmetric Ternary Quantizer, ATQ）

**动机**：LLM 的权重分布通常并不对称（均值非零），对称三值化在 PTQ 中会导致较大的表示偏差。

**非对称初始化**：引入逐行偏置 $\mu$（初始化为每行均值），使反量化权重为 $\widehat{\mathbf{W}} = \alpha \mathbf{T} + \mu$，更好地拟合非对称的权重分布。

**两个训练无关的优化阶段**：

**(a) 迭代三值拟合（Iterative Ternary Fitting, ITF）**
- 交替优化"三值网格参数"（$\alpha$ 和 $\mu$）与"三值矩阵"（$\mathbf{T}$）
- 固定 $\mathbf{T}$ 时，通过对量化误差 $\mathcal{E}_w = \|\mathbf{W} - \widehat{\mathbf{W}}\|_F^2$ 求导得到 $\alpha$ 和 $\mu$ 的**闭式解**（closed-form solution）
- 固定 $\alpha$ 和 $\mu$ 时，通过**灵活舍入（flexible rounding）**：$\mathbf{T}_{ij}^* = \arg\min_{t \in \{-1,0,1\}} |\frac{\mathbf{W}_{ij} - \mu_i}{\alpha_i} - t|$ 更新 $\mathbf{T}$
- 迭代直到 $\mathbf{T}$ 不再变化（实践中约 10 次收敛）

**(b) 激活感知网格对齐（Activation-aware Grid Alignment, AGA）**
- ITF 只最小化了权重的量化误差，但 LLM 的实际输出取决于权重和激活的交互
- AGA 转而优化输出误差：$\mathcal{E}_x = \|\mathbf{W}\mathbf{X} - \widehat{\mathbf{W}}\mathbf{X}\|_F^2$
- 同样通过对 $\alpha$ 和 $\mu$ 求导得到闭式解，引入校准数据的协方差矩阵 $\mathbf{C} = \sum_b \sum_i \mathbf{X}_{bi}\mathbf{X}_{bi}^\top$
- 仅更新网格参数（$\alpha, \mu$），不更新 $\mathbf{T}$（避免过拟合校准集）

### 2.2 结构相似性重排序（Structural Similarity-based Reordering, SSR）

**动机**：块级量化（GPTQ 框架）中，如果块内权重方差大或混入离群列，三值化会很粗糙。

**方法**：利用列间余弦相似度对权重列进行聚类重排列：
$$S_{ij} = \frac{\mathbf{W}_{:,i}^\top \mathbf{W}_{:,j}}{\|\mathbf{W}_{:,i}\|_2 \|\mathbf{W}_{:,j}\|_2}$$

将结构相似、数值相近的列放在同一个量化块中，形成更紧凑的分布，降低量化难度。同时将离群值聚集在一起，避免其破坏正常列的量化。

**高效集成**：采用轻量级策略——每量化一个块后，计算剩余列均值向量，选择与其余弦相似度最高的 top-$k$ 列作为下一个量化块。

### 2.3 整体流程

1. 非对称三值初始化
2. SSR 重排列（与 GPTQ 误差补偿结合）
3. ITF 迭代优化 $\alpha$、$\mu$、$\mathbf{T}$（最小化权重误差）
4. AGA 进一步优化 $\alpha$、$\mu$（最小化输出误差）

---

## 3. 实验

### 3.1 实验设置
- 使用 128 个 Wikitext2 校准样本，序列长度 2048
- 单卡 NVIDIA A800-80GB GPU
- 块大小固定为 128
- 评估模型：LLaMA-7B/13B/65B、LLaMA-2-7B/70B、LLaMA-3-8B、Qwen3-14B

### 3.2 主要结果

| 模型 | 方法 | 比特位 | WikiText2 PPL | 平均零样本准确率 |
|------|------|--------|--------------|----------------|
| LLaMA-7B | FP16 | 16 | 5.68 | 61.73 |
| LLaMA-7B | Slim-LLM | 2.0 | 14.58 | 39.74 |
| LLaMA-7B | **PT²-LLM** | **1.58** | **11.39** | **45.07** |
| LLaMA-13B | Slim-LLM | 2.0 | 9.12 | 48.04 |
| LLaMA-13B | **PT²-LLM** | **1.58** | **9.11** | **48.64** |
| LLaMA-65B | Slim-LLM | 2.0 | 6.15 | 54.69 |
| LLaMA-65B | **PT²-LLM** | **1.58** | 6.62 | **55.95** |

- 在使用最低比特位（1.58 bit）的情况下，几乎所有模型上超越了所有 2-bit 基线方法
- 相比同等比特位的 PB-LLM（1.7 bit），LLaMA-7B 平均准确率从 33.44 提升到 45.07（+11.63），WikiText2 PPL 降低 86%

### 3.3 消融实验

| 实验 | 结论 |
|------|------|
| ITF vs AGA | 两者各自有效，联合使用效果最佳。ITF 大幅降低 PPL，AGA 显著提升准确率 |
| SSR 策略 | SSR 优于随机重排和 Hessian 重排；随机重排几乎无效 |
| 校准集大小 | 64 样本即可，128/256 样本结果相近，表明方法对校准数据量鲁棒 |
| 校准集类型 | WikiText2 和 C4 效果相近，PTB 较差 |

### 3.4 压缩效率

| 方法 | LLaMA-7B 模型大小 | 压缩比 | 压缩时间 |
|------|-------------------|--------|----------|
| FP16 | 13.48 GB | - | - |
| GPTQ (2-bit) | 2.19 GB | 6.16× | 21 min |
| Slim-LLM (2-bit) | 2.30 GB | 5.86× | 182 min |
| PB-LLM (1.7-bit) | 2.91 GB | 4.63× | 22 min |
| **PT²-LLM (1.58-bit)** | **1.88 GB** | **7.17×** | **32 min** |

PT²-LLM 实现了最小的模型体积和最极致的压缩比，压缩时间仅 32 分钟，远快于 Slim-LLM。

### 3.5 推理加速
在 LLaMA-7B 到 65B 四个规模上测试，PT²-LLM（1.58-bit）相比标准 2-bit 量化在 prefill、decode 和端到端生成三个阶段均有明显吞吐量提升，LLaMA-65B 端到端加速最高达 **2.1×**。

---

## 4. 核心贡献总结

1. **首次系统探索 LLM 的后训练三值化**，提出 PT²-LLM 框架，无需任何重训练即可将预训练 LLM 高效压缩为三值形式
2. **非对称三值量化器（ATQ）**：包含 ITF（迭代三值拟合）和 AGA（激活感知网格对齐）两个训练无关的优化阶段，通过闭式解迭代优化三值参数
3. **结构相似性重排序（SSR）**：即插即用的列重排策略，利用列间结构相似性缓解离群值影响，降低块级三值化难度
4. 在 **1.58 bit** 的极致压缩比下，性能全面超越 2-bit SOTA PTQ 方法，同时模型更小、推理更快

---

## 5. 评价与思考

**优点**：
- 选题精准：后训练三值化是一个未被充分探索但极具实用价值的方向
- 方法精巧：ITF 和 AGA 的闭式解设计高效优雅，完全避免了梯度计算，实现了真正的 training-free
- SSR 思路简洁有效：将相似列聚集到同一块中的思想直观且实用
- 实验全面：覆盖多个主流 LLM 系列和模型规模，消融实验扎实

**潜在局限性**：
- 三值化仍属极端压缩，与 FP16 性能仍有明显差距（如 LLaMA-7B 准确率从 61.73 降至 45.07）
- AGA 阶段冻结 $\mathbf{T}$ 是一种折中策略，可能损失进一步优化的空间
- 推理加速依赖特定后端（llama.cpp），通用推理框架的支持可能需要额外适配工作

---

## 6. 深入讨论（Q&A）

### Q1: ATQ 中 ITF 和 AGA 都在更新 $\alpha$ 和 $\mu$ 吗？

是的，两者都在更新 $\alpha$ 和 $\mu$，但**配合方式**和**优化目标**不同：

- **ITF**：$\alpha$、$\mu$ **和** $\mathbf{T}$ **三者交替迭代更新**。固定 $\mathbf{T}$ → 闭式解更新 $\alpha, \mu$；固定 $\alpha, \mu$ → 灵活舍入更新 $\mathbf{T}$。目标是权重级误差 $\mathcal{E}_w$。
- **AGA**：**仅更新 $\alpha, \mu$，冻结 $\mathbf{T}$**。目标切换为输出级误差 $\mathcal{E}_x$，且只做一次更新。若连 $\mathbf{T}$ 也更新会导致严重过拟合校准集。

总结：ITF 负责找到好的三值分配（$\mathbf{T}$），AGA 在保持分配不变的前提下，进一步把网格对齐到实际输出。

### Q2: 本文是块量化吗？块大小是多少？

是的，本文采用 GPTQ 的**块量化（block-wise quantization）**框架：

- 大权重矩阵被切分为固定大小的块，逐块独立量化
- **块大小固定为 128**（论文明确声明 *"All quantized models use a fixed block size of 128"*）
- 每量化完一个块后，通过 Hessian 逆矩阵将残差传播到后续未量化列

SSR 在这个框架上工作：每轮选择下一个待量化块时，计算剩余列的均值向量，选取与其余弦相似度最高的 **top-128** 列作为下一块，使块内权重分布更紧凑。

### Q3: SSR 如何将 1280 列分成 10 个块？具体步骤？

SSR 是**动态逐步贪心选取**，而非一次性静态聚类。以 1280 列、块大小 128（共 10 个块）为例：

1. **第 1 轮**（还剩 1280 列）：计算剩余 1280 列的均值向量 $\bar{\mathbf{w}}$，计算每列与它的余弦相似度，选出 **top-128** → 第 1 块。对该块做 ATQ 量化，然后 GPTQ 误差补偿传播到剩余 1152 列。
2. **第 2 轮**（还剩 1152 列）：用补偿**更新后**的权重重新计算均值向量，再次选 top-128 → 第 2 块。量化 + 补偿 → 剩余 1024 列。
3. 依此类推，共 10 轮。

关键：每量化完一块、做完误差补偿后，剩余列的值**已经变化**，所以必须每轮重新计算均值、重新挑选，而非静态排序。

### Q4: 量化残差是如何传播的？

基于 GPTQ 的 Hessian 误差补偿机制：输出误差 $\mathcal{E} = \| \mathbf{W}\mathbf{X} - \widehat{\mathbf{W}}\mathbf{X} \|_F^2$ 的 Hessian 近似为 $\mathbf{H} \approx 2\mathbf{X}\mathbf{X}^\top$。

块级版本：量化第 $k$ 个块（128 列）后，计算误差 $\mathbf{E}_k = \mathbf{W}_{:,B_k} - \widehat{\mathbf{W}}_{:,B_k}$，对剩余列 $R$ 补偿：

$$\mathbf{W}_{:,R} \leftarrow \mathbf{W}_{:,R} - \mathbf{E}_k \cdot ([\mathbf{H}^{-1}]_{B_k,B_k})^{-1} \cdot [\mathbf{H}^{-1}]_{B_k,R}$$

直观理解：把本块量化产生的误差，按 Hessian 逆矩阵指示的相关性比例，"预扣"到后续列上，后续列量化时会自动吸收这个偏差。这也是 SSR 必须每轮重算的原因——补偿后的权重值已变化，下一轮均值向量和相似度也随之改变。

### 整体框架总结

> 在 GPTQ 框架下，**动态按列间余弦相似度贪心选取量化块**，对每块先用 ITF 交替求解 $(\alpha, \mu)$ 和 $\mathbf{T}$，再用 AGA 输出对齐微调 $(\alpha, \mu)$，最后通过 Hessian 逆矩阵将量化残差传播到剩余列。
