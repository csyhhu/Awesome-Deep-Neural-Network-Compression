# GLM-5: from Vibe Coding to Agentic Engineering

> **论文信息**
> - **arXiv**: [2602.15763](https://arxiv.org/abs/2602.15763)
> - **机构**: 智谱AI (Zhipu AI) & 清华大学
> - **发布日期**: 2026年2月
> - **代码 & 模型**: https://github.com/zai-org/GLM-5

---

## 1. 概述

GLM-5 是智谱 AI 的下一代旗舰基础模型，旨在将 AI 从 **"vibe coding"（氛围编程）** 范式推进到 **"agentic engineering"（智能体工程）** 范式。模型采用 **744B 总参数 / 40B 激活参数** 的 MoE 架构，在多项开源基准测试中达到 SOTA，并首次使开源模型在 Artificial Analysis Intelligence Index v4.0 上达到 50 分。

核心创新包括：
- **DSA 稀疏注意力**：大幅降低训练和推理成本，支持 128K+ 长上下文
- **异步 Agentic RL 框架**：解耦生成与训练，大幅提升 GPU 利用率
- **IcePop + 改进 RL 算法**：稳定的大规模强化学习训练
- **全栈国产芯片适配**：支持华为昇腾等 7 种国产芯片

---

## 2. 模型架构

### 2.1 模型规模

| 参数 | GLM-4.5 | GLM-5 |
|------|---------|-------|
| 总参数量 | 355B | 744B |
| 激活参数量 | 32B | 40B |
| 专家数 | 160 | 256 |
| 路由专家数 | 8 | 8 |
| 层数 | 92 (89 MoE + 3 Dense) | 78 (75 MoE + 3 Dense) |
| 隐藏维度 | 5120 | 6144 |
| 注意力头数 | 96 | 64 |

### 2.2 Multi-Latent Attention (MLA) 改进

GLM-5 采用 MLA 替代 GQA-8，并提出两项改进：

1. **Muon Split**：将 MLA 的上投影矩阵 $W^{UQ}, W^{UK}, W^{UV}$ 按注意力头拆分为独立小矩阵，分别进行矩阵正交化。该方法使 MLA 的性能匹敌 GQA-8，且注意力 logits 尺度在预训练中保持稳定。

2. **MLA-256**：将 head dimension 从 192 提升到 256，同时将注意力头数减少 1/3。训练计算量和参数量不变，但解码计算量降低。

### 2.3 Multi-Token Prediction (MTP) 参数共享

提出在训练时**共享 3 个 MTP 层的参数**，推理时仍用单层 MTP 预测下一个 token。该方法保持草稿模型显存成本不变，同时提升接受率（Accept Length: DeepSeek-V3.2 2.55 → GLM-5 2.76）。

### 2.4 DeepSeek Sparse Attention (DSA)

DSA 的核心思想是用动态、细粒度的 token 选择机制替代传统 $O(L^2)$ 密集注意力。GLM-5 在 mid-training 结束后引入 DSA 继续预训练：

- **Warm-up 阶段**（1000 步）：仅训练 indexer，冻结基座模型
- **稀疏适应阶段**（20B tokens）：联合训练模型和 indexer

仅需 20B tokens 即可使 DSA 模型长上下文性能接近 MLA 模型。相比从头训练 DSA（DeepSeek-V3.2 用了 943.7B tokens），成本大幅降低。

### 2.5 高效注意力变体消融研究

基于 GLM-9B 模型探索了多种高效注意力机制：

| 方法 | RULER@64K | RULER@128K |
|------|-----------|------------|
| Full Attention (基线) | 85.35 | 75.28 |
| SWA Interleave | 65.94 (-19.41) | 44.93 (-30.35) |
| SWA Pattern (搜索) | 83.72 (-1.63) | 69.59 (-5.69) |
| GDN (线性注意力) | 76.76 (-8.59) | 64.00 (-11.28) |
| **SimpleGDN** | 81.76 (-3.59) | 67.03 (-8.25) |

- **SWA Pattern**：基于 beam search 搜索最优层分配策略，显著优于固定交替模式
- **SimpleGDN**：去除 Conv1d 和显式门控，直接复用预训练权重，在 HELMET-ICL 上甚至超越全注意力基线

结论：DSA 是唯一无损的稀疏注意力方案，可应用于所有层而无质量退化。

---

## 3. 预训练与 Mid-Training

### 3.1 预训练数据

- **Web 数据**：改进 DCLM 分类器，引入 World Knowledge 分类器挖掘长尾知识
- **代码数据**：扩大代码语料库，模糊去重后 unique tokens 增长 28%
- **数学与科学**：从网页、书籍、论文中收集高质量数据，使用 LLM 评分筛选

总训练 token 预算：~28.5T

### 3.2 Mid-Training：长上下文扩展

分三阶段扩展上下文窗口：
1. 32K（1T tokens）
2. 128K（500B tokens）
3. 200K（50B tokens）

包含约 1000 万个 issue-PR 对的软件工程数据（~160B unique tokens）。

### 3.3 训练基础设施优化

- **Flexible MTP placement**：将 MTP 模块灵活分配到 pipeline 阶段
- **Pipeline ZeRO2 gradient sharding**：在 pipeline 内做梯度分片，降低梯度显存
- **Zero-redundant Muon communication**：仅 all-gather 各 rank 拥有的参数分片
- **Pipeline activation offloading**：前向完成后将激活 offload 到 CPU，反向前 reload
- **Sequence-chunked output projection**：分块计算输出投影，降低峰值显存
- **INT4 QAT**：在 SFT 阶段应用 INT4 量化感知训练

---

## 4. 后训练 (Post-Training)

### 4.1 整体 Pipeline

```
SFT → Reasoning RL → Agentic RL → General RL → On-Policy Cross-Stage Distillation
```

### 4.2 SFT 阶段

SFT 数据覆盖三大类：
- **General Chat**：问答、写作、角色扮演、翻译、多轮对话
- **Reasoning**：数学、编程、科学推理
- **Coding & Agent**：前后端工程、工具调用、代码智能体、搜索智能体

新增三种思维模式：
- **Interleaved Thinking**：每次回复和工具调用前都进行思考
- **Preserved Thinking**：跨多轮对话保留思考块，避免重复推理
- **Turn-level Thinking**：支持逐轮控制是否启用思考

### 4.3 Reasoning RL：IcePop 算法

基于 GRPO + IcePop 技术，显式区分 **训练策略** $\pi^{\text{train}}$ 和 **推理策略** $\pi^{\text{infer}}$，通过 $\operatorname{pop}$ 算子抑制训练-推理分布偏差过大的样本。

**DSA RL 关键发现**：
- DSA indexer 的 top-k 算子必须使用**确定性实现**（`torch.topk`），非确定性的 CUDA/TileLang top-k 会导致 RL 训练几步后性能急剧退化
- RL 阶段默认冻结 indexer 参数

混合域 RL：数学、科学、代码、工具集成推理（TIR）四个域联合训练。

### 4.4 Agentic RL：全异步解耦框架

**核心设计**：
- 推理引擎和训练引擎部署在不同 GPU 设备上
- 推理引擎持续生成轨迹，达到阈值后发送给训练引擎
- 推理引擎每 K 步梯度更新后同步训练引擎的最新权重
- 基于服务器的 Multi-Task Rollout Orchestrator 支持多任务联合训练

**训练稳定性优化**：
1. **Token-in-Token-out (TITO)**：避免重分词导致的 token 边界不匹配
2. **Direct Double-sided Importance Sampling**：用 rollout log-probability 直接作为行为代理，双边裁剪 $[1-\epsilon_\ell, 1+\epsilon_h]$
3. **丢弃过时/噪声样本**：记录 rollout 时的策略版本，丢弃版本差距过大的轨迹
4. **DP-aware Routing**：基于一致性哈希的路由，最大化 KV-cache 复用

### 4.5 General RL

三维优化目标：
- **基础正确性**：指令遵循、逻辑一致性、事实准确性、幻觉控制
- **情感智能**：共情、洞察力、自然人类交流风格
- **任务特定质量**：写作、文本处理、问答、角色扮演、翻译等

混合奖励系统：规则奖励 + 结果奖励模型 (ORM) + 生成式奖励模型 (GRM)

### 4.6 On-Policy Cross-Stage Distillation

作为最终阶段，将在前一阶段 RL checkpoint 作为教师模型，以 on-policy 蒸馏方式快速恢复前面各阶段获得的能力。优势项替换为：

$$\hat{A}_{i,t} = \text{sg}\left[\log\frac{\pi_{\text{teacher}}^{\text{infer}}(y_{i,t}|x,y_{i,<t})}{\pi_\theta^{\text{train}}(y_{i,t}|x,y_{i,<t})}\right]$$

### 4.7 SLIME 训练框架

- **可定制 Rollout**：支持多轮交互、工具调用、环境反馈、verifier 引导等
- **基于 HTTP API 的 Server Rollout**：解耦 rollout 逻辑与训练进程
- **FP8 推理 + MTP**：降低长尾延迟
- **PD Disaggregation**：prefill 和 decode 分离，避免干扰
- **心跳驱动容错**：自动检测并隔离故障服务器

---

## 5. 智能体环境扩展

### 5.1 软件工程 (SWE) 环境

基于 RepoLaunch 框架，自动分析仓库安装与依赖，构建可执行环境并生成测试。构建了超过 10K 个可验证环境，覆盖 9 种编程语言（Python, Java, Go, C, CPP, JavaScript, TypeScript, PHP, Ruby）。

### 5.2 终端环境

- 从种子任务出发，通过 LLM 生成终端任务草稿 → 构建 Harbor 格式任务 → 迭代优化
- 从 Web 语料中自动合成终端任务，采用闭环自验证机制
- Docker 构建准确率超过 90%

### 5.3 搜索任务

构建 Web Knowledge Graph (WKG)，通过三阶段 pipeline 生成高难度多跳 QA：
1. 过滤工具无关推理模型能答对的问题
2. 过滤基础搜索 agent 能解答的问题
3. 双向验证：确保答案唯一性、证据一致性、标签正确性

### 5.4 上下文管理策略

提出**分层上下文管理 (Hierarchical Context Management)**：
- **Keep-recent-k**：仅保留最近 k 轮交互的观察结果
- **Discard-all**：上下文超过阈值 T 时丢弃全部工具调用历史
- 组合使用，BrowseComp 从 55.3% 提升至 62.0%（w/ context manage 达 75.9%）

---

## 6. 国产芯片适配

以华为昇腾 Atlas 系列为例：
- **W4A8 混合精度量化**：标准 Attention/MLP 使用 W8A8，MoE Expert 压缩到 W4A8
- **高性能融合 Kernel**：Lightning Indexer（score+ReLU+TopK 融合）、Sparse Flash Attention、MLAPO（融合 13 个预处理算子）
- **推理引擎优化**：异步调度、RadixCache/Prefix Cache、混合 DP+EP 并行、MTP

单国产节点性能可比肩双 GPU 国际集群，长序列部署成本降低 50%。

---

## 7. 实验结果

### 7.1 ARC 基准测试（与前沿模型对比）

| Benchmark | GLM-5 | GLM-4.7 | DeepSeek-V3.2 | Kimi K2.5 | Claude Opus 4.5 | GPT-5.2 |
|-----------|-------|---------|---------------|-----------|-----------------|---------|
| HLE | 30.5 | 24.8 | 25.1 | 31.5 | 28.4 | 35.4 |
| HLE (w/ Tools) | 50.4 | 42.8 | 40.8 | 51.8 | 43.4* | 45.5* |
| SWE-bench Verified | 77.8 | 73.8 | 73.1 | 76.8 | 80.9 | 80.0 |
| SWE-bench Multilingual | 73.3 | 66.7 | 70.2 | 73.0 | 77.5 | 72.0 |
| Terminal-Bench 2.0 | 56.2 | 41.0 | 39.3 | 50.8 | 59.3 | 54.0 |
| BrowseComp | 62.0 | 52.0 | 51.4 | 60.6 | 37.0 | - |
| BrowseComp (w/ CM) | 75.9 | 67.5 | 67.6 | 74.9 | 57.8 | 65.8 |
| Vending-Bench 2 | $4,432 | $2,377 | $1,034 | $1,198 | $4,967 | $3,591 |

### 7.2 CC-Bench-V2（真实工程能力）

| 类别 | GLM-5 | GLM-4.7 | Claude Opus 4.5 |
|------|-------|---------|-----------------|
| Frontend HTML ISR | 38.9 | 35.4 | 52.2 |
| Frontend React ISR | 34.6 | 17.2 | 39.7 |
| Frontend Vue ISR | 32.7 | 24.5 | 46.9 |
| Backend Pass@1 | 25.8 | 19.6 | 26.9 |
| Repo Exploration | 65.6 | 47.8 | 64.5 |
| Chained Tasks | 52.3 | 43.0 | 61.6 |

### 7.3 LMArena

GLM-5 在 Text Arena 和 Code Arena 均为开源模型 #1。

### 7.4 消融研究关键发现

- **DSA vs MLA**：在长上下文 benchmark 上 DSA 与 MLA 性能接近（RULER@128K: DSA 78.86 vs MLA 79.21）
- **高效注意力**：SWA Pattern 搜索显著优于固定交替模式，SimpleGDN 在 HELMET-ICL 上超越全注意力基线
- **异步 RL 稳定性**：TITO + 双边重要性采样 + 版本过滤机制有效控制 off-policy bias

---

## 8. 核心贡献总结

1. **DSA 稀疏注意力**：使 744B 模型在 128K 长上下文下注意力计算降低 1.5-2×，且性能无损
2. **高效 MTP 参数共享**：训练时共享 3 层 MTP 参数，接受率从 2.55 提升至 2.76
3. **Muon Split 优化 MLA**：使 MLA 性能匹配 GQA-8
4. **全异步 Agentic RL 框架**：解耦生成与训练，1k+ 并发 rollout 支持
5. **IcePop 改进 GRPO**：稳定大规模 RL 训练，DSA indexer 确定性 top-k 是关键
6. **国产芯片全栈适配**：W4A8 量化 + 融合 kernel + 推理引擎优化，部署成本降低 50%
7. **开源模型首次 AA Intelligence Index v4.0 达 50 分**

---

## 9. 与模型压缩/高效推理的关系

GLM-5 在多个维度体现了模型压缩与高效推理的关键技术：

- **DSA 稀疏注意力**：通过动态 token 选择实现约 50% 注意力计算量压缩
- **MLA-256**：通过调整 head dim/head 数比例降低解码计算量
- **W4A8 量化**：MoE Expert 使用 INT4 权重，大幅降低显存占用
- **MTP 投机解码**：参数共享 + 多 token 预测加速推理
- **FP8 推理**：RL rollout 阶段使用 FP8 降低延迟
- **INT4 QAT**：在 SFT 阶段进行量化感知训练，确保低精度推理质量

---

## 10. 讨论与 Q&A

### Q1: GLM-5 在模型架构上使用了 DeepSeek 的 DSA 和 MLA，有其他模型架构上的创新吗？

**基本判断**：GLM-5 的核心架构组件（MoE、MLA、DSA、MTP）确实源自或沿袭了 DeepSeek 系列模型。它在**纯架构范式层面没有提出全新的 attention 机制**（不像 MLA 本身那样是范式级创新），但在**架构改进、训练方法论和高效注意力探索**上有实质贡献。

#### 一、对 DeepSeek 组件的关键改进

| 组件 | 来源 | GLM-5 的改进 | 改进价值 |
|------|------|-------------|---------|
| **MLA + Muon Split** | DeepSeek-V2 | 将 `W^{UQ},W^{UK},W^{UV}` 按 head 拆分为独立小矩阵，分别做矩阵正交化 | 解决了 MLA + Muon 优化器不兼容的问题，使 MLA 性能匹配 GQA-8（此前 MLA 明显更差） |
| **MLA-256** | DeepSeek-V2 | head dim 192→256，head 数减 1/3 | 训练计算量和参数量不变，但解码计算量降低；DeepSeek 的 head 配置是按 H800 roofline 选的，不适配其他硬件 |
| **MTP 参数共享** | DeepSeek-V3 | 训练时 3 个 MTP 层共享参数 | 保持草稿模型显存不变，接受率从 2.55 → 2.76 |
| **DSA + MLA 组合** | DeepSeek-V3.2 | 在 MLA 基座上做 DSA continue training（DeepSeek-V3.2 用的是 GQA） | 仅需 20B tokens 即完成稀疏适应，而 DeepSeek-V3.2 从头训练用了 943.7B tokens |

#### 二、GLM-5 独立的架构探索（消融实验）

虽然未用于最终 GLM-5 模型，但论文在 GLM-9B 上做了有价值的原创探索：

- **SWA Pattern（基于 Beam Search 的层选择）**：用 beam search 在 16K 上下文搜索最优的 SWA/Full Attention 层交替策略，找到的 pattern（如 `SFSSFFSSS...`）在所有长度上泛化良好，显著优于固定交替模式
- **SimpleGDN**：去除 Gated DeltaNet 的 Conv1d 和显式门控，直接复用预训练 QKV 权重映射到线性递归。在 HELMET-ICL 上甚至**超越全注意力基线**（+2.12@64K, +4.48@128K），且在 MRCR/RepoQA 上也优于原版 GDN

#### 三、综合评价

| 维度 | 评价 |
|------|------|
| 范式级架构创新 | ❌ 未提出全新的 attention 或 MoE 范式 |
| 组件级工程优化 | ✅ Muon Split、MLA-256、MTP 参数共享均解决了实际痛点 |
| 训练方法创新 | ✅ 20B tokens DSA 适应方案、DSA+MLA 组合验证具有方法论价值 |
| 高效注意力探索 | ✅ SimpleGDN 和 SWA Pattern 为社区提供了有价值的消研结论 |

**总结**：GLM-5 更像是在 DeepSeek 架构框架上的**深度工程优化与系统创新**，而非架构范式的原创者。其真正的核心竞争力在于**全异步 Agentic RL 框架、IcePop RL 算法、国产芯片全栈适配**等训练和部署层面的系统级贡献。
