# Kimi K3: Open Frontier Intelligence

> **来源**: [Kimi K3 发布博客](https://www.kimi.com/blog/kimi-k3)
> **机构**: Kimi（Moonshot AI）
> **发布日期**: 2026年7月
> **模型参数**: 2.8T 参数
> **上下文窗口**: 100万 token

---

## 1. 概述

Kimi K3 是 Kimi 推出的旗舰级大语言模型，是**世界首个开放的 3T 级模型**。核心特性：

- **2.8T 参数规模**：基于 Kimi Delta Attention (KDA) 和 Attention Residuals (AttnRes) 架构
- **原生视觉能力**：支持图像、截图、视频理解
- **100万 token 上下文窗口**：原生支持超长文本处理
- **Stable LatentMoE**：896 个专家中有效激活 16 个
- **整体缩放效率提升约 2.5×**：相比 Kimi K2

虽然整体性能仍落后于最强大的闭源模型（Claude Fable 5 和 GPT 5.6 Sol），但在评估套件中展现了前沿水平，持续超越其他测试模型。

---

## 2. 模型架构

### 2.1 核心架构组件

#### Kimi Delta Attention (KDA)

**早期论文**：[Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)（2025年11月）

**核心思想**：KDA 是一种**混合线性注意力机制**，扩展自 Gated DeltaNet，通过更细粒度的门控机制实现有限状态 RNN 内存的有效利用。

**技术细节**：
- **对角+低秩（DPLR）过渡矩阵**：使用专门设计的块级算法，大幅降低计算量
- **位置级门控**：将门控决策从层级别移动到 token 位置级别，允许不同信息通道有不同的遗忘速度
- **与标准 Attention 的区别**：标准 softmax attention 是二次复杂度 \(O(L^2)\)，而 KDA 是线性复杂度 \(O(L)\)

**训练阶段计算示例**：
```
输入序列 x = [x₁, x₂, x₃, ..., xₙ]

# KDA 前向传播
# 1. 计算 Q、K、V
Q = x · W_Q
K = x · W_K  
V = x · W_V

# 2. Delta 注意力核心计算（线性复杂度）
# 使用状态空间模型方式，维护一个持续更新的状态
state = 0
output = []
for i in 1..n:
    # 门控机制：决定当前 token 对状态的更新强度
    gate = σ(Q[i] · W_gate)
    
    # Delta 更新：选择性更新状态
    delta = K[i] · V[i]
    state = state * decay + gate * delta
    
    # 输出：从状态读取
    output[i] = Q[i] · state

# 3. 最终输出
y = output · W_out
```

**推理阶段计算示例**：
```
# Prefill 阶段：处理前缀序列
prefill_state = 0
for i in 1..prefix_len:
    gate = σ(Q[i] · W_gate)
    delta = K[i] · V[i]
    prefill_state = prefill_state * decay + gate * delta

# Decode 阶段：逐 token 生成（线性复杂度）
# 只需维护一个状态，无需重新计算整个前缀
current_state = prefill_state
for step in 1..max_new_tokens:
    # 仅计算当前 token 的 Q、K、V
    Q_t = x_t · W_Q
    K_t = x_t · W_K
    V_t = x_t · W_V
    
    # 更新状态
    gate = σ(Q_t · W_gate)
    delta = K_t · V_t
    current_state = current_state * decay + gate * delta
    
    # 生成下一个 token
    logits = Q_t · current_state · W_out
    x_{t+1} = argmax(logits)
```

**优势**：
- 1M token 上下文下解码加速可达 6.3×
- KV Cache 减少 75%
- 在短、长上下文和 RL 场景下均优于全注意力

---

#### Attention Residuals (AttnRes)

**早期论文**：[Attention Residuals](https://arxiv.org/abs/2603.xxxxx)（2026年3月），代码开源于 [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)

**核心思想**：将标准残差连接的固定权重累加替换为**深度方向的 softmax 注意力**，使每一层能够依据输入内容自适应地选择聚合哪些前序层的表征。

**标准残差连接的问题**：
```
# 标准 PreNorm 残差
hₗ = hₗ₋₁ + fₗ₋₁(hₗ₋₁)
# 展开后：hₗ = h₀ + f₀(h₀) + f₁(h₁) + ... + fₗ₋₁(hₗ₋₁)
# 所有层贡献权重相等（均为1），导致信息稀释
```

**AttnRes 解决方案**：
```
# AttnRes：用 softmax 注意力替代固定权重
# 每一层 l 有一个可学习的伪查询 wₗ ∈ ℝᵈ
# K、V 是所有前序层输出的 RMSNorm 结果

# 步骤 1：收集所有前序层输出
V = [b₀, b₁, ..., bₙ₋₁]  # b_i 是第 i 个块的压缩表征

# 步骤 2：RMSNorm 归一化（防止深层块主导注意力）
K = RMSNorm(V)

# 步骤 3：计算深度注意力分数
logitsᵢ = Kᵢ · wₗ  # 对每个块 i

# 步骤 4：Softmax 归一化（沿深度维度）
αᵢ = softmax(logits)ᵢ

# 步骤 5：加权组合
h = Σ αᵢ · Vᵢ
```

**Block AttnRes**（大规模模型实用方案）：
- 将 L 层划分为 N 个块（例如每 16 层一个块）
- 在块级别上进行注意力计算，而非层级别
- 内存复杂度从 \(O(Ld)\) 降至 \(O(Nd)\)
- 训练开销 < 4%，推理延迟增加 < 2%

**训练阶段计算示例**：
```
# 假设模型有 64 层，分为 4 个块（每块 16 层）

# 初始化：伪查询 wₗ 初始化为零，保证模型开始时是标准残差
w = [0, 0, 0, 0]  # 每个块一个伪查询

# 训练过程
for block_idx in 0..3:
    # 执行当前块内的 16 层 Transformer 计算
    block_output = process_block(block_idx, input)
    
    # 将块输出存入块缓存
    block_cache[block_idx] = block_output
    
    # 计算 AttnRes（如果不是第一个块）
    if block_idx > 0:
        # 收集所有已完成块的表征
        V = [block_cache[0], block_cache[1], ..., block_cache[block_idx]]
        
        # RMSNorm 归一化
        K = RMSNorm(V)
        
        # 计算注意力分数（使用当前块的伪查询）
        logits = K · w[block_idx]
        
        # Softmax
        α = softmax(logits)
        
        # 加权组合 → 作为下一个块的输入
        input = Σ αᵢ · Vᵢ
```

**推理阶段计算示例**：
```
# 推理时，块缓存可以预先计算并复用
# 对于长序列，只需更新最新块的表征

# 预计算所有块的输出（Prefill）
block_outputs = []
for block_idx in 0..3:
    output = process_block(block_idx, input)
    block_outputs.append(output)

# 计算 AttnRes 最终输出
V = block_outputs
K = RMSNorm(V)
logits = K · w[-1]  # 使用最后一个块的伪查询
α = softmax(logits)
final_output = Σ αᵢ · Vᵢ

# Decode 阶段：只需更新最后一个块，AttnRes 权重保持不变
# 因为伪查询 w 在推理时是固定的
```

**优势**：
- 缓解 PreNorm 稀释问题，使各层输出幅值和梯度分布更均匀
- 在 Kimi Linear 架构上预训练 1.4T token，所有下游任务性能提升
- 训练效率提升约 25%

---

#### Stable LatentMoE

**早期论文**：[LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts](https://arxiv.org/abs/2601.18089)（2026年1月）

**核心思想**：从软硬件协同设计角度重新审视 MoE 设计，优化单位计算的准确率。

**Kimi K3 中的 Stable LatentMoE**：
- **专家配置**：896 个专家，有效激活 16 个（仅 1.8% 的专家激活率）
- **关键技术**：

**1. Quantile Balancing（分位数平衡）**
```
# 标准 MoE 路由：每个 token 分配给 top-k 专家
router_logits = x · W_router  # [batch, num_experts]
top_k_experts = top_k(router_logits, k=16)

# Quantile Balancing：直接从路由分数分位数导出专家分配
# 消除启发式更新和敏感的平衡超参数
quantiles = compute_quantiles(router_logits)
expert_allocation = derive_from_quantiles(quantiles)
```

**2. Per-Head Muon（逐头 Muon 优化）**
```
# 标准 Muon：在整个注意力层级别进行优化
# Per-Head Muon：独立优化每个注意力头
for head_idx in 0..num_heads:
    # 对每个头独立应用 Muon 优化器
    W_head = muon_optimize(W_head, grad_head)
```

**3. Sigmoid Tanh Unit (SiTU)**：改进激活控制
```
# SiTU 激活函数
def situ(x):
    return sigmoid(x) * tanh(x)
```

**4. Gated MLA**：增强注意力选择性

**训练阶段计算示例**：
```
# MoE 前向传播
input = [x₁, x₂, ..., xₙ]

# 1. 路由计算
router_logits = input · W_router  # [n, 896]

# 2. Quantile Balancing：确定专家分配
expert_assignments = quantile_balance(router_logits, target_experts=16)

# 3. 专家前向
expert_outputs = []
for expert_idx in 0..895:
    # 只有被分配的专家才会被激活
    if expert_idx in expert_assignments:
        tokens_for_expert = select_tokens(input, expert_idx, expert_assignments)
        output = tokens_for_expert · W_expert_1
        output = swiGLU(output)
        output = output · W_expert_2
        expert_outputs.append((expert_idx, output))

# 4. 组合输出
# 使用路由权重进行加权
combined_output = weighted_sum(input, expert_outputs, router_logits)

# 5. 残差连接（配合 AttnRes）
final_output = attnres(combined_output, previous_block_outputs)
```

**推理阶段计算示例**：
```
# Prefill 阶段：处理前缀序列
prefix_expert_outputs = []
for expert_idx in 0..895:
    tokens = select_tokens(prefix, expert_idx)
    if tokens is not empty:
        output = tokens · W_expert_1 · swiGLU · W_expert_2
        prefix_expert_outputs.append((expert_idx, output))

# Decode 阶段：逐 token 生成
for step in 1..max_new_tokens:
    # 1. 路由计算（仅当前 token）
    router_logits = x_t · W_router
    
    # 2. 选择 top-16 专家
    top_16_experts = top_k(router_logits, k=16)
    
    # 3. 仅激活这 16 个专家
    expert_results = []
    for expert_idx in top_16_experts:
        output = x_t · W_expert_1[expert_idx]
        output = swiGLU(output)
        output = output · W_expert_2[expert_idx]
        expert_results.append(output)
    
    # 4. 组合输出
    combined = weighted_sum(expert_results, router_logits)
    
    # 5. 生成下一个 token
    logits = combined · W_out
    x_{t+1} = argmax(logits)
```

**优势**：
- 极端稀疏激活（仅 1.8%），保持推理计算可控
- Quantile Balancing 消除了专家负载不平衡问题
- 整体缩放效率相比 K2 提升约 2.5×

### 2.2 训练与优化

- **量化感知训练 (QAT)**：从 SFT 阶段开始应用，使用 MXFP4 权重 + MXFP8 激活，实现广泛硬件兼容性
- **完全平衡的专家并行训练**：静态形状，关键路径上无主机同步，防止专家不平衡导致的吞吐下降
- **推理部署建议**：推荐在 64 或更多加速器的超级节点配置上部署，以充分利用高带宽通信域
- **KDA 前缀缓存**：已向 vLLM 社区贡献对应实现，与预填充缓存配合实现具有竞争力的 token 价格

---

## 3. 核心能力展示

### 3.1 编码能力

#### 长程编码
- 可在最小人工监督下维持长时间工程会话
- 能够导航大规模代码仓库
- 可编排终端工具

#### 内核优化
- 在 GPU 内核优化任务上与 Claude Fable 5 竞争，大幅超越 Opus 4.8、GPT 5.6 Sol 和 GPT 5.5
- 支持 NVIDIA H200 和其他厂商 GPGPU 平台
- 在开发后期，早期版本的 K3 已处理团队大部分内核优化工作

#### GPU 编译器开发
- 从零构建 MiniTriton：一个紧凑的 Triton 类编译器
- 包含自己的 tile-level IR 层（基于 MLIR）、优化 pass 和 PTX 代码生成流水线
- 在 roofline 基准测试中性能与 Triton 持平或更好，部分工作负载超越 Triton
- 支持端到端 nanoGPT 训练，收敛稳定

#### 游戏开发与数字创作
- 结合 3D 推理、编码和视觉能力，将概念、图像和视频转化为可玩的交互式体验
- 实现"视觉在环"：在代码和实时截图之间无缝迭代

#### 芯片设计
- 作为早期概念验证，K3 设计了一个为自己架构上构建的纳米模型服务的芯片
- 在 48 小时内完成芯片的构建、优化和验证（使用开源 EDA 工具，基于 Nangate 45nm 工艺库）
- 芯片规格：4 mm² 面积，100 MHz 时序收敛，8,700+ tokens/s 解码吞吐，1.46M 标准单元，0.277 MB SRAM，带融合反量化的 INT4 MAC 阵列

#### 科研编码
- 桥接科学文献与可执行代码，自主实现、验证和分析复杂计算研究工作流
- 案例：在约 2 小时内完成通常需要 1-2 周的天体物理学研究（I-Love-Q 普适关系复现），包括：
  - 审查和交叉验证 20+ 篇论文
  - 实现完整数值流水线
  - 评估 300+ 状态方程
  - 识别已发表公式中的不一致
  - 生成 3,000+ 行 Python 代码
  - 生成交互式 HTML 仪表板

### 3.2 知识工作

#### 交互式研究可视化
- **案例 1：42年 AI ASIC 产业研究网站**：通过 120+ 轮递归自我改进创建，包含 2.8k+ 网页搜索/获取、1.1k+ 终端数据拉取，覆盖 11k+ 页面（87 份季度报告和 99 份原始 PDF）
- **案例 2：核聚变产业研究**：咨询风格的行业报告，包含时间线、漏斗图、范围条形图、甘特图和出版物级幻灯片
- **案例 3：GWTC-5 引力波分析**：使用 20+ 并发子智能体分析 391 个引力波事件，生成 7 个科学可视化、2 个表格和 10+ 篇论文的文献综述

#### Widgets 和 Dashboard
- **Widgets**：在聊天中直接生成交互式组件，可连接本地数据或外部插件实现持续更新
- **Dashboard**：将关注的 Widgets 整合到一个持久的个性化视图中，围绕主题、项目或目标组织

#### 视频编辑
- 利用原生多模态架构，在同一模型内理解文本、图像和视频
- 案例 1：创建 3Blue1Brown 风格的架构动画解释视频
- 案例 2：从 56 个源片段编辑自己的预告片，处理片段选择、动作匹配剪辑、帧级精确节拍同步、音频处理和多轮修订

---

## 4. 可用性

| 平台 | 说明 |
|------|------|
| **Kimi.com** | 网页版，已上线 |
| **Kimi Work** | 桌面应用，Windows 和 Apple Silicon Macs，版本 3.1.0+ |
| **Kimi Code** | 终端工具，使用 `/model` 命令选择 |
| **Kimi API** | 价格：缓存命中输入 $0.30/MTok，缓存未命中输入 $3.00/MTok，输出 $15.00/MTok |
| **Kimi Enterprise** | 企业级数据隐私和成员管理 |
| **模型权重** | 将于 2026年7月27日发布 |

---

## 5. 核心贡献总结

| 贡献 | 说明 |
|------|------|
| **首个开放 3T 级模型** | 2.8T 参数，打破开源模型规模上限 |
| **Kimi Delta Attention (KDA)** | 为注意力缩放提供高效基础 |
| **Attention Residuals (AttnRes)** | 跨深度选择性检索表示，改善信息流动 |
| **Stable LatentMoE** | 896→16 专家激活，通过 Quantile Balancing 和 Per-Head Muon 实现稳定高效训练 |
| **缩放效率提升 2.5×** | 相比 Kimi K2，更高效地将计算转化为智能 |
| **MXFP4/MXFP8 QAT** | 从 SFT 阶段开始的量化感知训练，支持广泛硬件 |
| **vLLM KDA 前缀缓存** | 社区贡献，实现高效长上下文推理 |
| **端到端 AI 工程能力** | 从内核优化到编译器开发到芯片设计的完整工程链路 |

---

## 6. 与模型压缩/高效推理的关系

Kimi K3 在多个维度体现了模型压缩与高效推理的关键技术：

- **KDA + AttnRes**：通过选择性注意力机制降低计算复杂度
- **MoE 稀疏激活**：896 个专家中仅激活 16 个，有效降低计算量
- **MXFP4/MXFP8 混合精度**：权重使用 MXFP4，激活使用 MXFP8，显著降低显存占用
- **量化感知训练**：从 SFT 阶段开始的 QAT，确保低精度推理质量
- **完全平衡的专家并行**：优化大规模专家并行训练的吞吐量
- **KDA 前缀缓存**：为 vLLM 贡献实现，优化长上下文推理效率

---

## 7. 讨论

### Q1: Kimi K3 在架构上的核心创新是什么？

K3 的核心架构创新在于三个方面：

1. **Kimi Delta Attention (KDA)**：为注意力机制提供高效的缩放基础
2. **Attention Residuals (AttnRes)**：有选择地跨深度检索表示，而非均匀累积
3. **Stable LatentMoE**：通过 Quantile Balancing（分位数平衡）和 Per-Head Muon（逐头 Muon 优化）解决大规模 MoE 的路由和优化挑战

这三者共同构成了能够良好扩展到万亿参数级别的架构基础。

### Q2: Kimi K3 的量化策略是什么？

K3 采用 **MXFP4/MXFP8 混合精度量化**：
- **权重**：MXFP4 格式
- **激活**：MXFP8 格式
- **量化起点**：从 SFT 阶段开始应用量化感知训练 (QAT)

这种策略实现了广泛的硬件兼容性，同时保持了模型性能。

### Q3: Kimi K3 在长上下文方面的能力如何？

K3 原生支持 **100万 token 上下文窗口**，配合 KDA 和 AttnRes 架构设计，能够有效处理超长序列。通过向 vLLM 社区贡献 KDA 前缀缓存实现，确保了长上下文推理的高效性。

### Q4: KDA、AttnRes、LatentMoE 有更早的论文吗？详细介绍这三个模块的内容，并通过例子说明训练和推理中的计算过程。

这三个模块都有独立的早期论文：

#### KDA 早期论文

- **论文**：[Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)（2025年11月）
- **代码**：https://github.com/MoonshotAI/Kimi-Linear

**核心设计**：
KDA 是一种混合线性注意力机制，扩展自 Gated DeltaNet。关键创新是**位置级门控**——将门控决策从层级别移动到 token 位置级别，允许不同信息通道有不同的遗忘速度。

**训练计算示例**（假设有 4 个 token 的序列）：
```
输入: x = [x₁, x₂, x₃, x₄]

# 计算 Q, K, V
Q = [q₁, q₂, q₃, q₄]
K = [k₁, k₂, k₃, k₄]
V = [v₁, v₂, v₃, v₄]

# Delta 注意力（线性复杂度）
state = 0
output = []

# i=1: 第一个 token
gate₁ = σ(q₁ · w_gate) = 0.6
delta₁ = k₁ · v₁ = [2, 3]
state = 0 * decay + 0.6 * [2, 3] = [1.2, 1.8]
output₁ = q₁ · state = 5.0

# i=2: 第二个 token
gate₂ = σ(q₂ · w_gate) = 0.8
delta₂ = k₂ · v₂ = [1, -1]
state = [1.2, 1.8] * 0.9 + 0.8 * [1, -1] = [1.88, 0.82]
output₂ = q₂ · state = 4.5

# ... 继续处理 x₃, x₄

# 最终：每个 token 的输出只依赖当前状态，无需与所有历史 token 计算点积
```

**推理计算示例**：
```
# Prefill：处理前缀
state = 0
for i = 1..prefix_len:
    gate = σ(q_i · w_gate)
    delta = k_i · v_i
    state = state * decay + gate * delta

# Decode：只需维护一个状态
# 生成第 1 个新 token
gate = σ(q_new · w_gate)
delta = k_new · v_new
state = state * decay + gate * delta
logits = q_new · state

# 生成第 2 个新 token（仅更新状态，无需重新计算前缀）
gate = σ(q_new2 · w_gate)
delta = k_new2 · v_new2
state = state * decay + gate * delta
logits = q_new2 · state
```

---

#### AttnRes 早期论文

- **论文**：[Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf)（2026年3月）
- **代码**：https://github.com/MoonshotAI/Attention-Residuals

**核心设计**：
将标准残差连接的固定权重累加替换为深度方向的 softmax 注意力。每一层有一个可学习的伪查询向量，用于选择性地聚合前序层的表征。

**训练计算示例**（假设有 4 个块，每块 16 层）：
```
# 初始：伪查询 w = [0, 0, 0, 0]（零初始化 = 标准残差）

# Block 0: 第一层 Transformer 块
block_output_0 = transformer_block(input)  # 输出 [10, 5, 3, ...]
block_cache = [block_output_0]

# Block 1: 第二层 Transformer 块
block_output_1 = transformer_block(block_output_0)  # 输出 [8, 6, 4, ...]
block_cache = [block_output_0, block_output_1]

# AttnRes 计算（Block 1 结束时）
V = [block_output_0, block_output_1]
K = RMSNorm(V)  # 归一化防止深层块主导
logits = K · w[1]  # w[1] 刚开始接近 0，所以 logits ≈ [0, 0]
α = softmax([0, 0]) = [0.5, 0.5]  # 均匀权重 = 标准残差

# 训练一段时间后，w[1] 学会选择性关注
# 假设 w[1] 变为 [0.5, -0.3]
logits = K · [0.5, -0.3] = [2.0, 0.8]
α = softmax([2.0, 0.8]) = [0.84, 0.16]  # 更多关注 Block 0

# Block 2 & 3: 继续相同过程，伪查询逐步学习选择性
```

**推理计算示例**：
```
# Prefill：计算所有块输出
block_outputs = []
for i in 0..3:
    output = transformer_block(input)
    block_outputs.append(output)

# AttnRes 输出（伪查询 w 已训练好，推理时固定）
V = block_outputs
K = RMSNorm(V)
logits = K · w[-1]
α = softmax(logits)
final = α₀·V₀ + α₁·V₁ + α₂·V₂ + α₃·V₃

# Decode：只需更新最后一个块，AttnRes 权重不变
```

---

#### LatentMoE 早期论文

- **论文**：[LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts](https://arxiv.org/abs/2601.18089)（2026年1月）

**核心设计**：
从软硬件协同设计角度重新审视 MoE，优化单位计算的准确率。K3 中的 Stable LatentMoE 进一步引入 Quantile Balancing 和 Per-Head Muon。

**训练计算示例**（896 个专家，激活 16 个）：
```
input = [x₁, x₂, ..., x₁₀₀]  # 100 个 tokens

# 路由计算
router_logits = input · W_router  # [100, 896]

# Quantile Balancing：基于分位数分配专家
# 假设计算后：
# token x₁ → 专家 [12, 45, 67, ...]（16个）
# token x₂ → 专家 [23, 56, 78, ...]（16个）
# ...

# 专家计算（仅激活的专家）
# 专家 12 收到 token x₁, x₅, x₁₀, ...
output_12 = [x₁, x₅, x₁₀] · W_expert_1[12] · swiGLU · W_expert_2[12]

# 专家 45 收到 token x₁, x₃, x₈, ...
output_45 = [x₁, x₃, x₈] · W_expert_1[45] · swiGLU · W_expert_2[45]

# ... 其他激活的专家

# 组合输出
# 每个 token 的输出是其分配到的专家输出的加权和
output_x₁ = weight_12(x₁) · output_12 + weight_45(x₁) · output_45 + ...
```

**推理计算示例**：
```
# Prefill：处理前缀，计算所有专家的输出
expert_outputs = {}
for expert in 0..895:
    tokens = select_tokens(prefix, expert)
    if tokens:
        expert_outputs[expert] = tokens · W_expert_1[expert] · swiGLU · W_expert_2[expert]

# Decode：逐 token 生成
for step in 1..max_tokens:
    # 路由：当前 token → top-16 专家
    router_logits = x_t · W_router
    top_16 = top_k(router_logits, k=16)  # 例如 [12, 45, 67, ...]
    
    # 仅计算这 16 个专家
    results = []
    for expert in top_16:
        output = x_t · W_expert_1[expert] · swiGLU · W_expert_2[expert]
        results.append(output)
    
    # 组合
    combined = weighted_sum(results, router_logits)
    logits = combined · W_out
    x_{t+1} = argmax(logits)
```

---

#### 三个模块的协同作用

| 模块 | 解决的问题 | 计算复杂度 | 协同作用 |
|------|-----------|-----------|---------|
| **KDA** | 序列维度的注意力效率 | \(O(L)\) | 使 1M token 上下文可行 |
| **AttnRes** | 深度维度的信息稀释 | \(O(Nd)\) | 改善梯度流动，提升训练效率 |
| **Stable LatentMoE** | 参数规模与推理效率的平衡 | \(O(E_{active} \cdot d)\) | 支持 2.8T 参数但仅激活 16/896 专家 |

三者结合使 K3 能够在 2.8T 参数规模下实现高效训练和推理。