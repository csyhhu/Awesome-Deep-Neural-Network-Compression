# 注意力高效化技术综述（讨论整理）

本文档整理自关于 **FlashAttention**、**PagedAttention** 与 **Linear Attention**（含 Linformer 等）的讨论，按问题层次与优化目标分类，便于与仓库内 `Paper/Large-Pretraining-Models/Attention-Acceleration.md` 对照阅读。

---

## 一、总览：三类技术解决的不是同一个问题

| 类别 | 代表 | 优化层次 | 是否改注意力公式 | 序列长度 \(n\) 渐近（注意力部分） | 典型场景 |
|------|------|----------|------------------|-----------------------------------|----------|
| **精确注意力加速** | FlashAttention | 算子 / HBM 访存 | 否（精确 Softmax） | \(O(n^2)\) 计算，\(O(n)\) 中间显存 | 训练、prefill、通用 Transformer |
| **推理 KV 内存管理** | PagedAttention (vLLM) | Serving 系统 / KV 布局 | 否 | 不改变 FLOPs 阶，提高显存利用率 | 多请求在线推理、动态 batch |
| **线性 / 近似注意力** | Linformer, Performer, RWKV, Mamba 等 | 算法 / 建模 | 是（或低秩近似 \(P\)） | \(O(n)\) 或 \(O(nk)\) | 超长序列、显存极紧、可接受近似 |

**关系**：FlashAttention 与 PagedAttention **可叠加**（vLLM 常见组合：分页 KV + Flash 类 kernel）；二者与 Linear Attention **正交**——前者算「标准注意力怎么算/怎么存」，后者算「要不要还算标准全局 Softmax」。

```text
                    ┌─────────────────────────────────────┐
                    │         标准 Softmax Attention       │
                    └─────────────────────────────────────┘
                           │                    │
              算法近似      │                    │  实现/系统优化
                           ▼                    ▼
              ┌────────────────────┐   ┌──────────────────────────┐
              │  Linear Attention   │   │ FlashAttention (怎么算)   │
              │  Linformer, Performer│   │ PagedAttention (KV 怎么存) │
              │  RWKV, Mamba, ...   │   └──────────────────────────┘
              └────────────────────┘
```

---

## 二、FlashAttention（精确注意力 · IO 优化）

### 2.1 是什么

- **论文**：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)（Dao et al., 2022）；后续 **FA-2**、**FA-3**（Hopper / FP8 等）。
- **目标**：在 **数学等价** 于标准 Softmax attention 的前提下，减少 **HBM 读写** 与 **\(n\times n\) 中间矩阵** 显存。

### 2.2 不是什么

- **不是** Linear Attention：复杂度对 \(n\) 仍为 **\(O(n^2)\)**（每个 query 仍覆盖所有 key，因果 mask 下为所有历史 key）。
- **不能** 在任意长 \(n\) 上达到 Linear Attention 的渐近速度；长 \(n\) 时设计良好的 \(O(n)\) 方法在算子时间上可能反超。

### 2.3 核心机制

1. **分块（Tiling）**：在片上 SRAM 分块加载 \(Q,K,V\)，不物化完整 \(S=QK^\top\)、\(P=\mathrm{softmax}(S)\)。
2. **Online Softmax**：按 \(K,V\) 块递推每行的 \(m\)（行 max）、\(\ell\)（归一化分母）、部分输出 \(O\)，多块合并后与整行 Softmax **等价**。
3. **Kernel Fusion**：`matmul → max/exp/sum → 累加 O` 融合，减少 kernel 与 HBM 往返。
4. **反向重计算**：存 \((m,\ell)\) 等 \(O(n)\) 统计量，反向在片上重算 \(P\)，不存完整 \(P\)。

### 2.4 Softmax 在 FA 中如何被「优化」

| 朴素实现 | FlashAttention |
|----------|----------------|
| 整行 \(S_{i,:}\) 进 HBM，再 softmax | 分块 \(S_{ij}\) 仅在 SRAM，**online 更新** \(m,\ell,O\) |
| 存完整 \(P\in\mathbb{R}^{n\times n}\) | **不写** \(P\) 到 HBM |
| softmax 与 \(PV\) 分离 | **融合**：新块 softmax 权重直接累加进 \(O\) |

数值稳定仍用 **减行最大值**；causal mask 在块内对非法位置置 \(-\infty\)。

### 2.5 实际加速比（量级，依赖基线与 \(n\)）

| 范围 | 相对 PyTorch 朴素 attention | 相对 Megatron 等强基线 | 整模训练 |
|------|----------------------------|------------------------|----------|
| 注意力算子 | 约 **2–4×**（\(n\) 约 128–2K）；GPT-2 上单算子可达 **~7×** | 差距缩小 | GPT-2 **~3×** vs HF；**~1.7×** vs Megatron；BERT **~15%** vs MLPerf |
| 显存（注意力相关） | \(n=2\text{K}\) 约 **10×**；\(n=4\text{K}\) 约 **20×** | — | 可更大 batch / 更长 context |

**FA-2**：相对 FA-1 约 **2×**；GPU 利用率约 **50–73%** 峰值（A100）。  
**注意**：decode 单 token 时收益常小于 prefill；短 \(n\) 时提升有限。

### 2.6 与 Linear Attention 的速度交叉（经验区间）

| 序列长 \(n\) | 相对常见 Linear / 低秩实现（当年论文 baseline） |
|--------------|--------------------------------------------------|
| \(\lesssim 512\) | FA 往往 **更快** |
| \(\approx 512\)–\(2048\) | **摇摆区**（实现质量影响大） |
| \(\gtrsim 2\text{K}\)–\(4\text{K}+\) | 成熟 \(O(n)\) 实现 **可能更快**（但是近似注意力） |

---

## 三、PagedAttention（推理 KV · 虚拟内存式管理）

### 3.1 是什么

- **论文 / 系统**：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（Kwon et al., vLLM, SOSP 2023）。
- **目标**：解决自回归推理中 **KV cache** 的动态增长、**碎片** 与 **过度预留**（传统实现可浪费约 **60%–80%** 显存）。

### 3.2 机制（类比 OS 分页）

- 将 KV cache 切成固定大小 **block（页）**。
- **逻辑上**序列连续；**物理上** block 可非连续存放。
- **Block table** 映射 logical block → physical block；按需分配 / 释放。
- 配套 **PagedAttention kernel**：按表 gather 非连续 \(K,V\) 做 attention。
- 支持 **prefix 共享**、beam search 等 **copy-on-write**。

### 3.3 与 FlashAttention 对比

| 维度 | FlashAttention | PagedAttention |
|------|----------------|----------------|
| 优化对象 | 单次 attention **计算** | 多请求 **KV 存储与分配** |
| 主要节省 | \(n\times n\) 中间矩阵 / 激活 | KV **碎片**、reserved 空间 |
| 训练 | 常用 | 主要针对 **推理 serving** |
| 是否替代对方 | **否**；vLLM 等常 **同时使用** |

**论文报告**：相对 FasterTransformer、Orca，相同延迟下吞吐约 **2–4×**（长序列、大模型更明显）。

### 3.4 问答：PagedAttention 只能用于推理吗？

**结论**：**不是数学上「仅限推理」**，但 **设计目标与主战场是 LLM 在线 serving（自回归推理）**；常规 **预训练 / 全序列训练** 几乎不用 PagedAttention。

#### 为何默认是推理技术？

PagedAttention 针对的是：

- KV cache 随 **decode 逐步变长**（每生成 1 token 追加 K/V）
- 请求长度 **不可预知**，连续预留 max_len → **碎片与浪费**（论文/博客约 **60%–80%**）
- **多请求 batch**、prefix 共享、beam search 等 serving 调度

标准 **训练 forward** 中这些特征不明显：

| | **训练（常见）** | **推理 decode** |
|---|------------------|-----------------|
| 序列 | 整段一次 forward（或 varlen packed） | **逐 token** 增长 |
| KV | 当前步即时算 QKV，一般不长期保留「分页 KV」 | **必须缓存** 历史 K/V |
| 显存瓶颈 | 激活、attention 计算（多用 **FlashAttention**） | **KV 总量** + 碎片 |

#### 训练里能否使用？

- **普通 LM 预训练 / SFT**：用 FlashAttention、varlen、checkpointing 等即可；**PagedAttention 收益很小**。
- **仍可能沾边的场景**（本质是生成/推理子问题）：
  - **RL / on-policy rollout** 自回归采样 → 常用 vLLM 等推理栈（含 PagedAttention）
  - **推测解码、多轮对话 KV 复用** → 推理管线内
- **Prefill**：在 vLLM 中 prefill 也会把 prompt KV 写入 **分页 block**，但仍属于 **推理管线**，不是 training backward。

#### 与 FlashAttention 的分工

- **训练**：主流是 **FlashAttention（类）** 算 attention。
- **推理 serving**：**PagedAttention 管 KV 布局** + 常搭配 **FlashAttention** 算 attention。

**一句话**：PagedAttention 解决的是 serving 里 **动态、多请求、非连续 KV**；标准训练没有同等痛点。长上下文 **预训练/中训** 应优先 **FA、GQA/CCA、varlen**，而非 PagedAttention。

---

## 四、Linear Attention（算法层 · 降复杂度）

### 4.1 共同目标

将 attention 对序列长度 \(n\) 的代价从 **\(O(n^2)\)** 降为 **\(O(n)\)**（或 \(O(nk)\) 且 \(k\ll n\) 固定），通常 **改变或近似** 标准 Softmax 全局注意力。

### 4.2 经典方法分类

#### A. 可分解核 / 线性 RNN（\(\phi(Q)(\phi(K)^\top V)\)）

| 方法 | 论文 | 要点 |
|------|------|------|
| **Linear Transformer** | [2006.16236](https://arxiv.org/abs/2006.16236) | 核技巧 + 结合律；推理可维护状态 |
| **Performer** | [2009.14794](https://arxiv.org/abs/2009.14794) | FAVOR+ 随机特征近似 Softmax 核 |
| **Kernel 统一视角** | [1911.03594](https://arxiv.org/abs/1911.03594) | 从核角度理解多种 attention |

#### B. 低秩 / 投影（仍可用 Softmax，但 \(P\) 低秩）

| 方法 | 论文 | 要点 |
|------|------|------|
| **Linformer** | [2006.04768](https://arxiv.org/abs/2006.04768) | 对 \(K,V\) 沿序列维投影到长度 \(k\)；\(\bar P\in\mathbb{R}^{n\times k}\) |
| **Nyströmformer** | [2102.03902](https://arxiv.org/abs/2102.03902) | Nyström 近似注意力矩阵 |

#### C. 状态空间 / 线性 RNN（LLM 长序列热点）

| 方法 | 论文 | 要点 |
|------|------|------|
| **RetNet** | [2307.08621](https://arxiv.org/abs/2307.08621) | 衰减线性 retention；可并行训、递推推 |
| **RWKV** | [2305.13048](https://arxiv.org/abs/2305.13048) | WKV 机制；推理步相对 \(n\) 省 KV |
| **Mamba** | [2312.00752](https://arxiv.org/abs/2312.00752) | 选择性 SSM；线性时间序列建模 |

#### D. 工程向线性 attention（大模型）

| 方法 | 说明 |
|------|------|
| **Lightning Attention-2** | 面向极长上下文的线性 attention 实现与分块 |
| **TransNormerLLM** | 线性 attention + Norm 训大模型 |

仓库 `Paper/Large-Pretraining-Models/Attention-Acceleration.md` 已列部分条目，可继续扩展 Performer、Linformer、RetNet、Mamba 等链接。

---

## 五、Linformer（Linear Attention 代表 · 详要）

### 5.1 动机

标准 attention 的 context 矩阵 \(P\in\mathbb{R}^{n\times n}\) 低秩（谱分析 + Theorem 1）；可用 **低秩** 近似，避免 \(O(n^2)\) 时空。

### 5.2 公式（单头）

\[
\overline{\mathrm{head}}_i
= \mathrm{softmax}\!\left(\frac{QW_i^Q (E_i K W_i^K)^\top}{\sqrt{d}}\right)
\cdot (F_i V W_i^V)
\]

- \(E_i,F_i\)：序列维投影 \(n \to k\)，\(k\ll n\)。
- \(\bar P\in\mathbb{R}^{n\times k}\)，复杂度 **\(O(nk)\)**；固定 \(k\) 时为 **\(O(n)\)**。
- **\(Q\) 不投影**；压缩在 \(K,V\) 侧。

### 5.3 实验结论（论文量级）

- 预训练：\(n{=}512,k{=}128\) 接近 RoBERTa；**性能主要由 \(k\) 决定**，而非 \(n/k\)。
- 下游：\(k{=}256\) + layerwise 共享可与 RoBERTa **打平或略好**。
- 推理（\(n{=}512,k{=}128\)）：约 **1.5×** 加速、**1.7×** 最大 batch；\(n{=}65536\) 可达约 **20×** 加速 / **60×** 显存（相对标准 Transformer）。

### 5.4 局限

- 全局交互瓶颈在 **\(k\)**；自回归 decode 需额外设计；产业界 LLM 主流仍是 **Softmax + FA**，Linformer 多作 baseline / 长序列 encoder 参考。

---

## 六、横向对照表（选型速查）

| 问题 | 优先考虑 |
|------|----------|
| 训练 / prefill 太慢、attention OOM（仍要精确注意力） | **FlashAttention** |
| 多用户推理、KV 浪费、batch 上不去 | **PagedAttention**（+ FA 类 kernel） |
| 标准预训练 / SFT（整段 forward） | **FlashAttention**；一般 **不用** PagedAttention |
| 训练流程中的 RL rollout / 批量生成 | 推理侧可用 **PagedAttention**（生成子阶段） |
| 序列极长、可接受非标准注意力 | **Linear / Linformer / Mamba** 等 |
| 既要长上下文又要 GPT 级 Softmax 质量 | **FA** + **KV 压缩（GQA/CCA）** 等；而非单靠 Linformer |
| 只优化 Softmax 实现、不换公式 | **FA**，不是 Linear |
| 只优化 KV 存放、不换 attention 定义 | **PagedAttention**，不是 FA |

---

## 七、概念辨析（易混点）

1. **FlashAttention ≠ 线性 attention**：FA 是 \(O(n^2)\) 精确实现的 IO 优化。
2. **PagedAttention ≠ 另一种 FlashAttention**：Paged 管 **KV 分页**；FA 管 **算子内 softmax/matmul**。
3. **Linformer 是 Linear 族，但不是核技巧线**：低秩投影 + Softmax，与 Performer 不同。
4. **FA 与 Linear 的「交叉区间」**：短–中 \(n\) FA 常更快；长 \(n\) 成熟 Linear 可能在算子上反超，但精度/能力需单独评估。
5. **Online softmax**：FA 的核心之一；与「线性 softmax 近似」无关。
6. **PagedAttention ≠ 仅推理的「硬限制」**：但标准训练不用；价值在 **动态 KV + serving**（见 §3.4）。

---

## 八、推荐阅读

| 主题 | 链接 |
|------|------|
| FlashAttention | https://arxiv.org/abs/2205.14135 |
| FlashAttention-2 | https://arxiv.org/abs/2307.08691 |
| PagedAttention / vLLM | https://arxiv.org/abs/2309.06180 |
| Linformer | https://arxiv.org/abs/2006.04768 |
| Linear Transformer | https://arxiv.org/abs/2006.16236 |
| Performer | https://arxiv.org/abs/2009.14794 |
| IO 最优性讨论 | *I/O Complexity of Attention, or How Optimal is FlashAttention?*（见 `Paper/Conference/2024.md`） |

---

## 九、修订说明

- **来源**：对话讨论整理（FlashAttention 原理与加速比、FA vs Linear 交叉区间、Linformer 详解、PagedAttention vs FA、Online Softmax、**PagedAttention 是否仅用于推理**）。
- **修订**：补充 §3.4 问答与 §六选型表、§七易混点第 6 条。
- **路径**：`Summary/Large Pretraining Models/summary_attention_efficiency.md`
- **关联**：`Paper/Large-Pretraining-Models/Attention-Acceleration.md`
