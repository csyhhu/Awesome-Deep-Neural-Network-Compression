# Speculative Decoding：Transformer 快速推理的无损加速

> **论文标题**: Fast Inference from Transformers via Speculative Decoding
>
> **作者**: Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research)
>
> **发表**: ICML 2023
>
> **链接**: https://arxiv.org/abs/2211.17192

---

## 1. 研究背景与动机

大语言模型（如 GPT-3、PaLM、LaMDA 等）的推理非常慢——解码 $K$ 个 token 需要 $K$ 次串行模型前向传播。此外，大模型的推理瓶颈往往不是**算术运算**，而是**内存带宽和通信**，这意味着额外的计算资源通常是可用的。

**核心观察**：
1. 困难的推理任务中常包含可以高效近似的"简单子任务"
2. 利用**投机执行（Speculative Execution）**的思想，可以让大模型并行验证小模型的预测，从而在保持输出分布不变的前提下加速推理

**目标**：无需改变模型架构、无需重新训练、不改变输出分布。

---

## 2. 核心方法

### 2.1 投机解码（Speculative Decoding）框架

```
┌──────────────────┐          ┌──────────────────┐
│  Draft Model M_q │ ──────→  │ Target Model M_p │ ──────→  Accept/Reject
│   (高效小模型)     │  候选token │   (要加速的大模型)   │         via MRS
└──────────────────┘          └──────────────────┘
```

1. 用更高效的小模型 $M_q$ 自回归生成 $\gamma$ 个候选 token
2. 用大模型 $M_p$ **并行**验证所有候选及其概率
3. 通过 **Speculative Sampling（投机采样）** 接受能在保持分布不变的前提下被接受的 token
4. 对第一个被拒绝的位置，从修正分布中重新采样

每轮 $M_p$ 的并行运行至少产生 1 个新 token（最坏情况不劣于标准解码），最优可产生 $\gamma+1$ 个 token。

### 2.2 投机采样（Speculative Sampling）—— 即 MRS

这是本文的核心创新，即我们之前讨论的 **MRS（Modified Rejection Sampling）**：

```python
def SpeculativeSample(p, q):
    """
    Input:  p(x) — target 分布 (来自 M_p)
            q(x) — draft 分布 (来自 M_q)
    Output: x ~ p(x), 保证输出严格服从 p

    步骤:
    1. 从 q 中采样 x ~ q(x)
    2. 如果 q(x) ≤ p(x): 接受 x
    3. 如果 q(x) > p(x): 以 1 - p(x)/q(x) 概率拒绝 → 从调整分布采样
       p'(x) = norm(max(0, p(x) - q(x)))
    """
    x = sample(q)
    r = uniform(0, 1)
    if r <= min(1.0, p(x) / q(x)):
        return x                    # 接受
    else:
        p_prime = normalize(max(0, p - q))  # 残差分布
        return sample(p_prime)      # 从残差重采样
```

**关键性质（Theorem 1）**：输出严格服从 $p$，**对任意 $q$ 成立**（$q$ 可以是任意差的小模型，甚至是 bigram 或 unigram）。

### 2.3 完整算法

```
Algorithm: SpeculativeDecodingStep
Input: M_p, M_q, prefix

1. [Draft] M_q 自回归生成 γ 个候选 token x_1,...,x_γ
2. [Verify] M_p 并行计算 p_1,...,p_{γ+1}（γ+1 个并行前向）
3. [Accept] 逐一应用 SpeculativeSample：
   - 接受率 β = P(接受 x ~ q) = Σ min(p(x), q(x)) = 1 - D_LK(p, q)
   - 其中 D_LK(p, q) = Σ|p(x) - M(x)| = 1 - Σ min(p(x), q(x))
4. 设 n 为连续被接受的 token 数，从调整后的 p_{n+1} 采样一个额外 token
5. 返回 prefix + [x_1, ..., x_n, t]，共 n+1 个新 token
```

---

## 3. 理论分析

### 3.1 期望生成 token 数

给定接受率 $\alpha = E(\beta) = E(\min(p, q))$，假设各 $\beta$ i.i.d：

$$E(\#\text{generated tokens}) = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}$$

### 3.2 总加速比

定义成本系数 $c$ = 一次 $M_q$ 运行时间 / 一次 $M_p$ 运行时间：

$$\text{Walltime Improvement} = \frac{1 - \alpha^{\gamma + 1}}{(1-\alpha)(\gamma c + 1)}$$

当 $\alpha > c$ 时存在 $\gamma$ 使得加速比 > 1，且至少为 $\frac{1+\alpha}{1+c}$。

### 3.3 最优 $\gamma$ 的选择

给定 $c$ 和 $\alpha$，最优 $\gamma$ 是使墙钟加速比最大化的整数。实际中可根据 $\alpha$ 动态调整 $\gamma$，上限额外提升可达 ~60%。

### 3.4 近似模型的选择

- **小 Transformer**（~2 个数量级更小）通常最佳，平衡 $\alpha$ 和 $c$
- **零成本模型**（$c \approx 0$）：n-gram、简单启发式等，加速比 $= \frac{1-\alpha^{\gamma+1}}{1-\alpha}$
- 有趣的是：即使最平凡的 bigram 模型也能在翻译任务中达到 $\alpha \approx 0.2$（1.25× 加速）
- **非自回归模型**也可用作 draft model

---

## 4. 实验结果

### 4.1 T5-XXL 墙钟加速

**任务**：英德翻译（WMT EnDe）、文本摘要（CNN/DM）

| 任务 | M_q | Temp | $\gamma$ | $\alpha$ | 加速比 |
|------|-----|------|-----------|----------|--------|
| EnDe | T5-small (77M) | 0 | 7 | 0.75 | **3.4×** |
| EnDe | T5-small (77M) | 1 | 7 | 0.62 | **2.6×** |
| CNNDM | T5-small (77M) | 0 | 5 | 0.65 | **3.1×** |
| CNNDM | T5-small (77M) | 1 | 5 | 0.53 | **2.3×** |

- T5-small（77M）作为 $M_q$ 平衡最好，而非更大的 T5-base 或 T5-large
- argmax 采样（temp=0）比标准采样加速更高（分布更尖锐 → $\alpha$ 更高）

### 4.2 各任务/模型的 $\alpha$ 值

| M_p | M_q | Temp | $\alpha$ |
|-----|-----|------|----------|
| GPT-like (97M) | GPT-like (6M) | 0 | 0.88 |
| GPT-like (97M) | Bigram | 0 | 0.05 |
| T5-XXL EnDe | T5-small | 0 | 0.75 |
| T5-XXL EnDe | Bigram | 0 | 0.20 |
| LaMDA (137B) | LaMDA (2B) | 0 | 0.71 |
| LaMDA (137B) | LaMDA (100M) | 0 | 0.61 |

**规律**：
- 近似模型越小，$\alpha$ 越低
- 分布越尖锐（temp=0），$\alpha$ 越高
- 即使最平凡的 n-gram 模型也有非零 $\alpha$

### 4.3 算术操作与内存带宽

- 每轮运行的**并行**算术操作增长 $\gamma+1$ 倍，但**总**算术操作可能增加（因拒绝造成的浪费）
- **总内存访问量下降**：模型权重和 KV cache 每轮只读一次，减少因子 = $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$
- 当内存带宽是瓶颈时，这一特性尤为重要

---

## 5. 核心贡献总结

1. **投机采样（Speculative Sampling = MRS）**：将投机执行推广到随机设置，保证输出分布严格不变
2. **投机解码（Speculative Decoding）**：训练免费、无输出变化的 AR 模型加速框架
3. **完整理论分析**：接受率、生成 token 数、墙钟加速比、算术操作增长的全套公式
4. **实验验证**：T5-XXL 上 2-3× 墙钟加速，跨多个任务和模型规模

---

## 6. 与后续工作的关系

本文是 **Speculative Decoding 体系的奠基之作**：

| 后续工作 | 改进方向 |
|---------|---------|
| **SJD** (Speculative Jacobi Decoding) | 免去 draft model，用自身的 Jacobi 迭代替代 |
| **SCD** (Speculative Coupled Decoding) | 在 SJD 基础上引入 Coupling，提升视觉加速到 4-13× |
| Chen et al. (2023) | 在 Chinchilla 70B 上独立复现 2-2.5× 加速 |

本文提出的 MRS 算法是所有 SD 变体的核心算子，在 SJD 中用作验证器、在 SCD 中被重新认识为 Maximal Coupling 的实现。

---

## 7. 论文局限与未来方向

- **算术操作增加**：加速以并发计算资源为代价，不适合计算资源紧张的场景
- **Beam Search 兼容性**：部分兼容但有性能损失
- **固定 $\gamma$**：动态调整 $\gamma$ 可获额外 ~60% 提升
- **定制近似模型**：用蒸馏等专门训练 $M_q$ 可能进一步提效
- **扩展模态**：论文仅在文本上验证，其他模态（图像、视频等）值得探索

---

## 8. Q&A

### Q1: Draft 模型自回归产生 γ 个连续的候选吗？还是对下一个位置产生 γ 个候选？

**Draft 模型自回归产生 γ 个连续的候选 token**，即序列上连续位置 `pos_1, pos_2, ..., pos_γ` 各自一个 token，**而非**对同一个位置产生 γ 个备选。

从论文 Algorithm 1（`SpeculativeDecodingStep`）的伪代码可知：

```python
# Sample γ guesses x_1,...,x_γ from M_q autoregressively.
for i = 1 to γ:
    q_i(x) = M_q(prefix + [x_1, ..., x_{i-1}])
    x_i ~ q_i(x)
```

每一步用前一步的采样输出作为新的 prefix，标准自回归串行生成，只是限定长度 γ。

#### 图解说明

```
prefix: "The quick brown"
                        │
    ┌───────────────────┤ M_q 自回归生成 γ=4 个连续 token
    ▼                   ▼
  pos_1      pos_2      pos_3      pos_4      pos_5
  "fox"      "jumps"    "over"     "the"      ???
   x_1        x_2        x_3        x_4       (target model 额外产生的 token)

生成顺序: x_1 → x_2 → x_3 → x_4 (串行，每次依赖前一个)
验证方式: M_p 一次性并行计算 p_1,...,p_5 并逐 token 做 MRS 裁决
```

#### 与 beam search 的区别

| | Speculative Decoding | Beam Search |
|---|---|---|
| **对同一位置** | 1 个候选 | k 个候选（beam width） |
| **对连续位置** | γ 个位置，各 1 个 token | 并行扩展 k 条 beam，各 1 个 token |
| **本质** | 投机：猜"未来序列" | 搜索：探索"当前可能性" |

所以 γ 的含义是 **"前瞻（lookahead）长度"**，即 draft model 一口气猜多少个连续 token，而非 beam 宽度。

### Q2: Target 模型如何并行验证候选概率？

表面上 Algorithm 1 写的是：

```python
p_1(x), ..., p_{γ+1}(x) = M_p(prefix), ..., M_p(prefix + [x_1, ..., x_γ])
```

看起来像是 γ+1 次独立的前向传播，但实际实现中**只需要一次前向传播**。这是 SD 能真正加速的关键。

#### 核心原理：利用 Transformer 的因果注意力

将 prefix 和所有 draft tokens 拼接成一个序列，一次性送入 M_p：

```
输入: prefix + [x_1, x_2, ..., x_γ]
```

Transformer decoder 的因果注意力（causal mask）确保了每个位置的输出只依赖于它之前的 token：

```
位置:   |prefix|-1    |prefix|      |prefix|+1        |prefix|+γ
        ──────────    ─────────     ────────────       ─────────
输入:   ...           x_1           x_2                x_γ
                        │              │                  │
                        ▼              ▼                  ▼
输出:   (忽略)         p_1(x)        p_2(x)             p_{γ+1}(x)
                    cond on:      cond on:            cond on:
                    prefix        prefix+[x_1]        prefix+[x_1,...,x_γ]
```

**关键点**：
- 位置 `|prefix|` 的 logits → $p_1$（即 prefix 后第一个 token 的目标分布）
- 位置 `|prefix|+1` 的 logits → $p_2$（prefix+x_1 后下一个 token 的分布）
- ...
- 位置 `|prefix|+γ` 的 logits → $p_{γ+1}$（全部 γ 个 draft 后的分布）

一次前向传播，γ+1 个位置的分布全部出来。

#### 为什么这是"并行"的？

```
传统自回归解码:
  M_p(prefix) → token_1
  M_p(prefix + [tok_1]) → token_2     ← 需要等 token_1
  M_p(prefix + [tok_1, tok_2]) → token_3  ← 需要等 token_2
  ...  K 次串行前向 ...

Speculative Decoding:
  draft model 串行生成: x_1, x_2, ..., x_γ  ← γ 次廉价前向
  target model 一次前向: M_p(prefix + [x_1, ..., x_γ])  ← 1 次前向得到 γ+1 个分布
  MRS 裁决: 连续接受 n 个，从 p_{n+1} 重采样 1 个
  产出: n+1 个 token（最少 1，最多 γ+1）
```

#### 以具体数值说明

```
prefix = [101, 202, 303]                    # 3 个 token
x_1=44, x_2=55, x_3=66, x_4=77              # γ=4 个 draft tokens
γ = 4

M_p 输入:  [101, 202, 303, 44, 55, 66, 77]  ← 拼接成 7 个 token 的序列
           位置0   1    2    3   4   5   6

M_p 输出每个位置对下一 token 的分布:
  位置 2 → logits_2 → softmax → p_1(x)    ← cond on [101,202,303]
  位置 3 → logits_3 → softmax → p_2(x)    ← cond on [101,202,303,44]
  位置 4 → logits_4 → softmax → p_3(x)    ← cond on [101,202,303,44,55]
  位置 5 → logits_5 → softmax → p_4(x)    ← cond on [...,44,55,66]
  位置 6 → logits_6 → softmax → p_5(x)    ← cond on [...,44,55,66,77]
                                        ↑ 即 γ+1 = 5 个分布

然后逐 token 做 MRS:
  MRS(p_1, q_1, x_1) → k=1 (接受) ✓
  MRS(p_2, q_2, x_2) → k=1 (接受) ✓
  MRS(p_3, q_3, x_3) → k=0 (拒绝) ✗
  → 接受 n=2 个: x_1, x_2
  → 从 adjust(p_3 - q_3) 重采样得到修正 token t
  → 本轮产出: x_1, x_2, t (共 3 个 token)
```

#### 与 draft 阶段的对比

| 阶段 | 执行方式 | 前向次数 | 产出 |
|------|---------|---------|------|
| **Draft** ($M_q$) | **串行**自回归 | γ 次 | γ 个 token + γ 个分布 q_i |
| **Verify** ($M_p$) | **并行**（一次前向） | **1 次** | γ+1 个分布 p_i |

这就是加速的本质：用廉价模型串行"猜"未来序列，用昂贵模型一次"验证"所有猜测。廉价模型的 γ 次前向 + 昂贵模型的 1 次前向，而非昂贵模型的 γ 次串行前向。

### Q3: Target 要计算 γ+1 个位置的概率，每个位置对应的 "mask"（可见前缀）长度不同，这些计算能复用吗？这真的只是一次前向吗？

**答案：确实只是一次前向，且计算大量复用。关键在于「不同前缀」并非需要不同的 mask——它们全部由同一个因果 mask 自然实现。**

#### 误解澄清：mask 是相同的，不是不同的

"不同的前缀"并不等于"不同的 mask"。在 Transformer decoder 中，**所有位置共享同一个 causal mask（下三角矩阵）**：

```
Q·K^T 矩阵 (7×7, 以 prefix 长度=3, γ=4 为例):

         k0  k1  k2  k3  k4  k5  k6
       [pre ][pre ][pre ][x1 ][x2 ][x3 ][x4 ]
q0 [pre]  ✓   ✗   ✗   ✗   ✗   ✗   ✗    ← 位置0只看到自己
q1 [pre]  ✓   ✓   ✗   ✗   ✗   ✗   ✗    ← 位置1看到[0,1]
q2 [pre]  ✓   ✓   ✓   ✗   ✗   ✗   ✗    ← 位置2看到[0,1,2] → p_1
q3 [x1 ]  ✓   ✓   ✓   ✓   ✗   ✗   ✗    ← 位置3看到[0,1,2,3] → p_2
q4 [x2 ]  ✓   ✓   ✓   ✓   ✓   ✗   ✗    ← 位置4看到[0..4] → p_3
q5 [x3 ]  ✓   ✓   ✓   ✓   ✓   ✓   ✗    ← 位置5看到[0..5] → p_4
q6 [x4 ]  ✓   ✓   ✓   ✓   ✓   ✓   ✓    ← 位置6看到[0..6] → p_5
```

**关键观察**：因果 mask 是同一个下三角矩阵，只是每一行自动"看到"了不同长度的历史。不需要为每个位置单独构造 mask。

#### 两次关键计算复用

**复用 1：prefix 的 KV Cache**

假设 prefix 的 K、V 已经缓存：

```
Q·K^T 的实际计算 (复用 prefix KV cache):

                      cached K                  fresh K (本次计算)
         [prefix  token 0,1,2] [x_1, x_2, x_3, x_4]
  fresh Q ↓                       ↓
  pos 3   [    attend...     ] [  ✓   ✗   ✗   ✗  ]
  pos 4   [    attend...     ] [  ✓   ✓   ✗   ✗  ]
  pos 5   [    attend...     ] [  ✓   ✓   ✓   ✗  ]
  pos 6   [    attend...     ] [  ✓   ✓   ✓   ✓  ]
```

- Prefix 的 K、V 已在之前计算过，**直接复用，无需重算**
- 只需为 γ 个 draft token 计算新的 Q、K、V
- 新的 Q 既 attend 到缓存的 prefix K，也 attend 到其他 draft token 的 K

**复用 2：这恰恰是训练时 teacher forcing 的做法**

训练时输入 `[x_1, x_2, ..., x_n]`，一次前向得到所有位置的 next-token 分布，用的就是同一个因果 mask。SD 的 verification 完全是同一套机制——只是输入从"真实序列"变成了"draft model 猜测的序列"。

#### 计算量对比：一次验证 vs 分别验证

假设 prefix 长度 $L$，draft token 数 $\gamma$：

| 方案 | Attention 计算量 | FFN 计算量 | KV Cache |
|------|:---:|:---:|:---:|
| **γ+1 次分别前向** | $\sum_{j=0}^{\gamma} O((L+j)^2 d) \approx O(\gamma L^2 d)$ | $\sum_{j=0}^{\gamma} O((L+j)d^2)$ | 各自独立 |
| **一次拼接前向** | $O((L+\gamma)^2 d)$ | $O((L+\gamma)d^2)$ | 复用 prefix KV |
| **一次拼接前向 + KV Cache** | $O((L+\gamma)\gamma d)$ 的新 token 部分 | $O(\gamma d^2)$ | 仅复用 prefix |

带 KV Cache 时，一次前向只需对新 token 做 attention，复杂度从 $O((L+\gamma)^2 d)$ 降到 $O((L+\gamma)\gamma d)$（因为 prefix 端的 K、V 只需读取不重算）。而分别前向每次都要重新编码不断变长的 prefix，毫无复用。

#### 一句话总结

> 这不是 γ+1 个不同 mask 的独立计算，而是**一个因果 mask 下的一次矩阵乘法**——每一行天然对应了不同长度的 prefix。加上 prefix KV Cache 复用和 teacher forcing 式的批量计算，这就是 SD 能加速的根本原因。

---

## 论文框架速览

1. **Introduction**: 动机 → 投机执行思想 → 双模型框架
2. **Speculative Decoding**: 算法框架 + Speculative Sampling (MRS) + 完整伪代码
3. **Analysis**: 接收率 $\alpha$、期望 token 数、墙钟加速公式、最优 $\gamma$、算术操作分析
4. **Experiments**: T5-XXL (2-3×)、GPT-like、LaMDA 的 $\alpha$ 和加速比
5. **Related Work**: 蒸馏、稀疏化、量化、自适应计算等
6. **Discussion**: Beam Search、层级加速、模态扩展等方向
