# Speculative Coupled Decoding（SCD）：自回归视觉生成的训练免费无损加速

> **论文标题**: Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation
>
> **作者**: Junhyuk So, Hyunho Kook, Chaeyeon Jang, Eunhyeok Park (POSTECH, 韩国)
>
> **发表**: ICML 2026
>
> **链接**: https://arxiv.org/abs/2510.24211v2
>
> **代码**: https://github.com/junhyukso/SCD

---

## 1. 研究背景与动机

自回归（AR）建模在视觉生成领域（图像、视频、3D、音频等）展现了巨大潜力，但其逐 token 生成的范式导致推理速度极慢——生成一张图可能需要数千步前向传播。

**已有加速方法的局限**：
- **Speculative Decoding (SD)**：需要训练单独的 draft 模型，开销大，且在视觉生成中加速有限
- **Speculative Jacobi Decoding (SJD)**：训练免费、无损的 SD 变体，将 Jacobi 迭代与 SD 结合，免除了 draft 模型的训练。但加速效果仅约 2×，远低于文本生成领域 4×+ 的水平

### 1.1 SD → SJD → SCD 关系梳理

三者是 **Speculative Decoding 框架下的递进关系**，核心区别在于 draft token 的来源：

#### SD（Speculative Decoding）—— 两个模型

```
Draft Model q (单独训练)  ──→  逐 token 生成 L 个候选  ──→  Target Model p 并行验证  ──→  MRS 接受/拒绝
```

- 需要额外训练一个 **draft model** $q$（通常是 $p$ 的小型副本）
- 两个模型存在 **通信/调度开销**（target 必须等 draft 产出）
- **文本**领域效果好（4×+），**视觉**领域困难：视觉 token 空间大（16384+）、分布平坦，训练好的 draft model 很难

#### SJD（Speculative Jacobi Decoding）—— 一个模型，自己当自己的 draft

```
Target Model p (唯一)
     │
     ├─ 上一轮 verify 的输出分布 p^{t-1} ──→ 当前轮作为 draft
     ├─ 基于 context X^{t-1} 并行 evaluate → 得到 p^t
     └─ MRS 验证 → 输出 p^{t+1} ──→ 下一轮的 draft
```

- **训练免费**：不需要额外 draft model
- **关键洞察**：验证阶段得到的分布 $p^t$ 已经是对 target 的良好近似，直接作为下轮 draft
- 视觉领域加速仅 ~2×（独立采样导致上下文剧烈波动）

#### SCD（Speculative Coupled Decoding）—— SJD + Coupling

```
SJD 框架
     │
     └─ Draft 阶段：独立采样 ──→ 耦合采样（Maximal / Gumbel Coupling）
```

- 在 SJD 基础上**仅修改 draft 采样方式**（1 行代码）
- 不改变分布演化机制，只让"相似的分布产出相似的 token"
- 视觉图像 4.2×、视频 13.6× 加速

| | SD | SJD | SCD |
|---|---|---|---|
| **Draft 来源** | 单独训练的小模型 $q$ | 自己的上轮分布 $p^{t-1}$ | 同 SJD + Coupling |
| **需要训练** | ✅ 需要训练 draft model | ❌ 训练免费 | ❌ 训练免费 |
| **模型数量** | 2 个（p + q） | 1 个（p） | 1 个（p） |
| **视觉图像加速** | 有限（draft 难训练） | ~2× | **4.2×** |
| **视觉视频加速** | — | ~3.3× | **13.6×** |
| **额外开销** | draft 训练 + 调度 | 几乎零 | < 5% |
| **核心修改** | 整套 pipeline | 基准 | 仅 1 行 |

---

## 2. 核心洞察

作者发现 SJD 的瓶颈在于 **draft token 采样的不稳定性**：

- SJD 中 draft token 是 **独立采样** 的，导致相邻迭代之间 token 碰撞概率极低
- 碰撞概率上界为 $\exp(-\frac{1}{2}(H_2(p) + H_2(q)))$，受 Rényi-2 熵的指数级约束
- 视觉 AR 模型输出分布较平坦（熵高），使得独立采样的碰撞概率极其微小
- 即使两个分布在 TV 距离上很接近，实际采样的 token 也几乎总会发生变化
- 这种不稳定导致 SJD 的接受率（acceptance rate）波动大且难以收敛

**关键观察**：高上下文相似性 → 高加速比。这意味着如果能提高连续迭代间 draft token 的一致性，就能显著提升加速效果。

## 3. 方法：Speculative Coupled Decoding (SCD)

### 3.1 核心思想：Coupling（耦合）

利用信息论中的 **Coupling** 概念，将 SJD 中独立的 draft 采样替换为耦合采样：

- **Coupling 定义**：给定两个分布 P 和 Q，联合分布 π(x,y) 是它们的 coupling，当且仅当其边缘分布分别等于 P 和 Q
- **关键性质**：从 coupling 中采样的 token 的边际分布与原分布一致，因此不破坏 SD 的无损性
- **目标**：最大化耦合代价（coupling cost）$C(\pi) = \Pr[X=Y]$，即 token 碰撞概率

### 3.2 两种 Coupling 实现

**① Maximal Coupling（π_MC）——理论最优**

- 耦合代价上界：$C(\pi_{MC}) = 1 - \mathcal{D}_{TV}(P, Q)$
- 实现与 Modified Rejection Sampling (MRS) 完全相同——仅需**一行代码修改**
- 可向量化实现，将 drafting 和 verification 合并为一步操作，几乎零开销

**② Gumbel Coupling（π_GS）——更简单的替代方案**

- 基于 Gumbel-Max Trick：通过**共享**相同的 Gumbel 噪声向量来耦合两个采样过程
- 耦合代价下界：$C(\pi_{GS}) \ge (1 - \mathcal{D}_{TV}) / (1 + \mathcal{D}_{TV})$
- 具有更好的**长程稳定性**（multi-step stability），在视频生成等任务中表现更优

### 3.3 与 SJD 的区别

| | SJD | SCD |
|---|---|---|
| Draft 采样 | 独立采样 $X^t \sim p^t(\cdot)$ | 耦合采样 $(X^t, X^{t-1}) \sim \pi(p^t, p^{t-1})$ |
| 碰撞概率 | 极低（熵约束） | 接近理论上界 |
| 接受率 | 波动大，不收敛 | 稳定上升趋势 |
| 实现复杂度 | 基准 | 仅需一行修改 |

### 3.4 MRS（Modified Rejection Sampling）详解

MRS 是 SD/SJD/SCD 全体系的核心算子，在 SCD 中承担**双重角色**。以下以图像/视频 AR 生成为例详细说明。

#### MRS 算法

```
输入: p (target 分布), q (draft 分布), x ~ q (draft 采样的 token)
输出: Y ~ p (修正后的 token), k ∈ {0,1} (接受标志)
```

```python
def MRS(p, q, x):
    u = random.uniform(0, 1)
    if u <= min(1.0, p[x] / q[x]):
        return (1, x)                              # accept: Y = x
    else:
        residual = normalize(max(0, p - q))         # p 中未被 q 覆盖的部分
        return (0, sample_categorical(residual))    # reject, 从残差重采样
```

#### 各参数在图像/视频 AR 中的实际含义

以 **Lumina-mGPT 生成 768×768 图像**（2390 tokens）为例：

| 参数 | 含义 | 实际数据 |
|------|------|---------|
| **p** | target 分布，即当前 prefix 下模型 $p_\theta$ 的 logit 输出经 softmax 后的概率向量 | shape = [vocab_size]，如 16384 维 |
| **q** | draft 分布，SD 中是 draft model 的输出；SJD/SCD 中是上一轮 verify 得到的 $p^{t-1}$ | 同 shape，同为 16384 维 |
| **x** | 从 draft 分布 q 中实际采出的 token（整数索引） | 标量，如 token_id = 3821 |
| **Y** | 修正后的 token，保证 $Y \sim p$（无损性） | 标量 token_id |
| **k** | 1=接受 x，0=拒绝（触发残差采样） | 布尔值 |

#### 角色一：SD 验证器（Verify）

在整个 SD 体系中的标准用法——验证 draft token 是否可被 target 分布接受：

```
位置:  SJD/SCD 的 Verify 阶段
调用:  k, X^{t+1}_j = MRS(p^{t+1}_j, p^t_j, X^t_j)

含义:
  输入  p^{t+1}_j : 模型基于新 context 计算的目标分布（"正确答案"）
        p^t_j     : draft 分布（"猜测"）
        X^t_j     : 从 p^t_j 中采样的 draft token（"猜测的值"）
  输出  k         : 是否接受 draft token
        X^{t+1}_j : 最终有效的 token（保证服从 p^{t+1}）

概率解释:
  Pr[k=1] = min(1, p(x)/q(x)) = 1 - D_TV(p, q)
  → 分布越接近，接受率越高
```

#### 角色二：Maximal Coupling 耦合器（SCD 的创新用法）

SCD 的核心洞察——MRS 本身就是 Maximal Coupling 的实现：

```
位置:  SCD-π_MC 的 Draft 阶段
调用:  _, X^t_j = MRS(p^t_j, p^{t-1}_j, X^{t-1}_j)

含义:
  输入  p^t_j     : 当前轮的分布（作为 coupling 的 P）
        p^{t-1}_j : 上一轮的分布（作为 coupling 的 Q）
        X^{t-1}_j : 上一轮采样的 token（服从 Q = p^{t-1}）
  输出  X^t_j     : 耦合后的 token（服从 P = p^t，但与 X^{t-1} 高度相关）

为什么 MRS 就是 Maximal Coupling:
  - MRS 保证了输出 Y ~ P（边际正确性）
  - MRS 的接受率 1-D_TV(p,q) 恰好等于 coupling cost 的理论上界
  - 因此 (Y, X) 的联合分布构成 Maximal Coupling
```

#### 视觉 AR 场景下 MRS 核心性质

**1. 无损性保证**（Theorem 1）：无论 draft 分布 q 和 token x 质量如何，MRS 输出 Y 严格服从 p。这意味着：
- SD 可以用任意差的 draft model，输出分布不变
- SJD 可以用自己的上一轮分布，输出分布不变
- SCD 用 MRS 做耦合采样，边际分布仍为 p^t，输出分布不变

**2. 接受率 = 1 - D_TV(p, q)**（Proposition 1）：视觉 AR 场景下：
- 视频生成相邻帧 token 分布高度相似 → D_TV 很小 → 接受率接近 1 → 大幅加速
- 图像生成全局 token 分布较分散 → D_TV 较大 → 接受率中等 → 加速效果次于视频

**3. 拒绝后的残差采样**：当 `p(x) < q(x)` 时可能拒绝，此时从 `max(0, p - q)` 归一化后的残差分布中重新采样。这保证了即使拒绝，输出仍服从 p——相当于 "切掉 q 中过度自信的部分，仅从 p 独有的概率质量中采样"。

#### 直观理解

以生成一张图片为例，假设某个位置 draft 预测 `token_id = 3821`（对应图片某区域的蓝色像素）：
- `p[3821] = 0.3`（target 认为蓝色概率 30%）
- `q[3821] = 0.4`（draft 认为蓝色概率 40%）
- `min(1, p/q) = min(1, 0.3/0.4) = 0.75`
- → **75% 概率直接接受** `token_id = 3821`
- → 25% 概率拒绝，从残差分布（p - q 的正部归一化）中重采样

这是 SD 无损性的关键：即使 draft 模型过于自信（q > p），也能通过概率拒绝来修正。

---

## 4. 实验结果

### 4.1 图像生成

**Lumina-mGPT (7B, 768×768) 在 MS-COCO 上**：

| 方法 | NFE | 延迟 (A100) | 加速比 | FID | IS |
|------|-----|------------|--------|-----|-----|
| Vanilla AR | 2390 | 102.03s | 1.00× | 30.79 | 32.81 |
| SJD (L=64) | 1035.9 | 42.98s | 2.37× | 30.81 | 32.76 |
| **SCD-π_MC (L=64)** | **567.7** | **24.41s** | **4.18×** | 30.83 | 33.43 |
| **SCD-π_GS (L=64)** | **568.0** | **24.24s** | **4.21×** | 30.90 | 32.80 |

- 相比 Vanilla AR 加速 **4.2×**，相比 SJD 加速 **1.8×**
- FID、IS、CLIP Score 均无损，保持质量一致性
- 对比有损方法 GSD（FID 33.21），SCD 在更快的速度下仍保持无损

**Janus-Pro (7B, 384×384)**：加速 3.0×（NFE 576 → 190），FID 保持一致

**Lumina-mGPT-2 和 Parti-Prompt 数据集**：加速达 4.4×，泛化能力强

### 4.2 视频生成

**Cosmos-1-AR (4B) 在 Real-Estate-10k 上**：

| 方法 | NFE | 延迟 | 加速比 | FVD |
|------|-----|------|--------|-----|
| Vanilla AR | 7680 | 157.25s | 1.00× | 156.9 |
| SJD (L=128) | 1789.9 | 47.73s | 3.3× | 158.3 |
| **SCD-π_GS (L=128)** | **564.4** | **13.60s** | **11.6×** | **152.4** |

- 加速高达 **13.6×**（实际延迟），NFE 加速 **13.6×**
- 视频帧间强时序冗余使 draft 预测更容易，窗口越大加速越多

### 4.3 开销分析

在 Janus-Pro 7B (RTX 3090) 上单步 NFE 的延迟分解：

| 操作 | 耗时 (L=64) |
|------|-----------|
| Transformer 前向 | 36.41ms |
| Token 采样 (GS) | 0.14ms |
| Vec. MRS (MC) | 1.66ms |
| 其他 | 2.67ms |

耦合采样的额外开销 < 5%，几乎可忽略。

### 4.4 关键消融研究

**耦合强度 α**：将耦合分布与独立分布进行线性插值 $\pi^\alpha = \alpha \pi_{cpl} + (1-\alpha)\pi_{ind}$。实验表明 α 从 0 到 1 单调提升性能，验证了耦合机制的有效性。

**多步行为**：π_GS 在 2-3 步迭代中比 π_MC 具有更小的 token 变化（更好的长程稳定性），解释了其在视频生成中更优的表现。

## 5. 理论贡献

1. **碰撞概率分析**：推导了 SJD 独立采样的碰撞概率上界，揭示其受 Rényi-2 熵的指数级约束
2. **Coupling 无损性证明**：证明了任何有效 coupling 的边际采样不改变 SD 的无损性质
3. **耦合代价上下界**：Maximal Coupling 达到理论上界 $1-\mathcal{D}_{TV}$，Gumbel Coupling 有紧致的下界 $(1-\mathcal{D}_{TV})/(1+\mathcal{D}_{TV})$
4. **α-Coupling**：提出了连续调节耦合强度的框架，可平滑过渡 SJD 和 SCD

## 6. 局限性与讨论

- 方法依赖并行计算能力，窗口/批次过大会导致并行开销显著
- 这是所有 SD 方法的共同局限，预计随硬件发展逐步消失
- π_MC 和 π_GS 在不同任务上各有优势，选择取决于 draft 预测的难易程度

## 7. 关键代码实现：SJD vs SCD

### 7.1 核心子程序

**MRS（Modified Rejection Sampling）**——既是 SD 验证机制，也是 Maximal Coupling 的实现：

```python
def MRS(p, q, x):
    """
    Input:  p (target dist), q (draft dist), x ~ q
    Output: Y ~ p, accept_flag k
    """
    u = random.uniform(0, 1)
    if u <= min(1.0, p[x] / q[x]):
        return (1, x)                    # accept: Y = x, 分布仍为 p
    else:
        residual = normalize(max(0, p - q))
        return (0, sample_categorical(residual))  # reject, 从残差重采样
```

**GS（Gumbel Sharing Coupling）**：

```python
def GS(p, q, G):
    """
    Input:  p, q (两个分布), G (共享 Gumbel 噪声)
    Output: (X, Y), X ~ p, Y ~ q, 噪声共享实现耦合
    """
    X = argmax(log(p) + G)    # 从 p 采样
    Y = argmax(log(q) + G)    # 从 q 采样（同一个 G！）
    return X, Y
```

### 7.2 完整迭代循环对比

**SJD（原始 Speculative Jacobi Decoding）**：

```python
# Algorithm: SJD (独立采样)
while i < N:
    # [Drafting] 独立采样，与上一轮的 X^{t-1} 无关
    for j in parallel(i, i+L):
        X_new[j] = sample_categorical(p_curr[j])   # X^t_j ~ p^t_j(x)

    # [Evaluate] 模型前向，基于新 context 计算下一轮分布
    for j in parallel(i, i+L):
        p_next[j] = model(X_new[:j])

    # [Verify] 逐个 token 接受/拒绝
    for j in range(i, i+L):
        k, X_next[j] = MRS(p_next[j], p_curr[j], X_new[j])
        if k == 0: break

    i = j + 1
    t += 1
    p_curr = p_next
```

**SCD-π_MC（Maximal Coupling）——仅一行修改**：

```python
# Algorithm: SCD-π_MC (Maximal Coupling) — 仅 Drafting 行改动
while i < N:
    # [Drafting] 耦合采样：与上一轮的 X_prev[j] 形成 Maximal Coupling
    for j in parallel(i, i+L):
        _, X_new[j] = MRS(p_curr[j], p_prev[j], X_prev[j])  # ← 仅此一行修改！
    #                                                       # MRS 的边际输出仍服从 p_curr[j]
    # [Evaluate] 以下完全同 SJD
    for j in parallel(i, i+L):
        p_next[j] = model(X_new[:j])

    # [Verify]
    for j in range(i, i+L):
        k, X_next[j] = MRS(p_next[j], p_curr[j], X_new[j])
        if k == 0: break

    i = j + 1
    t += 1
    p_prev, p_curr = p_curr, p_next
    X_prev = X_new
```

**SCD-π_GS（Gumbel Coupling）**：

```python
# Algorithm: SCD-π_GS (Gumbel Coupling)
while i < N:
    # [Drafting] 共享 Gumbel 噪声耦合两个采样过程
    for j in parallel(i, i+L):
        G_j = hash_gumbel_noise(j)                        # 按位置确定性生成
        X_new[j] = argmax(log(p_curr[j]) + G_j)           # ← 固定噪声耦合

    # [Evaluate] 完全同 SJD
    for j in parallel(i, i+L):
        p_next[j] = model(X_new[:j])

    # [Verify] 完全同 SJD
    for j in range(i, i+L):
        k, X_next[j] = MRS(p_next[j], p_curr[j], X_new[j])
        if k == 0: break

    i = j + 1
    t += 1
    p_curr = p_next
```

### 7.3 高效实现技巧

**π_MC 向量化合并**：注意 Draft 阶段的 `MRS(p^t, p^{t-1}, X^t)` 和 Verify 阶段的 `MRS(p^{t+1}, p^t, X^t)` —— 下一轮 Draft 的 p^t 就是当前轮 Verify 的 p^{t+1}。因此可合并为一次向量化 MRS，同时完成 Verification + Drafting：

```python
# 效率版本：MRS 一次调用同时做 Verify 和 Draft
while i < N:
    # [Evaluate]
    for j in parallel(i, i+L):
        p_next[j] = model(X_curr[:j])

    # [Verify & Draft 合并] 向量化 MRS
    Accept = i + L
    for j in parallel(i, i+L):
        k, X_next[j] = MRS(p_next[j], p_curr[j], X_curr[j])
        if k == 0 and Accept == i + L:   # 记录第一个 rejection 位置
            Accept = j
    # X_next 同时是验证输出 + 下一轮 draft token

    i = Accept + 1
    t += 1
    p_curr, X_curr = p_next, X_next
```

这样 π_MC 的额外开销仅为一次向量化 MRS（~1.66ms），对比 Transformer forward（~36ms）< 5%。

**π_GS 噪声生成**：Gumbel 噪声通过 token 全局索引的哈希确定性生成，保证同一位置在不同迭代中共享相同噪声向量，且计算代价等同于普通 multinomial 采样（~0.14ms）。

### 7.4 一行修改总结

| 方案 | SJD 原代码 | SCD 修改 | 改动量 |
|------|-----------|---------|--------|
| π_MC | `X_new[j] ~ p_curr[j]` | `_, X_new[j] = MRS(p_curr[j], p_prev[j], X_prev[j])` | 1 行 |
| π_GS | `X_new[j] ~ p_curr[j]` | `X_new[j] = argmax(log(p_curr[j]) + G_j)` | 1 行（+噪声初始化） |

---

## 8. 总结

SCD 提出了一个**极其简洁**（一行代码修改）但**效果显著**（图像 4.2×、视频 13.6× 加速）的训练免费、无损自回归视觉生成加速方法。核心思想是用信息论中的 Coupling 替代 SJD 中的独立采样，大幅提升连续迭代间的 token 碰撞概率，从而稳定 Jacobi 迭代轨迹、提高接受率。方法几乎零额外开销，并已在多个主流 AR 视觉模型上验证有效性。

---

## 9. Q&A

### Q1: 为什么 SJD 的 draft 采样和上一轮迭代的历史状态无关？如果是这样，那就不需要迭代采样了？

**简短回答**：SJD 的 **采样动作**（从分布中掷骰子）是独立的，但 **采样所依据的分布** 是依赖迭代历史的。迭代不是无用的——分布的演化才是迭代的意义所在。

**详细解释**：

SJD 的迭代过程中，有两层依赖关系：

```
迭代 t-1:
  p^{t-1}_j ← pθ(·| X^{t-2}_{<j})    ← 分布依赖上一轮的 context
  X^{t-1}_j ~ p^{t-1}_j(x)            ← 采样是独立的

迭代 t:
  p^t_j ← pθ(·| X^{t-1}_{<j})         ← 分布依赖更新后的 context
  X^t_j ~ p^t_j(x)                     ← 从新分布独立采样（不引用 X^{t-1}_j）
```

关键点：

1. **分布 p^t_j 是迭代依赖的**：p^t_j 由模型基于 context X^{t-1}_{<j} 重新计算得到。随着前缀 token 被逐步接受（稳定），分布 p^t 会越来越准确，这是迭代的核心价值。

2. **采样 X^t_j 是独立的**：给定分布 p^t_j 后，采样 `X^t_j ~ p^t_j` 是一个 "掷骰子" 的动作，它完全不看上一轮的采样结果 X^{t-1}_j。即使两个分布 p^t 和 p^{t-1} 非常相似（TV 距离很小），独立采样出的 token 也几乎必然不同。

3. **这正是 SCD 要解决的问题**：SCD 不是改变分布的演化（迭代的意义保留），而是改变采样方式——让 p^{t-1} → p^t 变化很小时，采出的 token 尽可能不变。这通过 Coupling 实现：
   - p^t 和 p^{t-1} 分布接近 → 耦合采样大概率产出相同 token → 接受率高
   - p^t 和 p^{t-1} 分布确实变了 → 耦合采样也会产出不同 token → 正确反映分布变化

4. **类比**：想象你在调整一个照片的曝光参数。每次调整后，你拍一张新照片。SJD 的独立采样相当于每次调完参数后完全重新构图取景——即使参数没怎么变，照片内容也可能完全不同。SCD 的耦合采样相当于保持相同的取景，只根据参数变化做微调——只有参数真的变了，照片才会变化。

**结论**：迭代采样不是无用的，分布 p^t 随 context 更新而演化正是 SJD/SD 工作的基础。SCD 的改进在于采样阶段，让相似分布产出相似 token，而非推翻迭代框架。

### Q2: SJD 和之前的 SD 相比有什么区别？

**SD** 需要额外训练一个 draft model $q$，由 draft model 逐步生成候选 token，target model $p$ 并行验证。问题在于视觉领域训练一个好的 draft model 非常困难（token 空间大、分布平坦），且存在双模型通信开销。

**SJD** 的核心创新是"自己当自己的 draft"——上一轮 verify 输出的分布 $p^t$ 直接作为下一轮的 draft 分布，省去了 draft model 的训练和通信成本。其迭代过程：

```
迭代 t:  p^t 作为 draft → 并行 evaluate → 得到 p^{t+1} → MRS 验证
迭代 t+1: p^{t+1} 作为 draft → 并行 evaluate → 得到 p^{t+2} → MRS 验证
```

但随着迭代进行，分布 $p^t$ 和 $p^{t+1}$ 虽然逐渐接近，独立采样出的 token 序列却剧烈波动，导致加速天花板仅 ~2×。

**SCD** 在 SJD 框架上仅修改 draft 采样方式（1 行代码），用 Coupling 替代独立采样，使相似分布产出相似 token，打破 ~2× 天花板。

| | SD | SJD | SCD |
|---|---|---|---|
| Draft 来源 | 单独训练的小模型 $q$ | 自己的上轮分布 $p^{t-1}$ | 同 SJD + Coupling |
| 需要训练 | ✅ | ❌ | ❌ |
| 视觉图像加速 | 有限 | ~2× | **4.2×** |
| 视觉视频加速 | — | ~3.3× | **13.6×** |

### Q3: SD 中也有 MRS 吗？MRS 全称是什么？SD 和 SJD 中 MRS 的具体输入输出含义和维度是怎样的？

**MRS 全称: Modified Rejection Sampling（修正拒绝采样）**

MRS 是 SD（Speculative Decoding）最先提出的核心算子，是整个 Speculative Decoding 体系的基石。它并非 SCD 发明，SCD 的贡献在于重新认识到 MRS 恰好也是 Maximal Coupling 的数学实现。

---

#### MRS 的标准算法形式

```python
def MRS(p, q, x):
    """
    输入:
        p: shape [V]     — target 分布（"正确答案"）
        q: shape [V]     — draft 分布（"猜测"）
        x: int, 0 ≤ x < V — draft token（从 q 中采样得到的值）
                           在批量版本中 shape [B]

    输出:
        k: int / bool     — 1=接受, 0=拒绝
        Y: int (0 ≤ Y < V) — 修正后的 token，保证 Y ~ p
    """
    u = random.uniform(0, 1)
    if u <= min(1.0, p[x] / q[x]):
        return (1, x)                           # 接受: Y = x
    else:
        residual = normalize(max(0, p - q))     # shape [V], p 中未被 q 覆盖的部分
        return (0, sample_categorical(residual)) # 拒绝: 从残差中重采样
```

分布维度 `V` = vocab_size（词表大小）。以 Lumina-mGPT 为例，`V = 16384`；以 768×768 图像为例，总 token 数 `N = 2390`。

---

#### 在 SD 中：MRS 的调用方式和维度

```
SD 架构:
  Draft Model q (小模型) → 逐 token 生成候选 → Target Model p (大模型) 并行验证 → MRS 逐一裁决
```

```python
# —— SD 的典型执行流程 ——
# Draft model 逐 token 生成 L 个候选 (自回归，串行)
X_draft = []                           # 存储 L 个 draft token ids
for j in range(L):
    q_j = draft_model(prefix + X_draft)   # q_j: shape [V]
    X_draft.append(sample_categorical(q_j))  # x_j 从 draft 分布中采样

# Target model 并行验证 (一次前向)
p_list = target_model(prefix + X_draft)    # p_list: list of L tensors, 每个 shape [V]

# MRS 逐 token 验证
for j in range(L):
    # 调用 MRS: target 分布 p_j vs draft 分布 q_j
    k, Y = MRS(
        p = p_list[j],        # shape [V], target model 的输出 (如 16384 维)
        q = q_list[j],        # shape [V], draft model 的对应输出 (也是 16384 维)
        x = X_draft[j]        # int, 从 q_j 中采样的 draft token
    )
    # 输出:
    #   k: 1 或 0, 表示接受该 draft token 与否
    #   Y: token id, 保证 Y ~ p_j (无损性)

    if k == 0:
        # 拒绝 → Y 来自残差重采样; 当前位置后的所有 draft tokens 报废
        X_final.append(Y)
        break
    else:
        # 接受 → 直接复用 draft token
        X_final.append(x)
```

在 SD 的批量推理中（处理多个 prompt 并行解码），各维度为：

| 参数 | 批量维度 [B] | 分布维度 [V] | 总 shape |
|------|:---:|:---:|---|
| **p** | B | 16384 | `[B, V]` |
| **q** | B | 16384 | `[B, V]` |
| **x** | B | — | `[B]` (每个样本一个整数 token id) |
| **k** | B | — | `[B]` (每个样本一个 0/1) |
| **Y** | B | — | `[B]` |

---

#### 在 SJD 中：MRS 的调用方式和维度

MRS 在 SJD 中仅用于 Verify 阶段，输入分布来自同一模型的相邻两次迭代：

```python
# —— SJD 的一轮迭代 ——
while i < N:
    # [Draft]  独立采样: X^t_j ~ p^t_j
    X_new[j] = sample_categorical(p_curr[j])   # p_curr: list, 每个 [V]

    # [Evaluate] 模型并行前向: p^{t+1}_j = pθ(·|X^t_{<j})
    p_next = model(X_new)                       # p_next: list, 每个 [V]

    # [Verify] MRS 验证
    for j in range(i, i+L):
        k, X_next[j] = MRS(
            p = p_next[j],    # shape [V], 目标是"新 context 基于本轮 draft 后的结果"
            q = p_curr[j],    # shape [V], draft 分布 = 上轮 verify 的 p^t (历史的"预测")
            x = X_new[j]      # int, 从 draft 分布 p_curr[j] 中独立采样的 token
        )
        if k == 0: break

    i = j + 1
    p_curr = p_next           # 本轮 p^{t+1} 成为下一轮的 draft 分布
```

**SJD 与 SD 的核心区别**：MRS 的三个输入都来自同一个模型：

| 对比 | SD | SJD |
|------|-----|------|
| **p（target）** | `p_θ` 大型 target model 的输出 | 模型基于本轮新 context 计算的 `p^{t+1}` |
| **q（draft）** | 单独训练的 `q_φ` draft model 的输出 | 上轮 verify 得到的 `p^t`（**同一个模型，上一轮**） |
| **x（draft token）** | `q_φ` 逐 token 自回归生成 | 从 `p^t` **独立采样** 得到（不引用历史 X^{t-1}） |
| **维度** | 完全相同：[V]=16384, B=1~N | 完全相同：[V]=16384, B=1~N |

---

#### MRS 的关键数学性质

```
公式:
  p_target = [0.05, 0.10, 0.30, 0.15, 0.08, ...]   ← shape [16384], sum=1
  q_draft  = [0.03, 0.12, 0.40, 0.10, 0.06, ...]   ← shape [16384], sum=1
  x        = 2  (假设从 q 中采样得到 token_id=2)

计算:
  q[x] = q[2] = 0.40
  p[x] = p[2] = 0.30
  p[x]/q[x] = 0.30/0.40 = 0.75

  with u ~ Uniform(0,1):
    75% 概率 → k=1 (接受), Y=x=2
    25% 概率 → k=0 (拒绝), Y ~ normalize(max(0, p - q))
                                   = normalize([0.02, 0.00, 0.00, 0.05, 0.02, ...])
                                   → 从残差中采样新 token
```

**核心不变性**：无论 $p$ 与 $q$ 是否来自同一个模型、无论 x 是否在上一轮被接受过，MRS 输出的 Y 永远严格服从 p。这也是 SD/SJD/SCD 全部无损的根本原因。

### Q4: 图像解码时不是全部 token 一起解码吗（如 Diffusion）？SCD 作用在哪个阶段？

SCD 作用的对象是 **自回归（AR）视觉生成模型**，而非扩散模型（Diffusion Models）。两者有本质区别：

#### 两类图像生成模型的对比

| | 扩散模型 (Diffusion) | 自回归视觉模型 (AR Visual) |
|---|---|---|
| **代表** | Stable Diffusion, DALL·E 2, Flux | Lumina-mGPT, Janus-Pro, Parti, Cosmos-AR |
| **解码方式** | 所有 token 并行去噪 | Token 逐 token 串行生成（光栅顺序） |
| **推理步骤** | 典型 20-50 步去噪 | 一张 768×768 图 = **2390 步**串行前向 |
| **瓶颈** | 每步计算量大 | **串行步数多**，延迟极高 |
| **SCD 适用？** | ❌ 不适用 | ✅ 适用 |

#### AR 视觉模型的生成过程

```
图像 tokenization (VQ-VAE / VQGAN):
  原始图像 768×768 → 离散 token 序列 (如 2390 tokens, vocab_size=16384)

AR 解码 (逐 token):
  [BOS] → token_1 → token_2 → ... → token_2390
    ↑        ↑         ↑               ↑
  起始   第1个像素块  第2个像素块    最后像素块
         的离散编码   的离散编码     的离散编码

每次前向：输入 prefix tokens，输出下一个 token 的分布
总共需要：2390 次串行模型前向
```

SCD 作用的正是这个 **AR 逐 token 解码阶段**——用 Jacobi 迭代 + Coupling 将 2390 次串行前向压缩到约 568 次（4.2× 加速），同时保证输出分布完全不变。

#### 为什么 SCD 不用于 Diffusion？

Diffusion 模型所有 token 一起去噪，没有"逐 token 验证 / draft / accept"的串行依赖结构，因此 SD/SJD/SCD 系列方法不适用。

反过来，AR 视觉模型的推理瓶颈恰恰是"数千步串行"——这与 LLM 面临的"逐 token 自回归解码"是同一个困境，这正是 SD 系列方法的用武之地。

### Q5: 在 SJD 中，上一轮 verify 的输出分布为什么可以当做本轮的 draft？维度是多少？Verify 的输入输出到底是什么？

#### 一句话回答

SJD 的核心设计就是 `p_curr = p_next`——把本轮的 Evaluate/Verify 产生的目标分布直接"传递"给下一轮做 draft 分布。因为随着迭代收敛，相邻两轮的条件分布越来越接近，上一轮的分布就是下一轮分布的极好近似。维度上，所有分布的 shape 都是 `[vocab_size]`（如 16384）。Verify 操作在每个 position 上**独立**进行，输入是两个 `[V]` 分布 + 一个整数 token，输出一个 0/1 标志 + 一个整数 token。

---

#### 详细解释

**（1）为什么可以「自己当自己的 draft」？**

从 SJD 的伪代码（§7.2）看最关键的一行：

```python
p_curr = p_next    # ← 迭代结束时，本轮 verify 中的 target 分布成为下轮的 draft
```

整个 SJD 的单轮迭代流程和维度变化如下：

```
┌─────────────────────────────────────────────────────────────────────┐
│  第 t 轮迭代 (token 索引从 i 开始，窗口长度 L)                        │
│                                                                      │
│  [Draft]     X^t_j ~ p_curr[j]         ← 从分布中采样整数 token      │
│              X^t:  [L] 个整数, 每个 ∈ [0, V-1]                       │
│                                                                      │
│  [Evaluate]  p_next = model(prefix + X^t[:j])                       │
│              p_next: [L, V] 概率分布                                 │
│              ↑ 这是模型基于 "draft token 构成的新 context"            │
│                计算出的 next-token 分布                              │
│                                                                      │
│  [Verify]    对每个位置 j:                                          │
│                k, X_next[j] = MRS(                                   │
│                    p = p_next[j],    # [V] target 分布（"正确答案"）  │
│                    q = p_curr[j],    # [V] draft 分布（"猜测"）       │
│                    x = X^t_j         # int draft token（"猜测的值"）  │
│                )                                                     │
│              遇到 k=0 就 break                                       │
│                                                                      │
│  结束:       p_curr = p_next   ← p_next 成为下一轮的 "p_curr"        │
│                                 （即下一轮的 draft 分布 q）           │
└─────────────────────────────────────────────────────────────────────┘
```

**为什么用上一轮的分布做 draft 是合理的**：

- 迭代初期，prefix 中很多位置 token 还没确定，分布可能不准确
- 但随着 token 被逐步接受（MRS accept），context 趋向稳定
- 当 context 几乎不变时（本轮 accept 率很高），$p^{t}$ 和 $p^{t+1}$ 的 TV 距离很小
- 因此 $p^{t}$ 是 $p^{t+1}$ 的良好近似 → 下一轮直接拿 $p^{t}$ 作 draft
- 这正是 SJD 名称中 **"Jacobi"** 的含义：类似于 Jacobi 迭代解方程，用上一步的解作为下一步的初始猜测，逐步收敛

**与 SD 的本质对比**：

```
SD:   Draft Model q (单独训练)  →  q_j  [V]  →  采样 x_j  →  MRS(p_j, q_j, x_j)
SJD:  自己上一轮的 p^{t-1}    →  p^{t-1}_j  [V]  →  采样 X^t_j  →  MRS(p^{t+1}_j, p^{t-1}_j, X^t_j)
```

SD 中 q 是从另一个模型来的；SJD 中 q 就是自己上一轮算出来的分布——**同一个模型、同一个参数、只是条件 prefix 略有不同**。

---

**（2）维度相同吗？维度是多少？**

是的，**所有分布维度完全相同**：

| 变量 | 含义 | Shape | 示例值 |
|------|------|-------|--------|
| `p_curr[j]` | 本轮 draft 分布（= 上轮 verify 的 target 分布） | `[V]` | `[16384]` |
| `p_next[j]` | 模型基于新 context 计算的目标分布 | `[V]` | `[16384]` |
| `X^t_j` | 从 `p_curr[j]` 中采样的 draft token | 标量 int | `3821` |
| `X_next[j]` | MRS 修正后的最终 token | 标量 int | `3821`（accept）或 `7253`（reject 重采样） |
| `k` | 接受标志 | 标量 0/1 | `1` 或 `0` |

所有分布（`p_curr`, `p_next`）都是长度为 `V`（词表大小, vocab_size）的向量，和为 1。Lumina-mGPT 的 V=16384，LLaMA 的 V≈32000。

当并行验证 L=64 个窗口时，以 batch 形式处理：

| 批量变量 | Shape |
|----------|-------|
| `p_curr`（draft 分布） | `[L, V]` 如 `[64, 16384]` |
| `p_next`（target 分布） | `[L, V]` 如 `[64, 16384]` |
| `X^t`（draft tokens） | `[L]` 如 `[64]` |
| `X_next`（最终 tokens） | `[L]` 如 `[64]` |
| `k`（接受标志） | `[L]` 如 `[64]` |

---

**（3）Verify 的输入输出具体是什么？不是 `[\gamma, d]` 吗？**

这里容易产生一个关键误解。逐层澄清：

**误解：Verify 输入是 "target model 的 output, 比如说 γ 个 token 的 embedding `[\gamma, d]`，输出是 `[\gamma]` 个判定结果"**

**实际：Verify（MRS）操作的不是 token embedding，而是概率分布。**

整个过程分两个阶段：

```
阶段 1: Evaluate（模型前向）       ← 这是你所说的 "target model output"
  输入: [prefix_tokens + X^t] 的 embeddings, shape [prefix_len + γ, d]
        d = 模型隐藏维度（如 4096）
  输出: logits, shape [γ, V]    ← 每个位置对下一个 token 的概率原始值
        p_next = softmax(logits) → shape [γ, V]

阶段 2: Verify（MRS 验证）        ← 这才是 verify 操作
  输入:  p_next    [γ, V]    ← 模型算出的目标分布（注意：是 [γ, V] 不是 [γ, d]）
         p_curr    [γ, V]    ← draft 分布（上一轮的 p_next）
         X^t       [γ]       ← draft token IDs（整数）
  输出:  k         [γ]       ← accept/reject flags
         X_next    [γ]       ← corrected token IDs（整数）
  操作:  对每个 j ∈ [0, γ)，独立调用 MRS(p_next[j], p_curr[j], X^t[j])
```

**关键澄清**：

- 模型的 raw output 是 `[γ, V]` 的 logits（对数概率），经过了 softmax 才是概率分布 `[γ, V]`
- Verify（MRS）操作的是 **`[γ, V]` 的概率分布**，不是 `[γ, d]` 的 token embeddings
- MRS 不"看到" token 的语义内容，只"看到"概率值——它只关心 `p[x]/q[x]` 这个比值
- Verify 的 "Metric" 是 $p_j(x)/q_j(x)$，配合均匀随机数 u 做接受/拒绝判决
- 输出是 `[\gamma]` 个整数 token IDs（不是 embedding），以及 `[\gamma]` 个 0/1 接受标志

**图解**：

```
模型前向 (Evaluate):
  embeddings [prefix_len+64, 4096]
       │
       ▼  Transformer forward
  logits [64, 16384]
       │
       ▼  softmax
  p_next [64, 16384]  ← 即 "verify 的输出分布"
  ↑ 在下轮迭代中, p_curr = p_next

MRS Verify:
  输入: p_next[64,16384], p_curr[64,16384], X^t[64]
        │           │              │
        │           │              └─ 从 p_curr 采样的整数 token
        │           └─ 上一轮的 p_next（作为本轮的 draft 分布 q）
        └─ 模型基于新 context 算出的分布（作为 target 分布 p）
        
  输出: k[64], X_next[64]
        │       └─ 接受/修正后的 token IDs
        └─ accept/reject flags (遇到第一个 reject 即截断)
```

---

#### 总结

| 问题 | 答案 |
|------|------|
| 为什么上轮分布能做下轮 draft？ | Jacobi 迭代思想：相邻轮分布 TV 距离很小，$p^{t-1}$ 是 $p^t$ 的良好近似 |
| 分布维度？ | `[V] = [vocab_size]`，如 16384。p_curr 和 p_next 维度完全一致 |
| Verify 输入？ | `p_next [γ, V]`（目标分布）+ `p_curr [γ, V]`（draft 分布）+ `X^t [γ]`（draft token IDs），**不是** `[γ, d]` 的 embeddings |
| Verify 输出？ | `k [γ]`（0/1）+ `X_next [γ]`（整数 token IDs）。遇到第一个 rejection 后截断 |
| Verify 的 Metric？ | $p_j(X^t_j) / q_j(X^t_j)$，配合随机数 u 做 MRS 判决 |

---

## 论文框架速览

1. **Introduction**: AR 视觉生成慢 → SD/SJD 局限 → Coupling 方案
2. **Preliminaries**: SD 原理、MRS 算法、SJD 框架
3. **Motivation and Analysis**: 低碰撞概率导致 SJD 不稳定 → 需要提高上下文相似性
4. **Methods**: Coupling 定义 → Maximal Coupling → Gumbel Coupling → 实现细节
5. **Experimental Results**: 图像生成（Lumina/Janus）+ 视频生成（Cosmos）+ 消融分析
6. **Conclusion**: 简洁高效的训练免费无损加速方案
