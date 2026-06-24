# Muon 与 LLM 训练优化（讨论整理）

本文档整理自关于 **Muon** 优化器及其在 **大语言模型（LLM）训练** 中用法、动因与 **Muon + AdamW 混合** 实践的讨论。主题属于 **LLM Training Optimization**，与仓库内 `Summary/Optimization/summary_hessian_optimization.md`（Hessian / Adam 曲率视角）互补：**Muon 不用 Hessian，而是对矩阵更新的正交化 + 谱范数标度**。

---

## 一、总览：Muon 在优化器谱系中的位置

| 优化器 | 主要机制 | 适用参数 | 与 Hessian |
|--------|----------|----------|------------|
| **SGD / SGD-momentum** | 固定或全局 lr × 梯度（+ 动量） | 全部 | 无 |
| **AdamW** | 动量 \(m_t\) + 对角 \(v_t\) + 解耦 weight decay | 全部 | 不显式；\(v_t\) 近似对角经验 Fisher |
| **Muon** | 动量 → **Newton–Schulz 正交化** → 谱范数 lr | **主要为 2D 隐藏权重** | 无 |
| **K-FAC / 截断牛顿** | 块结构或 \(Hv\) 曲率 | 按层 / 全模型 | 显式或近似二阶 |

```text
  LLM 参数
      │
      ├── 2D 矩阵 W（Linear, Attention 投影）──► Muon：正交化更新
      │
      └── 向量/标量/Embedding/Head ───────────► AdamW：逐元素自适应
```

**实践共识**：端到端 LLM 训练几乎总是 **Muon 与 AdamW 混用**，而非单独 Muon 或单独 AdamW 包打天下。

---

## 二、Muon 是什么

### 2.1 名称与提出

- **Muon** = **M**oment**U**m **O**rthogonalized by **N**ewton–**Schulz**。
- 由 Keller Jordan 等推广（2024 起 NanoGPT speedrun、博客与开源实现）；大模型扩展见 [*Muon is Scalable for LLM Training*](https://arxiv.org/abs/2502.16982)（Moonlight 等）。
- 参考实现：[KellerJordan/Muon](https://github.com/KellerJordan/Muon) · [博客说明](https://kellerjordan.github.io/posts/muon/)。

### 2.2 核心思想（一句话）

对 **二维权重矩阵** 的更新：先做 **SGD 式动量**，再把更新矩阵 **近似正交化**，最后用 **按谱范数（算子范数）理解的学习率** 作用到权重上。

**正交化的是「更新 \(\Delta W\)」**，不是把参数 \(W\) 本身约束成正交矩阵。

### 2.3 单步算法（矩阵参数 \(W \in \mathbb{R}^{m\times n}\)）

设 \(G_t = \nabla_W L\) 为当前 batch 梯度。

**Step 1 — 动量**

\[
M_t = \beta M_{t-1} + (1-\beta) G_t, \quad \beta \approx 0.95
\]

常用 **Nesterov** 形式：先用 \(G_t\) 与 \(M_t\) 组合得到用于正交化的矩阵（实现见 `muon_update`）。

**Step 2 — Newton–Schulz 近似正交化**

对 \(M_t\) 迭代若干步（常 5 步），在 **bfloat16** 上近似：

\[
M_t \;\mapsto\; U V^\top \quad (\text{若 } M_t = U\Sigma V^\top)
\]

即把更新换成 **最近的半正交矩阵**（奇异值压到 ~1 附近）。实现 **不做完整 SVD**（太贵），而用 NS 迭代；结果可为 \(US'V^\top\)，\(S'_{ii} \in [0.5, 1.5]\) 量级，作者称对效果影响很小。

**Step 3 — 矩形缩放（可选）**

\[
\widehat{\Delta W} \;\leftarrow\; \widehat{\Delta W} \cdot \sqrt{\max(1,\, m/n)}
\]

**Step 4 — Weight decay + 更新**

\[
W \leftarrow W \cdot (1 - \eta\,\lambda) - \eta\,\widehat{\Delta W}
\]

（AdamW 式解耦 decay：先衰减 \(W\)，再加更新。）

**学习率 \(\eta\)**：按 **每步谱范数变化 \(\eta\)** 理解，与 Adam 的逐坐标 lr **不是同一套标度**。

### 2.4 伪代码

```text
输入: 梯度 G, 动量缓冲 M, lr η, momentum β≈0.95, NS 步数=5
1. 更新动量 M（含可选 Nesterov）
2. Δ ← NewtonSchulz(M, steps=5)      # 近似正交，bf16
3. Δ ← Δ * sqrt(max(1, m/n))          # 非方阵
4. W ← W * (1 - η*wd) - η * Δ
```

### 2.5 与 Adam、Hessian 的区别

| | Muon | AdamW |
|--|------|-------|
| 更新几何 | 矩阵整体、正交化 | 逐元素 \(g/\sqrt{v}\) |
| 状态 | 主要是 **动量矩阵** \(M\) | \(m_t\) + \(v_t\) |
| 曲率 | **不用** Hessian / Fisher 完整信息 | \(v_t\) 对角经验 Fisher |
| 参数类型 | **仅建议 2D 隐藏权重** | 通用 |

### 2.6 动量为何几乎必需（Muon 语境）

正交化会 **重塑更新方向**；若对 **单步裸梯度** 正交化，噪声过大、不稳定。  
**先动量、再正交**：\(M_t\) 是多步梯度信号的聚合，再正交化，经验上明显优于无动量（预训练默认）。  
**例外**：部分 **RL / 异步** 栈为省 state 使用 **Muon 零动量**（见第五节 Zaya1 例）。

---

## 三、为什么 Muon 适合 LLM 训练

### 3.1 LLM 里绝大多数可学习量都是「矩阵算子」

Transformer 中：

- Attention 的 \(W_Q, W_K, W_V, W_O\)
- MLP 的 \(W_{\mathrm{up}}, W_{\mathrm{down}}\)（及 gate）

均为 **大矩阵**。一次前向是 **线性算子** 的复合；Adam 把 \(W\) 拆成独立标量，忽略 **奇异值结构** 与 **方向耦合**。

Muon 显式把 **\(W\) 当作整体** 更新，缓解 **谱偏置（spectral bias）**：更新长期沿 **少数主导奇异方向** 挤压，有效秩塌缩、宽度方向利用不均。

### 3.2 正交更新 ≈ 方向均衡的矩阵步

正交（或近似正交）的 \(\Delta W\) 使各奇异方向上的更新幅度更均衡，避免「只拧最大奇异方向」。  
理论上有 **谱范数下最陡下降** 的表述（Bernstein & Newhouse 等）：在算子范数约束下选最大下降方向，与「标量 Adam」是不同几何。

### 3.3 样本效率与训练步数

社区与论文报告（小模型 speedrun → 十亿级 Moonlight）：

- 在相同 token / 相同算力预算下，**达到同等验证 loss 所需步数更少**；
- 对 **预训练 + SFT** 均可使用（大规模扩展需分布式 NS、lr 迁移等工程）。

注意：不是「免费午餐」——正交化有 **额外算力**（NS 迭代），净收益体现在 **收敛步数 / 最终质量**。

### 3.4 与宽度缩放（μP）、学习率迁移

Muon 配合 **μP（maximal update parametrization）** 类讨论：矩阵更新的谱范数标度与 **随宽度变化的学习率规则** 更易对齐。  
宽模型上 **Muon lr 与 Adam 组 lr 需分别调**；Moonlight 等工作中包含 **跨宽度 / 跨规模迁移** 的经验。

### 3.5 与 LLM 训练「 pathology」的匹配

| LLM 训练特点 | Muon 的对应 |
|--------------|-------------|
| 超高维、mini-batch 梯度极噪 | 动量平滑后再正交化 |
| 损失面病态、各向异性 | 矩阵级更新而非仅逐元素缩放 |
| 极长训练、微弱单步信号 | 动量时间累积（与 Adam \(\beta_1\) 类似动机） |
| 参数海量、无法二阶 | \(O(mn)\) 每矩阵，NS 固定小步数，无 \(d\times d\) Hessian |

### 3.6 大规模训练上的可扩展性（论文要点）

[*Muon is Scalable for LLM Training*](https://arxiv.org/abs/2502.16982) 针对：

1. 十亿参数 + 万亿 token 上稳定预训练 / SFT；
2. **分布式** 下近似正交化的实现；
3. 与 AdamW 混合时的 **全局超参** 与 **RMS 对齐** 等工程细节。

### 3.7 不适合用 Muon 的部分（必须交给 AdamW）

| 参数 | 原因 |
|------|------|
| **Token embedding** | 非「隐藏线性层」；行向量语义特殊 |
| **LM head** | 常与 embedding tying；输出层尺度敏感 |
| **Bias、LayerNorm/RMSNorm \(\gamma\)** | 1D，无矩阵 SVD/NS 结构 |
| **MoE router、标量门控** | 1D / 特殊 |

官方与社区实现均强调：**隐藏 2D 权重用 Muon，其余用 AdamW**。

---

## 四、Muon 与 AdamW 的结合用法（实践）

### 4.1 参数分组（标准模板）

```python
# 概念分组（名称因模型而异）
hidden_2d = [p for n, p in model.named_parameters()
             if p.ndim >= 2 and "embed" not in n and "lm_head" not in n]
embed     = [p for n, p in model.named_parameters() if "embed" in n]
head      = [model.lm_head.weight]  # 若未 tie
scalars   = [p for p in model.parameters() if p.ndim < 2]

# Muon 组
muon_group  = dict(params=hidden_2d, lr=η_muon, momentum=0.95, use_muon=True)
# AdamW 组（可多组不同 lr）
adam_embed  = dict(params=embed,  lr=η_embed,  betas=(0.9, 0.95), use_muon=False)
adam_head   = dict(params=head,   lr=η_head,   betas=(0.9, 0.95), use_muon=False)
adam_scalar = dict(params=scalars, lr=η_scalar, betas=(0.9, 0.95), use_muon=False)
```

**卷积**：将 kernel reshape 为 2D 后归入 Muon 组（参考 `muon.py` 注释）。

### 4.2 单一 Optimizer：`MuonWithAuxAdam`

[KellerJordan/Muon](https://github.com/KellerJordan/Muon) 提供 **`MuonWithAuxAdam`** / **`SingleDeviceMuonWithAuxAdam`**：

- `use_muon=True` 的 param group → 内部走 `muon_update`；
- `use_muon=False` → 内部走 `adam_update`（含 \(m_t, v_t\)、偏置校正逻辑可简化版）；
- 训练循环只需 **`optimizer.step()` 一次**。

避免维护两个 optimizer 的 step 顺序、lr scheduler 同步等问题。

### 4.3 学习率：两套标度，勿直接照搬

| 组 | lr 量级（仅作数量级参考，需按模型重调） |
|----|----------------------------------------|
| Muon（隐藏矩阵） | 常 **更大** 的谱范数 lr（如 ~0.02–0.05 量级，与实现强相关） |
| AdamW（embed/head/1D） | 接近常规 LLM AdamW（如 \(10^{-4}\)–\(10^{-3}\) 或 μP 规则导出） |

**原则**：Muon 的 \(\eta\) 不是 Adam 的 \(\eta\)；从「全 AdamW 基线」迁到 Muon 时，**矩阵组与 Adam 组分别搜索**，并配合 **warmup、wd、全局 batch**。

### 4.4 Weight decay

- Muon 组、Adam 组均可设 **解耦 weight decay** \(\lambda\)；
- 大模型预训练常 **矩阵与 embedding 使用不同 \(\lambda\)**（与纯 AdamW  recipe 类似）。

### 4.5 AdamW RMS matching（工业变体）

仓库内 **Zaya1-8B** 等 recipe 提到：**Muon（含 AdamW RMS matching）**。

**含义（工程向）**：Muon 更新与 AdamW 更新的 **RMS 尺度** 对齐，避免：

- 矩阵层（Muon）与 embedding/head（Adam）**有效步长差几个数量级**；
- 切换宽度或 depth 时一组过快、一组过慢。

具体公式因实现而异，常见思路是对 Muon 组或 Adam 组乘 **共享或慢变的 RMS 比例系数**（训练框架内部统计 gradient/update RMS）。  
**落地以各训练栈文档为准**；概念上属于 **混合优化器校准**，不是 Muon 论文必选项。

### 4.6 分布式训练

官方 `Muon` 类在 `step()` 内对参数做 **分片 + `all_gather`**，使 NS 与更新在数据并行下一致。  
大模型需使用 **支持分布式的 Muon 分支**（或 Moonlight / 内部 fork），勿直接把单卡 `SingleDeviceMuon` 裸用到百亿参数。

### 4.7 变体：Muon 零动量（RL / 异步场景）

**Zaya1-8B**（`summary_zaya1_8b.md`）在 **PipelineRL** 等后训练阶段：

- **矩阵参数**：Muon **零动量**（每步仅用当前 batch \(G_t\) 做 NS，无 \(M_{t-1}\) 累积）；
- **Embedding / LM head**：仍 **AdamW**。

**动机**：

| 因素 | 说明 |
|------|------|
| 省 optimizer state | RL 多阶段、多 worker，动量缓冲占显存 |
| 非平稳目标 | policy 分布随 rollout 变，长记忆动量可能拖尾 |
| 与稀疏子网络更新类比 | 引用 RL 文献中「每步 fresh 梯度」类做法 |

**注意**：这是 **后训练工程变体**；**预训练 / 中训练** 仍推荐 **\(\beta \approx 0.95\) 动量** 的标准 Muon。

### 4.8 推荐检查清单（上线前）

- [ ] 仅 **2D 隐藏权重** 进 Muon；embed / head / norm / bias 进 AdamW  
- [ ] 使用 **MuonWithAuxAdam** 或等价封装，单 `step()`  
- [ ] Muon lr、Adam lr、wd、warmup **分开调**  
- [ ] 分布式 NS 与 **梯度同步** 已验证  
- [ ] 与 **μP / 宽度缩放** 文档对照（若做宽模型迁移）  
- [ ] RL 阶段若用零动量，确认 **不与预训练 checkpoint 的 optimizer state 强绑定**

---

## 五、与仓库其他文档的关系

| 文档 | 关系 |
|------|------|
| `summary_hessian_optimization.md` | Adam / Hessian / Fisher；Muon **不走** Hessian 路线 |
| `summary_zaya1_8b.md` | 实例：Muon + AdamW RMS matching；RL 阶段 Muon 零动量 |
| `Paper/...` 中 HAWQ、Q-BERT 等 | 「Hessian-aware」用于 **量化敏感度**，与训练优化器 Muon **目标不同** |

---

## 六、讨论要点速查

| 问题 | 结论 |
|------|------|
| Muon 关键是不是「每步更新尽量正交」？ | **是**（对 **更新矩阵** 近似正交），但不是对 \(W\) 本身，且为 **近似** NS |
| Muon 是否近似 Hessian？ | **否** |
| 为何 LLM 常用 Muon？ | 矩阵权重大、谱偏置、样本效率与 μP 标度；大模型已有可扩展实现 |
| 能否全参数 Muon？ | **不推荐**；embed/head/1D 用 AdamW |
| 动量在 LLM 里为何重要？ | 降噪 + 病态景观中方向累积；Muon 上 **正交化前几乎必需**（预训练） |
| 与 Adam 关系？ | **互补混用**，非替代；同一 `optimizer` 内分组最常见 |

---

## 七、延伸阅读

- Keller Jordan, [Muon blog](https://kellerjordan.github.io/posts/muon/)  
- [KellerJordan/Muon](https://github.com/KellerJordan/Muon)（`muon.py`, `MuonWithAuxAdam`）  
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)  
- Bernstein & Newhouse, 谱范数最陡下降与 Muon 的理论联系（2024–2025）  
- 仓库：`Summary/Large Pretraining Models/summary_zaya1_8b.md`（混合优化器与 RL 变体）

---

*文档类型：讨论整理 · 主题：LLM Training Optimization / Muon · 与 `summary_hessian_optimization.md` 并列于 `Summary/Optimization/`。*
