# 优化中的 Hessian 使用（讨论整理）

本文档整理自关于 **Hessian**、**牛顿法**、**各类近似** 以及 **Adam 与二阶信息关系** 的讨论，面向深度学习中的参数优化，便于与一阶自适应方法（Adam、AdamW）及近年二阶/准二阶方法对照阅读。

---

## 一、总览：为什么大模型很少显式用 Hessian

| 对象 | 规模 | 典型代价 |
|------|------|----------|
| 参数维度 \(d\) | \(10^8\)–\(10^{12}+\) | — |
| 完整 Hessian \(H \in \mathbb{R}^{d\times d}\) | 存储 \(O(d^2)\) | 不可行 |
| 求 \(H^{-1}\) | \(O(d^3)\) | 不可行 |
| **Hessian–向量积 \(Hv\)** | 每步 \(O(d)\) | 可行（两次反向传播） |

**实践结论**：深度学习中几乎从不 **物化并求逆完整 \(H\)**；要么用 **\(Hv\)** 隐式使用曲率，要么用 **结构化近似**（块对角、Kronecker、对角 Fisher 等），要么 **只对少量参数** 用二阶。

```text
                    目标：利用曲率信息
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   显式完整 H          结构化 H 近似        不形成 H
  （小 d 可行）      （K-FAC, Shampoo）    （Hv, L-BFGS）
         │                 │                 │
         ▼                 ▼                 ▼
     牛顿法            块/层预条件          拟牛顿 / CG
```

---

## 二、Hessian 的推导

### 2.1 定义

设目标 \(f:\mathbb{R}^d \to \mathbb{R}\)（如损失 \(L(\theta)\)），梯度 \(g(\theta) = \nabla f(\theta)\)。**Hessian 矩阵**为梯度对参数的 Jacobian（对称）：

\[
H(\theta) = \nabla^2 f(\theta), \qquad H_{ij} = \frac{\partial^2 f}{\partial \theta_i \partial \theta_j}
\]

### 2.2 由泰勒展开得到

在 \(\theta_t\) 处二阶泰勒展开：

\[
f(\theta_t + \Delta\theta) \approx f(\theta_t) + g^\top \Delta\theta + \frac{1}{2}\Delta\theta^\top H(\theta_t)\,\Delta\theta
\]

- 一阶项 \(g^\top \Delta\theta\)：沿梯度下降方向的变化。
- 二阶项 \(\frac{1}{2}\Delta\theta^\top H \Delta\theta\)：刻画 **不同方向上的曲率**（碗口陡峭程度、方向间耦合）。

**对角元** \(H_{ii} = \partial^2 f / \partial \theta_i^2\)：仅沿坐标轴 \(\theta_i\) 的曲率。  
**非对角元** \(H_{ij}\)：参数 \(\theta_i\) 与 \(\theta_j\) 的耦合曲率。

### 2.3 与梯度的关系（重要区分）

| 量 | 定义 | 阶数 | 含义 |
|----|------|------|------|
| \(g_i\) | \(\partial f / \partial \theta_i\) | 一阶 | 当前坡度 |
| \(g_i^2\) | \((\partial f / \partial \theta_i)^2\) | — | 坡度大小的平方 |
| \(H_{ii}\) | \(\partial^2 f / \partial \theta_i^2\) | 二阶 | 沿第 \(i\) 维的曲率 |

**一般不等价**：\(g_i^2 \neq H_{ii}\)。

**例 1**：\(f(\theta)=\frac{1}{2}h\theta^2\) → \(g=h\theta\)，\(H=h\) 常数；\(g^2=h^2\theta^2\) 随 \(\theta\) 变。  
**例 2**：在极小值 \(g=0 \Rightarrow g^2=0\)，但 \(H_{ii}\) 常为正（正定极小点）。

### 2.4 负对数似然下的期望关系（Fisher）

对 **负对数似然** \(L(\theta)=-\log p(y|x,\theta)\)，得分 \(s=\nabla_\theta \log p\)。在正则条件下：

\[
F(\theta) = \mathbb{E}\big[s\, s^\top\big] = \mathbb{E}\big[\nabla L\, \nabla L^\top\big] = \mathbb{E}\big[H_{\text{data}}(\theta)\big]
\]

即 **Fisher 信息矩阵 = 损失 Hessian 对数据的期望**（完整矩阵意义）。  
**对角元**：\(F_{ii} = \mathbb{E}[g_i^2] = \mathbb{E}[H_{ii}]\)（期望意义），仍 **不等于** 当前点的 \(H_{ii}(\theta_t)\)。

### 2.5 Gauss–Newton 与 Hessian

对最小二乘、交叉熵等，真实 Hessian 可写为：

\[
H = J^\top J + \sum_k r_k \,\nabla^2 r_k
\]

**Gauss–Newton（GN）** 丢弃第二项，用 \(H \approx J^\top J\)。  
该近似常 **半正定**，且与 **Fisher / 外积 \(g g^\top\)** 结构相近，是许多“廉价二阶”方法的基础。

---

## 三、牛顿法

### 3.1 推导

将泰勒展开在 \(\theta_t\) 处对 \(\Delta\theta\) 求极小（设 \(H\) 正定）：

\[
\min_{\Delta\theta}\; f(\theta_t) + g^\top \Delta\theta + \frac{1}{2}\Delta\theta^\top H \Delta\theta
\quad \Rightarrow \quad
H\,\Delta\theta = -g
\]

**牛顿更新**：

\[
\theta_{t+1} = \theta_t - H^{-1} g
\]

等价于在度量 \(H\) 下最陡下降；在二次型目标上 **一步到达极小点**。

### 3.2 阻尼牛顿 / Levenberg–Marquardt

当 \(H\) 不定或远离极小值时，使用：

\[
\theta_{t+1} = \theta_t - (H + \lambda I)^{-1} g
\]

\(\lambda\) 增大时趋近梯度下降，提高稳健性。

### 3.3 深度学习中的障碍

| 问题 | 说明 |
|------|------|
| 规模 | \(d\) 巨大，无法存 \(H\) 或求 \(H^{-1}\) |
| 非凸 | \(H\) 不定，鞍点、负曲率方向 |
| 随机梯度 | mini-batch 的 \(H\) 噪声大 |
| 计算 | 完整二阶导代价过高 |

因此大模型 **端到端训练** 很少用经典牛顿法；多用于 **小参数子集**（最后一层、LoRA）或配合 **\(Hv\) + CG** 的截断牛顿。

### 3.4 截断牛顿（Truncated Newton）

不解完整线性系统，用共轭梯度（CG）迭代近似解 \((H+\lambda I)\Delta\theta=-g\)，每步只需 **\(Hv\)**，存储 \(O(d)\)。这是 **显式使用 Hessian 信息** 且可扩展的主要路径之一。

---

## 四、其他方法对 Hessian 的近似

### 4.1 分类总表

| 类别 | 代表方法 | 近似对象 | 是否物化 \(d\times d\) 的 \(H\) | 典型代价 |
|------|----------|----------|--------------------------------|----------|
| **拟牛顿** | BFGS, L-BFGS | \(H^{-1}\) 低秩递推 | 否（存 \(m\) 对 \(s,y\)） | \(O(md)\) |
| **Hessian–向量积** | Pearlmutter; CG + 截断牛顿 | 隐式 \(Hv\) | 否 | \(O(d)\)/步 |
| **随机估计** | Hutchinson | \(\mathrm{tr}(H)\), 谱 | 否 | 多次 \(Hv\) |
| **对角** | AdaGrad, RMSprop, Adam 分母 | \(\mathrm{diag}(F)\) 或 \(\mathbb{E}[g^2]\) | 否 | \(O(d)\) |
| **块对角 / 按层** | 分层牛顿 | \(H_{\ell\ell}\)，跨层为 0 | 每层小块 | 取决于层宽 |
| **Kronecker** | K-FAC, Shampoo | \(F_\ell \approx A_\ell \otimes G_\ell\) | 因子矩阵 | 层内 \(O(n^2+m^2)\) |
| **Gauss–Newton** | GN, LM | \(J^\top J\) | 可选 \(Jv\) | 同 \(Hv\) 量级 |
| **低秩** | Lanczos + \(Hv\) | top-\(r\) 特征方向 | 否 | \(r\) 次 \(Hv\) |
| **子集参数** | Head/LoRA + L-BFGS | 仅 \(d'\ll d\) 维的 \(H'\) | 小矩阵可显式 | \(O(d'^2)\) |
| **Sophia 等** | \(Hv\) 估对角曲率 | \(\mathrm{diag}(H)\) 的估计 | 否 | \(O(d)\) |

### 4.2 拟牛顿法（Quasi-Newton）

用梯度差分 \(\Delta g\)、参数差分 \(\Delta\theta\) 更新对 \(H^{-1}\) 的近似，满足割线方程。**L-BFGS** 只保留最近 \(m\) 步历史，适合 **中等规模、全批量或低噪声** 问题（小模型微调、传统 ML）。小 batch 深度学习中需方差缩减等才较稳。

### 4.3 Hessian–向量积 \(Hv\)

**不形成 \(H\)**，用两次反向传播计算 \(v \mapsto Hv\)。用于：

- 截断牛顿 + CG；
- Hutchinson：\(\mathrm{tr}(H) \approx \mathbb{E}_z[z^\top H z]\)；
- 曲率谱、最大特征值估计。

### 4.4 结构化近似：“只算一部分参数之间的 Hessian”

常见含义：

1. **块对角**：\(H \approx \mathrm{blockdiag}(H^{(1)}, H^{(2)}, \ldots)\)，层 \(\ell\) 只算 \(H_{\ell\ell}\)，忽略 \(H_{\ell\ell'}\)。
2. **K-FAC**：层内 \(F_\ell \approx A_\ell \otimes G_\ell\)，求逆用 \(A^{-1}\otimes G^{-1}\)。
3. **Shampoo**：每层左右预条件矩阵，避免完整 \(nm\times nm\) 矩阵。
4. **子集参数**：仅对 head、LoRA 等 \(d'\ll d\) 的参数做牛顿或 L-BFGS。

### 4.5 对角预条件与 Fisher

| 名称 | 公式（对角） | 与 Hessian 关系 |
|------|----------------|-----------------|
| **Fisher** | \(F_{ii}=\mathbb{E}[g_i^2]\) | \(F=\mathbb{E}[H]\)（NLL） |
| **经验 Fisher** | \(\hat{F}_{ii}=\frac{1}{B}\sum_b g_{b,i}^2\) 或 EMA | 用观测梯度估计 \(F_{ii}\) |
| **Hessian 对角** | \(H_{ii}=\partial^2 L/\partial\theta_i^2\) | 当前点真曲率 |

**经验预条件**：用训练过程中观测到的 \(g\) 构造 \(\hat{P}\)（如 \(\hat{P}_{ii}=\mathrm{EMA}(g_i^2)\)），更新 \(\theta \leftarrow \theta - \alpha\,\hat{P}^{-1/2} m\)。

### 4.6 实用选型（经验）

```text
大模型默认训练        → Adam / AdamW（对角经验 Fisher 式预条件）
按层、可接受实现成本  → K-FAC, Shampoo
要“真”曲率且可研究   → Hv + 截断牛顿；Sophia（Hv 估对角 H）
小参数子集微调        → L-BFGS 或显式小 Hessian（head / LoRA）
```

---

## 五、Adam 与 Hessian 的关系

### 5.1 Adam 在做什么（简要）

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\]
\[
\theta_{t+1} = \theta_t - \alpha \,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\]

\(v_t\) 跟踪 **梯度逐元素平方的 EMA**，即 **未中心化** 的 \(\mathbb{E}[g_i^2]\) 估计。

### 5.2 核心结论（讨论归纳）

| 说法 | 是否正确 |
|------|----------|
| Adam **显式计算或使用** 当前点的 Hessian \(H(\theta_t)\) | ❌ **否** |
| Adam 用 \(g_i^2\) **近似** 当前 \(H_{ii}\) | ❌ **一般不成立** |
| Adam 用 \(\mathrm{EMA}(g_i^2)\) 近似 **对角经验 Fisher** \(\hat{F}_{ii}\) | ✅ **最贴切** |
| 在 NLL 下，\(F=\mathbb{E}[H]\)，故 \(v_t\) 与 **期望意义下的平均曲率** 间接相关 | ⚠️ **有条件**（见下） |

**一句话**：Adam 是 **带对角经验 Fisher 预条件的一阶优化器**，**不是** Hessian 近似器。

### 5.3 为何常与“二阶”混淆

1. 分母 \(\sqrt{v_t}\) 起 **按坐标缩放步长** 的作用，类似对角预条件 \(\mathrm{diag}(F)^{-1/2}g\)。
2. 对数似然损失下，**Fisher 在期望上等于 Hessian**，但 Adam 估计的是 **\(F_{ii}=\mathbb{E}[g_i^2]\)**，不是 **\(H_{ii}(\theta_t)\)**。
3. Gauss–Newton 与 \(g g^\top\) 结构相近，文献中常将自适应方法与自然梯度/Fisher 联系在一起。

### 5.4 与牛顿 / 真二阶的对比

| | 牛顿法 | Adam |
|--|--------|------|
| 使用信息 | \(H^{-1}g\)（当前点二阶） | \(g/\sqrt{\mathbb{E}[g^2]}\)（一阶 + 统计缩放） |
| 耦合 | 完整 \(H\) 含非对角 | 仅对角，忽略 \(F_{ij}\) |
| 代价 | 不可行（大 \(d\)） | \(O(d)\) |
| 与 \(H_{ii}\) | 直接（若可算） | 无直接对应 |

### 5.5 何时“更像曲率”、何时不像

**更接近平均曲率尺度的情形**：

- 损失为交叉熵 / 负对数似然；
- batch 较大，\(v_t\) 已充分平滑；
- 接近收敛，\(\mathbb{E}[g]\approx 0\)，\(v_t\) 更接近 \(\mathrm{Var}(g_i)\approx F_{ii}\)。

**差别大的情形**：

- MSE、强化学习、对抗训练等，\(F=\mathbb{E}[H]\) 解释变弱；
- 远离极小值，\(v_t\) 含 \((\mathbb{E}[g])^2\) 偏差（Adam 用偏置校正缓解）；
- 需要参数间耦合时，对角方法无法替代完整 \(H^{-1}\)。

---

## 六、概念关系图

```text
  泰勒展开 ──► 牛顿法:  Δθ = -H⁻¹g
       │
       ├──► 完整 H 不可行（大 d）
       │
       ├──► 隐式:  Hv + CG（截断牛顿）
       │
       ├──► 拟牛顿: L-BFGS（低秩 H⁻¹）
       │
       ├──► 结构化: K-FAC, Shampoo, 块对角
       │
       └──► 对角统计: E[g²] ──► 经验 Fisher ──► Adam 分母
                    │                    │
                    │                    └── 期望下 F = E[H]（NLL）
                    └── 一般 ≠ 当前 H_ii
```

---

## 七、讨论中的易错点速查

| 易错点 | 正确理解 |
|--------|----------|
| “\(g_i^2\) 就是 \(H_{ii}\)” | 否；\(g\) 是一阶，\(H_{ii}\) 是二阶 |
| “Adam 近似 Hessian” | 更准确：对角 **经验 Fisher** 预条件 |
| “Adam 完全与曲率无关” | 在 NLL 下与 **\(\mathbb{E}[H]\)** 的对角有统计联系，非当前 \(H\) |
| “Fisher 对角 = Hessian 对角” | 仅 **\(F_{ii}=\mathbb{E}[H_{ii}]\)**，非逐点相等 |
| “用大模型也要上完整牛顿” | 实践中用 \(Hv\)、块结构或子集参数 |

---

## 八、延伸阅读方向

- **Pearlmutter (1994)**：快速 \(Hv\) 算法。
- **Martens & Grosse, K-FAC**：Kronecker 因子化 Fisher/GN。
- **Kingma & Ba, Adam**：自适应学习率与 \(v_t\) 动机。
- **Sophia** 等：用 \(Hv\) 估计对角曲率，比 \(g^2\) 更贴近 \(H_{ii}\) 的一阶可扩展方案。
- **自然梯度 / Amari**：\(F^{-1}g\) 与统计流形上的最陡下降。

---

*文档类型：讨论整理 · 主题：Optimization / Hessian · 与仓库内压缩、量化文献中 “Hessian-aware”（如 HAWQ、Q-BERT）可对照：后者常估 **对角 \(H_{ii}\)** 或 **\(Hv\)** 用于剪枝/量化敏感度，与训练优化器中的 Hessian 使用同属曲率信息，但目标不同。*
