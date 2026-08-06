# Parallel Decoding Distillation (PDD) —— 面向快速图像与视频生成的并行解码蒸馏

> 论文：Parallel Decoding Distillation for Fast Image and Video Generation
> 作者：Neta Shaul (Weizmann), Chao Liu, Arash Vahdat†, Julius Berner† (NVIDIA)
> arXiv: 2607.26004
> 项目页: https://research.nvidia.com/labs/genair/pdd

## 0. 综合理解与点评

### 综合理解

PDD 在预训练 flow/diffusion 模型的 backbone 上**增加 N 个并行输出头**（N = 时间网格大小），每次前向同时输出 N 个子区间的**平均速度**预测。**训练时**：采样块起始 n 和块内索引 k，用 student 前 (k−n) 个头的速度预测做累加位移，推进到中间状态 $\bar X_k$；然后让冻结的 teacher 在 $\bar X_k$ 处用一步 Runge-Kutta（Euler 1 次或 Midpoint 2 次前向）估计该子区间的平均速度，作为 student 第 k 个头的回归目标（MSE）。**推理时**：按块大小 L 取出连续 L 个头的速度做一次累加位移跳过 L 步，实现 N/L 次 NFE 的快速采样；并可层融合为单一线性层消除额外开销。Teacher 采用 RK2（Midpoint）是为了**更精确地估计监督目标**（平均速度），而非加速 teacher 自身推理。

## LLM总结

PDD 是一种**纯轨迹蒸馏**方法：把教师模型的多步去噪轨迹离散化为若干"块"，让学生用一个"并行解码头"在**一次前向**中预测块内所有子区间的平均速度；只需一个简单的回归损失，**不需要 VSD/GAN/JVP/有限差分**，支持推理时动态切换 NFE，在 LTX-2.3 / Wan2.1 14B / Qwen-Image 上达到 SOTA，且**显著保留了视频多样性与运动量**。

---

## 2. 研究背景与动机

扩散 / 流匹配模型生成成本高昂（数百次网络评估）。当前加速方法分两类：

- **轨迹类**（Progressive Distillation、Consistency Models、Mean Flow、Pi-Flow 等）：把教师的多步轨迹压缩成少步。在图像上有效，但在视频大模型上质量退化明显，常依赖 JVP 或有限差分，训练昂贵且不稳定。
- **分布类**（ADD/LADD、DMD/DMD2、VSD、f-distill 等）：只对齐学生与教师的边缘分布。是当前视频大模型蒸馏的主流（rCM、AnyFlow、TwinFlow、PhaseDMD、TMD），但**易出现模式崩溃**——多样性下降、视频静态、缺乏运动。

PDD 的目标：**用纯轨迹蒸馏在大规模视频模型上达到分布类方法的 SOTA，同时避免其多样性损失**。

---

## 3. 方法核心

### 3.1 并行解码器（Parallel Decoder）

- 时间域离散为 $N$ 段：$0=t_0<\dots<t_N=1$，按块大小 $L$ 分组。
- 块起始 $n$ 处状态 $X_n\sim p_{t_n}$，并行解码器 $\bar u^\theta_n(\cdot\mid X_n)\in\mathcal X^L$ 在**单次网络评估**内预测块内全部 $L$ 个子区间的平均速度：
  $$\bar u^\theta_n(k\mid X_n)\approx u_k(X_k),\quad k=n,\dots,n+L-1.$$
- 关键点：因为流过程在该块内由初值 $X_n$ 与精确解完全决定，故该并行预测是**良定义**的。

### 3.2 块步采样（Block-step Sampling）

求解递推得到跳过 $L$ 步的更新规则：
$$\bar X_{n+L}=X_n+\sum_{k=n}^{n+L-1}(t_{k+1}-t_k)\bar u^\theta_n(k\mid X_n).$$
重复 $N/L$ 次即得样本，从而 NFE $=N/L$。

### 3.3 PD 损失（单目标、on-policy）

$$\mathcal L_{\text{PD}}(\theta)=\mathbb E\big\|\bar u^\theta_n(k\mid X_n)-u_k(\mathrm{sg}(\bar X_k))\big\|^2,$$
- 块起 $n\in\{0,L,\dots,N-L\}$、块内 $k\in\{n,\dots,n+L-1\}$ 均匀采样；
- $X_n\sim p_{t_n}$ 用插值过程采样；
- $\bar X_k$ 由学生自身（parallelized process）展开；
- 教师 $u_k$ 用 **Runge-Kutta 一步**（Euler 1 次评估 / Midpoint 2 次评估）近似。
- 学生 $\bar u^\theta_n$ 与 $\bar X_k$ 来自同一前向 → 训练成本与教师评估数同阶。
- **Proposition 1**：PD 损失的全局极小满足并行解码条件，且（在 RK 误差范围内）采样精确重合教师轨迹 $\bar X_n=X_n$。

### 3.4 架构与可变块大小

- 共享教师 backbone $H_t$，仅在**最后线性层复制 $N$ 份**：$\bar u^\theta_n(k\mid x_n)=W^\theta_k H^\theta_{t_n}(x_n)$。
- $W^\theta_k$ 由教师末层初始化，**无需第二时间坐标**（区别于 flow map）。
- 训练时设定 $L_{\min},L_{\max}$，按 $L_{\min}$ 倍数选 $n$、块内按 $L_{\max}$ 采 $k$，**单模型即可在多种 NFE 下推理**。

### 3.5 层融合（Layer Fusion）—— 与 flow map 的联系

- 推理时只用加权平均方向，把块内 $L$ 个线性层融合成一个：
  $$W^\theta_{n:n+L}=\sum_{k=n}^{n+L-1}\Delta_k W^\theta_k,\quad \Delta_k=\frac{t_{k+1}-t_k}{t_{n+L}-t_n}.$$
- → **推理时无额外计算/显存开销**；训练时梯度通过共享 backbone 在期望意义下学习全区间平均速度。
- 与 Lagrangian flow map 相比：PDD 把"学习位移导数对齐瞬时速度"换成"直接回归数值积分速度"，**免 JVP / 有限差分**。

### 3.6 Data-free 训练

无数据场景下采用在线 on-policy：每次前向既做优化又推进状态（对推进操作 stop-gradient），无需用插值过程采 $X_n$。对 Qwen-Image / Wan2.1 / LTX-2.3 均用此模式（仅 ImageNet 用真数据）。

### 3.7 与同类方法差异（表 1）

| | Eulerian/Lagrangian Flow Maps | Pi-Flow | **PDD** |
|---|---|---|---|
| NFE | 可变 | 固定 | **可变** |
| JVP / 有限差分 | 必需 | 无 | **无** |
| 推理头 | Linear | 高斯混合 | **融合线性** |

---

## 4. 实验

### 4.1 ImageNet-256（教师：SiT-XL+REPA）

- $N=128$ (Euler) / $N=64$ (Midpoint)，batch 2048，300k iter，lr 5e-5，EMA 0.99995。
- 1-NFE FID：PDD-Euler 2.73、PDD-Midpoint 2.69；对比 Pi-Flow 2.85、FreeFlow 1.45。**在简化目标 + 支持多 NFE 的前提下极具竞争力**。

### 4.2 Qwen-Image 文生图（OneIG / DPG / GenEval）

- $N=256$ (Euler) / $128$ (Midpoint)，shift $s=5$，CFG $w=4$，batch 2048，3k iter，无 EMA，data-free。
- NFE∈{2,4,8}：PDD-Midpoint 在 OneIG-EN/DPG/GenEval 整体分**几乎全部第一**（如 4-NFE: 0.538 / 88.66 / 0.86）。
- HPSv2/PickScore 略次于 DMD2(Lightning-v2)，但 **OneIG 多样性显著领先**（0.18–0.20 vs DMD2 的 0.09–0.11），证明缓解了模式崩溃。

### 4.3 Wan2.1 文生视频（VBench）

- 1.3B / 14B；$N=256$ (Euler) / $128$ (Midpoint)，shift $s=6$，CFG $w=5$，**跳过 1 层 unconditional 分支**（1.3B 跳第 10 层、14B 跳第 12 层）。
- 1.3B @ 4-NFE：**Overall 84.94、Quality 86.45 第一**；多样性 V-JEPA2/VideoMAE V2 同时第一。
- 14B @ 4-NFE：PDD-short (200 iter) Quality 85.71 第一、Overall 84.92 次于 AnyFlow；PDD-long (3k iter) 多样性最高但 VBench 略低 → 训练后期运动增强。
- 8-NFE 同样表现优异。
- 与 DMD2/AnyFlow 相比，PDD 视频运动量与多样性明显更好。

### 4.4 LTX-2.3 文生视频/音频（22B，720p，10s，含音频）

- 仅 Euler、$N=256$、单区间，250 iter；视频/音频 latent 分别施加 PD 损失后取均值。
- 教师 CFG + cross-modal guidance + spatiotemporal skip guidance 折算为单次 student 前向 → **8 NFE** 与官方 distilled 8-step 相当或更好（Gemini 3.1 Pro 评判：四轴平均 PDD 胜 142 / 平 35 / 负 123）。

### 4.5 关键消融

- **Midpoint 普遍优于 Euler**（一致性提升）。
- **Batch size 越大越好**、**时间重参数化（shift）**是两个最敏感超参。
- 不同 NFE 适合不同 CFG 尺度（ImageNet 上 8-NFE 反而需要更低 guidance）。

---

## 5. 优势 / 局限

### 优势
1. 训练目标极简：单一回归 MSE，无 VSD/GAN/JVP/有限差分/多阶段。
2. 显存与计算友好：师生共享 backbone，仅末层复制 $N$ 份；推理通过层融合**零额外开销**。
3. **变量 NFE**：一模型支持 2/4/8 步自适应，无需第二时间条件。
4. **多样性 / 运动量**显著优于分布类方法。
5. 大规模可扩展：从 ImageNet 到 22B 多模态视频音频一致有效。

### 局限 / 未来工作
- 大规模文生图/视频实验均依赖 **data-free**，数据相关设定（除 ImageNet 外）未充分探索。
- 块大小在推理时静态选择；**自适应块大小**（基于 verifier / 置信度）是未来方向。
- 可推广至**离散自回归模型**的并行解码。

---

## 6. 个人启发

1. **核心洞察**：流过程块内由初值唯一决定 → 可让一个网络同时输出整块所有子步速度，无需在结构上引入"目标时间"条件。这与 mean flow / shortcut model 的"两时间坐标"路径形成鲜明对比，把表达力交给**多个独立末层 + 共享 backbone**。
2. **训练-推理解耦**：训练用 $N$ 个并行头监督子区间，推理用融合头跳整块 —— 这是把"学平均速度"转化为"学可加权和的子速度分量"的漂亮技巧，等价于让 backbone 在期望意义下吸收全区间梯度。
3. **避免 JVP/有限差分**：把 flow map 的"位移导数 = 速度"约束换成"子区间平均速度 = 教师 RK 估计"，本质上是把连续时间优化改写为离散回归，从而避开高阶自动微分的不稳定性。
4. **多样性保留**：相比 DMD2，PDD 不引入对抗/score-matching 目标，因而不会把分布"塌缩"到教师轨迹的局部模态，这对视频运动量尤其关键。
5. **实践提示**：跳一层 unconditional CFG、shift 时间重参数化、Midpoint 监督，是迁移到新模型时可直接尝试的强 baseline 配置。

---

## 7. 关键公式速查

| 名称 | 公式 |
|---|---|
| 流过程 | $\frac{d}{dt}X_t=v_t(X_t),\ X_0\sim p_0$ |
| 精确解 | $X_{n+1}=X_n+(t_{n+1}-t_n)u_n(X_n)$ |
| 平均速度 | $u_n(X_n)=\frac{1}{t_{n+1}-t_n}\int_{t_n}^{t_{n+1}}v_t(X_t)dt$ |
| Euler 近似 | $u_n\approx v_{t_n}(X_n)$ |
| Midpoint 近似 | $u_n\approx v_{t_\text{mid}}(X_\text{mid})$ |
| 并行解码器 | $\bar u^\theta_n(k\mid X_n)\approx u_k(X_k)$ |
| 块步采样 | $\bar X_{n+L}=X_n+\sum_k(t_{k+1}-t_k)\bar u^\theta_n(k\mid X_n)$ |
| PD 损失 | $\mathbb E\|\bar u^\theta_n(k\mid X_n)-u_k(\mathrm{sg}(\bar X_k))\|^2$ |
| 融合层 | $W^\theta_{n:n+L}=\sum_k\Delta_k W^\theta_k$ |
| CFG 引导速度 | $v_t^w=v_t+w(v_t(\cdot\mid c)-v_t)$ |

---

## 8. 深度讨论 (Q&A)

### Q1：用一个例子说明训练与推理的完整流程

**配置**：Wan2.1 14B, Midpoint, $N=128$, $L=32$ → NFE=4（论文实际配置）。Grid 上 128 个时间点（shift 变换），student 与 teacher 共享 DiT backbone，student 末层参数复制 128 份 $W_0^\theta,...,W_{127}^\theta$.

**训练（data-free PD loss，Algorithm 2，单步）**：
1. 输入 $X_n$（首轮 $X_0\sim\mathcal N(0,I)$）。
2. **Student 一次前向**：backbone 出 hidden $H^\theta_{t_n}(X_n)$，128 个头并行输出 128 个速度预测 $\bar u^\theta_n(0|X_n),...,\bar u^\theta_n(127|X_n)$。
3. **块内采样 k**（如 k=n+15）：用前 15 个学生预测累积推进 $\bar X_{n+15}=X_n+\sum_{i=n}^{n+14}(t_{i+1}-t_i)\bar u^\theta_n(i|X_n)$。
4. **Teacher Midpoint** 在 $\bar X_{n+15}$ 处估计该子区间平均速度 $u_{n+15}$（2 次 teacher 前向）。
5. **MSE loss** $=\|\bar u^\theta_n(n+15|X_n)-u_{n+15}\|^2$，反传更新 student（第 n+15 个头 + backbone）。
6. **推进状态**：$X_n\leftarrow X_n+\sum_{i=n}^{n+31}(t_{i+1}-t_i)\bar u^\theta_n(i|X_n)$（stop-gradient），$n\leftarrow n+32$，进入下一轮。

**推理（Algorithm 1）**：
1. $X_0\sim\mathcal N(0,I)$。
2. **for** n=0, 32, 64, 96:
   - Student 前向一次，取头 $\bar u^\theta_n(n..n+31|X_n)$。
   - 块步更新 $X_{n+32}=X_n+\sum_{k=n}^{n+31}(t_{k+1}-t_k)\bar u^\theta_n(k|X_n)$。
3. 共 **4 次前向 → NFE=4**，得到 $X_{128}$。

**部署优化（层融合）**：把每块的 32 个头融合成 1 个线性层 $W^\theta_{n:n+32}=\sum_{k=n}^{n+31}\Delta_k W^\theta_k$，推理时无额外计算/显存开销。

### Q2："训练时一次前向，多头生成多次 timestamp decode 结果，再和 teacher 实际 decode 结果逼近" —— 这个理解对吗？

**大方向对，需三点澄清**：

1. **头输出的是"速度"而非"decode 后样本"**：每个头预测该子区间的**平均速度** $u_k$，不是直接生成样本 $X_k$。样本通过"学生头输出速度 + 累积求和"间接得到。

2. **监督目标是"teacher 一步 RK 估计"而非"teacher 实际 decode 结果"**：teacher 不真的从 $X_n$ 多步 decode 到 $X_k$ 再对比，而是只在 $\bar X_k$（学生推进到的中间态）处用**一次 Runge-Kutta 步**（Euler 1 次 / Midpoint 2 次 teacher 前向）估计该子区间的平均速度。这是 PDD 训练成本低的关键——teacher 评估次数 O(1)，不随 $L$ 增长。

3. **每次 loss 只监督一个随机采样的 k**（不是 128 个头一起监督）：单次 loss 只对第 k 个头生效，但 backbone 通过第 k 个头间接得到监督，不同 k 在不同 batch 上轮转，**期望意义下 backbone 学到所有子区间的表示**。

**精妙之处**：结构上并行头实现一次前向多步预测；监督上用 teacher 一步 RK 估计避免多步 rollout；优化上 on-policy + 单一 MSE 稳定高效。

### Q3：加速方法分类的理解 + PDD 如何用纯轨迹蒸馏达到 SOTA

**分类理解纠正**：两类**目标都是减少采样步数**，区别在**约束形式**：

| | 轨迹类 | 分布类 |
|---|---|---|
| **约束** | student 必须沿 teacher 轨迹走 | 只对齐边缘分布 |
| **训练信号** | 轨迹上每点明确目标 | 分布层面隐式对齐（score/GAN） |
| **典型方法** | Progressive Distill, Consistency, Mean Flow, Pi-Flow, **PDD** | ADD/LADD, DMD/DMD2, VSD, rCM, AnyFlow |
| **优点** | 训练稳定、保留多样性 | 灵活、质量高、易扩展大模型 |
| **缺点** | 大模型质量退化、依赖 JVP | 模式崩溃、多样性损失、超参敏感 |

- "分布类压缩 student model"——**不准确**，DMD2 的 student 与 teacher 同 backbone，不压缩模型。
- "轨迹类减少步数"——对，但两类都在减步数，**真正区别是约束强度**：轨迹类强约束（走轨迹），分布类弱约束（分布对即可）。

**PDD 如何用纯轨迹蒸馏达到 SOTA**：PDD **没有"结合分布类优势"**，而是**用更聪明的轨迹监督解决了传统轨迹方法在大模型上的几个瓶颈**：

1. **解决 JVP/有限差分瓶颈**：传统轨迹方法（Mean Flow、Flow Map、AYF）需要 Jacobian-vector product 估计平均速度，大模型上昂贵且不稳定。PDD 用**并行头**把"学平均速度"转成"学可加权和的子速度分量"，**完全免 JVP**。

2. **解决多阶段训练瓶颈**：Progressive Distillation 需要多阶段（每阶段步数减半），PDD 单阶段训练，$L_{\min},L_{\max}$ 灵活，一模型支持多 NFE。

3. **解决质量退化瓶颈**：传统轨迹方法大模型上质量差，主因是监督稀疏 + JVP 不稳定。PDD 通过 **on-policy + 稠密子区间监督 + RK 估计**，让每个子区间都有明确 teacher 目标，监督信号比分布类还稠密（分布类只在分布层面隐式对齐）。

4. **解决模式崩溃瓶颈**（相对分布类的最大优势）：分布类通过 score 对齐易把多模态分布塌缩到主模态（DMD2 多样性 0.20→0.09）。PDD 学"如何走 teacher 轨迹"，teacher 轨迹本身含多样性（不同噪声→不同轨迹），student 自然继承。

5. **大规模可扩展**：data-free + 单一损失 + 共享 backbone + 无额外网络，直接扩展到 22B（LTX-2.3），无需数据集、无需 fake/real 网络、无需交替训练。

**一句话总结**：PDD 不是"结合两类优势"，而是"用更聪明的轨迹监督（并行头 + RK 估计 + on-policy）让纯轨迹方法在大模型上首次达到/超过分布类 SOTA，同时保留轨迹方法固有的多样性优势"。

### Q4："前 15 个学生预测累积推进"公式的意义

公式 $\bar X_{n+15}=X_n+\sum_{i=n}^{n+14}(t_{i+1}-t_i)\bar u^\theta_n(i|X_n)$ 的物理意义就是 **"速度 × 时间 = 位移"的累加**。流过程 ODE $\frac{d}{dt}X_t=v_t(X_t)$ 的精确解离散形式 $X_{t_{k+1}}=X_{t_k}+(t_{k+1}-t_k)u_k$，从 $X_n$ 推进 15 个子区间就是把 15 段位移累加：

$$\bar X_{n+15} = X_n + \underbrace{(t_{n+1}-t_n)u_n}_{\text{第0段位移}} + \underbrace{(t_{n+2}-t_{n+1})u_{n+1}}_{\text{第1段位移}} + \dots + \underbrace{(t_{n+15}-t_{n+14})u_{n+14}}_{\text{第14段位移}}$$

**关键澄清**：这里的 $u_i$ 由 student 的**第 i 个头**预测 $\bar u^\theta_n(i|X_n)$ 给出，且**所有 15 个头共享同一个输入 $X_n$**（不是递推！）。这是 PDD 与传统多步 rollout 的本质区别——一次前向并行预测，而非序列化逐步推进。

### Q5："Teacher Midpoint"是什么？为什么两次前向？

**Teacher**：预训练好的、待蒸馏的大模型（如 Wan2.1 14B 原始模型）。训练中 **teacher 参数冻结**，只为 student 提供监督目标。

**Midpoint 方法**（二阶 Runge-Kutta）：估计子区间 $[t_k,t_{k+1}]$ 的平均速度 $u_k$：
$$u_k \approx v_{t_{mid}}(X_{mid}),\quad t_{mid}=\frac{t_k+t_{k+1}}{2},\quad X_{mid}=X_k+\frac{t_{k+1}-t_k}{2}v_{t_k}(X_k)$$

**两次 teacher 前向**：
1. **第一次**：在 $X_k$、时间 $t_k$ 评估 $v_{t_k}(X_k)$ → 算中点状态 $X_{mid}$
2. **第二次**：在 $X_{mid}$、时间 $t_{mid}$ 评估 $v_{t_{mid}}(X_{mid})$ → 这就是 Midpoint 估计的 $u_k$

**对比 Euler**：Euler 只用一次前向 $u_k\approx v_{t_k}(X_k)$（把起点速度当整段平均），一阶精度；Midpoint 二阶精度，更准确。论文实验验证 **Midpoint 普遍优于 Euler**（Table 2-4）。

**在 PD loss 中的位置**：student 一次前向 → 采样 k → student 累积推进到 $\bar X_k$ → **teacher 在 $\bar X_k$ 处用 Midpoint 估计 $u_k$** → 作为 student 第 k 个头的监督目标。

### Q6：训练过程理解确认（"和 teacher 真实结果对比"对吗？）

理解"单次推理得 student 所有输出 → 累加得 student 解码结果 → 和 teacher 真实结果对比"——**基本正确，第三步需关键澄清**：

1. ✅ 单次前向得 128 个头输出（速度）
2. ✅ 累加得 student 推进状态 $\bar X_k$（累加用 student 自己预测的速度）
3. ⚠️ 对比的**不是**"teacher 多步 decode 后的样本 $X_k$"，而是"teacher **在 $\bar X_k$ 处用一步 Midpoint 估计的平均速度** $u_k$"

**关键区别**：
- 若用 teacher 真实多步 decode 结果对比，teacher 要从 $X_n$ 走 15 步到 $X_{n+15}$，成本随 k 线性增长 ❌
- PDD 做法：student 推进到 $\bar X_{n+15}$，teacher **只在这一点的 Midpoint 估计**（2 次前向），得到"teacher 认为该子区间应有的平均速度" ✅

**为什么合理**：流过程由初值决定，理想情况下 $\bar X_k$（student 推进）= $X_k$（teacher 轨迹），此时 student 第 k 个头预测的平均速度应等于 teacher 在 $X_k$ 处的平均速度。这是 **on-policy** 的体现——student 在自己推进的 $\bar X_k$ 上接受监督，避免 distribution shift。

### Q7："沿轨迹走" vs "对齐边缘分布"的本质区别

**核心区别：约束对象是"单个样本路径"还是"样本集合的分布"**

- **轨迹（trajectory）**：从 $X_0$ 到 $X_1$ 的**一条具体路径** $(X_0, X_{t_1}, \dots, X_1)$，是**单个样本**的演化
- **边缘分布（marginal distribution）**：$p_t(X)$，时间 $t$ 时状态 $X$ 的**分布**，是**样本集合**的统计性质

| | 轨迹类约束（强） | 分布类约束（弱） |
|---|---|---|
| **约束对象** | 同一 $X_0$ 下 $\bar X_t = X_t$（逐点对齐） | $p_t^{student} = p_t^{teacher}$（分布一致） |
| **是否限定同一 $X_0$** | 是 | 否 |
| **多样性保证** | 强（每个 $X_0$ 必须对应特定 $X_1$） | 弱（可塌缩到主模态） |

**举例**：teacher 用 50 步从噪声 $X_0$ 生成猫图 $X_1$。
- **轨迹类**：student 用 4 步从**同一 $X_0$** 生成**同一张猫图** $X_1$，且中间状态 $\bar X_{0.25}\approx X_{0.25}$
- **分布类**：student 用 4 步从**同一 $X_0$** 生成**一张猫图** $X_1'$，$X_1'$ 可以和 $X_1$ 不同（甚至不同的猫），只要 student 在大量样本上的**分布**与 teacher 一致

**回答具体疑问**："让 student 第 1 次输出和 teacher 第 15 次输出一致"——
- 若来自**同一 $X_0$** 且要求逐样本对齐 → **轨迹约束**
- 若只要求**分布**一致（不限定同一 $X_0$）→ **分布约束**

分布约束不关心"哪个噪声对应哪个输出"，只关心"输出的整体统计性质"。

**为什么分布类容易模式崩溃**：分布类只要求分布一致，student 可以"塌缩到 teacher 分布的主模态"满足约束——比如无论输入什么噪声都生成最常见的猫。分布上看似一致（都是猫图），但多样性丧失。轨迹类要求逐点对齐，student 无法塌缩（每个 $X_0$ 必须对应特定 $X_1$）。

**PDD 属于轨迹类**：同一 $X_n$ 下，student 第 k 个头预测的平均速度对齐 teacher 在 $\bar X_k$ 处的平均速度——逐点（轨迹）对齐，故天然保留多样性。

### Q8：为什么 teacher 在 student 预估的 $\bar X_k$ 上做 Midpoint 估计是"teacher 认为该子区间应有的平均速度"？如果 teacher 只做一次正常采样呢？

**Teacher 在状态 x、时间 $t_k$ 处做 Midpoint 估计的语义**：得到"如果当前状态是 x、时间到了 $t_k$，那么从 $t_k$ 到 $t_{k+1}$ 这个子区间，teacher 模型认为的平均速度是多少"——这是一个**条件平均速度**（以状态 x 为条件）。

**为什么在 student 推进的 $\bar X_k$ 上估计，而非 teacher 真实轨迹的 $X_k$ 上？** 这是 **on-policy** 的关键：

- **若 student 已学好**（$\bar X_k \approx X_k$）：teacher 在 $\bar X_k$ 处的估计 \approx teacher 在真实轨迹 $X_k$ 处的平均速度 = 正确监督信号 ✅
- **若 student 还没学好**（$\bar X_k$ 偏离 $X_k$）：teacher 在 $\bar X_k$ 处给出"在你当前到达的这个状态下，你应该怎么走"的指导——避免了 student 永远追不上 teacher 的 **distribution shift** 问题

**Proposition 1 的自洽保证**：PD loss 的全局极小满足 $\bar X_k = X_k$，此时监督信号恰好等于真实轨迹的平均速度。所以目标是**自洽的**——当 student 学好时，监督自动变为正确。

**"如果 teacher 只做一次正常采样呢？"** 两种理解：

- **理解 A（Euler 一步，1 次前向）**：$u_k \approx v_{t_k}(X_k)$，把起点瞬时速度当整段平均，一阶精度，偏差大——这就是 PDD 的 Euler 版本，能用但不如 Midpoint
- **理解 B（teacher 多步真实采样到 $X_{n+k}$）**：成本随 k **线性增长** ❌（对 14B/22B 模型不可行）；且给出的是"整段 $[t_n,t_{n+k}]$ 的平均速度"，**不是子区间 $[t_k,t_{k+1}]$ 的平均速度**，监督粒度不对 ❌

| 方案 | teacher 前向次数 | 精度 | 成本随 k |
|---|---|---|---|
| Euler 一步 | 1 | 一阶 | O(1) |
| Midpoint 一步 | 2 | 二阶 | O(1) |
| Teacher 多步真实采样 | k | "精确" | O(k) ❌ |

**PDD 选 Midpoint 的妙处**：成本固定 O(1) 不随块大小增长 + 二阶精度 + on-policy 自洽——这是大模型可扩展的关键。
