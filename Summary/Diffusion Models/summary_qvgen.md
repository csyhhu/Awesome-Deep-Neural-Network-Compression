# QVGen: Pushing the Limit of Quantized Video Generative Models

- **论文链接**: [arXiv:2505.11497](https://arxiv.org/abs/2505.11497)
- **作者**: Yushi Huang (HKUST), Ruihao Gong (Beihang), Jing Liu (Monash), Yifu Ding (Beihang), Chengtao Lv (NTU/SenseTime), Haotong Qin (ETH Zürich), Jun Zhang (HKUST)
- **会议**: ICLR 2026

---

## 1. 研究动机

视频扩散模型（Video Diffusion Models）如 CogVideoX、Wan 等在视频生成方面取得了显著突破，但其巨大的计算和内存开销严重限制了实际部署 —— 例如 Wan-14B 在单张 H100 上生成 10 秒 720p 视频需要超过 30 分钟和 50GB 显存。

**量化是自然的解决方案**，但存在以下挑战：
- 现有的图像 DM 量化方法（Q-DM、EfficientDM、LSQ 等）直接应用于视频 DM 时效果极差，在 4-bit 下视频质量严重退化
- 视频生成引入了复杂的时序建模，使得量化比图像生成更加困难
- **首次探索**视频 DM 的 QAT（Quantization-Aware Training）

---

## 2. 核心理论洞察：梯度范数是收敛关键

作者首先从理论上分析了 QAT 的收敛性：

**定理 1（Regret 上界）**：在凸假设下，QAT 的平均 regret 满足：
$$\frac{R(T)}{T} \leq \frac{dD_{\infty}^2}{2T\eta_T^m} + \frac{1}{T}\sum_{t=1}^T\frac{\eta_t^M}{2}\|\mathbf{g}_t\|_2^2$$

当训练步数 $T$ 足够大时，第一项可忽略，**降低梯度范数 $\|\mathbf{g}_t\|_2$ 是提升 QAT 收敛性的关键**。

实验验证：与 Q-DM 相比，QVGen 在所有模型上展现出**持续更低的 $\|\mathbf{g}_t\|_2$ 和更低的训练 loss**，证实了理论分析。

---

## 3. 方法

QVGen 包含两个核心组件：

### 3.1 辅助模块 $\Phi$：降低梯度范数

在每个量化线性层后引入可学习的辅助模块 $\Phi$（本质是一个**线性层**，无激活函数）：

$$\hat{\mathbf{Y}} = \mathcal{Q}_b(\mathbf{W}) \mathcal{Q}_b(\mathbf{X}) + \Phi(\mathcal{Q}_b(\mathbf{X}))$$

其中 $\Phi(\mathcal{Q}_b(\mathbf{X})) = \mathbf{W}_{\Phi} \mathcal{Q}_b(\mathbf{X})$，即一个简单的矩阵乘法。

- $\mathbf{W}_{\Phi}$ 使用**权重量化误差** $\mathbf{W} - \mathcal{Q}_b(\mathbf{W})$ 初始化
- $\Phi$ 通过补偿量化误差来稳定训练，降低 $\|\mathbf{g}_t\|_2$
- 对多种初始化策略不敏感（权重误差、层级训练、零初始化均表现相近）

### 3.2 Rank-Decay 策略：渐进消除 $\Phi$

$\Phi$ 在推理时会引入额外开销（全精度矩阵乘法 + 额外存储），为此需要将其消除。直接衰减所有参数效果差（$\lambda=1$ 导致严重性能下降）。

**关键观察**：随着 QAT 进行，$\mathbf{W}_{\Phi}$ 中**小奇异值的比例不断增加**（从初始 73% 增长到 99%），意味着越来越多成分贡献微弱、可以安全移除。这表明主权重 $\mathbf{W}$ 在逐渐吸收量化误差，$\Phi$ 的有效秩自然坍缩。

#### 3.2.1 SVD 分解与低秩重构

每轮衰减开始时，对**当前最新的** $\mathbf{W}_{\Phi}$ 做 SVD：

$$\mathbf{W}_{\Phi} = \sum_{s=1}^{d} \sigma_s \mathbf{u}_s \mathbf{v}_s^\top \quad (\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_d)$$

取前 $r$ 个最大奇异值分量，构造低秩矩阵（默认 $r=32 \ll d$）：

$$\mathbf{L} = \big[\sqrt{\sigma_1}\mathbf{u}_1, \ldots, \sqrt{\sigma_r}\mathbf{u}_r\big] \in \mathbb{R}^{n \times r}$$

$$\mathbf{R} = \big[\sqrt{\sigma_1}\mathbf{v}_1, \ldots, \sqrt{\sigma_r}\mathbf{v}_r\big]^\top \in \mathbb{R}^{r \times m}$$

则 $\mathbf{W}_{\Phi} \approx \mathbf{L}\mathbf{R}$，$\Phi$ 的前向计算变为 $\Phi(\mathcal{Q}_b(\mathbf{X})) = \mathbf{L}\mathbf{R}\mathcal{Q}_b(\mathbf{X})$。将 $\sqrt{\sigma_s}$ 各分一半给 $\mathbf{L}$ 和 $\mathbf{R}$，使二者大小均衡。

**参数量优势**：以 $\mathbf{W}_\Phi \in \mathbb{R}^{4096 \times 4096}$ 为例，全秩训练需要 1680 万参数，而 $\mathbf{L}\mathbf{R}$ 形式仅需 $(4096+4096) \times 32 \approx 26$ 万参数，约 **64 倍**差距，且天然限制了 $\Phi$ 的表达能力，防止过度补偿后难以消除。

#### 3.2.2 乘性衰减系数 $\gamma$

在前向传播中对 $\mathbf{L}$ 施加逐元素衰减：

$$\hat{\mathbf{Y}} = \mathcal{Q}_b(\mathbf{W})\mathcal{Q}_b(\mathbf{X}) + \underbrace{(\gamma \odot \mathbf{L})\ \mathbf{R}\ \mathcal{Q}_b(\mathbf{X})}_{\Phi \text{ 模块（含衰减）}}$$

其中 $\gamma$ 的构造为：

$$\gamma = \mathrm{concat}\big(\underbrace{[1]_{n \times (1-\lambda)r}}_{\text{前 }(1-\lambda)r\text{ 列：恒为 }1},\ \underbrace{[u]_{n \times \lambda r}}_{\text{后 }\lambda r\text{ 列：衰减}}\big) \in \mathbb{R}^{n \times r}$$

- $\lambda = 1/2$：每次衰减一半的列
- $u$ 按余弦退火从 $1 \to 0$（固定调度，**不可学习**）
- $\mathbf{L}$ 和 $\mathbf{R}$ 正常参与梯度更新，$\gamma$ 仅作为门控系数

**$\gamma$ 的作用机制**：

```
L (n × 32):           γ (n × 32):              γ ⊙ L:
┌────────┬────────┐   ┌────────┬────────┐   ┌────────┬──────────┐
│ 列1~16 │列17~32 │   │   1    │ u·1→0  │   │  不变   │ 逐步→0   │
│ 大σ    │ 小σ    │   │ (保留) │ (衰减) │ = │ (保留)  │ (被抑制)  │
└────────┴────────┘   └────────┴────────┘   └────────┴──────────┘
```

反向传播时，后 $\lambda r$ 列的梯度被 $u$ 缩放，$u \to 0$ 时梯度也 $\to 0$，这些列**先被"冻结"再被截断**，整个过程平滑无突变。

**注意**：这不是经典的核范数（Nuclear Norm）正则化（即在 loss 中加 $\|\mathbf{W}_\Phi\|_*$ 惩罚项），而是一种**直接作用于参数分解的 structured multiplicative decay**——利用 SVD 天然的大小排序，精确掐掉信息量最小的方向。

#### 3.2.3 完整衰减调度（多轮迭代 SVD）

**重点**：不是做一次 SVD 后就一直训练，而是**每轮衰减前都重新做 SVD**。时序如下：

```
      SVD①               SVD②               SVD③
        ↓                   ↓                   ↓
  ┌──────────┐        ┌──────────┐        ┌──────────┐
  │ L,R(32)  │ 截断   │ L,R(16)  │ 截断   │ L,R(8)   │ 截断
  │ +γ 训练  │ ───▶  │ +γ 训练  │ ───▶  │ +γ 训练  │ ───▶  ...
  └──────────┘        └──────────┘        └──────────┘
       ↑                   ↑                   ↑
   基于原始 W_Φ       基于截断后的 W_Φ     基于再次截断后的 W_Φ
```

每轮流程（以 $r=32 \to 16$ 为例）：

1. **重新 SVD**：对当前 $\mathbf{W}_\Phi$ 做 SVD，取前 $r$ 个分量构造 $\mathbf{L},\mathbf{R}$（此时正交）
2. **构造 $\gamma$**：前 $(1-\lambda)r$ 列 $\times 1$，后 $\lambda r$ 列 $\times u$
3. **训练**：`it_per_decay_phase` 轮，梯度更新 $\mathbf{L},\mathbf{R}$（正交性被打破，但不影响衰减）
4. **截断**：$u=0$ 后截断为 $\mathbf{L}' = \mathbf{L}[:,\ :(1-\lambda)r]$，$\mathbf{R}' = \mathbf{R}[:(1-\lambda)r,\ :]$
5. **重构**：$\mathbf{W}_\Phi = \mathbf{L}'\mathbf{R}'$，$r \leftarrow (1-\lambda)r = r/2$

**完整 r 衰减路径**（$\lambda=1/2$，初始 $r=32$）：

$$32 \xrightarrow{\text{SVD+训练+截断}} 16 \xrightarrow{\text{SVD+训练+截断}} 8 \xrightarrow{\text{SVD+训练+截断}} 4 \xrightarrow{\text{SVD+训练+截断}} 2$$

退出 while 循环（条件 $r > 1/\lambda = 2$）后，最后阶段 $\gamma = [u]_{n \times r}$（全部列衰减），训练至 $u=0$ 后 $\mathbf{W}_\Phi \to \varnothing$，$\Phi$ 完全消失。

**为什么必须多次 SVD？** 每轮训练改变了 $\mathbf{W}_\Phi$ 的参数值，奇异值分布已完全不同。只做一次 SVD，后续截断依据的是过时的排序，可能砍错方向。重新 SVD 保证每次衰减都对准**当前真正最弱**的成分。

#### 3.2.4 平滑衰减 vs 硬截断

直接 SVD 截断（即 $\lambda=1$，一次砍到底）会在截断点产生性能断崖：

```
硬截断: 训练 W_Φ(rank=32) ──▶ 突然砍到 rank=16 ──▶ 性能骤降 ──▶ 重新训练恢复
平滑:   u=1 → 0.5 → 0，后16列逐步退场，前16列同步接管其职能
```

消融实验证实：$\lambda=1$（硬截断）性能大幅下降，$\lambda=1/2$（平滑衰减）性能损失 < 0.6%。

#### 3.2.5 设计要点汇总

- **关键设计**：只衰减与小奇异值关联的成分（低贡献），保留高贡献成分继续训练
- **$\mathbf{L},\mathbf{R}$ 不需保持正交**：SVD 仅在每轮开始瞬间用于定位弱方向，训练中正交性自然打破，不影响衰减
- **$\gamma$ 无可训练参数**：纯粹的门控系数，不参与梯度更新
- 比直接衰减（线性）、稀疏化（Sparse）、残差量化（Res. Q.）等策略**速度更快（1.8×-2.6× GPU 天）且性能显著更优**

---

## 4. 实验设置

| 项目 | 详情 |
|------|------|
| **模型** | CogVideoX-2B、1.5-5B；Wan 1.3B、14B |
| **量化配置** | W4A4、W3A3（per-channel 权重 + per-token 激活） |
| **基线** | PTQ：ViDiT-Q、SVDQuant；QAT：LSQ、Q-DM、EfficientDM |
| **训练数据** | OpenVidHQ-4M 中 16K 带字幕视频 |
| **评估** | VBench（8 维度）、VBench-2.0（物理/推理） |
| **训练资源** | 8×~128× H100 GPU |

---

## 5. 主要结果

### 5.1 W4A4 量化（核心结果）

| 模型 | 指标 | QVGen vs Q-DM / SVDQuant |
|------|------|--------------------------|
| CogVideoX-2B | VBench Total | **80.62%** vs Q-DM 78.44%（与 BF16 80.95% 几乎持平） |
| Wan 1.3B | Aesthetic Quality | 相比 W4A4 SVDQuant 提升 **+8.37** |
| Wan 1.3B | Subject Consistency | 相比 W4A4 SVDQuant 提升 **+14.61** |
| CogVideoX1.5-5B | VBench | **首个**在大模型上实现 W4A4 FP-comparable 性能 |
| Wan 14B | VBench-2.0 | W4A4 总体得分仅下降 **~1%** |

**首次在 4-bit 设置下达到全精度可比质量。**

### 5.2 W3A3 量化

- CogVideoX-2B W3A3：动态程度（Dynamic Degree）+25.28，场景一致性（Scene Consistency）+8.43
- 极端压缩下仍显著优于此前所有方法

### 5.3 推理效率

| 场景 | 加速比 |
|------|--------|
| Wan 1.3B (A800) | **1.21×**（vs BF16） |
| Wan 14B (A800) | **1.44×**（vs BF16） |
| +SVG 稀疏注意力 | **1.70× / 2.63×**（1.3B / 14B） |
| 内存节省 | **~4×**（vs BF16） |

QVGen 生成标准均匀量化模型，可直接适配现有 W4A4 推理 Kernel。

### 5.4 训练开销

QVGen 相比 KD-based QAT（Q-DM）仅增加 **~1.02× GPU 天**和 **~1.01× 峰值显存**。

---

## 6. 消融实验

| 消融项 | 结论 |
|--------|------|
| **$\Phi$ 模块** | 核心贡献，所有指标大幅提升 |
| **Rank-Decay** | 消除 $\Phi$ 的推理开销，性能损失 <0.6%，部分指标甚至提升 |
| **收缩比 $\lambda$** | $\lambda=1/2$ 最优；$\lambda=1$（一次性衰减）性能大幅下降 |
| **初始秩 $r$** | $r=32$ 最优；$r=64$ 因每个阶段训练时间不足反而变差 |
| **衰减策略对比** | Rank > Sparse w/ MaskLLM > Sparse w/ Wanda > Sparse > Res. Q. > Linear |
| **退火函数** | Cosine、Linear、Cubic 等 5 种函数结果接近，方法鲁棒 |
| **$\gamma$ 方向** | 衰减小奇异值成分 > 衰减大奇异值成分 > 随机衰减 |
| **梯度裁剪** | 适度裁剪（0.5）有益，过度裁剪（0.1）有害 |

---

## 7. 关键创新点总结

1. **首次系统研究视频 DM 的 QAT**：揭示了梯度范数 $\|\mathbf{g}_t\|_2$ 是收敛瓶颈
2. **辅助模块 $\Phi$**：通过补偿量化误差降低梯度范数，实现稳定 QAT
3. **Rank-Decay 策略**：利用 SVD 识别低贡献成分，通过秩正则化渐进消除 $\Phi$，实现零推理开销
4. **理论+实证双重验证**：凸和非凸条件下都证明降低梯度范数对 QAT 收敛的关键作用
5. **端到端可部署**：最终产出标准 W4A4/W3A3 量化模型，兼容现有推理 Kernel

---

## 8. 潜在局限

- 训练数据依赖（仅使用 OpenVidHQ 的 16K 视频，更多数据可能带来提升）
- 3-bit 下大模型（14B）的场景一致性等指标仍有明显下降
- 加速比（1.21×-1.44×）相比理论极限仍有提升空间（可通过 kernel fusion 和形状优化改进）
- 目前仅在 DiT 架构的视频生成中验证，推广到更多任务（NLP 等）待探索
