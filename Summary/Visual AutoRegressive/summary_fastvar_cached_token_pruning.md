# FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning

> **Paper**: [arXiv:2503.23367](https://arxiv.org/abs/2503.23367)
> **Authors**: Hang Guo, Yawei Li, Taolin Zhang, Jiangshan Wang, Tao Dai, Shu-Tao Xia, Luca Benini
> **Affiliations**: Tsinghua University, ETH Zürich, Shenzhen University, Peng Cheng Laboratory
> **Venue**: ICCV 2025
> **Code**: https://github.com/csguoh/FastVAR
> **Tag**: `fastvar_cached_token_pruning`

---

## 1. 核心问题

视觉自回归建模（Visual Autoregressive, VAR）将传统 AR 的 **"next-token prediction"** 范式转换为 **"next-scale prediction"**：每个自回归单元是一个多尺度 token map，而非单个 token。这种方式大幅减少了生成步数，但带来一个新的瓶颈——**计算复杂度和延迟随分辨率急剧放大**。

- 第 $k$ 步需一次性处理整个 $h_k \times w_k$ token map，token 数随分辨率 $n\times n$ 按 $\mathcal{O}(n^2)$ 增长，注意力层复杂度达到 $\mathcal{O}(n^4)$。
- 即使启用 FlashAttention，VAR 的推理延迟仍是超线性的；最后两步甚至占据总运行时间的 60%。
- 这导致现有 VAR 模型（HART、Infinity）无法扩展到 2K 等更高分辨率。

## 2. 核心思想

FastVAR 是一个 **post-training、training-free、plug-and-play** 的加速方法，核心是 **"cached token pruning"（缓存式 token 剪枝）**：在大尺度步只前传少量关键 token，并用早期尺度步的缓存 token 复原被剪枝的位置，从而把 VAR 的复杂度近似为线性。

该方法基于对预训练 VAR 模型的三个关键观察：

1. **大尺度步是速度瓶颈但鲁棒**：最后几步占主要耗时，但对 token 丢弃的鲁棒性高于小尺度步，因此适合剪枝。
2. **大尺度步主要建模高频信息**：生成过程可分解为两阶段——
   - **Structure Construction Stage（小尺度步）**：生成主体轮廓（低频）；
   - **Texture Filling Stage（大尺度步）**：基于前一阶段草图填充细节纹理（高频），此时低频 token 几乎已收敛。因此可剪掉冗余的低频 token，只前传关键的高频 token。
3. **跨尺度 token 强相关**：注意力图呈现对角稀疏性，即当前尺度的一个 token 不仅关注同尺度邻居，也与前一尺度对应位置的 token 强相关。因此可用前序尺度步的缓存 token 估计被剪枝位置的输出。

## 3. 方法详解

### 3.1 VAR 预备与两阶段划分

VAR 将图像特征量化为 $K$ 个多尺度 token map $\mathcal{R}=\{r_1,\dots,r_K\}$，用因果形式联合建模：
$$p(r_1,\dots,r_K)=\prod_{k=1}^{K}p(r_k\mid r_1,\dots,r_{k-1})$$

训练时采用残差策略降低学习难度：
$$\tilde{r}_k = \mathrm{interpolate}(\tilde{r}_{k-1},(h_k,w_k)) + f_k$$
展开为累加形式 $\tilde{r}_k = \sum_{i=1}^{k}\mathrm{interpolate}(f_i,(h_k,w_k))$。最后一步的预测用于生成最终图像；推理时用 KV-Cache 避免重复计算。

FastVAR 将 $K$ 步划分为两个集合：
- $\mathcal{S}=\{1,2,\dots,K-N\}$：Structure Construction Stage，保持标准 VAR 不变；并将第 $(K-N)$ 步的每层输出作为后续复原的缓存。
- $\mathcal{T}=\{K-N+1,\dots,K\}$：Texture Filling Stage，应用 FastVAR 剪枝 + 快速前传 + 复原。

### 3.2 Pivotal Token Selection (PTS)

如何识别"关键 token"是难点：常用频域算子（如 FFT）在频域操作，难以直接定位原 token map 中各 token 的频率特性。PTS 给出近似解——**直接在空间域用直流分量估计低频成分**：

1. 低频成分 $\bar{x}_k$ 用空间全局平均池化近似直流分量：
   $$\bar{x}_k = \mathrm{global\_avg\_pool}(x_k)$$
2. 高频成分为 $x_k - \bar{x}_k$，关键度分数取其 L2 范数：
   $$s_k = \|x_k - \bar{x}_k\|_2$$
3. 取 Top-K 得到索引集 $\mathcal{I}$，通过索引对 $x_k$ 剪枝进行快速前传。

附加收益：PTS 减少了 transformer 层的输入 token 数，**KV-cache 也相应缩减**，同时优化了 GPU 显存占用和后续尺度步的跨尺度注意力。

### 3.3 Cached Token Restoration (CTR)

剪枝后需还原原始 token 数以维持 2D 图像格结构。基于跨尺度对角稀疏性，CTR 用缓存步骤对应位置的 token 近似被剪位置的输出：

1. 取缓存 token map $y_{K-N}$（大小 $(h_{K-N}, w_{K-N})$），上采样到当前尺度：
   $$y^{cache}_k = \mathrm{interpolate}(y_{K-N}, (h_k,w_k))$$
2. 用索引集 $\mathcal{I}$ 将 $y^{cache}_k$ 散射到 $y_k$ 的被剪位置，得到与 $x_k$ 同规模的恢复输出 $y'_k$。

**为何选 $(K-N)$ 作为缓存步**：消融实验表明，缓存步越靠近被剪尺度步、与剪枝位置 gap 越小，效果越好；$(K-N)$ 是 $\mathcal{S}$ 的最后一个元素，最贴近 $\mathcal{T}$，因此最优。

### 3.4 整体实现与渐进式剪枝

- 由于 PTS+CTR 的 token 数不变性，将这对算子作用于 $\mathcal{T}$ 中每个 Attention & FFN 层。
- **渐进式剪枝比例调度**：大尺度步对剪枝更鲁棒，故给更大的尺度步分配更大的剪枝比例。
- FastVAR **不访问 attention map**，因此可与 FlashAttention 正交组合进一步加速。

### 3.5 PyTorch 伪代码

```python
def pivotal_token_selection(x, topk):
    # 直流分量近似低频
    pool_x = rearrange(x, 'b (h w) c -> b c h w')
    pool_x = adaptive_avg_pool2d(x, (1, 1))
    pool_x = rearrange(pool_x, 'b c 1 1 -> b 1 c')
    score = sum((x - pool_x)**2, dim=-1)          # 高频 L2 范数
    pivotal_idx = argsort(score, dim=1, descending=True)[:, :topk, :]
    return gather(x, dim=1, index=pivotal_idx)

def cached_token_restoration(x, cache):
    restored_x = interpolate(cache)              # 上采样缓存到当前尺度
    restored_x = rearrange(restored_x, 'b c h w -> b h w c')
    restored_x.scatter_(dim=1, index=pivotal_idx, src=x)  # 复原被剪位置
    return restored_x
```

## 4. 实验

### 4.1 设置

- **模型**：HART、Infinity（均可生成 1024×1024），超参数保持默认。
- **评测**：GenEval（高层语义一致性）、MJHQ30K（感知质量）；效率指标含运行时间、吞吐、加速比、显存。
- **剪枝配置**：Infinity 取 $N=4$，比例 $\{40\%,50\%,100\%,100\%\}$；HART 取 $N=2$，比例 $\{50\%,75\%\}$。100% 表示跳过该步、直接插值得到最终输出。
- **硬件**：单张 NVIDIA RTX 3090 (24GB)。除特别说明，baseline 均已用 FlashAttention 加速；FastVAR 在其之上叠加。推理速度不含 VAE（各方法共享）。

### 4.2 主要结果（1024×1024 GenEval）

| 方法 | Steps | Speedup | Latency | Memory | GenEval Overall |
|---|---|---|---|---|---|
| SDXL | 40 | – | 4.3s | – | 0.55 |
| SD3-medium | 28 | – | 4.4s | – | 0.62 |
| LlamaGen (AR) | 1024 | – | 37.7s | – | 0.32 |
| HART | 14 | 1.0× | 0.95s | – | 0.51 |
| **HART + FastVAR** | 14 | **1.5×** | **0.63s** | 14.7GB | **0.51** |
| Infinity | 13 | 1.0× | 2.61s | 16.1GB | 0.73 |
| **Infinity + FastVAR** | 13 | **2.7×** | **0.95s** | 11.9GB | **0.72** |

- 相比 LlamaGen，Infinity+FastVAR 达 **39.7×** 加速，GenEval 提升 125%。
- 相比 FlashAttention baseline 的 18.9GB，FastVAR **显存降 22.2%** 到 14.7GB。
- 性能损失 **< 1%**。

### 4.3 与 FlashAttention 正交组合

| Setup | Speedup | Latency | Memory |
|---|---|---|---|
| SlowAttn | – | – | OOM |
| FlashAttn only | 1.0× | 2.61s | 16.1GB |
| SlowAttn + FastVAR | 2.1× | 1.25s | 12.8GB |
| **FlashAttn + FastVAR** | **2.7×** | **0.95s** | **11.9GB** |

- FastVAR-only 相对 FlashAttn-only 有 **2.1×** 加速、**20.4%** 显存下降。
- 因 FastVAR 不读 attention map，能与 FlashAttention 无缝组合。

### 4.4 与 ToMe 对比

ToMe 把多个 token 合并成一个再 unmerge 还原 2D shape。在 VAR 上 ToMe 难以高加速：1.36× 加速就出现明显 FID 退化；而 FastVAR 能以 1.7× 加速获得更好 FID。

### 4.5 零样本扩展到更高分辨率

把 Infinity 的尺度调度追加额外步实现零样本高分辨率：
- 1344×1344：15GB / 1.3s（FlashAttn baseline 在 24GB 3090 上 OOM）。
- **2K 图像**：15GB 显存 / **1.5s**，单张 3090 即可生成。

### 4.6 消融实验

- **剪枝比例**：随比例增大运行时稳定下降，但过高会导致关键高频 token 丢失、纹理不连续。HART 上 40%~75% 是性能-效率的甜点。
- **尺度敏感性**：在小尺度步剪枝加速有限且明显掉点（破坏结构构造阶段，误差会向后传播放大）；在大尺度步剪枝加速显著且鲁棒。例如 48、64 尺度 50% 剪枝比 16、21 尺度 75% 剪枝更快且 FID 低 5.71。
- **剪枝槽可视化**：FastVAR 优先保留高频边缘/纹理槽（人眼、头发、嘴），剪掉已收敛的平坦区域（脸颊），且随比例提升保序——验证了频域关键 token 选择的合理性。

## 5. 补充材料要点

- **ImageNet 256×256 类条件生成**：FastVAR 在 VAR(d=24/d=30) 上仍有竞争力，但 256 图像最大尺度仅 16×16，小尺度剪枝鲁棒性弱于 1024 场景，留作未来工作。
- **更细粒度效率分析**：r=75% 时单层注意力 4.6×、FFN 3.8× 加速；PTS 仅 0.59ms、CTR 仅 0.24ms，合计 0.63ms，约为原注意力模块的 5%，远小于带来的加速收益。
- **更多基准**：HPSv2.1、ImageReward 上 FastVAR 均优于 ToMe 且更高效。
- **极端剪枝比例的适用性**：100% 比例对 Infinity (2B) 可行，但对 HART (700M) 严重退化。原因是大模型能力强、能在更早尺度步建模复杂纹理；小模型依赖 test-time scaling 用更长尺度步生成细节，极端剪枝会破坏。
- **缓存步消融**：$(K-N)$ 在所有指标上一致最优；缓存步越靠前，与被剪尺度步 gap 越大，效果越差。

## 6. 局限与未来工作

1. 当前聚焦大尺度步加速；小分辨率场景需更通用的剪枝策略以纳入小尺度 token map。
2. 当前用渐进式剪枝比例调度；未来可引入更细粒度先验（layer-wise、自适应比例）。
3. FastVAR 可与 FlashAttention 正交组合；进一步可与量化、更少解码步数等方法组合继续加速。

## 7. 个人总结 / 启示

- **频域先验的轻量空间域实现**：用 global avg pool 近似直流分量、用高频 L2 范数作为 token 重要性分数，绕开了 FFT 的频域-空域对齐难题，且对 attention map 无依赖——这是能与 FlashAttention 正交的关键。
- **缓存复原而非重计算**：利用跨尺度对角稀疏性，用前一阶段缓存 token 直接近似被剪位置输出，避免了"剪枝后必须重新生成"的开销，维持了 2D 格结构。
- **与 diffusion 加速思路的对照**：FastVAR 与 ToMeSD（token merging）、DeepCache（特征缓存复用）、TokenCache（跨步缓存）同属"剪枝/缓存"加速家族，但它是面向 next-scale 范式定制的；其"两阶段划分 + 大尺度步剪枝"的思路对其他多尺度生成模型也有借鉴价值。
- **实用价值**：在单张消费级 3090 上实现 2K 图像 1.5s 生成，把 VAR 从"算不动高分辨率"推向"可量产"。

## 8. 讨论与答疑

### Q1：Visual Autoregressive 和 Visual Generation 有什么区别？VAR 是不是"从低分辨率渲染到高分辨率"？

**概念关系：任务 vs 范式。**
Visual Generation 是一个**任务**（生成图像/视频），下面包含 GAN、VAE、Diffusion、next-token AR、next-scale AR 等多种范式。Visual Autoregressive (VAR) 是视觉生成下的一个**建模范式**（next-scale prediction），是 Visual Generation 的子集，而非并列关系。

**"从低分辨率渲染到高分辨率"的直觉基本正确，但需两点精细化：**

1. **VAR 工作在离散量化 token 空间，不是直接在像素/图像空间渲染。**
   多尺度 VQVAE 把图像量化成 $K$ 个分辨率递增的 token map $\mathcal{R}=\{r_1,\dots,r_K\}$，Transformer 逐尺度因果预测 $p(r_k\mid r_1,\dots,r_{k-1})$，**只有最后一个尺度的 token map $r_K$ 被 VAE decoder 解码成像素图像**。中间尺度不单独解码成图（论文 Figure motivation (a) 把 $\tilde{r}_k$ 解码出来只是可视化）。准确说法：VAR 在离散 token 空间里从粗尺度预测到细尺度，最后把最细尺度的 token map 解码成图——是"token map 尺度"上的从低到高，而非"图像分辨率渲染放大"。

2. **从粗到细靠残差累加实现，不是渲染。**
   $\tilde{r}_k = \mathrm{interpolate}(\tilde{r}_{k-1}) + f_k$，展开为 $\tilde{r}_k = \sum_{i=1}^{k}\mathrm{interpolate}(f_i,(h_k,w_k))$。每步加一个残差 $f_k$ 并提升分辨率，这正是论文两阶段划分的来源：小尺度步累加出低频结构（Structure Construction Stage），大尺度步叠加高频纹理（Texture Filling Stage）。

**与论文方法的直接关系（关键）：**
正是因为 VAR 是 coarse-to-fine 的残差累加，大尺度步的低频结构 token 已收敛、残差 $f_k$ 主要承载高频纹理。FastVAR 才敢在大尺度步：用 PTS 只前传高频关键 token（avg pool 估低频、高频 L2 范数排序）、把已收敛低频 token 剪掉；用 CTR 把被剪位置用 $(K-N)$ 步缓存 token 上采样填回（跨尺度对角稀疏性保证近似有效）。**VAR 的从粗到细特性，使"大尺度步只需补纹理、结构可从前序尺度步缓存复原"成立——这是 FastVAR 能剪枝+缓存复原而不掉点的根本原因。**

**与 Diffusion coarse-to-fine 的对比：**
扩散模型也可看作某种从粗到细（早期去噪恢复结构、晚期补细节），但机制不同——Diffusion 是**同一分辨率下沿时间步迭代去噪**，VAR 是**跨尺度、token map 分辨率递增地一次性生成**。所以 Diffusion 加速靠减步数/缓存相邻步特征（DeepCache、TokenCache），VAR 加速靠减大尺度步 token 数（FastVAR），加速对象完全不同，这也是 ToMeSD/DeepCache 这类方法不能直接搬到 VAR 上的原因。

### Q2：Visual Autoregressive 和 Diffusion Model 之间的关系是什么？

**定位：都是 Visual Generation 的子范式，但底层哲学不同。**
Diffusion 在**连续 latent 空间**做去噪 score/流匹配（连续 SDE/ODE）；VAR 在**离散量化 token 空间**做自回归似然的链式分解 $p(r_1\cdots r_K)=\prod p(r_k\mid\cdots)$。一个是连续数值 ODE 求解，一个是离散 GPT 式自回归。

**核心区别："步"指的不是一回事。**
- Diffusion 的 step = **时间步** $t$（噪声水平），所有步处理**同一分辨率**的 latent，是同一对象沿时间的连续细化；步数多（DDPM 1000 / SDXL ~20），单步代价均匀。
- VAR 的 step = **尺度步** $k$（token map 分辨率），每步处理**不同分辨率**的 token map，靠残差累加 $\tilde{r}_k=\sum_i\mathrm{interpolate}(f_i)$ 拼成；步数少（VAR 10 / HART 14 / Infinity 13），但单步代价随分辨率 $\mathcal{O}(n^4)$ 爆炸，最后两步占 60% 时延。
- 一句话：Diffusion 是"同分辨率、沿时间步迭代"，VAR 是"跨尺度、分辨率递增地一次性生成"——这正好接上 Q1 的"从低分辨率到高分辨率"直觉，对 VAR 成立、对 Diffusion 不成立。

**生成过程拓扑：**
```
Diffusion: x_T(噪声) ─同分辨率去噪─> x_0(干净 latent) ─VAE decode─> 图像
VAR:       r_1(文本) ─逐尺度新增 token map─> r_K(最细 token map) ─VAE decode─> 图像
```
Diffusion 链上每个节点分辨率相同、只是噪声水平变化；VAR 链上每个节点分辨率不同、逐级放大。这导致缓存逻辑根本不同：Diffusion 可缓存相邻步的**同尺寸**特征（DeepCache/TokenCache）；VAR 不能直接缓存相邻尺度步特征（token map 大小都不同），所以 FastVAR 选择缓存 $(K-N)$ 步后**上采样**到当前尺度来复原（CTR）。

**加速方向互补（呼应 FastVAR）：**
| | Diffusion 加速 | VAR 加速 |
|---|---|---|
| 主攻方向 | 减步数（distillation、consistency、DMD2、progressive） | 减大尺度步 token 数（FastVAR） |
| 缓存对象 | 相邻时间步的同尺寸特征 | 前序尺度步 token map（上采样复原） |
| 量化对象 | 连续 latent 量化（PTQD、QDiffusion） | Transformer 权重量化（token 本身已离散） |
| 经典方法 | DeepCache、ToMeSD、TokenCache、TCD | FastVAR、CoDe |
ToMeSD/DeepCache 搬不到 VAR 上的根本原因：Diffusion 没有"大尺度步"概念，VAR 没有"相邻同尺寸时间步"概念，加速对象错位。

**两者并非互斥，可融合：**
- **HART** 本身就用 continuous diffusion module 补偿 VQ 量化误差——主干 VAR（离散 token 自回归）、细节校正用 diffusion head。
- **MAR**（Li 2024）用自回归建模连续 token + diffusion head 生成。
- **Show-o、Emu3** 等试图统一理解与生成。
所以 VAR 与 Diffusion 是两种可组合的"积木"：VAR 提供 LLM 式自回归骨架与离散 token 建模，Diffusion 提供连续细化能力，HART 是典型拼法。

**生态类比（建立直觉）：**
- VAR ≈ 图像版 GPT：离散 token、自回归、有 scaling law、可复用 LLM 架构与训练 recipe、能用 KV-cache、支持 in-context。
- Diffusion ≈ 数值 ODE 求解器：连续去噪、依赖步数与 schedule、加速靠减步数/蒸馏，与 LLM 生态几乎无关。
这也解释了近期 AR/VAR 在视觉生成上重新崛起——它把视觉生成拉回 LLM 统一技术栈，而 Diffusion 是独立的连续建模路线。