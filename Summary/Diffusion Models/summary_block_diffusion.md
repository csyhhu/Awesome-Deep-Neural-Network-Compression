# Block Diffusion（BD3-LMs）论文总结

- **论文**: *Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models*  
- **arXiv**: [2503.09573](https://arxiv.org/abs/2503.09573)（ICLR 2025 Oral）

## 1. 想解决什么问题

作者指出离散扩散语言模型（discrete diffusion / masked diffusion）在文本生成上有三类关键限制：

- **固定长度生成**：很多离散扩散架构天然一次生成固定长度向量，不适合对话等“可变长度输出”。
- **KV cache 困难**：扩散常用双向上下文（或非因果注意力），难以像 AR 那样复用历史 KV，推理效率受限。
- **likelihood / perplexity 落后**：在标准语言建模指标上往往明显差于强 AR。

Block Diffusion 的目标是同时缓解这些问题：既保留扩散的并行潜力，又引入 AR 的“可变长度、可缓存、likelihood 更直接”的优势。

## 2. 核心建模：块级 AR + 块内扩散（semi-autoregressive）

把长度为 \(L\) 的序列分成 \(B=L/L'\) 个 block，每个 block 长度 \(L'\)。

- **块级因果分解（AR over blocks）**：

\[
\log p_\theta(x)=\sum_{b=1}^{B}\log p_\theta(x^b \mid x^{<b})
\]

- **块内条件分布用离散扩散建模**：对每个 \(p_\theta(x^b \mid x^{<b})\)，在 block 内跑一个离散扩散/去噪过程（常用 masked diffusion）。

直观理解：

- 当 \(L'=1\) 时，接近标准 AR（每次只生成 1 个 token）。
- 当 \(L'\) 很大时，更像“扩散一次并行生成一大段”，但块与块之间仍是因果串行。

这就是标题里“在 AR 和 diffusion 之间插值（interpolate）”的意思：用 block size 控制速度-质量/可控性的折中。

## 3. 模型与注意力：block-causal mask + 精确 KV cache

作者用一个共享参数的 transformer（同一网络）作为所有 block 的去噪器，并配 **block-causal attention mask**：

- 第 \(b\) 个 block 的 token 可以看见 **1..b** 的 blocks（前面的块作为条件）。
- 从而可以像 AR 一样缓存前面 blocks 的 KV。

文中把模型写成会输出 logits + KV：

\[
\text{logits}^b,\;K^b,\;V^b \leftarrow f_\theta(x_t^b,\;K^{1:b-1},V^{1:b-1})
\]

## 4. 训练目标：把 block diffusion 的 NELBO 加总

对每个条件项 \(\log p_\theta(x^b \mid x^{<b})\) 套用离散扩散的 NELBO（负 ELBO），得到整体训练目标（一个上界）：

\[
 -\log p_\theta(x)\;\le\;\mathcal{L}_{BD}(x;\theta)=\sum_{b=1}^B \mathcal{L}(x^b, x^{<b};\theta)
\]

论文也讨论 masked diffusion 的简化目标（基于 MDLM / simplified diffusion objective），形式上是对不同噪声水平 \(t\) 的期望，加上一个权重 \(\frac{\alpha_t'}{1-\alpha_t}\)：

\[
\mathcal{L}_{BD}(x;\theta)=\sum_{b=1}^B \mathbb{E}_{t}\mathbb{E}_{q}\;\frac{\alpha_t'}{1-\alpha_t}\log p_\theta(x^b \mid x_t^b, x^{<b})
\]

要点：块内是扩散去噪，块间是因果条件。

## 5. 高效训练：为什么需要“专门的训练算法/向量化”

难点在于：要去噪第 \(b\) 个 block，需要前面 blocks 的“干净”上下文；但训练里每个 block 又会被加噪，所以朴素实现会变成 **按 block 循环多次 forward**，很慢。

作者给了两种思路：

- **两次 forward 的最小方案**：先对整段干净序列做一次 forward，预计算所有 blocks 的 KV；再对每个 block 的 noisy 输入做去噪预测（每个 token 过网络两次）。
- **向量化/单次 forward 近似实现**：把 “noisy blocks 拼起来” 与 “clean sequence” 拼成一个更长输入 \(x_\text{noisy}\oplus x\)，再设计特殊 attention mask，让 noisy tokens 只看“自己 block 的 noisy + 之前 blocks 的 clean”，从而用一次注意力 kernel 同时算出所有 blocks 的损失。论文报告这种做法训练上比两次 forward 更高效（减少 memory 带宽瓶颈，利用高效 attention kernel）。

## 6. 采样（推理）：按 block 生成，块内可并行

推理时：

- **block 之间仍然是串行**（因为条件依赖前面 blocks）。
- **block 内可以并行采样多个 token**（扩散/去噪的一步通常会同时给出 block 内所有位置的 logits）。
- 使用 KV cache：生成下一个 block 时复用前面 blocks 的 KV，避免重复计算。

因此它能做到：

- **任意长度生成**：不断往后生成 block 即可（不像很多固定长度扩散一次生成固定维度）。
- **比纯双向扩散更“可服务”**：至少在块级具备 AR 的缓存结构。

## 7. 论文最强调的一点：perplexity gap 来自“高方差训练”，噪声日程可以优化

作者做了一个很关键的分析：即使在 \(L'=1\)（理论上与 AR 等价的极限）下，扩散式目标的 **Monte Carlo 估计** 也会因为“每个 batch 有效监督 token 数少/权重变化大”等因素导致 **梯度方差更大**，从而出现 perplexity gap。

他们提出：

- 用 **训练方差（或梯度估计方差）** 的估计量来解释/诊断 diffusion 与 AR 的差距。
- 设计 **clipped noise schedules**：不从全区间 \([0,1]\) 均匀采样 mask rate，而是采样一个子区间 \([\beta,\omega]\)，避免极端 mask rate（太轻或太重都“学习信号差且方差高”）。
- **数据驱动选择 \(\beta,\omega\)**：训练中定期做网格搜索，选择能最小化 \(\mathrm{Var}_{X,t}[\mathcal{L}]\) 的区间；论文展示这个方差与测试 perplexity 相关，并在不同 block size 下最优区间不同。

直观效果：

- 对较小 block（如 \(L'=4\)）可能更偏好“更重的 masking”；
- 对较大 block（如 \(L'=16\) 或更大）更偏好“更轻的 masking”；

这部分也是 TiDAR 等后续工作经常引用 Block Diffusion 的原因之一：它把“扩散 LLM 训练/likelihood 差距”的一个核心来源拆成了**方差与日程**问题。

## 8. 实验结论（论文主张）

在 LM1B、OpenWebText 等基准上：

- Block diffusion（不同 \(L'\)）在离散扩散模型里拿到更好的 perplexity（SOTA 级别），并显示 block size 带来的折中。
- 通过方差最小化/噪声日程（clipped schedule + 数据驱动选择），进一步缩小与 AR 的差距。

## 9. 与 TiDAR 的关系（怎么“被 TiDAR 用上”）

TiDAR 把 Block Diffusion 的“因果 + 块内双向”的结构视为一种实现 **KV cache** 与并行预测的基础，但 TiDAR 更进一步：

- Block Diffusion 的采样仍是“按 block 串行推进”，块内并行；
- TiDAR 则把“并行草稿（diffusion/marginal）+ AR 验证/采样（joint）”并行塞进同一次 forward（利用 free token slots），并用 rejection sampling 保质量。

## 10. 局限

- 训练比常规 diffusion 更贵（作者强调其向量化算法把代价压到 <2x diffusion 训练速度，并可先用标准 diffusion loss 预训练再微调）。
- 生成仍要按 block 串行推进：当 block 很小时，速度/可控性会更接近 AR；block 的最优值与任务、硬件并行度有关。

