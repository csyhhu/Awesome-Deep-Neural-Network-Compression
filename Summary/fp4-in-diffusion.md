# TrajComp: Trajectory-Level Optimization for Quantization + Step-Cache in Diffusion Models

## 目录

- [TrajComp: Trajectory-Level Optimization for Quantization + Step-Cache in Diffusion Models](#trajcomp-trajectory-level-optimization-for-quantization--step-cache-in-diffusion-models)
  - [目录](#目录)
  - [目标](#目标)
  - [相关工作](#相关工作)
    - [NVFP4 / FP4](#nvfp4--fp4)
    - [Diffusion 量化](#diffusion-量化)
    - [Diffusion 缓存](#diffusion-缓存)
  - [现有工作的 Gap](#现有工作的-gap)
  - [TrajComp：Layer-wise \& Step-wise Composite Compression](#trajcomplayer-wise--step-wise-composite-compression)
    - [Baseline Study](#baseline-study)
      - [Sana](#sana)
  - [Next Step](#next-step)

---

## 目标

使用 **NVFP4 量化** + **Step-level Caching** 加速 T2I Diffusion 模型。

核心命题：
> **不做 per-step 局部优化，而是直接优化多步缓存+量化后的轨迹终点与 FP16 轨迹终点的对齐。**

---

## 相关工作

### NVFP4 / FP4

| 工作 | 出处 | 核心思路 |
|------|------|---------|
| [NVFP4 QAD](../Large%20Pretraining%20Models/summary_nvfp4_qad.md) | NVIDIA | NVFP4 + KL 蒸馏；block-16 E4M3 + tensor FP32 |
| [Metis](../Large%20Pretraining%20Models/summary_metis_fp4_training.md) | ICLR 2026 | FP4 训练：SVD 分解 → 窄分布 → 适应 FP4 |
| [HiFloat4](../Large%20Pretraining%20Models/summary_hifloat4_fp4_pretraining.md) | NeurIPS 2025 | 三级缩放 FP4；RHT 消除异常值 |

### Diffusion 量化

| 工作 | 出处 | 核心思路 |
|------|------|---------|
| [PTQD](../Diffusion%20Models/summary_ptqd_post_training_quantization_diffusion.md) | NeurIPS 2023 | 量化噪声解耦；CNC+BC+VSC 三步 per-step 校正 |
| [Q-Sched](../Diffusion%20Models/summary_qsched_quantization_aware_scheduling.md) | NeurIPS 2025 | 修改采样调度器 (c_x, c_ε) 补偿量化误差 |

### Diffusion 缓存

| 工作 | 出处 | 核心思路 |
|------|------|---------|
| [ERTACache](../Diffusion%20Models/summary_ertacache.md) | ICLR 2026 | 离线标定缓存步 + 闭式 K,B 误差修正 |
| [BudCache](../Diffusion%20Models/summary_budcache_step_level_caching.md) | ICML 2026 | 离散策略搜索（SA+HC）+ **Trajectory Matching** 损失 |

---

## 现有工作的 Gap

> **没有任何工作将 NVFP4 量化与 step-level caching 联合用于 T2I Diffusion。**

三条技术路线各自独立：
- **NVFP4** → LLM 训练/推理，未进入扩散模型
- **Diffusion 量化** (PTQD/Q-Sched) → INT4/INT8，未用 FP4；**所有补偿方法用 per-step loss 训练**
- **Diffusion 缓存** (ERTACache/BudCache/TeaCache) → 全精度下做缓存，不考虑量化

**关键盲区**：
- PTQD/Q-Sched 都做 **per-step 局部对齐**：让量化后每一步的输出逼近 FP16 同一步的输出
- BudCache 虽然用了 trajectory matching，但**只用于搜索离散的缓存 mask**，不涉及量化和补偿参数的学习
- 没有人尝试：**在量化 + 缓存联合场景下，用 trajectory-level 损失端到端学习补偿参数**

---

## TrajComp：Layer-wise & Step-wise Composite Compression

### Baseline Study

#### Sana

| 项目 | 内容 |
|------|------|
| **模型** | Sana-0.6B / Sana-1.6B |
| **架构** | Flow Matching + Linear DiT |
| **Solver** | Flow-DPM-Solver（基于 DPM-Solver++ 修改） |
| **推理步数** | **20 steps**（默认，论文中所有 benchmark 以此为准） |
| **VAE** | F32C32P1（高压缩 autoencoder） |
| **分辨率** | 512×512 和 1024×1024 |

**Evaluation Metrics**（5 项主流指标）：

| 指标 | 数据集/规模 | 衡量维度 |
|------|------------|---------|
| **FID↓** | MJHQ-30K (30K 张 Midjourney 图像) | 图像质量与多样性 |
| **CLIP Score↑** | MJHQ-30K | 图文对齐 |
| **GenEval↑** | 533 prompts | 对象级图文对齐（单对象/双对象/计数/颜色/位置/属性） |
| **DPG-Bench↑** | 1,065 prompts | 密集图文对齐（实体/属性/关系） |
| **ImageReward↑** | 100 prompts | 人类偏好 |

**Sana-0.6B 在 512×512 上的 Baseline**（FP16，20 steps）：

| FID↓ | CLIP↑ | GenEval↑ | DPG↑ | Latency (A100, bs=1) |
|------|-------|----------|------|----------------------|
| 5.67 | 27.92 | 0.64 | 84.3 | 0.8s |

**选择 Sana 作为实验目标的原因**：
- 仅 20 步推理，实验迭代快（vs FLUX 的 50 步）
- 0.6B 参数，FP4 量化和缓存实验资源友好
- Flow Matching 框架，与主流 DiT（SD3/FLUX）架构一致
- 已有完整的 5 项 eval metric pipeline

## Next Step

- 选取某个benchmark, 看下多步下使用不同的step cache策略的效果（随机选择，固定选择，erta）
- 直接加入FP4，在不同step cache下的效果
- Learnable FP4在单步生成的效果

---

*Last updated: 2026-07-09*
