# Token Merging for Fast Stable Diffusion

> **原文**: [arXiv:2303.17604](https://arxiv.org/abs/2303.17604)\
> **作者**: Daniel Bolya, Judy Hoffman (Georgia Tech)\
> **发表**: CVPR 2023\
> **代码**: <https://github.com/dbolya/tomesd>

---

## 目录

- [Part I: ToMeSD 论文分析](#part-i-tomesd-论文分析)
  - [1. 综合理解](#1-综合理解)
  - [2. 核心问题与动机](#2-核心问题与动机)
  - [3. 核心思想](#3-核心思想)
    - [3.1 Token Merging 基础](#31-token-merging-基础)
    - [3.2 扩散模型的特殊挑战](#32-扩散模型的特殊挑战)
  - [4. 朴素方法与局限性](#4-朴素方法与局限性)
  - [5. 关键改进](#5-关键改进)
    - [5.1 新的 Token 分区方法](#51-新的-token-分区方法)
    - [5.2 设计选择消融实验](#52-设计选择消融实验)
      - [5.2.1 对哪些模块应用 ToMe（What）](#521-对哪些模块应用-tomewhat)
      - [5.2.2 对哪些层应用 ToMe（Where）](#522-对哪些层应用-tomewhere)
      - [5.2.3 何时应用 ToMe（When）](#523-何时应用-tomewhen)
  - [6. 最终结果](#6-最终结果)
    - [6.1 定量结果](#61-定量结果stable-diffusion-v15512512)
    - [6.2 与 xFormers 叠加](#62-与-xformers-叠加)
  - [7. 关键贡献总结](#7-关键贡献总结)
  - [8. 局限性与未来方向](#8-局限性与未来方向)
  - [9. 讨论问答（Q&A）](#9-讨论问答qa)
  - [10. 与本项目的关联](#10-与本项目的关联)
- [Part II: DiT 加速方法全面对比](#part-ii-dit-加速方法全面对比)
  - [11. 方法总览](#11-方法总览)
  - [12. 各维度详细对比](#12-各维度详细对比)
    - [12.1 压缩什么（What to Compress）](#121-压缩什么what-to-compress)
    - [12.2 何时压缩（When to Compress）](#122-何时压缩when-to-compress)
    - [12.3 如何恢复（How to Recover）](#123-如何恢复how-to-recover)
  - [13. 实验效果对比](#13-实验效果对比)
    - [13.1 ImageNet 类别条件生成](#131-imagenet-类别条件生成)
    - [13.2 跨任务通用性](#132-跨任务通用性)
  - [14. 方法退化关系图](#14-方法退化关系图统一框架的动机)
- [Part III: 统一框架设计](#part-iii-统一框架设计)
  - [15. 设计理念与四大正交维度](#15-设计理念与四大正交维度)
    - [15.1 WHAT — 压缩对象](#151-what--压缩对象)
    - [15.2 WHEN — 压缩时机](#152-when--压缩时机)
    - [15.3 WHERE — 压缩位置](#153-where--压缩位置)
    - [15.4 HOW — 恢复机制](#154-how--恢复机制)
  - [16. TokenCompress 统一公式](#16-tokencompress-统一公式)
    - [16.1 统一框架](#161-统一框架)
    - [16.2 退化形式：ToMe](#162-退化形式1tomeattention-based-merge)
    - [16.3 退化形式：SparseDiT-Poolingformer](#163-退化形式2sparsedit-poolingformerglobal-average-pooling)
    - [16.4 退化形式：SparseDiT-SDTM](#164-退化形式3sparsedit-sdtmadaptive-spatial-pooling)
    - [16.5 统一视角下的关键区别](#165-统一视角下的关键区别)
    - [16.6 可学习的 TokenCompress](#166-更深层的统一可学习的-tokencompress)
  - [17. Scheduling 统一公式](#17-scheduling-统一公式)
    - [17.1 统一 Scheduling 函数](#171-统一-scheduling-函数)
    - [17.2 退化形式](#172-退化形式)
    - [17.3 统一视角下的关键区别](#173-统一视角下的关键区别)
    - [17.4 可学习 Scheduler](#174-统一的可学习-scheduler)
    - [17.5 TokenCompress + Scheduling 联合](#175-更深层的统一tokencompress--scheduling-联合)
  - [18. 统一框架理论推导](#18-统一框架理论推导)
    - [18.1 TokenCompress 统一公式](#181-tokencompress-统一公式)
    - [18.2 Scheduling 统一公式](#182-scheduling-统一公式)
    - [18.3 四个正交自由度](#183-统一框架的四个正交自由度)
  - [19. 五种方法的降维矩阵与升维矩阵](#19-五种方法的降维矩阵与升维矩阵深度分析)
    - [19.1 统一矩阵表达](#191-统一矩阵表达)
    - [19.2 各方法的矩阵形式](#192-各方法的矩阵形式)
    - [19.3 矩阵汇总表](#193-矩阵汇总表)
    - [19.4 核心洞察](#194-核心洞察)
  - [20. DiT 中 N 是否固定？](#20-dit-中-n-是否固定)
    - [20.1 标准 DiT 的 token 数量](#201-标准-dit-的-token-数量)
    - [20.2 N 变化的场景](#202-n-变化的场景)
    - [20.3 对统一框架的影响](#203-对统一框架的影响)
  - [21. Scheduler 的输入设计](#21-scheduler-的输入设计)
    - [21.1 现有方法的 Scheduler 输入](#211-现有方法的-scheduler-输入)
    - [21.2 加入 Compressor 的 Loss 反馈](#212-用户提议加入-compressor-的-loss)
  - [22. Compressor 与 Scheduler 的耦合分析](#22-compressor-与-scheduler-的耦合分析)
    - [22.1 耦合关系](#221-耦合关系)
    - [22.2 三种解耦策略](#222-三种解耦策略)
    - [22.3 推荐方案](#223-推荐方案)
    - [22.4 统一框架的最终形式](#224-统一框架的最终形式)

---

## Part I: ToMeSD 论文分析

### 1. 综合理解

> **用户理解**：本文是个非训练的 token merge 方法，在已训练好的模型上，手动执行 token merge 操作，将相似的 token 取平均合并。

#### 点评

这个理解**基本正确**，但有几点值得补充和细化：

1. **"手动执行"→ 更准确说是"规则驱动的推理时操作"**。不是人为手动操作，而是在推理前向传播中按固定规则（2×2 区域随机分区 + 贪心合并相似 token）自动执行。无需训练，但也不是人为逐个调整，而是**程序化的固定策略**。
2. **"取平均合并"只是 merge 操作的一半**。完整流程是 **merge → 计算 → unmerge** 三步：
   - Merge：相似 token 取平均合并（减少 token 数，加速计算）
   - 计算：在减少后的 token 序列上执行 self-attention
   - Unmerge：计算完成后将合并结果还原回原 token 位置（信息从 dst 分发回 src）
3. **无训练 ≠ 无学习**。所有超参数（合并比例 r%、应用哪些模块、哪些层、何时合并）是通过**消融实验**在验证集上人工选定的最优值，而非通过梯度下降学习。这是本文最大的优势——**零训练成本即可获得显著加速**。

### 2. 核心问题与动机

扩散模型（如 Stable Diffusion）虽然能生成高质量图像，但其核心基于 Transformer，计算量随 token 数量呈**平方级增长**，导致推理速度慢、内存开销大。

现有加速方法（Flash Attention、xFormers 等）虽然优化了实现效率，但**并未减少实际工作量**——仍然对每个 token 都进行计算。然而，大多数图像（包括扩散模型生成的图像）本身具有很高的**冗余性**，对每个 token 都计算是资源浪费。

### 3. 核心思想

本文将 **Token Merging (ToMe)** 引入 Stable Diffusion，利用图像的自然冗余性，将相似的 token 合并，从而**减少实际计算量**，且**无需任何额外训练**。

#### 3.1 Token Merging 基础

Token Merging 通过将 token 划分为**源集合（src）和**目标集合（dst），然后将 src 中最相似的 $r$ 个 token 合并到 dst 中，从而减少 token 数量，加速后续计算。

#### 3.2 扩散模型的特殊挑战

与分类任务不同，扩散模型是**稠密预测任务**——每个 token 都需要输出去噪信息。因此需要引入 **Unmerging** 机制：

- **合并（Merging）**：将两个相似 token $x\_1, x\_2$ 取平均合并为 $x^\*\_{1,2}$
- **逆合并（Unmerging）**：在计算完成后，将合并结果重新分配回原始位置

虽然逆合并会丢失信息，但由于 token 本身相似，误差很小。

### 4. 朴素方法与局限性

朴素地将 ToMe 应用到 Stable Diffusion：在每个组件（自注意力、交叉注意力、MLP）之前合并 token，之后逆合并。

| 合并比例 r% | FID ↓ | 速度（秒/张）↓ | 内存（GB/张）↓ |
| :-----: | :---: | :------: | :-------: |
|  0 (基线) | 33.12 |   3.09   |    3.41   |
|    10   | 33.14 |   2.60   |    2.99   |
|    30   | 33.60 |   2.11   |    1.71   |
|    50   | 38.95 |   1.53   |    0.89   |

朴素方法可获得约 2× 加速和约 4× 内存节省，但 **FID 显著上升**（图像质量下降）。

### 5. 关键改进

#### 5.1 新的 Token 分区方法

原始 ToMe 通过交替排列 src/dst token 进行分区，这在扩散模型中导致 dst token 形成**规则的列**，图像沿行方向分辨率减半。

本文提出改进方案：

|         方法         |         描述         |    FID    |
| :----------------: | :----------------: | :-------: |
|      交替分区（原始）      |    src/dst 交替排列    |   38.95   |
|       2D 步长采样      |   以 2×2 步长采样 dst   |   36.12   |
|      随机采样（无修正）     |      随机选择 dst      |   46.08   |
|      随机采样（有修正）     |    固定 batch 内随机性   |   36.00   |
| **2×2 区域随机**（最终方案） | 每个 2×2 区域随机选一个 dst | **35.66** |

**关键发现**：使用无分类器引导（classifier-free guidance）时，有 prompt 和无 prompt 的样本需要**以相同方式分配 dst token**。固定 batch 内的随机性可解决此问题。

#### 5.2 设计选择消融实验

##### 5.2.1 对哪些模块应用 ToMe（What）

|  自注意力 | 交叉注意力 |  MLP  |    FID    |    速度    |
| :---: | :---: | :---: | :-------: | :------: |
|   ✓   |   ✓   |   ✓   |   35.66   |   1.56   |
|   ✓   |   ✗   |   ✓   |   36.10   |   1.57   |
| **✓** | **✗** | **✗** | **33.73** | **1.64** |
|   ✗   |   ✗   |   ✓   |   34.70   |   2.81   |

**结论**：仅对**自注意力**应用 ToMe 效果最好。

##### 5.2.2 对哪些层应用 ToMe（Where）

| 最少 token 数 | 受影响 block 数 |    FID    |    速度    |
| :--------: | :---------: | :-------: | :------: |
|     64     |    15（全部）   |   35.66   |   1.56   |
|    1024    |      9      |   34.37   |   1.56   |
|  **4096**  |    **4**    | **33.29** | **1.63** |

**结论**：仅对 token 数量最多的前几层应用 ToMe 即可获得大部分加速收益，且 FID 更低。

##### 5.2.3 何时应用 ToMe（When）

|  起始 r% |  结束 r% |    FID    |    速度    |
| :----: | :----: | :-------: | :------: |
|   70   |   30   |   35.89   |   1.65   |
| **60** | **40** | **35.53** | **1.58** |
|   50   |   50   |   35.66   |   1.56   |
|   30   |   70   |   36.45   |   1.61   |

**结论**：早期扩散步骤合并更多 token、后期少合并略好，但差异不显著。

### 6. 最终结果

#### 6.1 定量结果（Stable Diffusion v1.5，512×512）

|        方法       |   r%   |   FID ↓   |   秒/张 ↓  |  GB/张 ↓  |
| :-------------: | :----: | :-------: | :------: | :------: |
|        基线       |    0   |   33.12   |   3.09   |   3.41   |
| **ToMe for SD** |   10   |   32.86   |   2.56   |   2.99   |
|      <br />     |   20   |   32.86   |   2.29   |   2.17   |
|      <br />     |   30   |   32.80   |   2.06   |   1.71   |
|      <br />     |   40   |   32.87   |   1.85   |   1.26   |
|      <br />     |   50   |   33.02   |   1.65   |   0.89   |
|      <br />     | **60** | **33.37** | **1.52** | **0.60** |

在 60% token  reduction 下，实现 **2× 加速** 和 **5.6× 内存节省**，且 FID 仍保持较低水平。

#### 6.2 与 xFormers 叠加

ToMe 的加速效果可与 xFormers 等高效实现**叠加**。对于 2048×2048 图像，ToMe + xFormers 组合可实现 **5.4× 加速**（28 秒完成生成）。

### 7. 关键贡献总结

1. **首次将 Token Merging 应用于扩散模型**，无需任何训练即可实现显著加速
2. **提出了针对扩散模型的 Token 分区方法**：2×2 区域随机采样 + batch 内随机性固定
3. **系统性消融实验**：明确了 ToMe 应仅应用于自注意力模块、仅在 token 最多的层使用
4. **实用工具**：已开源实现（tomesd），可直接用于现有 Stable Diffusion 模型

### 8. 局限性与未来方向

- 更优的 unmerging 策略（如学习式逆合并）
- 探索基于 key 的相似度和比例注意力对扩散模型是否有效
- 将 ToMe 扩展到其他稠密预测任务（如视频生成）
- 合并比例超过 60% 后收益递减

### 9. 讨论问答（Q&A）

#### Q1: 原始 ToMe 的"交替排列 src/dst"具体是怎么交替的？一个 src 下一个就是 dst？

是的，就是**逐 token 交替分配**。将展平后的 token 序列按位置交替：

```
token 0 → src,  token 1 → dst,  token 2 → src,  token 3 → dst, ...
```

问题在于：Stable Diffusion 的 token 来自 **2D 空间网格**（如 64×64）。将 2D 网格按行展平后交替分配，如果宽度 W 是偶数，则**每一行的同一列位置类型相同**——dst 会形成**竖直列**（每隔一列出现一列 dst）。当合并 50% token 时（src 全部合并进 dst），相当于沿水平方向将分辨率减半，导致图像出现明显的垂直伪影。

#### Q2: 最终方案中"2×2"是什么意思？latent space 是 4 维（H×W×D×C）吗？

**关键澄清：Stable Diffusion 是图像生成，不是视频，没有时间维度 D。**

- **Latent space 维度**：VAE 将 512×512×3 的图像编码为 **64×64×4**（空间 H×W = 64×64，通道 C = 4）。所以在 transformer 中，token 排列成 **2D 空间网格**（H×W = 64×64 = 4096 个 token），每个 token 的特征维度为 C（经过 patch embedding 后通常更高，如 320 或 640）。
- **"2×2"的含义**：指空间网格上的 **2×2 区域**（4 个空间相邻的 token）。最终方案是：在每个 2×2 区域中**随机选 1 个 token 作为 dst**（dst 占 25%），其余 3 个为 src（占 75%）。

```
2×2 区域示例（两种可能的分配）：
[ src  dst ]        [ dst  src ]
[ src  src ]        [ src  src ]
```

这样既保证了 dst 在空间上**均匀分散**（不会聚集），又引入了**随机性**（避免规则网格伪影）。

> **注意**：H×W×D×C 的 4D 结构是**视频生成**模型的 token 排列（如 USV 论文），本文是纯图像生成，只有 H×W×C。

#### Q3: "对 token 数量最多的前几层应用"——为什么每层 token 数量不一样？每一层会执行一次 ToMe 吗？

因为 Stable Diffusion 使用 **U-Net 架构**，有多个分辨率层级。U-Net 通过下采样/上采样在不同分辨率间转换：

|  U-Net 层级  | 空间分辨率 | Token 数量 |
| :--------: | :---: | :------: |
| 输入层（最高分辨率） | 64×64 |   4096   |
|    下采样 1   | 32×32 |   1024   |
|    下采样 2   | 16×16 |    256   |
|     最底层    |  8×8  |    64    |

**每一层都会执行一次 ToMe**（如果该层被选中）。在朴素方法中，ToMe 应用于所有 15 个 block；但消融实验发现，**低分辨率层（如 8×8 = 64 token）合并意义不大**——token 本来就少，合并后信息损失大但加速收益小。最终方案选择 `min_tokens = 4096`，即**只对最高分辨率的 4 个 block** 应用 ToMe。

#### Q4: 消融实验是手动进行的吗？"何时应用"是每个 step 单独测试吗？

是的，**全部是手动设置不同配置进行实验**，不是自动搜索。

- **What（对哪些模块）**：手动组合 {self attn, cross attn, mlp} 的 4 种方案
- **Where（对哪些层）**：手动设置 4 个不同的 min\_tokens 阈值（64/256/1024/4096）
- **When（何时合并）**：手动设置 5 种线性插值方案

对于"When"实验，**不是每个 step 单独测试**，而是设置一个**跨所有 diffusion step 的线性调度**。例如 `r% start=60, r% end=40` 表示：50 个 diffusion step 中，第 1 步合并 60% token，第 50 步合并 40%，中间线性插值。每个配置生成 2000 张 ImageNet 图像后计算 FID，选 FID 最低的方案。

#### Q5: 本文涉及训练吗？

**完全不涉及训练**，这是本文最大的卖点之一。

ToMe 是一个**纯推理时的 token 处理策略**——在已有的 Stable Diffusion 模型上，通过修改前向传播过程（在 attention 之前合并 token，之后逆合并）来减少计算量。

所有选择（merge 方式、模块、层、step 调度）都是**推理时的配置参数**，通过消融实验在验证集（2000 张 ImageNet 图像 + FID 指标）上确定最佳配置，而非通过梯度下降学习得到。

### 10. 与本项目的关联

本文是 Token Merging 在扩散模型领域的开创性工作，与本项目中其他 Token Merging 相关研究（如 USV 视频生成统一稀疏化）有直接渊源。ToMe 的核心思想——利用 token 冗余性实现无训练加速——已被后续工作广泛借鉴和扩展。

---

## Part II: DiT 加速方法全面对比

> 本部分系统对比了五种主流 DiT 加速方法的设计哲学、技术细节和实验效果，为构建统一框架提供参考。

### 11. 方法总览

| 方法              | 论文                         | 核心思想                      | 压缩维度                   | 训练需求               |
| --------------- | -------------------------- | ------------------------- | ---------------------- | ------------------ |
| **ToMeSD**      | Bolya et al., CVPR 2023    | Token Merging + Unmerging | Token 数量               | 无训练（纯推理时）          |
| **DyDiT**       | Zhao et al., ICLR 2025     | 动态宽度 + Token 跳过           | Head + Channel + Token | 微调 + Router 训练     |
| **TokenCache**  | Lou et al., IEEE TSSP 2025 | Token 结果缓存复用              | Token 结果               | Cache Predictor 训练 |
| **DiTFastAttn** | Yuan et al., NeurIPS 2024  | 注意力冗余压缩                   | Attention 计算           | 无训练（纯推理时）          |
| **SparseDiT**   | Chang et al., NeurIPS 2025 | 空间三段式 + 时间动态剪枝            | Token 密度               | 微调                 |

### 12. 各维度详细对比

#### 12.1 压缩什么（What to Compress）

| 方法          | 压缩对象                                 | 压缩方式                                             |
| ----------- | ------------------------------------ | ------------------------------------------------ |
| ToMeSD      | Token 数量                             | 相似 token merge → 计算 → unmerge                    |
| DyDiT       | Attention Head + MLP Channel + Token | Router 选择激活的 head/channel；Token Router 跳过 MLP    |
| TokenCache  | Token 计算结果                           | 重要性评分 → 低重要性 token 复用缓存结果                        |
| DiTFastAttn | Attention 计算量                        | Window Attention + 残差共享；Step/CFG 间输出共享           |
| SparseDiT   | Token 密度                             | 底层 Poolingformer + 中层 Sparse-Dense 交替 + 顶层 Dense |

#### 12.2 何时压缩（When to Compress）

| 方法          | 时间步策略                       | 空间策略                              |
| ----------- | --------------------------- | --------------------------------- |
| ToMeSD      | 简单线性调度（60%→40%）             | 仅前几层（token 最多时）                   |
| DyDiT       | TDW：Router 根据 $E\_t$ 动态调整宽度 | SDT：Router 根据 token 难度跳过 MLP      |
| TokenCache  | I-step/P-step 调度（基于累积 α）    | Block 间自适应分配 cache ratio          |
| DiTFastAttn | AST：相邻步 attention 输出共享      | 每个 (step, layer) 独立选择策略           |
| SparseDiT   | 剪枝率线性递减（r\_min→r\_max）      | 三段式固定架构（Poolingformer/SDTM/Dense） |

#### 12.3 如何恢复（How to Recover）

| 方法          | 恢复机制                                   | 恢复粒度                |
| ----------- | -------------------------------------- | ------------------- |
| ToMeSD      | Unmerge：平均分发回原 token 位置                | Token 级（近似恢复）       |
| DyDiT       | 无需恢复（动态跳过，下一层重新计算）                     | Head/Channel 级（硬跳过） |
| TokenCache  | 直接复用上一步计算结果                            | Token 级（精确缓存）       |
| DiTFastAttn | 残差共享/输出共享（无需恢复）                        | Attention 级（共享计算）   |
| SparseDiT   | SDTM 内：上采样 + 线性融合 + cross-attention 恢复 | 模块级（稀疏→稠密循环）        |

### 13. 实验效果对比

#### 13.1 ImageNet 类别条件生成

| 方法          | 模型           | 分辨率       | FLOPs 变化 | 速度提升      | FID 变化 |
| ----------- | ------------ | --------- | -------- | --------- | ------ |
| ToMeSD      | DiT-XL       | 256×256   | -        | +66%      | +12.47 |
| DyDiT       | DiT-XL       | 512×512   | **-51%** | **+73%**  | -0.20  |
| TokenCache  | DiT-XL/2     | 256×256   | -39%     | +201%     | -0.02  |
| DiTFastAttn | PixArt-Sigma | 2048×2048 | **-76%** | **+80%**  | 微小     |
| SparseDiT   | DiT-XL       | 512×512   | **-55%** | **+175%** | +0.09  |

> 注：不同方法的实验设置（模型、分辨率、采样器）不一致，结果仅供参考

#### 13.2 跨任务通用性

| 方法          | 图像生成           | 视频生成         | 文生图        | 跨架构                    |
| ----------- | -------------- | ------------ | ---------- | ---------------------- |
| ToMeSD      | ✓ (SD v1.5)    | ✗            | ✗          | ✗ (仅 U-Net)            |
| DyDiT       | ✓ (DiT)        | ✓ (Latte)    | ✓ (PixArt) | ✓ (DiT/Latte/SD3/FLUX) |
| TokenCache  | ✓ (DiT)        | ✓ (OpenSora) | ✓ (PixArt) | ✓ (多种 DiT)             |
| DiTFastAttn | ✓ (DiT/PixArt) | ✓ (OpenSora) | ✗          | ✓ (DiT/MMDiT)          |
| SparseDiT   | ✓ (DiT)        | ✓ (Latte)    | ✓ (PixArt) | ✓ (DiT/Latte/PixArt)   |

### 14. 方法退化关系图（统一框架的动机）

```
                    ┌─────────────┐
                    │  统一框架 U  │
                    └──────┬──────┘
           ┌───────────┬───┴───┬───────────┐
           ▼           ▼       ▼           ▼
    ┌─────────┐  ┌─────────┐ ┌─────────┐  ┌─────────┐
    │ ToMeSD  │  │  DyDiT  │ │TokenCache│  │DiTFastAttn│  ← 各方法为 U 的特例
    └─────────┘  └─────────┘ └─────────┘  └─────────┘
         │
         ▼
    ┌─────────┐
    │SparseDiT│
    └─────────┘
```

每种方法都可视为统一框架在特定参数配置下的退化形式：
- **ToMeSD**：U 在 `{token_merge=true, dynamic_scheduling=false, attention_compress=false, no_training}` 下的退化
- **DyDiT**：U 在 `{dynamic_width=true, token_skip_mlp=true, time_aware=true, router_training=true}` 下的退化
- **TokenCache**：U 在 `{token_cache=true, step_scheduling=true, block_allocation=true, cache_predictor_training=true}` 下的退化
- **DiTFastAttn**：U 在 `{window_attention=true, step_sharing=true, cfg_sharing=true, no_training}` 下的退化
- **SparseDiT**：U 在 `{sparse_dense_alternation=true, poolingformer_bottom=true, timestep_pruning=true, fine_tuning=true}` 下的退化

---

## Part III: 统一框架设计

### 15. 设计理念与四大正交维度

构建一个**多轴稀疏化空间**的统一框架，将五种方法的设计维度抽象为**正交的控制旋钮**，每种方法都是这些旋钮的一种特定配置。

#### 15.1 WHAT — 压缩对象

决定压缩什么计算：

| 选项                      | 描述               | 代表方法        |
| ----------------------- | ---------------- | ----------- |
| `token_count`           | 减少 token 数量      | ToMeSD      |
| `token_density`         | 分块使用不同 token 密度  | SparseDiT   |
| `token_results`         | 复用 token 计算结果    | TokenCache  |
| `attention_computation` | 减少 attention 计算量 | DiTFastAttn |
| `model_width`           | 动态调整模型宽度         | DyDiT       |

#### 15.2 WHEN — 压缩时机

决定何时施加压缩：

| 选项                   | 描述                   | 代表方法              |
| -------------------- | -------------------- | ----------------- |
| `static`             | 固定不变                 | ToMeSD            |
| `timestep_linear`    | 随时间步线性变化             | SparseDiT         |
| `timestep_dynamic`   | Router 根据时间步动态决策     | DyDiT             |
| `timestep_scheduled` | 预计算 I-step/P-step 调度 | TokenCache        |
| `step_sharing`       | 相邻步共享计算              | DiTFastAttn (AST) |
| `cfg_sharing`        | 条件/无条件推理共享           | DiTFastAttn (ASC) |

#### 15.3 WHERE — 压缩位置

决定在网络的哪些位置施加压缩：

| 选项                   | 描述                     | 代表方法        |
| -------------------- | ---------------------- | ----------- |
| `bottom_only`        | 仅底层压缩                  | ToMeSD      |
| `three_segment`      | 三段式（bottom/middle/top） | SparseDiT   |
| `per_layer_dynamic`  | 每层独立决策                 | DiTFastAttn |
| `head_channel_token` | Head/Channel/Token 三维  | DyDiT       |
| `block_adaptive`     | Block 间自适应分配           | TokenCache  |

#### 15.4 HOW — 恢复机制

决定如何从压缩状态恢复信息：

| 选项                   | 描述          | 代表方法        |
| -------------------- | ----------- | ----------- |
| `merge_unmerge`      | 合并→计算→逆合并   | ToMeSD      |
| `sparse_dense_cycle` | 稀疏→稠密交替循环   | SparseDiT   |
| `result_reuse`       | 直接复用缓存结果    | TokenCache  |
| `residual_sharing`   | 残差共享 + 输出共享 | DiTFastAttn |
| `dynamic_skip`       | 硬跳过（无需恢复）   | DyDiT       |

### 16. TokenCompress 统一公式

**核心问题**：ToMe 用 Attention 权重做 Token Merge（相似 token 加权求和），SparseDiT 用卷积/池化做降采样——能否用一个统一公式表达？

#### 16.1 统一框架

设原始 token $T \in \mathbb{R}^{N \times d}$，压缩后 token $T' \in \mathbb{R}^{M \times d}$（$M < N$），统一表达为：

$$T' = \text{TokenCompress}(T; W, \Pi)$$

其中：

- $W \in \mathbb{R}^{M \times N}$ 是**聚合权重矩阵**（每个目标 token 从哪些源 token 聚合信息）
- $\Pi$ 是**模式参数**（控制聚合方式）

#### 16.2 退化形式1：ToMe（Attention-based Merge）

$$W_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d})}{\sum_{k} \exp(q_i \cdot k_k / \sqrt{d})} \cdot \mathbb{1}[\text{sim}(i, j) > \tau]$$

- $W$ 由 Attention 权重**动态生成**（依赖 token 内容）
- $\Pi = \tau$（merge 阈值：只合并相似度超过阈值的 token）
- 本质：**内容感知的加权聚合**——相似的 token 被分配高权重

#### 16.3 退化形式2：SparseDiT-Poolingformer（Global Average Pooling）

$$W_{ij} = \frac{1}{N}, \quad \forall i, j$$

- $W$ 为均匀权重矩阵（全局平均）
- $\Pi = \text{none}$（无模式参数，每个 token 平等贡献）
- 本质：**位置无关的均匀聚合**——所有 token 信息融合为全局特征

#### 16.4 退化形式3：SparseDiT-SDTM（Adaptive Spatial Pooling）

$$W_{ij} = \frac{\text{softmax}(\text{Linear}(p_j))_i}{\sum_{k} \text{softmax}(\text{Linear}(p_j))_k} \cdot \mathbb{1}[\text{cell}(j) = \text{cell}(i)]$$

- $W$ 由**可学习的空间位置编码**生成（依赖 token 位置）
- $\Pi = \text{spatial\_grid}$（空间网格划分：每个 cell 内的 token 聚合到一个目标 token）
- 本质：**位置感知的空间聚合**——空间上相邻的 token 被聚合

#### 16.5 统一视角下的关键区别

| 维度       | ToMe            | SparseDiT (Poolingformer) | SparseDiT (SDTM) |
| -------- | --------------- | ------------------------- | ---------------- |
| $W$ 生成方式 | Attention（内容驱动） | 均匀（固定）                    | 位置编码（学习驱动）       |
| 聚合粒度     | Token 间（不规则）    | 全局（一个池）                   | 空间 cell（规则网格）    |
| Token 数量 | 动态（随 step 变化）   | 不变（N→N）                   | 固定（N→M）          |
| 信息损失     | 低（保留内容结构）       | 高（丢失位置信息）                 | 中（保留空间结构）        |

#### 16.6 更深层的统一：可学习的 TokenCompress

如果我们将 $W$ 也设为可学习的，那么三者都可以统一为：

$$T' = \text{softmax}(\text{Router}(T, t)) \cdot T$$

- **Router** 是一个可学习函数（可以是 Attention、MLP、Conv 等）
- 当 Router = Attention + Top-K mask → ToMe 退化
- 当 Router = Linear(1) + uniform mask → Poolingformer 退化
- 当 Router = Linear(pos_embed) + spatial mask → SDTM 退化

**结论**：ToMe 和 SparseDiT 的 Token 压缩本质上都是**加权聚合**，区别仅在于权重矩阵 $W$ 的生成方式（内容 vs 位置 vs 固定）。可以用统一公式 $T' = W \cdot T$ 表达，不同方法对应不同的 $W$ 生成策略。

### 17. Scheduling 统一公式

**核心问题**：ToMe 用固定压缩率，SparseDiT 用线性函数，DiTFastAttn 用步间共享（等价于删除某些步的计算）——能否用一个函数统一表达？

#### 17.1 统一 Scheduling 函数

设 $r(t)$ 为第 $t$ 步的**计算比率**（$r(t) \in [0, 1]$，1 = 全计算，0 = 跳过计算），统一表达为：

$$r(t) = \sigma(f_{\psi}(t, T, x_t))$$

其中：

- $t$：当前时间步，$T$：总步数
- $x_t$：当前步的输入特征（可选，用于内容自适应）
- $f_{\psi}$：参数化函数（可学习或预设）
- $\sigma$：映射到 $[0, 1]$ 的单调函数（Sigmoid 或 ReLU6）

#### 17.2 退化形式

**退化形式 1：ToMe（固定压缩率）**

$$f_{\psi}(t) = c \quad \text{(常数函数)}, \quad r(t) = \sigma(c) = r_{\text{fixed}}$$

**退化形式 2：SparseDiT（线性函数）**

$$f_{\psi}(t) = r_{\min} + (r_{\max} - r_{\min}) \cdot \frac{t}{T}, \quad r(t) = \sigma(f_{\psi}(t))$$

**退化形式 3：DiTFastAttn AST（步间共享）**

$$f_{\psi}(t) = \begin{cases} 0 & t \in S_{\text{share}} \\ \infty & t \in S_{\text{full}} \end{cases}, \quad r(t) = \begin{cases} 0 & t \in S_{\text{share}} \\ 1 & t \in S_{\text{full}} \end{cases}$$

**退化形式 4：TokenCache（I-step/P-step 调度）**

$$f_{\psi}(t) = \begin{cases} r_{\text{cache}} & t \in P\text{-steps} \\ 1 & t \in I\text{-steps} \end{cases}$$

#### 17.3 统一视角下的关键区别

| 方法                | $f_{\psi}$ 形式 | $r(t)$ 曲线          | 自适应         |
| ----------------- | -------------- | ------------------ | ----------- |
| ToMe              | 常数 $c$         | 水平线                | 无           |
| SparseDiT         | 线性函数           | 单调递增曲线             | 时间步自适应      |
| DiTFastAttn (AST) | 阶跃函数           | 阶梯状（0/1 交替）        | 步组自适应       |
| TokenCache        | 阶跃函数           | 阶梯状（r_cache/1 交替） | 步类型自适应      |
| DyDiT (TDW)       | 神经网络           | 复杂非线性              | 时间步 + 内容自适应 |

#### 17.4 统一的可学习 Scheduler

如果 $f_{\psi}$ 用一个轻量级神经网络（如 MLP）参数化：

$$r(t) = \sigma(\text{MLP}([t/T, \text{time\_embed}(t)]))$$

则：

- **ToMe 退化**：MLP 输出常数（权重=0，偏置=const）
- **SparseDiT 退化**：MLP 学习线性函数
- **DiTFastAttn 退化**：MLP 学习阶跃函数（权重→∞，产生 sharp 0/1 切换）
- **新能力**：MLP 可以学习任意非线性调度曲线，发现更优的时间分配策略

#### 17.5 更深层的统一：TokenCompress + Scheduling 联合

将两个维度统一：

$$T_t' = \text{TokenCompress}(T_t; W_t, \Pi_t) \cdot r(t)$$

- $W_t, \Pi_t$：由 Scheduling 函数 $r(t)$ 决定的压缩策略
- 不同时间步 $t$ 采用不同的压缩方法和比率
- 例如：早期用 ToMe-style merge（内容相似），中期用 SparseDiT-style pooling（空间结构），晚期全计算（细节）

### 18. 统一框架理论推导

#### 18.1 TokenCompress 统一公式

$$\boxed{T' = W \cdot T, \quad W_{M \times N} = g_{\phi}(T, \text{pos})}$$

- $g_{\phi}$ 为权重生成函数：
  - $g_{\phi} = \text{softmax}(\text{Attention}(T)) + \text{TopK mask}$ → **ToMe**
  - $g_{\phi} = \frac{1}{N}\mathbf{1}^T$ → **SparseDiT-Poolingformer**
  - $g_{\phi} = \text{softmax}(\text{Linear}(\text{pos\_embed})) + \text{Spatial mask}$ → **SparseDiT-SDTM**
  - $g_{\phi} = \sigma(\text{MLP}(T))$ → **可学习统一形式**（待探索）

#### 18.2 Scheduling 统一公式

$$\boxed{r(t) = \sigma(f_{\psi}(t, T, x_t))}$$

- $f_{\psi}$ 为调度函数：
  - $f_{\psi} = c$ (常数) → **ToMe** 退化
  - $f_{\psi} = r_{\min} + (r_{\max} - r_{\min}) \cdot t/T$ (线性) → **SparseDiT** 退化
  - $f_{\psi} = \text{step\_function}(t; S_{\text{share}})$ (阶跃) → **DiTFastAttn** 退化
  - $f_{\psi} = \text{MLP}(t/T, \text{time\_embed}(t))$ (可学习) → **统一目标**

#### 18.3 统一框架的四个正交自由度

| 自由度              | 含义        | ToMe            | SparseDiT     | DiTFastAttn  | 统一形式              |
| ---------------- | --------- | --------------- | ------------- | ------------ | ----------------- |
| **压缩方式** (How)   | $W$ 的生成方式 | Attention-based | Pooling-based | Window-based | $g_{\phi}$ (可学习) |
| **压缩位置** (Where) | 网络中哪些层压缩  | 底层 only         | 三段式           | 每层独立         | 可学习分配             |
| **压缩时机** (When)  | 时间步调度     | 固定              | 线性            | 步组共享         | $f_{\psi}$ (可学习) |
| **压缩对象** (What)  | 什么计算被压缩   | Token 数量        | Token 密度      | Attention 计算 | 可学习组合             |

### 19. 五种方法的降维矩阵与升维矩阵（深度分析）

#### 19.1 统一矩阵表达

所有 token 压缩操作可统一为矩阵乘法：

$$T' = W_{\downarrow} \cdot T \quad (W_{\downarrow} \in \mathbb{R}^{M \times N})$$
$$\hat{T} = W_{\uparrow} \cdot T' \quad (W_{\uparrow} \in \mathbb{R}^{N \times M}, \text{if applicable})$$

#### 19.2 各方法的矩阵形式

##### 19.2.1 ToMeSD — 二分匹配合并矩阵

**降维** $W_{\downarrow} \in \mathbb{R}^{M \times N}$（$M = N - r$，$r$ 为合并对数）：

$$W_{\downarrow}[i, j] = \begin{cases} 1 & \text{token } j \text{ 未合并，映射到 } i \\ 0.5 & \text{token } j \text{ 与另一 token 合并到 } i \\ 0 & \text{otherwise} \end{cases}$$

**示例**（N=6, 合并 token 3&4 → 3'）：
```
W_down = [[1, 0, 0,   0,   0, 0],    # token 0 → 0'
          [0, 1, 0,   0,   0, 0],    # token 1 → 1'
          [0, 0, 1,   0,   0, 0],    # token 2 → 2'
          [0, 0, 0, 0.5, 0.5, 0],    # token 3&4 → 3' (平均)
          [0, 0, 0,   0,   0, 1]]    # token 5 → 4'
```

**升维** $W_{\uparrow} \in \mathbb{R}^{N \times M}$（Unmerge：合并值复制回原位置）：

$$W_{\uparrow}[j, i] = \begin{cases} 1 & \text{token } j \text{ 来自合并 token } i \\ 0 & \text{otherwise} \end{cases}$$

```
W_up = [[1, 0, 0, 0,   0],    # token 0 ← 0'
        [0, 1, 0, 0,   0],    # token 1 ← 1'
        [0, 0, 1, 0,   0],    # token 2 ← 2'
        [0, 0, 0, 1,   0],    # token 3 ← 3' (复制)
        [0, 0, 0, 1,   0],    # token 4 ← 3' (复制)
        [0, 0, 0, 0,   1]]    # token 5 ← 4'
```

**关键**：$W_{\uparrow} \neq W_{\downarrow}^T$。合并时用 0.5（平均），恢复时用 1（复制）。$W_{\uparrow} \cdot W_{\downarrow}$ 对未合并 token 为 1（恒等），对合并 token 对为 $\begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}$（平均滤波器）。

##### 19.2.2 SparseDiT-Poolingformer — 秩-1 均匀矩阵

**降维** $W_{\downarrow} \in \mathbb{R}^{N \times N}$（注意：M = N，不减少 token 数量）：

$$W_{\downarrow} = \frac{1}{N} \mathbf{1}_{N \times N} = \frac{1}{N} \begin{pmatrix} 1 & 1 & \cdots & 1 \\ 1 & 1 & \cdots & 1 \\ \vdots & & \ddots & \vdots \\ 1 & 1 & \cdots & 1 \end{pmatrix}$$

- 秩为 1（所有行相同），将所有 token 聚合为全局均值后广播
- 实际操作：$T_{\text{out}} = T + W_{\downarrow} \cdot T = (I + W_{\downarrow}) \cdot T$

**升维**：无（输出已是 $N$ 维）。但信息从 $N$ 维压缩到 $1$ 维（均值）再广播回 $N$ 维，信息损失 $= N - 1$ 个自由度。

##### 19.2.3 SparseDiT-SDTM — 块对角空间池化矩阵

**降维** $W_{\downarrow} \in \mathbb{R}^{M \times N}$（自适应空间池化）：

将 $N = H \times W$ 个 token 划分为 $M = H' \times W'$ 个空间 cell，每个 cell 含 $k = N/M$ 个 token：

$$W_{\downarrow} = \text{blkdiag}\left(\frac{1}{k}\mathbf{1}_{1 \times k}, \frac{1}{k}\mathbf{1}_{1 \times k}, \ldots, \frac{1}{k}\mathbf{1}_{1 \times k}\right)$$

**示例**（N=9, M=4, 每个 cell 含 2-3 个 token）：
```
W_down = [[0.33, 0.33, 0.33,  0,    0,    0,    0,    0,    0   ],
          [0,    0,    0,     0.5,  0.5,  0,    0,    0,    0   ],
          [0,    0,    0,     0,    0,    0,    0.33, 0.33, 0.33],
          [0,    0,    0,     0,    0,    0.5,  0,    0,    0.5 ]]
```

若可学习版本：$W_{\downarrow} = \text{softmax}(\text{Linear}(\text{pos\_embed})) \odot M_{\text{spatial}}$

**升维** $W_{\uparrow} \in \mathbb{R}^{N \times M}$（上采样 + 线性融合）：

- 最近邻插值：$W_{\uparrow} = \text{blkdiag}(\mathbf{1}_{k \times 1}, \ldots, \mathbf{1}_{k \times 1})$
- 可学习版本：$W_{\uparrow} = \text{Linear}(M \to N)$ 或转置卷积
- SDTM 实际操作：$\hat{T} = W_{\uparrow} \cdot T' \cdot W_1 + T \cdot W_2$（线性融合原始 token）

##### 19.2.4 DyDiT — 对角掩码矩阵（非降维）

DyDiT **不减少 token 数量**，而是跳过部分计算：

**SDT（Token 级跳过）**：$W \in \mathbb{R}^{N \times N}$

$$W = \text{diag}(m_1, m_2, \ldots, m_N), \quad m_i \in \{0, 1\}$$

$$T' = W \cdot \text{MLP}(T) + (I - W) \cdot T$$

- $m_i = 1$：token $i$ 过 MLP
- $m_i = 0$：token $i$ 跳过 MLP（保留原值）
- 有效计算量 $= \sum_i m_i$，但输出维度仍为 $N$

**TDW（Head/Channel 级跳过）**：不涉及 token 矩阵

- Head mask：$S_h \in \{0,1\}^H$，选择激活的 attention head
- Channel mask：$S_c \in \{0,1\}^H$，选择激活的 MLP channel group
- 压缩维度：$d \to d'$（模型宽度），不是 $N \to M$（token 数量）

##### 19.2.5 TokenCache — 对角混合矩阵（时间维度）

TokenCache **不减少 token 数量**，而是在时间维度上复用：

$$W_t = \text{diag}(\alpha_1^t, \alpha_2^t, \ldots, \alpha_N^t), \quad \alpha_i^t \in [0, 1]$$

$$T'_t = W_t \cdot f(T_t) + (I - W_t) \cdot f(T_{t-1})$$

- $\alpha_i = 1$：token $i$ 重新计算
- $\alpha_i = 0$：token $i$ 复用上一步结果
- 这是**时间维度的对角矩阵**，不是空间降维

##### 19.2.6 DiTFastAttn — 注意力矩阵掩码

DiTFastAttn 压缩的是**注意力矩阵** $A \in \mathbb{R}^{N \times N}$，而非 token：

**WA（Window Attention）**：

$$A' = A \odot M_{\text{window}}$$

$M_{\text{window}} \in \{0,1\}^{N \times N}$ 为块对角掩码：
```
M_window = [[1, 1, 0, 0, 0, 0, ...],
            [1, 1, 0, 0, 0, 0, ...],
            [0, 0, 1, 1, 0, 0, ...],
            [0, 0, 1, 1, 0, 0, ...],
            ...]
```

**AST（步间共享）**：$A_t = A_{t-1}$（时间维度复制，$r(t) = 0$）

**ASC（CFG 共享）**：$A_{\text{cond}} = A_{\text{uncond}}$（条件/无条件共享）

#### 19.3 矩阵汇总表

| 方法 | 矩阵类型 | 形状 | 是否降维 | 升维矩阵 | 关键特征 |
|---|---|---|---|---|---|
| **ToMeSD** | 稀疏二值分数矩阵 | $M \times N$ | ✓ ($N \to M$) | $N \times M$（二值复制） | 内容驱动，动态生成 |
| **Poolingformer** | 秩-1 均匀矩阵 | $N \times N$ | ✗ ($N \to N$) | 无 | 全局平均，信息损失大 |
| **SDTM** | 块对角池化矩阵 | $M \times N$ | ✓ ($N \to M$) | $N \times M$（可学习） | 位置驱动，空间结构化 |
| **DyDiT-SDT** | 对角 0/1 掩码 | $N \times N$ | ✗ ($N \to N$) | 无（跳过=保留原值） | Token 级跳过 |
| **DyDiT-TDW** | Head/Channel mask | $H \times 1$ | ✗ (宽度压缩) | 无 | 非 token 级压缩 |
| **TokenCache** | 对角 $\alpha$ 混合 | $N \times N$ | ✗ ($N \to N$) | 无（复用上一步） | 时间维度复用 |
| **DiTFastAttn** | 块对角 0/1 掩码 | $N \times N$ | ✗ ($N \to N$) | 无 | 注意力矩阵稀疏化 |

#### 19.4 核心洞察

从矩阵视角看，五种方法可分为**三大类**：

1. **真降维**（$N \to M$）：ToMeSD、SDTM — 输出 token 数减少，需要升维恢复
2. **掩码跳过**（$N \to N$，对角矩阵）：DyDiT-SDT、TokenCache — token 数不变，部分计算跳过
3. **注意力稀疏化**（$N \times N$ 矩阵）：DiTFastAttn-WA — token 数不变，注意力矩阵被掩码

**统一公式**：

$$T' = \underbrace{W_{\downarrow}}_{\text{空间压缩}} \cdot T \cdot \underbrace{D}_{\text{宽度压缩}} + \underbrace{(I - D)}_{\text{跳过}} \cdot T_{\text{cache}}$$

其中 $W_{\downarrow}$ 可退化成各方法的形式，$D$ 为宽度掩码（DyDiT），$T_{\text{cache}}$ 为时间复用（TokenCache）。

### 20. DiT 中 N 是否固定？

#### 20.1 标准 DiT 的 token 数量

在标准 DiT 中，输入图像 $H \times W \times C$ 经 patchify（patch size $p$）后：

$$N = \frac{H}{p} \times \frac{W}{p}$$

**对于固定分辨率和 patch size，N 是确定的**。例如：
- 256×256, p=16 → N = 16×16 = 256
- 512×512, p=16 → N = 32×32 = 1024
- 1024×1024, p=8 → N = 128×128 = 16384

#### 20.2 N 变化的场景

| 场景 | N 是否变化 | 原因 |
|---|---|---|
| 不同分辨率 | ✓ 变化 | $N = (H/p) \times (W/p)$ 随分辨率变化 |
| 不同长宽比 | ✓ 变化 | PixArt/FLUX 支持变长宽比，$H \neq W$ |
| 不同 patch size | ✓ 变化 | p=8 vs p=16 导致 4× 差异 |
| 文本条件注入 | ✓ 变化 | 文本 token 拼接：$N_{\text{total}} = N_{\text{image}} + N_{\text{text}}$，$N_{\text{text}}$ 随 prompt 长度变化 |
| 视频生成 | ✓ 变化 | $N = T_{\text{frames}} \times (H/p) \times (W/p)$ |
| 同一模型同一分辨率 | ✗ 固定 | $N$ 由 $(H, W, p)$ 唯一确定 |

#### 20.3 对统一框架的影响

**关键结论**：N 在推理时对同一配置是固定的，但跨配置会变化。

这意味着：
1. **降维矩阵 $W_{\downarrow}$ 不能硬编码为固定大小** — 需要是分辨率自适应的函数
2. **ToMe 的合并比例** 可以用比例 $r/N$ 而非绝对数 $r$ 来适应不同 $N$
3. **SDTM 的空间网格** 需要根据 $H \times W$ 动态调整池化核大小
4. **Poolingformer 的 $W = \frac{1}{N}\mathbf{1}^T$** 天然适应任意 $N$（只需知道 $N$ 的值）

**统一框架设计建议**：压缩矩阵应表示为**生成函数**而非固定矩阵：

$$W_{\downarrow} = g_{\phi}(N, M, \text{pos\_embed}, T)$$

### 21. Scheduler 的输入设计

#### 21.1 现有方法的 Scheduler 输入

| 方法 | 输入 | 输出 | 特点 |
|---|---|---|---|
| ToMe | 无 | 固定 $r$ | 完全静态 |
| SparseDiT | $t$（时间步） | $r(t)$ 线性函数 | 仅时间步感知 |
| DyDiT | $E_t$（时间步 embedding） | Head/Channel mask | 时间步 + 可学习路由 |
| TokenCache | $x_t$（当前特征）+ $t$ | $\alpha \in [0,1]^N$ | 内容 + 时间步感知 |
| DiTFastAttn | $t$ + layer index | 策略选择 | 离散决策 |

#### 21.2 用户提议：加入 Compressor 的 Loss

**这是一个很好的想法！** 将 compressor 的反馈信息加入 scheduler，形成**闭环调度**。

**设计方案**：

$$r(t) = \sigma\Big(f_{\psi}\big(\underbrace{t, T}_{\text{时间步}}, \underbrace{x_t}_{\text{当前特征}}, \underbrace{L_{\text{recon}}(t)}_{\text{压缩器反馈}}\big)\Big)$$

其中 $L_{\text{recon}}(t)$ 为压缩器的重构损失：

$$L_{\text{recon}}(t) = \|T_t - W_{\uparrow} \cdot W_{\downarrow} \cdot T_t\|^2$$

**反馈逻辑**：

- $L_{\text{recon}}$ **高** → 压缩损失大 → scheduler 降低压缩率（增加 token 数）
- $L_{\text{recon}}$ **低** → 压缩安全 → scheduler 提高压缩率（减少 token 数）

**可用的反馈信号**：

| 信号 | 计算方式 | 开销 | 信息量 |
|---|---|---|---|
| $L_{\text{recon}}$ | $\|T - W_\uparrow W_\downarrow T\|^2$ | 低（一次矩阵乘法） | 高（直接衡量压缩质量） |
| Attention 熵 | $H(A) = -\sum A_{ij} \log A_{ij}$ | 中（需计算 attention） | 中（衡量注意力集中度） |
| Token 方差 | $\text{Var}(T) = \frac{1}{N}\sum \|T_i - \bar{T}\|^2$ | 低 | 中（衡量 token 多样性） |
| 频域能量比 | $\frac{E_{\text{low-freq}}}{E_{\text{total}}}$ | 中（需 FFT） | 高（衡量低/高频分布） |
| 历史累积误差 | $\sum_{\tau=0}^{t} L_{\text{recon}}(\tau)$ | 低（累加） | 高（全局质量监控） |

**闭环调度的优势**：

1. **自适应**：不同图像自动调整压缩策略（简单图像压缩更多，复杂图像保留更多 token）
2. **自校正**：压缩过度时 loss 增大，scheduler 自动回退
3. **内容感知**：超越时间步感知，实现真正的内容感知调度

**潜在风险**：

1. **计算开销**：计算 $L_{\text{recon}}$ 需要先做一次压缩-解压，增加约 5% 开销
2. **训练不稳定**：闭环反馈可能导致振荡（压缩率在高低之间震荡）
3. **冷启动**：推理开始时无历史信息可用

**缓解方案**：用 EMA 平滑 $L_{\text{recon}}$，或设置上下界 $r \in [r_{\min}, r_{\max}]$ 防止极端情况。

### 22. Compressor 与 Scheduler 的耦合分析

#### 22.1 耦合关系

**是的，compressor 和 scheduler 的联合学习是耦合的。** 原因：

1. **Scheduler 依赖 Compressor 的行为**：
   - 若 compressor 是强恢复力的 SDTM（有 cross-attention 恢复），scheduler 可以更激进
   - 若 compressor 是弱恢复力的 Poolingformer（仅全局平均），scheduler 需保守

2. **Compressor 依赖 Scheduler 的调度**：
   - 若 scheduler 在早期大量压缩，compressor 需要处理更粗糙的全局特征
   - 若 scheduler 在后期保持全计算，compressor 的恢复模块可以更轻量

3. **梯度耦合**：联合训练时，compressor 和 scheduler 的梯度互相影响，可能陷入局部最优

#### 22.2 三种解耦策略

**策略 1：联合训练（完全耦合）**

```
L_total = L_diffusion + λ_1 · L_FLOPs(compressor) + λ_2 · L_FLOPs(scheduler)
```

- 同时优化 compressor 参数 $\phi$ 和 scheduler 参数 $\psi$
- **优点**：理论最优
- **缺点**：训练不稳定，梯度耦合，超参数敏感

**策略 2：交替训练（半解耦）**

```
Phase 1: 固定 compressor φ，训练 scheduler ψ
Phase 2: 固定 scheduler ψ，训练 compressor φ
重复直至收敛
```

- **优点**：每个子问题更稳定，类似 EM 算法
- **缺点**：可能不收敛到全局最优

**策略 3：条件 Scheduler（用户提议的方案）**

**核心思想**：Scheduler 不直接学习压缩策略，而是学习**如何根据 compressor 的状态调整策略**。

$$r(t) = f_{\psi}\big(t, \underbrace{\text{state}_{\text{compressor}}(t)}_{\text{压缩器状态}}\big)$$

其中 $\text{state}_{\text{compressor}}(t)$ 可以是：
- 压缩比率 $M/N$
- 重构损失 $L_{\text{recon}}$
- 压缩类型（Pooling / SDTM / ToMe / ...）
- 压缩器输出的统计量（均值、方差、熵）

**优点**：
- **Compressor 可换**：不同 compressor 无需重新训练 scheduler
- **Scheduler 通用**：一个 scheduler 适配多种 compressor
- **解耦设计**：compressor 和 scheduler 可以独立开发和验证

**类比**：这类似 Actor-Critic 架构：
- Compressor = Actor（执行压缩动作）
- Scheduler = Critic（评估并调度，基于 actor 的表现）

#### 22.3 推荐方案

**采用策略 3（条件 Scheduler）+ 策略 2（交替训练）的组合**：

1. **阶段 1**：固定一个简单 compressor（如 SDTM），训练条件 scheduler
2. **阶段 2**：固定 scheduler，微调 compressor 适配 scheduler 的调度
3. **阶段 3**：换用不同 compressor，验证 scheduler 的泛化能力

这样既保证了解耦的灵活性，又通过微调实现了协同优化。

#### 22.4 统一框架的最终形式

$$\boxed{T'_t = \underbrace{g_{\phi}(T_t, \text{pos})}_{\text{Compressor}} \cdot \underbrace{\sigma\Big(f_{\psi}\big(t, T, L_{\text{recon}}(t)\big)\Big)}_{\text{条件 Scheduler}}}$$

- $g_{\phi}$：权重生成函数（可退化为 ToMe / Poolingformer / SDTM / ...）
- $f_{\psi}$：调度函数（可退化为常数 / 线性 / 阶跃 / 可学习）
- $L_{\text{recon}}$：压缩器反馈信号（连接两个模块的桥梁）
- 训练方式：交替优化 $\phi$ 和 $\psi$，以 $L_{\text{recon}}$ 作为通信信道
