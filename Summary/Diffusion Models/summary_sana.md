# Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers

- **论文链接**: [arXiv:2410.10629](https://arxiv.org/abs/2410.10629)
- **作者**: Enze Xie\*, Junsong Chen\*, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, Song Han
- **机构**: NVIDIA, MIT, 清华大学
- **发表**: ICLR 2025
- **代码/项目页**: [nvlabs.github.io/Sana](https://nvlabs.github.io/Sana)

---

## 1. 研究动机

当前主流文生图模型（如 SD3 8B、Flux 12B、Playground v3 24B）参数量不断膨胀，训练和推理成本极高，普通消费者难以使用。本文提出 Sana，目标是构建一个**高效、低成本、可在笔记本 GPU 上部署**的高质量高分辨率文生图框架（最大支持 4096×4096）。

---

## 2. 核心贡献

### 2.1 深度压缩自编码器（AE-F32C32）

- 传统自编码器压缩率为 8×（AE-F8），Sana 将其激进提升至 **32×（AE-F32C32）**，潜在 token 数量减少 16 倍
- 关键发现：让 AE 专注于高倍压缩、DiT 专注于去噪是最优设计（结论：F32C32P1 > F16C32P2 > F8C16P4）
- AE-F32C32 的重建质量媲美 SDXL 的 AE-F8C4

### 2.2 线性 DiT（Linear DiT）

- 将 DiT 中所有 vanilla self-attention **替换为 ReLU 线性注意力**，计算复杂度从 O(N²) 降至 O(N)
- 引入 **Mix-FFN**：在 MLP 中嵌入 3×3 深度可分离卷积，增强局部信息聚合能力，补偿线性注意力的不足
- **首次在 DiT 中完全移除位置编码（NoPE）**：3×3 卷积隐式提供了位置信息，且无性能损失
- 使用 Triton 进行算子融合加速

### 2.3 解码器架构小模型 LLM 作为文本编码器

- 用 **Gemma-2（解码器架构 LLM）** 替代 T5-XXL 作为文本编码器，推理速度提升 6 倍
- 解决 LLM 文本嵌入方差过大的训练不稳定问题：添加 RMSNorm + 可学习缩放因子
- 设计 **复杂人类指令（CHI, Complex Human Instruction）**：利用 LLM 的上下文学习能力增强图文对齐

### 2.4 高效训练与采样策略

- **多 VLM 自动标注流水线**：用 4 个 VLM（VILA-3B/13B、InternVL2-8B/26B）为图像生成多样化标注
- **基于 CLIP-Score 的标注采样器**：按概率采样高质量文本，引入温度参数控制
- **Flow-DPM-Solver**：基于 Rectified Flow 改造 DPM-Solver++，将采样步数从 28-50 步降至 14-20 步，同时质量更优
- **级联分辨率训练**：直接从 512px 开始训练，跳过 256px 预训练阶段

### 2.5 端侧部署

- W8A8 INT8 量化（per-token 激活 + per-channel 权重），关键层保留 FP16
- CUDA 算子融合（线性注意力 ReLU(K)^T×V 与 QKV 投影融合、GLU 与量化融合等）
- 在笔记本 4090 GPU 上生成 1024×1024 图像仅需 **0.37 秒**

---

## 3. 模型规格

| 模型 | 宽度 | 深度 | FFN | 头数 | 参数量 |
|------|------|------|-----|------|--------|
| Sana-0.6B | 1152 | 28 | 2880 | 36 | 590M |
| Sana-1.6B | 2240 | 20 | 5600 | 70 | 1.6B |

---

## 4. 主要实验结果

- **Sana-0.6B** 相比 Flux-12B：模型小 20 倍，吞吐量快 100+ 倍，多项指标竞争相当
- **4K 生成**：吞吐量比 Flux 快 100 倍以上
- **端侧推理**：W8A8 量化后，生成 1024px 图像仅需 0.37s（加速 2.4×），画质几乎无损
- 在 FID、Clip Score、GenEval、DPG-Bench、ImageReward 等指标上与 SOTA 方法竞争

---

## 5. 关键消融发现

- **线性注意力**：在 1024px 分辨率下略慢于原始注意力（Triton 融合后可略快），但在 4K 分辨率下加速 1.7 倍
- **AE-F32C32P1** 比 AE-F8C4P2 节省 4× MACs 和延迟
- **CHI** 可进一步提升图文对齐，尤其是对短 prompt（如 "a cat"）生成更稳定
- **NoPE** 完全移除位置编码不损失性能

---

## 6. 局限与展望

- 图像内容安全性和可控性无法完全保证
- 在文本渲染、人脸和手部生成等复杂场景仍有挑战
- 未来将基于 Sana 构建高效视频生成流水线
