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

---

## 7. 讨论与问答

### Q1: 介绍一下本文所用的深度压缩自编码器

Sana 的深度压缩自编码器（AE-F32C32）是整个高效流水线的基石，核心是将传统 8× 压缩激进提升至 32×。

**传统 LDM 的两阶段压缩**：传统方法（PixArt、SD3、Flux）采用 F=8 的 AE + P=2 的 DiT Patchify。以 1024×1024 图像为例，产生 (1024/16)² = 4096 个 token。Sana 的 AE-F32P1 产生 (1024/32)² = 1024 个 token，**token 数量减少 4 倍**。

**核心设计哲学**：自编码器应完全承担压缩任务，扩散模型只负责去噪。作者对比了三种将图像压缩到相同 token 数（32×32）的配置：

| 配置 | AE 压缩 | Patch | 重建 rFID | 生成 FID |
|------|---------|-------|-----------|----------|
| AE-F8C16P4 | 8× | 4 | 最好 | 最差 |
| AE-F16C32P2 | 16× | 2 | 中等 | 中等 |
| AE-F32C32P1 | 32× | 1 | 最差 | **最优** |

**反直觉结论**：AE 重建质量与最终生成质量并非正相关。将压缩任务集中在 AE 上（F32P1），虽然潜在空间更粗糙，但 DiT 可以纯粹专注于去噪，最终生成效果最好。

**通道数选择**（C=32）：C=16 重建太差，C=64 重建虽好但 DiT 训练收敛太慢，C=32 是最佳平衡点。

**与 SDv1.5 F32C64 对比**：SD 也曾尝试 32× 压缩，但 rFID 高达 0.82。Sana AE-F32C32 的 rFID 仅 0.34，逼近 SDXL F8C4 的 0.31，这得益于 AE 内部使用线性注意力 + 多阶段训练策略（先在低分辨率训练，再在 1024×1024 上微调）。

**Python 伪代码演示 AE 压缩 + Patchify（维度变化）**：

```python
import torch
import torch.nn as nn

B = 2                    # batch size
H, W = 1024, 1024        # input image resolution

# ============================================================
# 1. Traditional pipeline: AE-F8 + Patchify P=2 (e.g. PixArt, SD3, Flux)
# ============================================================

class TraditionalAutoEncoder(nn.Module):
    """Traditional F=8 autoencoder"""
    def encode(self, x):
        # x: (B, 3, 1024, 1024)
        x = self.conv_in(x)       # -> (B, 128, 1024, 1024)
        x = self.down_blocks(x)   # each block downsamples 2x
        # Block1: (B, 128, 512, 512)
        # Block2: (B, 256, 256, 256)
        # Block3: (B, 512, 128, 128)   # 1024 / 8 = 128
        x = self.conv_out(x)      # -> (B, 4, 128, 128)  C_out=4 or 16
        return x

class TraditionalPatchify(nn.Module):
    """Traditional P=2 patch embed"""
    def __init__(self, in_channels=4, embed_dim=1152, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        # flatten each patch and project to embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, z):
        # z: (B, 4, 128, 128)
        z = self.proj(z)           # -> (B, 1152, 64, 64)  128/2=64
        B, D, Hp, Wp = z.shape
        z = z.flatten(2)           # -> (B, 1152, 4096)
        z = z.transpose(1, 2)      # -> (B, 4096, 1152)   # 4096 tokens!
        return z

# Traditional pipeline end-to-end:
x = torch.randn(B, 3, H, W)                                # (B, 3, 1024, 1024)
z_trad = TraditionalAutoEncoder().encode(x)                # (B, 4,  128,  128)  spatial 8x compression
tokens_trad = TraditionalPatchify(in_channels=4).forward(z_trad)
# => (B, 4096, 1152)   # 4096 = (128/2) x (128/2) = 64x64


# ============================================================
# 2. Sana pipeline: AE-F32 + Patchify P=1
# ============================================================

class SanaAutoEncoder(nn.Module):
    """Sana F=32 deep compression autoencoder"""
    def encode(self, x):
        # x: (B, 3, 1024, 1024)
        x = self.conv_in(x)       # -> (B, 128, 1024, 1024)
        x = self.down_blocks(x)   # each block downsamples 2x
        # Block1: (B, 128, 512, 512)
        # Block2: (B, 256, 256, 256)
        # Block3: (B, 512, 128, 128)
        # Block4: (B, 512, 64,  64)
        # Block5: (B, 512, 32,  32)    # 1024 / 32 = 32
        x = self.conv_out(x)      # -> (B, 32, 32, 32)   C_out=32
        return x

class SanaPatchify(nn.Module):
    """Sana P=1 pointwise projection (no spatial merging)"""
    def __init__(self, in_channels=32, embed_dim=1152, patch_size=1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, z):
        # z: (B, 32, 32, 32)
        z = self.proj(z)           # -> (B, 1152, 32, 32)  32/1=32
        B, D, Hp, Wp = z.shape
        z = z.flatten(2)           # -> (B, 1152, 1024)
        z = z.transpose(1, 2)      # -> (B, 1024, 1152)   # only 1024 tokens!
        return z

# Sana pipeline end-to-end:
x = torch.randn(B, 3, H, W)                                # (B, 3, 1024, 1024)
z_sana = SanaAutoEncoder().encode(x)                       # (B, 32, 32,   32)  spatial 32x compression
tokens_sana = SanaPatchify(in_channels=32).forward(z_sana)
# => (B, 1024, 1152)  # 1024 = (32/1) x (32/1) = 32x32


# ============================================================
# 3. Summary: token count comparison
# ============================================================
print("=" * 55)
print(f"{'Pipeline':<20} {'AE Output':<15} {'#Tokens':<12}")
print("-" * 55)
print(f"{'Traditional F8P2':<20} {'(B,4,128,128)':<15} {'(B, 4096, D)':<12}")
print(f"{'Sana F32P1':<20} {'(B,32,32,32)':<15} {'(B, 1024, D)':<12}")
print("-" * 55)
print(f"Token reduction: 1024 / 4096 = 1/4, i.e. 75% less")
print("=" * 55)

# ============================================================
# 4. 4K resolution (4096x4096) — advantage is even larger
# ============================================================
H4k, W4k = 4096, 4096
# Traditional F8P2: (4096/16) x (4096/16) = 256 x 256 = 65536 tokens
# Sana F32P1:      (4096/32) x (4096/32) = 128 x 128 = 16384 tokens
print(f"\n4K scenario (4096x4096):")
print(f"  Traditional F8P2: {(4096//16)**2:,} tokens  (256x256)")
print(f"  Sana F32P1:       {(4096//32)**2:,} tokens  (128x128)")
print(f"  => reduction to {(4096//32)**2 / (4096//16)**2:.1%}")
# => 16384 / 65536 = 25.0%, i.e. 75% less
```

### Q2: down_blocks 一般是怎么实现的？Sana 和传统 encoder 比较只是采样更大吗？

**down_block 的典型实现**（以 SDXL/SD3 等主流 LDM 为例）：

每个 down_block 通常包含以下组件组合：

```python
class DownBlock(nn.Module):
    """
    A typical downsampling block in an LDM autoencoder.
    Consists of: ResNet blocks + optional self-attention + downsampling.
    """
    def __init__(self, in_channels, out_channels, num_resblocks=2,
                 use_attn=False, num_heads=4, downsample=True):
        super().__init__()
        # 1. Stack of ResNet blocks (GroupNorm + SiLU + Conv2d + residual)
        self.resblocks = nn.ModuleList([
            ResnetBlock2D(in_channels if i == 0 else out_channels,
                          out_channels)
            for i in range(num_resblocks)
        ])
        # 2. Optional vanilla self-attention at certain resolutions
        self.attn = nn.ModuleList([
            # SelfAttention: QKV projection + softmax(QK^T/sqrt(d))V
            nn.MultiheadAttention(out_channels, num_heads,
                                  batch_first=True)
            for _ in range(num_resblocks)
        ]) if use_attn else None
        # 3. Downsampling via stride-2 convolution
        self.downsample = (nn.Conv2d(out_channels, out_channels,
                                     kernel_size=3, stride=2, padding=1)
                           if downsample else nn.Identity())

    def forward(self, x):
        # x: (B, in_c, H, W)
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x)                         # ResNet block
            if self.attn is not None:
                B, C, H_, W_ = x.shape
                x_flat = x.flatten(2).transpose(1,2)  # (B, H_*W_, C)
                x_flat = self.attn[i](x_flat, x_flat, x_flat)  # self-attn
                x = x_flat.transpose(1,2).view(B, C, H_, W_)
        x = self.downsample(x)                      # stride-2 → H/2, W/2
        return x
```

**传统 encoder (F=8) 的完整结构**：

```
                  input: (B, 3, 1024, 1024)
                       │
    conv_in ─────────────────────────────────────► (B, 128, 1024, 1024)
                       │
    DownBlock_1 (ResNet×2, no attn, stride2) ────► (B, 128, 512, 512)
    DownBlock_2 (ResNet×2, attn,     stride2) ────► (B, 256, 256, 256)
    DownBlock_3 (ResNet×2, attn,     stride2) ────► (B, 512, 128, 128)
                       │
    MidBlock   (ResNet×2, attn) ─────────────────► (B, 512, 128, 128)
                       │
    conv_out ────────────────────────────────────► (B, 4,   128, 128)
```

**Sana 和传统 encoder 的区别绝不只是"堆更多采样层"**，有以下四个本质差异：

| 维度 | 传统 AE (SDXL F8C4) | Sana AE (F32C32) |
|------|---------------------|-------------------|
| **下采样层数** | 3 层 → 8× 压缩 | 5 层 → 32× 压缩 |
| **自注意力类型** | vanilla softmax attention (O(N²)) | **linear attention** (O(N)) |
| **输出通道数** | C=4 | C=32 |
| **训练策略** | 单阶段训练 | **多阶段训练**：先低分辨预训练，再在 1024² 上微调 |

**关键差异 1：线性注意力替代普通自注意力**

论文原文明确指出：*"We replace the vanilla self attention mechanism with linear attention blocks."* 即 Sana 的 AE 内部也做了结构优化 — 在高分辨率特征图（如 512×512 中间层）上，vanilla attention 的计算量是 (512²)² = 68B，使用线性注意力可降至 O(N)，既提效又降低显存。

**关键差异 2：多阶段训练策略**

仅靠堆层数是不够的。SDv1.5 也尝试过 F32C64（rFID 高达 0.82），证明纯架构改动无法弥补质量损失。Sana 的做法是：
- 先在较低分辨率（如 256²）上预训练 AE，让模型学会基础压缩
- 再在 1024² 分辨率上微调，让模型适应高分辨率的细节重建

**关键差异 3：输出通道数的权衡**

传统 F=8 AE 只用 C=4 或 C=16 就足够，因为空间分辨率还较高（128×128）。但对 F=32 来说，空间只有 32×32，必须用更大的 C 来保留足够的信息容量 — 但又不能太大（C=64 会让 DiT 训练收敛显著变慢），所以 C=32 是最优平衡。

**总结**：Sana 的 AE 不只是"多采样两层"，而是**结构（线性注意力）+ 通道策略（C=32）+ 训练策略（多阶段）**三方面的系统改进，三者缺一不可。

### Q3: Encoder 阶段就会用 attention 了吗？

是的，**AE encoder 阶段确实会使用 self-attention**，这是主流 LDM 自编码器的标准设计。Attention 通常出现在分辨率较低的 down_block 和中间瓶颈层：

```
传统 AE (F=8) 的结构:
DownBlock_1 (1024² → 512²):  ResNet × 2, no attn      ← 分辨率太高不用
DownBlock_2 (512²  → 256²):  ResNet × 2, + self-attn   ← 开始引入
DownBlock_3 (256²  → 128²):  ResNet × 2, + self-attn   ← 语义建模
MidBlock    (128²):           ResNet × 2, + self-attn   ← 全局建模核心
```

**为什么在高分辨率层不用 attention？** 因为 vanilla attention 的复杂度是 O(N²)。以 512² 的特征图为例，N = 512² = 262K tokens，一次 self-attention 就需要 68G 的计算量，显存和延迟都不可接受。

**Sana 的创新**在于将这些 vanilla self-attention 全部替换为**线性注意力**（O(N) 复杂度）。由于 Sana F=32 的 AE 层数更多、中间特征图分辨率更高（如 256²），换成线性注意力能有效控制 AE 自身的计算开销。这一点与 Sana 在 DiT 中"全线替换线性注意力"的思路是一脉相承的。

---

### Q4: DiT 架构上，Sana 有什么创新？

Sana 在 DiT 架构上有 **四个核心创新**，贯穿了从 token 输入、注意力机制、FFN 到位置编码的整个计算图：

```python
# ============================================================
# Sana DiT Block — 创新全览
# ============================================================

class SanaDiTBlock(nn.Module):
    """A single Sana DiT block showing all architectural innovations."""
    def __init__(self, dim, num_heads, mlp_ratio=2.5):
        super().__init__()
        # ── 创新 1: Patchify P=1 (AE 承担全部压缩, DiT 只负责去噪) ──
        # Token 数仅为传统 DiT 的 1/4

        # ── 创新 2: Linear Attention 替代 Vanilla Attention ──
        self.norm1 = nn.LayerNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim * 3)
        # vanilla attention: O_i = softmax(Q_i K^T / sqrt(d)) V  → O(N^2)
        # Sana linear attn:  O_i = ReLU(Q_i)(Σ ReLU(K_j)^T V_j) / ReLU(Q_i)(Σ ReLU(K_j)^T) → O(N)
        self.out_proj = nn.Linear(dim, dim)

        # ── 创新 3: Mix-FFN (MLP + 3×3 depthwise conv + GLU) ──
        # vanilla FFN:    Linear → GELU → Linear
        # Sana Mix-FFN:   Linear → GELU → 3×3 DWConv → GLU gating
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn_linear1 = nn.Linear(dim, hidden_dim * 2)  # GLU splits in half
        self.ffn_dwconv = nn.Conv2d(hidden_dim, hidden_dim,
                                     kernel_size=3, padding=1, groups=hidden_dim)
        self.ffn_linear2 = nn.Linear(hidden_dim, dim)

        # ── 创新 4: NoPE — 无位置编码 ──
        # 没有 sin/cos PE, 没有 learnable PE, 没有 RoPE
        # 3×3 DWConv 隐式提供了位置信息

    def forward(self, x, c):
        """
        x: (B, N, D)  latent tokens  (N=1024 for 1024² image with F32P1)
        c: (B, Nc, D) text embeddings
        """
        # --- Linear Attention (创新 2) ---
        x_norm = self.norm1(x)
        qkv = self.qkv_proj(x_norm)        # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        # ReLU-based linear attention: compute shared KV once, reuse for all Q
        kv = torch.einsum('bnd,bne->bde',  # (B, D, D) — computed only once!
                          torch.relu(k), v)
        k_sum = torch.relu(k).sum(dim=1)    # (B, D) — computed only once!
        num = torch.einsum('bnd,bde->bne', torch.relu(q), kv)
        den = torch.einsum('bnd,bd->bn', torch.relu(q), k_sum).unsqueeze(-1) + 1e-6
        attn_out = self.out_proj(num / den)  # O(N) complexity!
        x = x + attn_out

        # --- Mix-FFN (创新 3) ---
        x_norm2 = self.norm2(x)
        ffn_x = self.ffn_linear1(x_norm2)   # -> (B, N, 2 * hidden_dim)
        gate, ffn_x = ffn_x.chunk(2, dim=-1)
        # Reshape for spatial depthwise conv
        B_sq = int(math.sqrt(ffn_x.shape[1]))  # 32 for 1024², 64 for 2048²
        ffn_x = ffn_x.reshape(B, B_sq, B_sq, -1).permute(0, 3, 1, 2)
        ffn_x = self.ffn_dwconv(ffn_x)          # 3×3 DWConv → 聚合局部信息
        ffn_x = ffn_x.permute(0, 2, 3, 1).reshape(B, -1, ffn_x.shape[1])
        ffn_out = self.ffn_linear2(ffn_x * F.gelu(gate))  # GLU gating
        x = x + ffn_out

        return x  # No positional encoding added anywhere!
```

下面逐一展开四个创新点：

---

**创新 1：Patchify P=1 — 职责分离**

| | 传统 DiT (PixArt, SD3) | Sana DiT |
|---|---|---|
| Token 来源 | AE 压缩 8× + Patch 合并 2× | **AE 全权压缩 32×** |
| Patch size | P=2 | **P=1** |
| 1024² → tokens | 4096 | **1024** |
| 设计哲学 | 压缩任务分散 | **单向压缩流** |

这是架构层面的关键决策：DiT 不再参与压缩，每个 latent 像素直接是一个 token。消融实验证实，虽然 AE-F32 重建质量略差于 AE-F8，但生成质量反而更好 — 因为 DiT 可以**纯粹专注于去噪**。

---

**创新 2：Linear Attention — O(N²) → O(N)**

```
Vanilla Self-Attention:
    O_i = Σ_j softmax(Q_i K_j^T / sqrt(d)) V_j    ← 对每个 i 都要和所有 j 算相似度
    复杂度: O(N²d)

Sana ReLU Linear Attention:
    O_i = ReLU(Q_i) (Σ_j ReLU(K_j)^T V_j) / ReLU(Q_i) (Σ_j ReLU(K_j)^T)
          ~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~    ~~~~~~~~~~~~~~~~~~~~~
            逐 query      共享项(只算一次!)           归一化项(只算一次!)
    复杂度: O(Nd²)

    当 N >> d 时 (高分辨率生成, N=16384 在 4K 场景), O(Nd²) << O(N²d)
    实测 4K 生成加速 1.7×
```

注意这里的权衡：线性注意力用特征值的 ReLU 非线性替代了 softmax 的成对相似度计算，**失去了 token 间精确的交互建模**。这就是为什么需要 Mix-FFN 来弥补。

---

**创新 3：Mix-FFN — 补足线性注意力的局部建模短板**

```
vanilla FFN:
    x → Linear → GELU → Linear → output

Sana Mix-FFN:
    x → Linear → 3×3 DWConv → GLU → Linear → output
                  ~~~~~~~~~~   ~~~
                  局部信息      门控
```

- **3×3 Depthwise Conv**：线性注意力缺乏 softmax 的精确 token 交互，DWConv 通过局部邻域聚合来补偿
- **GLU (Gated Linear Unit)**：引入门控机制，提升非线性表达能力
- **副作用**：由于 3×3 卷积 + zero padding 本身"携带"了隐式位置信息，使得 NoPE 成为可能

---

**创新 4：NoPE — 首次在 DiT 中完全移除位置编码**

| 方法 | 位置编码方式 |
|------|-------------|
| DiT (Peebles) | 可学习绝对 PE |
| PixArt | 可学习绝对 PE |
| SD3 / Flux | RoPE (旋转位置编码) |
| **Sana** | **无位置编码 (NoPE)** |

论文中表示"we are surprised that we can remove Positional Embedding without any loss in performance"。原理：
- 3×3 DWConv (zero padding) 隐式编码了像素的空间位置关系
- 与 LLM 领域 NoPE 的最新发现一致（`Kazemnejad et al. 2024`），NoPE 可能具有更好的长度泛化能力
- 2K/4K 微调时重新引入 PE 做插值，但 1K 训练完全不需 PE

---

**四个创新之间的关系**：

```
Patchify P=1 ──→ Token 数减少 4× ──→ DiT 计算量大幅下降
                                         │
Linear Attn ───→ O(N²)→O(N) ────────────┤ 双重加速
                                         │
Mix-FFN ──────→ 补足线性注意力的局部建模 ─┤ 保证质量
                推动 NoPE 成为可能 ───────┘
```

这四个设计不是孤立的，而是互相耦合的系统：P=1 减少 token 数让我们敢用更整体的计算；线性注意力消除了 O(N²) 瓶颈；Mix-FFN 弥补线性注意力的能力损失同时推动 NoPE，形成了**设计闭环**。

### Q5: 综合理解 — Sana 是否采用了更激进的 encoder + 所有 Attention 做 Linear Attention + MQA？

**用户综合观点**：Sana 主要是采用了更激进的 encoder，以及对所有的 Attention 都做了 Linear Attention + MQA。

**澄清与纠正**：

- ✅ **"更激进的 encoder"** — 正确。AE-F32C32（32× 压缩），AE 内部也是线性注意力。
- ✅ **"所有 Attention 都做了 Linear Attention"** — 正确。AE 中的 self-attention 和 DiT 中的 self-attention 全部替换为 ReLU linear attention。
- ❌ **"MQA"** — **Sana 论文中并未使用 MQA**。全文搜索 "MQA / multi-query / GQA" 结果为 0。

**关键区分：Linear Attention ≠ MQA**

| | Linear Attention (Sana 实际用的) | MQA (Multi-Query Attention) |
|---|---|---|
| **是否多头** | 仍是多头（Sana-0.6B 有 36 head，1.6B 有 70 head） | 多 Q head 共享 1 组 K/V head |
| **每个 head 内部计算** | 每头独立做 ReLU 线性注意力：`ReLU(Q)(Σ ReLU(K)ᵀV) / ReLU(Q)(Σ ReLU(K)ᵀ)` | 每头仍做标准 softmax 注意力 |
| **核心目的** | 改变计算顺序，将复杂度从 O(N²) 降到 O(N) | 减少 KV cache 显存，不改变计算量 |
| **复杂度** | O(Nd²) | 仍是 O(N²d) |

补充：用户提问「多个 Q head 共享一组 K、V head 不就是 MQA 吗？」—— **该定义完全正确**，这正是 MQA 的标准定义（如 PaLM 所用）。但 Sana 实际做的是「多头 + 每头线性化」，不是「多 Q 头共享 K/V」。两者都能省资源，但路径不同：Linear Attention 省的是 FLOPs（改变计算顺序），MQA 省的是 KV cache 内存（减少参数量）。Sana 通过 Linear Attention 实现了 O(N) 复杂度，与 MQA 是两套独立的技术路线。
