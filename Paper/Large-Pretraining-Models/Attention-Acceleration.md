# Attention Acceleration

> 讨论整理综述：[summary_attention_efficiency.md](../../Summary/Large%20Pretraining%20Models/summary_attention_efficiency.md)（FlashAttention、PagedAttention、Linear Attention / Linformer 分类对比）

## Attention Approximation
- [WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](../../Summary/Diffusion%20Models/summary_worldcache.md)

## Linear Attention
Linear attention aims at reducing computation complexity in attention from $O(N^2)$ to $O(N)$:

- Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
- Transformer Dissection: A Unified Understanding of Transformer's Attention via the Lens of Kernel
- Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
- Exact Conversion of In-Context Learning to Model Weights in Linearized-Attention Transformers


- TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer (Lightning Attention-1)
- Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models


- Attention-Free Transformers
- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)

## Attention Optimization (Exact Attention)

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM)
- FlashTransformers
- I/O Complexity of Attention, or How Optimal is FlashAttention?


## Block Sparse Attention 
- [SVOO: Attention Sparsity is Input-Stable —— Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](../../Summary/Diffusion%20Models/summary_svoo_training_free_sparse_attention.md)
- [DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](../../Summary/Diffusion%20Models/summary_dfsattn_sparse_attention.md)
- [VEDA: Scalable Video Diffusion via Distilled Sparse Attention](../../Summary/Diffusion%20Models/summary_veda_sparse_attention.md)

## Low-Rank / Linear Complexity
- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)

# Token Compression
- Token Merging for Fast Stable Diffusion