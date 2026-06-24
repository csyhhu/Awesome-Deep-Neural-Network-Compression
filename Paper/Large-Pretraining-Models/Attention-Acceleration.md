# Attention Acceleration

> 讨论整理综述：[summary_attention_efficiency.md](../../Summary/Large%20Pretraining%20Models/summary_attention_efficiency.md)（FlashAttention、PagedAttention、Linear Attention / Linformer 分类对比）

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

## Low-Rank / Linear Complexity

- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)

# Token Compression
- Token Merging for Fast Stable Diffusion