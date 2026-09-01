# Token Skip

## Motivation
Given a pre-trained DiT model, generate a **token skip cost map** for inference with token skip under different budget.

## Related Work
- [Probe and Skip: Self-Predictive Token Skipping for Efficient Long-Context LLM Inference](../Summary/Large%20Pretraining%20Models/summary_self_predictive_token_skipping.md): Light attention scores for attention token skip. Re-run for FFN token skip.
- [CoReDiT: Spatial Coherence-Guided Token Pruning and Reconstruction for Efficient Diffusion Transformers](../Summary/Diffusion%20Models/summary_coredit_spatial_coherence_pruning.md): Similarity in a small patch for recovery as a proxy for token importance.
 