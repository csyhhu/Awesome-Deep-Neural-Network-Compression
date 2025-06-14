# Quantization

* BinaryConnect: Training Deep Neural Networks with binary weights during propagations
* Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM
* [Loss-aware Binarization of Deep Networks](https://arxiv.org/abs/1611.01600)
* [Loss-aware Weight Quantization of Deep Networks](https://arxiv.org/abs/1802.08635)
* Overcoming Challenges in Fixed Point Training of Deep Convolutional Networks
* CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization
* Two-Step Quantization for Low-bit Neural Networks
* [Training deep neural networks with low precision multiplications](https://arxiv.org/abs/1412.7024)
* Explicit Loss-Error-Aware Quantization for Low-Bit Deep Neural Networks
* Value-aware Quantization for Training and Inference of Neural Networks
* Training Competitive Binary Neural Networks from Scratch
* Linear Symmetric Quantization of Neural Networks for Low-precision Integer Hardware 
- AutoQ: Automated Kernel-Wise Neural Network Quantization 
- Additive Powers-of-Two Quantization: A Non-uniform Discretization for Neural Networks
- Learned Step Size Quantization
- Sampling-Free Learning of Bayesian Quantized Neural Networks
- Gradient $\ell_1$ Regularization for Quantization Robustness
- BinaryDuo: Reducing Gradient Mismatch in Binary Activation Network by Coupling Binary Activations 
- Training binary neural networks with real-to-binary convolutions 
- Critical initialisation in continuous approximations of binary neural networks 
- Trained Ternary Quantization [[code]](https://github.com/czhu95/ternarynet)
- MetaQuant: Learning to Quantize by Learning to Penetrate Non-differentiable Quantization [[paper]](https://github.com/csyhhu/MetaQuant/blob/master/MetaQuant-Preprint.pdf) [[codes]](https://github.com/csyhhu/MetaQuant)
- Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization [[paper]](https://papers.nips.cc/paper/8971-latent-weights-do-not-exist-rethinking-binarized-neural-network-optimization.pdf)
- Fully Quantized Network for Object Detection
- Learning to Quantize Deep Networks by Optimizing Quantization Intervals With Task Loss
- Quantization Networks
- SeerNet: Predicting Convolutional Neural Network Feature-Map Sparsity Through Low-Bit Quantization
- Simultaneously Optimizing Weight and Quantizer of Ternary Neural Network Using Truncated Gaussian Approximation
- Binary Ensemble Neural Network: More Bits per Network or More Networks per Bit?
- A Main/Subsidiary Network Framework for Simplifying Binary Neural Networks
- Regularizing Activation Distribution for Training Binarized Deep Networks
- Structured Binary Neural Networks for Accurate Image Classification and Semantic Segmentation
- Learning Channel-Wise Interactions for Binary Convolutional Neural Networks
- Circulant Binary Convolutional Networks: Enhancing the Performance of 1-Bit DCNNs With Circulant Back Propagation
- Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks
- Proximal Mean-Field for Neural Network Quantization
- Relaxed Quantization for Discretized Neural Networks
- ProxQuant: Quantized Neural Networks via Proximal Operators 
- Per-Tensor Fixed-Point Quantization of the Back-Propagation Algorithm
- Defensive Quantization: When Efficiency Meets Robustness
- Learning Recurrent Binary/Ternary Weights 
- Combinatorial Attacks on Binarized Neural Networks
- ARM: Augment-REINFORCE-Merge Gradient for Stochastic Binary Networks 
- An Empirical study of Binary Neural Networks' Optimisation 
- Integer Networks for Data Compression with Latent-Variable Models 

## Weights & Activation Quantization

* Quantized Neural Networks Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations
* Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1
* XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks [[codes]](https://github.com/jiecaoyu/XNOR-Net-PyTorch)
* Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm
* LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks

## Weights & Activation & Error & Gradient  Quantization

* DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
* Training and Inference with Integers in Deep Neural Networks

## Activation Quantization

- PACT: Parameterized Clipping Activation for Quantized Neural Networks

## Non-symmetric Quantization

* Weighted-Entropy-based Quantization for Deep Neural Networks
* Towards the Limit of Network Quantization
* LSQ++: Lower running time and higher recall in multi-codebook quantization

## Probabilistic (Bayesian) Quantization

* Relaxed Quantization for Discretized Neural Networks
* [Learning Discrete Weights Using the Local Reparameterization Trick](https://arxiv.org/abs/1710.07739)
* Expectation Backpropagation: Parameter-Free Training of Multilayer Neural Networks with Continuous or Discrete Weights

## Quantization Theory

* Analysis of Quantized Models
* On the Universal Approximability and Complexity Bounds of Quantized ReLU Neural Networks
- Dimension-Free Bounds for Low-Precision Training [[paper]](https://papers.nips.cc/paper/9346-dimension-free-bounds-for-low-precision-training.pdf)
- A Mean Field Theory of Quantized Deep Networks: The Quantization-Depth Trade-Off [[paper]](https://arxiv.org/pdf/1906.00771.pdf)
- On the Universal Approximability and Complexity Bounds of Quantized ReLU Neural Networks 
- Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets

## Quantization with Distillation

* [Model compression via distillation and quantization](https://arxiv.org/abs/1802.05668)


## Adaptive Quantization

* [Adaptive Quantization of Neural Networks](https://openreview.net/forum?id=SyOK1Sg0W)


## Gradient Quantization

* SIGNSGD: compressed optimisation for non-convex problems
* QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding
- Qsparse-local-SGD: Distributed SGD with Quantization, Sparsification, and Local Computations [[paper]](https://arxiv.org/pdf/1906.02367.pdf)


## Relaxed Quantization

* Incremental Network Quantization [[code]](https://github.com/AojunZhou/Incremental-Network-Quantization/tree/master/src/caffe)
* BinaryRelax: A Relaxation Approach For Training Deep Neural Networks With Quantized Weights
* ProxQuant: Quantized Neural Networks via Proximal Operators
* Self-Binarizaing Networks

## Specified Application
* Quantized Convolutional Neural Networks for Mobile Devices


## Co-Design (Hardware/Energy/Memory)
* Espresso: Efficient Forward Propagation for BCNNs
- HAQ: Hardware-Aware Automated Quantization With Mixed Precision
- ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Model
- Double Viterbi: Weight Encoding for High Compression Ratio and Fast On-Chip Reconstruction for Deep Neural Network 
- Energy-Constrained Compression for Deep Neural Networks via Weighted Sparse Projection and Layer Input Masking 

## Data-Free / Post-training Quantization
* Data-Free Quantization through Weight Equalization and Bias Correction
* Quantizing deep convolutional networks for efficient inference: A whitepaper
* Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
- Post-training 4-bit quantization of convolution networks for rapid-deployment
- Improving Neural Network Quantization without Retraining using Outlier Channel Splitting
- Same, Same But Different: Recovering Neural Network Quantization Error Through Weight Factorization

## Parameterized Quantizer
- Learned Step Size Quantization
- Mixed Precision DNNs: All you need is a good parametrization

## Unclassified
* Scalable Methods for 8-bit Training of Neural Networks
* Efficient end-to-end learning for quantizable representations ([code](https://github.com/maestrojeong/Deep-Hash-Table-ICML18))
* Network Sketching: Exploiting Binary Structure in Deep CNNs
* DNQ: Dynamic Network Quantization
* Training Hard-Threshold Networks with Combinatorial Search in a Discrete Target Propagation Setting
* An Empirical study of Binary Neural Networks' Optimisation
* Heterogeneous Bitwidth Binarization in Convolutional Neural Networks
* HitNet: Hybrid Ternary Recurrent Neural Network