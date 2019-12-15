# 2019

## NeurIPS

- Efficient and Effective Quantization for Sparse DNNs
- Focused Quantization for Sparse CNNs [[paper]](https://papers.nips.cc/paper/8796-focused-quantization-for-sparse-cnns.pdf)
- Point-Voxel CNN for Efficient 3D Deep Learning [[paper]](https://arxiv.org/abs/1907.03739)
- Model Compression with Adversarial Robustness: A Unified Optimization Framework [[paper]](https://arxiv.org/abs/1902.03538)

### Quantization
- MetaQuant: Learning to Quantize by Learning to Penetrate Non-differentiable Quantization [[paper]](https://github.com/csyhhu/MetaQuant/blob/master/MetaQuant-Preprint.pdf) [[codes]](https://github.com/csyhhu/MetaQuant)
- Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization [[paper]](https://papers.nips.cc/paper/8971-latent-weights-do-not-exist-rethinking-binarized-neural-network-optimization.pdf)

#### Post Quantization
- Post-training 4-bit quantization of convolution networks for rapid-deployment


#### Gradient Compression
- Qsparse-local-SGD: Distributed SGD with Quantization, Sparsification, and Local Computations [[paper]](https://arxiv.org/pdf/1906.02367.pdf)
- PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization [[paper]](https://arxiv.org/pdf/1905.13727.pdf)


### Pruning
- AutoPrune: Automatic Network Pruning by Regularizing Auxiliary Parameters


#### Unstructure Pruning
- Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask
- Global Sparse Momentum SGD for Pruning Very
Deep Neural Networks [[paper]](https://arxiv.org/pdf/1909.12778.pdf)[[codes]](https://github.com/DingXiaoH/GSM-SGD)

#### Structrue Pruning
- Channel Gating Neural Network
- Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks

### Distillation
- Positive-Unlabeled Compression on the Cloud [[paper]](https://arxiv.org/abs/1909.09757)


### Factorization
- Einconv: Exploring Unexplored Tensor Decompositions for Convolutional Neural Networks [[paper]](https://arxiv.org/abs/1908.04471) [[codes]](https://github.com/pfnet-research/einconv)
- A Tensorized Transformer for Language Modeling [[paper]](https://arxiv.org/pdf/1906.09777.pdf)


### Efficient Model Design
- Shallow RNN: Accurate Time-series Classification on Resource Constrained Devices [[paper]](https://papers.nips.cc/paper/9451-shallow-rnn-accurate-time-series-classification-on-resource-constrained-devices.pdf)
- CondConv: Conditionally Parameterized Convolutions for Efficient Inference [[paper]](https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf)


### Dynamic Inference
- SCAN: A Scalable Neural Networks Framework Towards Compact and Efficient Models [[paper]](https://papers.nips.cc/paper/8657-scan-a-scalable-neural-networks-framework-towards-compact-and-efficient-models.pdf)


### Neural Architecture Search
- Constrained deep neural network architecture search for IoT devices accounting hardware calibration [[paper]](https://papers.nips.cc/paper/8838-constrained-deep-neural-network-architecture-search-for-iot-devices-accounting-for-hardware-calibration.pdf)
- DATA: Differentiable ArchiTecture Approximation [[paper]](https://papers.nips.cc/paper/8374-data-differentiable-architecture-approximation.pdf)
- Efficient Forward Architecture Search [[paper]](https://arxiv.org/pdf/1905.13360.pdf)

### Cost (Energy, Memory, Time) Saving Training
- Hybrid 8-bit Floating Point (HFP8) Training and
Inference for Deep Neural Networks [[paper]](https://papers.nips.cc/paper/8736-hybrid-8-bit-floating-point-hfp8-training-and-inference-for-deep-neural-networks.pdf)
- E2-Train: Training State-of-the-art CNNs with Over
80% Energy Savings [[paper]](https://arxiv.org/pdf/1910.13349.pdf)
- Backprop with Approximate Activations for Memory-efficient Network Training [[paper]](https://arxiv.org/pdf/1901.07988.pdf)


### Theory
- Dimension-Free Bounds for Low-Precision Training [[paper]](https://papers.nips.cc/paper/9346-dimension-free-bounds-for-low-precision-training.pdf)
- A Mean Field Theory of Quantized Deep Networks:
The Quantization-Depth Trade-Off [[paper]](https://arxiv.org/pdf/1906.00771.pdf)


## CVPR

- Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss
- Simultaneously Optimizing Weight and Quantizer of Ternary Neural Network using Truncated Gaussian Approximation
- Structured Pruning of Neural Networks with Budget-Aware Regularization
- Towards Optimal Structured CNN Pruning via Generative Adversarial Learning
- Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure
- Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration
- ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Model
- Cascaded Projection: End-to-End Network Compression and Acceleration
- Accelerating Convolutional Neural Networks via Activation Map Compression
- Energy-Constrained Compression for Deep Neural Networks via Weighted Sparse Projection and Layer Input Masking
- Factorized Convolutional Neural Networks
- Exploiting Kernel Sparsity and Entropy for Interpretable CNN Compression
- A Main/Subsidiary Network Framework for Simplifying Binary Neural Networks
- Binary Ensemble Neural Network: More Bits per Network or More Networks per Bit?
- Cross Domain Model Compression by Structurally Weight Sharing

## ICML

- Improving Neural Network Quantization without Retraining using Outlier Channel Splitting
- Same, Same But Different-Recovering Neural Network Quantization Error Through Weight Factorization
- Parameter Efficient Training of Deep Convolutional Neural Networks by Dynamic Sparse Reparameterization
- Variational inference for sparse network reconstruction from count data
- Collaborative Channel Pruning for Deep Networks

## ICLR
- Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets
- Minimal Random Code Learning: Getting Bits back from Compressed Model Parameters

# 2018

## NeurIPS

* Scalable Methods for 8-bit Training of Neural Networks
* Heterogeneous Bitwidth Binarization in Convolutional Neural Networks
* HitNet: Hybrid Ternary Recurrent Neural Network

## ICML
- WSNet: Compact and Efficient Networks Through Weight Sampling

## ICLR

* Espresso: Efficient Forward Propagation for BCNNs
* An Empirical study of Binary Neural Networks' Optimisation
* [Learning Discrete Weights Using the Local Reparameterization Trick](https://arxiv.org/abs/1710.07739)
* On the Universal Approximability and Complexity Bounds of Quantized ReLU Neural Networks
* Learning To Share: Simultaneous Parameter Tying and Sparsification in Deep Learning

## ECCV

* Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm
* Value-aware Quantization for Training and Inference of Neural Networks
* LSQ++: Lower running time and higher recall in multi-codebook quantization
* LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks



## CVPR ##

- NISP: Pruning Networks using Neuron Importance Score Propagation
- SYQ: Learning Symmetric Quantization For Efficient Deep Neural Networks



## IJCAI ##

- Improving Deep Neural Network Sparsity through Decorrelation Regularization



# 2016

## ICML

* Fixed Point Quantization of Deep Convolutional Networks
* 





# 2014

## NIPS

* Expectation Backpropagation: Parameter-Free Training of Multilayer Neural Networks with Continuous or Discrete Weights