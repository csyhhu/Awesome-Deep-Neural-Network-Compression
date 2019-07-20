# My Re-implementation of Trained Ternary Quantization

## How to use it

### Specify your Dataset Root
Please refer to [here](./README.md)

### Train a full-precision model

Run `train_base_model.py` by specifying your model architecture and dataset used. Such as:

```
python train_base_model.py -m ResNet20 -d CIFAR10
```

Normally, I will train a fp model under 3 different learning rate: 1e-1, 1e-2, 1e-3:

```
python train_base_model.py -m ResNet20 -d CIFAR10
python train_base_model.py -m ResNet20 -d CIFAR10 --resume --lr=0.01
python train_base_model.py -m ResNet20 -d CIFAR10 --resume --lr=0.001
```

By using `train_base_model.py`, it will generate a folder named as `../Results/model-dataset` (such as 
`../Results/ResNet20-CIFAR10`) in the upper level folder, with the pretrain network named as `model-dataset-pretrained.pth`

### Run the Codes

```
python TTQ.py -m ResNet20 -d CIFAR10 -tf 0.05
```

### Results Visualization