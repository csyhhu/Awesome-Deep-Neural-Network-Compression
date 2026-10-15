# My Re-implementation of Trained Ternary Quantization

## How to use it

### Specify your Dataset Root
Please refer to [here](../README.md#specify-your-dataset-root)

### Train a full-precision model

From `Codes/TTQ`, run `../train-base-model.py` from the `Codes` directory, specifying your model architecture and dataset:

```
cd ..
python train-base-model.py -m ResNet20 -d CIFAR10
cd TTQ
```
This generates `Codes/Results/model-dataset` (such as
`Codes/Results/ResNet20-CIFAR10`), which is `../Results/model-dataset` relative to `Codes/TTQ`, with the pretrained weights named `model-dataset-pretrain.pth`.

### Put utils in

```
ln -s ../utils ./utils
```

### Run the Codes

```
python TTQ.py -m ResNet20 -d CIFAR10 -tf 0.05
```

### Results Visualization
Specify the training log path you want to see in `visualize_training_log.py`
```
python visualize_training_log.py
```

## Experiment
All layers are ternarized:

| Model    | Dataset |  Quantized Acc | FP Acc |
| :-------:|:-------:|:-------------:|:--------:|
| ResNet20 | CIFAR10 | 90.3| 91.5|
| ResNet20 | STL10 | 69.825 | 70.3 |

### Training Log of ResNet20-STL10
![](./TTQ-ResNet20-STL10.png) 

## Customization
Build your own model with `nn.Conv2d`/`nn.Linear` replaced by `TTQ_CNN`/`TTQ_Linear`.

## Acknowledgement
This reimplementation is based on [Trained Ternary Quantization](https://arxiv.org/pdf/1612.01064v1.pdf).