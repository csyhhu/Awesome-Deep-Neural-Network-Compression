# My Re-Implementation of Dynamic Channel Pruning: Feature Boosting and Suppression

Under Improvement

## How to use it

### Specify your Dataset Root
Please refer to [here](../../Codes)

### Train a full-precision model

Please refer to [here](../TTQ#train-a-full-precision-model)

### Put utils in

```
ln -s ../utils ./utils
```

### Run the Codes

```
python FBS.py -m ResNet20 -d CIFAR10 -CR 0.5
```

## Experiment
| Model    | Dataset |  Compression Rate(%) | Pruned Acc | FP Acc |
| :-------:|:-------:|:-------:|:-------------:|:--------:|
| ResNet20 | CIFAR10 | 50 | 88.5 | 91.5|

