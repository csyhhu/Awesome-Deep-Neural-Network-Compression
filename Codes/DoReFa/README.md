# My Re-Implementation of DoReFa-Net / BWN

Note: Currently only weights quantization is finished.

Updated [Sept.16th, 2021]: Weights/Activation quantization is supported.

## How to use it

### Specify your Dataset Root
Please refer to [here](./README.md)

### Train a full-precision model
Please refer to [here](../TTQ/README.md)

### Run the code

#### Weight Quantization Only (Old Version)
For k-bit quantization using `dorefa`
```
python DoReFa.py -m ResNet20 -d CIFAR10 -q dorefa -bw k
```
For 1-bit quantization using `BWN`
```
python DoReFa.py -m ResNet20 -d CIFAR10 -q BWN
```

#### Weight/Activation Quantization
For k-bit quantization using `dorefa`
```
python main.py -m ResNet20 -d CIFAR10 -bw k -ba k -o SGD
```

## Experiments

| Method | W | A | G | E | Quantized Acc | FP Acc |
| :-------:|:---:|:---:|:---:|:---:|:-------------:|:--------:|
| DoReFa | 1 | 32|32|32| 89.69 | 91.5|
| DoReFa | 2 | 32|32|32| 91.19| 91.5
| BWN    | 1 | 32|32|32| 90.3 | 91.5|

### Visualization of Training Log
ResNet20-CIFAR10 using dorefa / BWN with 1-bit quantization.
![](./dorefa-BWN-training-log.png)

## Acknowledgement