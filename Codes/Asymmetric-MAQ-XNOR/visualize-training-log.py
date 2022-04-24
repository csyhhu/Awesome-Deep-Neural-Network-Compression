import numpy as np
from numpy import genfromtxt

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='Asymmetric-MAQ-XNOR/LTH Visualization')
parser.add_argument('--selected_layer', '-l', type=str, default='conv1', help='Threshold information shown by layer.')
args = parser.parse_args()

layer_name = args.selected_layer

method_list = {

    "bg-8":{
        'name': 'bg-8',
        'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-1e+00-bitA-4-1e+00-bitG-8-lr-adjust-1-epoch-1',
        'color': 'k'
    },

    "bg-9":{
        'name': 'bg-9',
        'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-1e+00-bitA-4-1e+00-bitG-9-lr-adjust-1-epoch-1',
        'color': 'r'
    },

    # "asymmetric-MAQ-XNOR-32-32":{
    #     'name': '32-32',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-MAQ-XNOR/bitW-32-bitA-32-lr-adjust-10-epoch-30',
    #     'color': 'k'
    # },

    # "asymmetric-MAQ-XNOR":{
    #     'name': 'asymmetric-MAQ-XNOR',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-MAQ-XNOR/SGD-bitW-4-bitA-4-lr-adjust-10-epoch-30-compression-1e-0',
    #     'color': 'r'
    # },

    # "dorefa":{
    #     'name': 'dorefa',
    #     'path': 'Results/ResNet20-CIFAR10/dorefa/SGD-bitW-4-bitA-4-lr-adjust-10-epoch-30',
    #     'color': 'y'
    # },

    # "asymmetric-MAQ-XNOR-1e-1":{
    #     'name': 'asymmetric-MAQ-XNOR-1e-1',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-MAQ-XNOR/SGD-bitW-4-bitA-4-lr-adjust-10-epoch-30-compression-1e-1',
    #     'color': 'y'
    # },
    #
    # "asymmetric-MAQ-XNOR-compression-7e-1":{
    #     'name': 'asymmetric-MAQ-XNOR-compression-7e-1',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-MAQ-XNOR/SGD-bitW-4-bitA-4-lr-adjust-10-epoch-30-compression-7e-1',
    #     'color': 'g'
    # },

    # "asymmetric-LTH-Adam":{
    #     'name': 'asymmetric-LTH-Adam',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/Adam-bitW-4-bitA-4-lr-adjust-10-epoch-30-init-lr-1e-2',
    #     'color': 'g'
    # },

    # "asymmetric-MAQ-XNOR-compression-5e-1":{
    #     'name': 'asymmetric-MAQ-XNOR-compression-5e-1',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-MAQ-XNOR/SGD-bitW-4-bitA-4-lr-adjust-10-epoch-30-compression-5e-1',
    #     'color': 'b'
    # },

    # "asymmetric-LTH":{
    #     'name': 'asymmetric-LTH',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/bitW-4-bitA-4-lr-adjust-10-epoch-30',
    #     'color': 'b'
    # },
    #
    # "asymmetric-LTH-domain-factor-2e-1":{
    #     'name': 'asymmetric-LTH-domain-factor-2e-1',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-bitA-4-lr-adjust-10-epoch-30-out-domain-grad-factor-2e-1',
    #     'color': 'g'
    # },
    #
    # "asymmetric-LTH-domain-factor-5e-1":{
    #     'name': 'asymmetric-LTH-domain-factor-5e-1',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-bitA-4-lr-adjust-10-epoch-30-out-domain-grad-factor-5e-1',
    #     'color': 'k'
    # }

    # ---
    # Pretrained on Well-Train Model
    # ---
    # "asymmetric-MAQ-XNOR":{
    #     'name': 'asymmetric-MAQ-XNOR',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-MAQ-XNOR/SGD-bitW-4-bitA-4-lr-adjust-30-epoch-90-pretrain-init-lr-1e-3',
    #     'color': 'g'
    # },
    #
    # "asymmetric-LTH":{
    #     'name': 'asymmetric-LTH',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-bitA-4-lr-adjust-30-epoch-90-pretrain-init-lr-1e-3',
    #     'color': 'b'
    # },

}

file_name_list = ['loss', 'train-acc', 'test-acc']
selected_layer_list = ['conv1', 'layer1.0.conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'layer2.downsample']

# n_row = np.ceil(len(file_name_list) / 2)
n_row = len(file_name_list)
n_col = int(len(file_name_list) / n_row)
"""
for idx, data_name in enumerate(file_name_list):

    ax = plt.subplot(n_row, n_col, idx + 1)

    for method_name, info in method_list.items():

        color = info['color']
        label_name = info['name']

        data = genfromtxt("%s/%s.txt" % (info['path'], data_name), delimiter=',')[:, 1]
        if data_name == 'loss':
            data = np.convolve(data, np.ones(100)/100, mode='valid')

        xaxis = range(len(data))

        ax.plot(xaxis, data, color=color, linestyle='-', label=label_name, markersize=1)

        if data_name == 'test-acc':
            print('[%s] Best test acc: %.3f' % (label_name, np.max(data)))

    ax.set_ylabel(data_name)
    if data_name in ['loss']:
        ax.set_yscale('log')
    ax.grid()
    if data_name in ['loss']:
        ax.legend()

plt.tight_layout()
plt.show()
"""

file_name_list = [
    # 'weight_threshold', 'input_threshold',
    # 'weight_quantization_error', 'input_quantization_error',
    # 'input_bit_allocation', 'weight_bit_allocation',
    'grad_quant_info_collection'
]
n_row = np.ceil(len(file_name_list) / 2)
n_col = len(file_name_list) // n_row

for layer_name in selected_layer_list:

    plt.figure()

    for idx, data_name in enumerate(file_name_list):

        ax = plt.subplot(n_row, n_col, idx + 1)

        for method_name, info in method_list.items():

            # ax = plt.subplot(n_row, 2, idx + 1)

            color = info['color']
            label_name = info['name']

            try:
                data = genfromtxt('%s/%s/%s.txt' % (info['path'], layer_name, data_name), delimiter=',')
            except Exception as e:
                print('Path [%s/%s/%s.txt] not found' % (info['path'], layer_name, data_name))
                continue

            if data_name == 'input_threshold':
                xaxis = range(data.shape[0])
                ax.plot(xaxis, data[:, 0], color=color, linestyle='-', label=label_name, markersize=1)
                ax.plot(xaxis, data[:, 1], color=color, linestyle='-', label=label_name, markersize=1)
            elif data_name == 'grad_quant_info_collection':
                xaxis = range(data.shape[0])
                ax.plot(xaxis, data[:, 2], color=color, linestyle='-', label=label_name, markersize=1)
            else:
                xaxis = range(len(data))
                ax.plot(xaxis, data, color=color, linestyle='-', label=label_name, markersize=1)
            """
            if method_name != 'asymmetric-MAQ-XNOR-32-32' and False:

                if data_name == 'weight_quantization_error':
                    weight_threshold = genfromtxt('%s/%s/weight_threshold.txt' % (info['path'], layer_name), delimiter=',')
                    theoretical_gap = weight_threshold / (2**4 - 1)
                    ax.plot(range(len(theoretical_gap)), theoretical_gap, color=color, linestyle='--', markersize=1)

                elif data_name == 'input_quantization_error':
                    input_threshold = genfromtxt('%s/%s/input_threshold.txt' % (info['path'], layer_name), delimiter=',')
                    theoretical_gap = abs(input_threshold[:, 1] - input_threshold[:, 0]) / ((2 ** 4 - 1) * 2)
                    ax.plot(range(len(theoretical_gap)), theoretical_gap, color=color, linestyle='--', markersize=1)
            """

        if data_name in ['weight_threshold'] or idx == 0:
            ax.legend()

        ax.title.set_text(data_name)

    plt.tight_layout()
    plt.suptitle('Threshold and error in [%s]' % layer_name)
    plt.show()
    flag = input('')
    if flag == 'q':
        break
    # plt.savefig('./Results/ResNet20-CIFAR10/figs/%s.pdf' % layer_name)
# """