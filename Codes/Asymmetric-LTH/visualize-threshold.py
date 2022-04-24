import numpy as np
from numpy import genfromtxt

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='Asymmetric-LTH Visualization')
parser.add_argument('--selected_layer', '-l', type=str, default='conv1', help='Threshold information shown by layer.')
args = parser.parse_args()

layer_name = args.selected_layer

method_list = {

    "pk-1-pa-1":{
        'name': '1-1',
        'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-bitA-4-bitG-32-lr-adjust-10-epoch-30',
        'color': 'k'
    },

    "pk-1-pa-1-bitG--1":{
        'name': 'shift-grad',
        'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-1e+00-bitA-4-1e+00-bitG--1-lr-adjust-10-epoch-30',
        'color': 'r'
    },

    # "pk-10-pa-1":{
    #     'name': '10-1',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-1e+01-bitA-4-1e+00-bitG-32-lr-adjust-10-epoch-30',
    #     'color': 'r'
    # },
    #
    # "pk-100-pa-1":{
    #     'name': '100-1',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-1e+02-bitA-4-1e+00-bitG-32-lr-adjust-10-epoch-30',
    #     'color': 'g'
    # },
    #
    # "pk-1e-1-pa-1":{
    #     'name': '1e-1-1',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-1e-01-bitA-4-1e+00-bitG-32-lr-adjust-10-epoch-30',
    #     'color': 'y'
    # },
    #
    # "pk-1e-2-pa-1":{
    #     'name': '1e-2-1',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-1e-02-bitA-4-1e+00-bitG-32-lr-adjust-10-epoch-30',
    #     'color': 'm'
    # },

    # "pk-1-pa-10":{
    #     'name': '1-10',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-1e+00-bitA-4-1e+01-bitG-32-lr-adjust-10-epoch-30',
    #     'color': 'b'
    # },
    #
    # "pk-1-pa-100":{
    #     'name': '1-100',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-1e+00-bitA-4-1e+02-bitG-32-lr-adjust-10-epoch-30',
    #     'color': 'r'
    # },
    #
    # "pk-1-pa-1e-1":{
    #     'name': '1-1e-1',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-1e+00-bitA-4-1e-01-bitG-32-lr-adjust-10-epoch-30',
    #     'color': 'y'
    # },
    #
    # "pk-1-pa-1e-2":{
    #     'name': '1-1e-2',
    #     'path': 'Results/ResNet20-CIFAR10/asymmetric-LTH/SGD-bitW-4-1e+00-bitA-4-1e-02-bitG-32-lr-adjust-10-epoch-30',
    #     'color': 'm'
    # },

}

selected_layer_list = ['conv1', 'layer1.0.conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'layer2.downsample']

"""
metrics_list = ['loss', 'train-acc', 'test-acc']
# n_row = np.ceil(len(file_name_list) / 2)
n_row = len(metrics_list)
n_col = int(len(metrics_list) / n_row)

for idx, data_name in enumerate(metrics_list):

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

metrics_list = [
    'loss',
    # 'train-acc',
    'test-acc',
    'weight_threshold', 'input_threshold', 'gradient_threshold',
    # 'weight_quantization_error', 'input_quantization_error',
    'input_bit_allocation', 'weight_bit_allocation'
]
n_col = 2
n_row = int(np.ceil(len(metrics_list) / n_col))


for layer_name in selected_layer_list:

    plt.figure()

    for idx, data_name in enumerate(metrics_list):

        ax = plt.subplot(n_row, n_col, idx + 1)

        for method_name, info in method_list.items():

            # ax = plt.subplot(n_row, 2, idx + 1)

            color = info['color']
            label_name = info['name']

            try:
                if data_name in ['loss', 'train-acc', 'test-acc']:
                    data = genfromtxt('%s/%s.txt' % (info['path'], data_name), delimiter=',')
                else:
                    data = genfromtxt('%s/%s/%s.txt' % (info['path'], layer_name, data_name), delimiter=',')
            except Exception as e:
                print('Path [%s/%s/%s.txt] not found' % (info['path'], layer_name, data_name))
                continue

            if data_name in ['input_threshold']:
                xaxis = range(data.shape[0])
                ax.plot(xaxis, data[:, 0], color=color, linestyle='-', label=label_name, markersize=1)
                ax.plot(xaxis, data[:, 1], color=color, linestyle='-.', label=label_name, markersize=1)
            elif data_name in ['gradient_threshold']:
                chosen_point = np.linspace(0, data.shape[0], 100, dtype=int, endpoint=False)
                # print(chosen_point)
                ax.plot(chosen_point, np.take(data[: ,0], indices=chosen_point), color=color, linestyle='-', label=label_name, markersize=1)
                ax.plot(chosen_point, np.take(data[: ,1], indices=chosen_point), color=color, linestyle='-.', label=label_name, markersize=1)
            elif data_name == 'loss':
                data = data[:, 2]
                xaxis = range(len(data))
                ax.plot(xaxis, data, color=color, linestyle='-', label=label_name, markersize=1)
            elif data_name in ['test-acc', 'train-acc']:
                data = data[:, 1]
                xaxis = range(len(data))
                ax.plot(xaxis, data, color=color, linestyle='-', label=label_name, markersize=1)
            else:
                xaxis = range(len(data))
                ax.plot(xaxis, data, color=color, linestyle='-', label=label_name, markersize=1)


        if data_name in ['weight_threshold']:
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