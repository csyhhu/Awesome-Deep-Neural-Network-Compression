"""
A script to visualize training log for TTQ
"""

import numpy as np
from numpy import genfromtxt

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='INQ Visualization')
args = parser.parse_args()

method_list = {

    '../Results/ResNet20-CIFAR10/runs-INQ/Adam-1bits-25-50-75-100':
        {'name': 'INQ-1bit-25-50-75-100', 'color': 'k'},
}

file_name_list = ['loss', 'lr', 'train-acc', 'test-acc']

n_row = np.ceil(len(file_name_list) / 2)

for idx, data_name in enumerate(file_name_list):
    ax = plt.subplot(n_row, 2, idx + 1)

    for method_path, method_vis_info in method_list.items():
        color = method_vis_info['color']
        label_name = method_vis_info['name']

        data = genfromtxt('%s/%s.txt' % (method_path, data_name), delimiter=',')[:, 1]
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

plt.show()