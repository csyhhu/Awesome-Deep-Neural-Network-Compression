"""
A script to visualize training log for TTQ
"""

import numpy as np
from numpy import genfromtxt

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='TTQ Visualization')
args = parser.parse_args()

method_list = {

    '../Results/ResNet20-STL10/runs-TTQ/TheshFactor-0.050-Optimizer-Adam':
        {'name': 'TTQ-0.05', 'color': 'k'},
}

file_name_list = ['loss', 'lr', 'train-acc', 'test-acc', 'conv1_neg', 'conv1_pos', 'conv1_prune']

n_row = np.ceil(len(file_name_list) / 2)

for idx, data_name in enumerate(file_name_list):
    ax = plt.subplot(n_row, 2, idx + 1)

    for method_path, method_vis_info in method_list.items():
        color = method_vis_info['color']
        label_name = method_vis_info['name']

        if data_name in ['conv1_neg', 'conv1_pos']:
            data = genfromtxt('%s/%s.txt' % (method_path, data_name), delimiter=',')
            xaxis = range(len(data))
            value, portion = data[:, 1], data[:, 2]

            ax.plot(xaxis, value, color=color, linestyle='-',
                    label='%s-%s-value' %(label_name, data_name), markersize=1)
            ax.plot(xaxis, portion, color=color, linestyle=':',
                    label='%s-%s-portion' % (label_name, data_name), markersize=1)

        else:
            data = genfromtxt('%s/%s.txt' % (method_path, data_name), delimiter=',')[:, 1]
            xaxis = range(len(data))
            ax.plot(xaxis, data, color=color, linestyle='-', label=label_name, markersize=1)

        if data_name == 'test-acc':
            print('[%s] Best test acc: %.3f' % (label_name, np.max(data)))

    ax.set_ylabel(data_name)
    if data_name in ['loss']:
        ax.set_yscale('log')
    ax.grid()
    if data_name in ['loss', 'conv1_neg', 'conv1_pos']:
        ax.legend()

plt.show()