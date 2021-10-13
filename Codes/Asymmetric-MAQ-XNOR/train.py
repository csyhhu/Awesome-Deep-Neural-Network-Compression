import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

import os
import time
import sys
from datetime import datetime
import numpy as np

from utils.train import  progress_bar
from utils.recorder import Recorder

def train(
        _net, _train_loader, _optimizer, _criterion, _device = 'cpu',
        _recorder: Recorder = None,
        _weight_threshold_recorder_collection = None, _input_threshold_recorder_collection = None,
        _weight_quantization_error_collection = None, _input_quantization_error_collection = None,
        _weight_bit_allocation_collection = None, _input_bit_allocation_collection = None
):

    _net.train()
    _train_loss = 0
    _correct = 0
    _total = 0

    for batch_idx, (inputs, targets) in enumerate(_train_loader):

        inputs, targets = inputs.to(_device), targets.to(_device)

        _optimizer.zero_grad()
        outputs = _net(inputs)
        losses = _criterion(outputs, targets)
        losses.backward()
        _optimizer.step()

        _train_loss += losses.data.item()
        _, predicted = torch.max(outputs.data, 1)
        _total += targets.size(0)
        _correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(
            batch_idx, len(_train_loader),
            'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (_train_loss / (batch_idx + 1), 100. * _correct / _total, _correct, _total)
        )

        if _recorder is not None:
            _recorder.update(loss=losses.data.item(), acc=[_correct / _total], batch_size=inputs.size(0), is_train=True)

        if _weight_threshold_recorder_collection and _input_threshold_recorder_collection is not None:
            for name, layer in _net.quantized_layer_collections.items():
                _weight_threshold_recorder_collection[name].write('%.8e\n' % layer.weight_threshold.item())
                _input_threshold_recorder_collection[name].write('%.8e, %.8e\n' % (layer.left_input_threshold.item(), layer.right_input_threshold.item()))

        if _weight_quantization_error_collection and _input_quantization_error_collection is not None:
            for name, layer in _net.quantized_layer_collections.items():
                _weight_quantization_error = torch.abs(layer.quantized_weight - layer.weight).mean().item()
                _input_quantization_error = torch.abs(layer.quantized_input - layer.fp_input).mean().item()
                _weight_quantization_error_collection[name].write('%.8e\n' % _weight_quantization_error)
                _input_quantization_error_collection[name].write('%.8e\n' % _input_quantization_error)

        if _weight_bit_allocation_collection and _input_bit_allocation_collection is not None:
            for name, layer in _net.quantized_layer_collections.items():
                _weight_bit_allocation_collection[name].write(
                    '%.2f\n' % (torch.abs(layer.quantized_weight_bit).mean().item())
                )
                _input_bit_allocation_collection[name].write(
                    '%.2f\n' % (torch.abs(layer.quantized_input_bit).mean().item())
                )

    return _train_loss / (len(_train_loader)), _correct / _total