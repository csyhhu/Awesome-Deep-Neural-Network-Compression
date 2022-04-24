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

# from module import LearnableThresHold_Conv2d
from torch.utils.tensorboard import SummaryWriter

def train(
    _net, _train_loader, _optimizer, _criterion,
    _n_batch_used=-1, _epoch_idx=0,
    _device = 'cpu',
    _recorder: Recorder = None,
    _weight_threshold_recorder_collection = None, _input_threshold_recorder_collection = None, _gradient_threshold_recorder_collection = None,
    _weight_quantization_error_collection = None, _input_quantization_error_collection = None,
    _min_max_weight_collection = None, _min_max_input_collection = None,
    _weight_bit_allocation_collection = None, _input_bit_allocation_collection = None,
    _gradient_out_of_domain_collection = None, _gradient_quantization_error_collection = None,
    _writer: SummaryWriter = None
):

    _net.train()
    _train_loss = 0
    _correct = 0
    _total = 0

    _this_n_batch = len(_train_loader) if _n_batch_used == -1 else _n_batch_used

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
            batch_idx, _this_n_batch,
            'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (_train_loss / (batch_idx + 1), 100. * _correct / _total, _correct, _total)
        )

        if _recorder is not None:
            _recorder.update(loss=losses.data.item(), acc=[_correct / _total], batch_size=inputs.size(0), is_train=True)

        if _weight_threshold_recorder_collection and _input_threshold_recorder_collection is not None:
            for name, layer in _net.quantized_layer_collections.items():
                _weight_threshold_recorder_collection[name].write('%.8e\n' % layer.weight_threshold.item())
                _input_threshold_recorder_collection[name].write('%.8e, %.8e\n' % (layer.left_input_threshold.item(), layer.right_input_threshold.item()))

        if _gradient_threshold_recorder_collection is not None:
            for name, layer in _net.quantized_layer_collections.items():
                _gradient_threshold_recorder_collection[name].write('%.8e, %.8e\n' % (layer.gradient_left_threshold.item(), layer.gradient_right_threshold.item()))

        if _weight_quantization_error_collection and _input_quantization_error_collection is not None:
            for name, layer in _net.quantized_layer_collections.items():
                _weight_quantization_error = torch.abs(layer.quantized_weight - layer.weight).mean().item()
                _input_quantization_error = torch.abs(layer.quantized_input - layer.fp_input).mean().item()
                _weight_quantization_error_collection[name].write('%.8e\n' % _weight_quantization_error)
                _input_quantization_error_collection[name].write('%.8e\n' % _input_quantization_error)

        if _min_max_weight_collection and _min_max_input_collection is not None:
            for name, layer in _net.quantized_layer_collections.items():
                _min_max_weight_collection[name].write('%.8e, %.8e\n' % (layer.min_weight, layer.max_weight))
                _min_max_input_collection[name].write('%.8e, %.8e\n' % (layer.min_input, layer.max_input))

        if _weight_bit_allocation_collection and _input_bit_allocation_collection is not None:
            for name, layer in _net.quantized_layer_collections.items():
                _weight_bit_allocation_collection[name].write('%.2f\n' % (torch.abs(layer.quantized_weight_bit).mean().item()))
                _input_bit_allocation_collection[name].write('%.2f\n' % (torch.abs(layer.quantized_input_bit).mean().item()))

        if _gradient_out_of_domain_collection is not None:
            for name, layer in _net.quantized_layer_collections.items():
                _gradient_out_of_domain_collection[name].write('%.2f, %.2f\n' % (layer.grad_quant_info['out_of_min_error'], layer.grad_quant_info['out_of_max_error']))

        if _gradient_quantization_error_collection is not None:
            for name, layer in _net.quantized_layer_collections.items():
                _gradient_quantization_error_collection[name].write('%.8f\n' % (layer.grad_quant_info['in_domain_error']))

        if _writer is not None:
            _iter = batch_idx + _this_n_batch * _epoch_idx
            _writer.add_scalar("train/loss", losses.item(), _iter)
            _writer.add_scalar("train/acc", _correct / _total, _iter)
            # selected_layer = ['conv1', 'layer1.0.conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'layer2.downsample']
            # selected_layer = ['conv1']
            selected_layer = []
            for name, layer in _net.quantized_layer_collections.items():
                _writer.add_scalar("train/%s/weight_threshold" % name, layer.weight_threshold, _iter)
                _writer.add_scalar("train/%s/left_input_threshold" % name, layer.left_input_threshold, _iter)
                _writer.add_scalar("train/%s/right_input_threshold" % name, layer.right_input_threshold, _iter)
                # try:
                _writer.add_scalar("train/%s/gradient/left_threshold" % name, layer.gradient_left_threshold, _iter)
                _writer.add_scalar("train/%s/gradient/right_threshold" % name, layer.gradient_right_threshold, _iter)
                _writer.add_scalar("train/%s/gradient/max_gradient" % name, torch.max(layer.weight.grad).item(), _iter)
                _writer.add_scalar("train/%s/gradient/min_gradient" % name, torch.min(layer.weight.grad).item(), _iter)
                _writer.add_scalar("train/%s/gradient/out_of_min" % name, layer.grad_quant_info['out_of_min_error'], _iter)
                _writer.add_scalar("train/%s/gradient/out_of_max" % name, layer.grad_quant_info['out_of_max_error'], _iter)
                _writer.add_scalar("train/%s/gradient/in_domain_error" % name, layer.grad_quant_info['in_domain_error'], _iter)
                _writer.add_scalar("train/%s/gradient/in_domain_error_pre" % name, layer.grad_quant_info['in_domain_error_pre'], _iter)
                _writer.add_scalar("train/%s/gradient/quantization_error" % name, layer.grad_quant_info['quantization_error'], _iter)
                _writer.add_scalar("train/%s/gradient/quantization_error_pre" % name, layer.grad_quant_info['quantization_error_pre'], _iter)
                try:
                    _writer.add_scalar("train/%s/gradient/grad_loss_wrt_left_threshold" % name, layer.grad_quant_info['grad_loss_wrt_left_threshold'], _iter)
                    _writer.add_scalar("train/%s/gradient/grad_loss_wrt_right_threshold" % name, layer.grad_quant_info['grad_loss_wrt_right_threshold'], _iter)
                except:
                    pass
                if name in selected_layer and (_iter + 1) % 40 == 0:
                    _writer.add_histogram("train/%s/pre_quantized_grad" % name, layer.grad_quant_info['pre_quantized_grad'], _iter)
                    _writer.add_histogram("train/%s/clipped_grad" % name, layer.grad_quant_info['clipped_grad'], _iter)
                    _writer.add_histogram("train/%s/quantized_grad" % name, layer.grad_quant_info['quantized_grad'], _iter)

        if batch_idx > _n_batch_used != -1:
            break

    return _train_loss / (_this_n_batch), _correct / _total