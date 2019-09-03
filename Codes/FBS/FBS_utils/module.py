import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time
import numpy as np

from utils.train import AverageMeter, accuracy, progress_bar



# class winner_take_all(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx, x, d):
#
#
#
#         return ternary_weight
#
#     @staticmethod
#     def backward(ctx, grad_outputs):
#
#         return


class FBS_CNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(FBS_CNN, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)

        # self.phi = nn.Parameter(torch.rand([in_channels, out_channels]))
        # self.rho = nn.Parameter(torch.Tensor([1.]))
        self.saliency_predictor = nn.Linear(in_features=in_channels, out_features=out_channels, bias=True)
        self.saliency = None
        self.sparse_output_masks = None

        self.data_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        print('Initialize FBS CNN')

    def forward(self, x, CR=0.5):

        # Subsample input features x \in [N, C, H, W] => [N, C] by L1 norm
        subsample_x = torch.abs(x).mean(-1).mean(-1)
        self.saliency = torch.abs(self.saliency_predictor(subsample_x))
        # Use wta to attain sparisity
        # self.pi = winner_take_all.apply(self.saliency, self.CR)
        threshold = self.saliency.topk(dim=1, k=int(np.round(self.out_channels * CR)))[0][:, -1]
        self.sparse_output_masks = \
            self.saliency * (self.saliency > threshold.view(-1, 1)).type(self.data_type)
        # [N,C] ==> [N,C,1,1]
        self.sparse_output_masks = self.sparse_output_masks.unsqueeze(dim=-1).unsqueeze(dim=-1)

        return self.sparse_output_masks * F.conv2d(x, self.weight, self.bias, self.stride,
                                                   self.padding, self.dilation, self.groups)


class FBS_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(FBS_Linear, self).__init__(in_features, out_features, bias=bias)

        self.saliency_predictor = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        self.saliency = None
        self.sparse_output_masks = None

        self.data_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        print('Initialize FBS Linear')

    def forward(self, x, CR=0.5):

        self.saliency = torch.abs(self.saliency_predictor(torch.abs(x)))

        threshold = self.saliency.topk(dim=1, k=int(np.round(self.out_features * CR)))[0][:, -1]

        self.sparse_output_masks = \
            self.saliency * (self.saliency > threshold.view(-1, 1)).type(self.data_type)
        self.sparse_output_masks = self.sparse_output_masks

        return self.sparse_output_masks * F.linear(x, self.weight, self.bias)


class testNet(nn.Module):

    def __init__(self):
        super(testNet, self).__init__()
        self.conv1 = FBS_CNN(3, 3, 5, 1, 2)
        self.fc1 = FBS_Linear(in_features=3*32*32, out_features=10)

    def forward(self, x):

        x = self.conv1(x)
        print(x.shape)
        x = x.view(-1, 3*32*32)
        x = self.fc1(x)
        return x


def test(net, CR, test_loader, use_cuda = True, dataset_name='CIFAR10', n_batches_used=None):

    net.eval()

    if dataset_name not in ['ImageNet']:
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                outputs = net(inputs, CR)

            _, predicted = torch.max(outputs.data, dim=1)
            correct += predicted.eq(targets.data).cpu().sum().item()
            total += targets.size(0)
            progress_bar(batch_idx, len(test_loader), "Test Acc: %.3f%%" % (100.0 * correct / total))

        return 100.0 * correct / total

    else:

        batch_time = AverageMeter()
        train_loss = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            end = time.time()
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                losses = nn.CrossEntropyLoss()(outputs, targets)

                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                train_loss.update(losses.data.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % 200 == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        batch_idx, len(test_loader), batch_time=batch_time, loss=train_loss,
                        top1=top1, top5=top5))

                if n_batches_used is not None and batch_idx >= n_batches_used:
                    break

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg, top5.avg


if __name__ == '__main__':

    net = testNet().cuda()
    inputs = torch.rand([10, 3, 32, 32]).cuda()
    outputs = net(inputs)
    losses = torch.nn.MSELoss()(outputs, torch.rand([10, 10]).cuda())
    losses.backward()