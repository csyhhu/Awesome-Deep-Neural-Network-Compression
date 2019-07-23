import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFARNet(nn.Module):

    def __init__(self, num_classes=10):

        super(CIFARNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)  # 3 * 32 * 5 * 5 =
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)

        self.fc1 = nn.Linear(576, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 3, 2))

        x = self.conv2(x)
        x = F.avg_pool2d(F.relu(x), 3, 2)

        x = self.conv3(x)
        x = F.avg_pool2d(F.relu(x), 3, 2)

        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == '__main__':

    net = CIFARNet()
    inputs = torch.rand([1, 3, 32, 32])
    outputs = net(inputs)
