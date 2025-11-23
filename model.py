import torch
import torch.nn as nn
from functools import reduce
from gcnn import Conv2dZ2P4, Conv2dP4P4, MaxPoolingP4


# ResBlock P4
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = Conv2dP4P4(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = Conv2dP4P4(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.res_conv = Conv2dP4P4(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = None

    def forward(self, x):
        identity = x
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.res_conv:
            identity = self.res_conv(identity)
        return self.relu(y + identity)


# ResNet G-CNN P4
class ResNetGPN4(nn.Module):
    def __init__(self, structure=[(64,1), (128,2), (256,2)], input_dim=(3,32,32), device="cuda"):
        super().__init__()
        self.device = device

        # couche d'entrée Z2 -> P4
        self.init_conv = Conv2dZ2P4(input_dim[0], 64, kernel_size=7, stride=2, padding=3)
        self.pool = MaxPoolingP4(kernel_size=(3,3), stride=2)

        # empilement des ResBlocks
        self.blocks = nn.ModuleList()
        in_ch = 64
        for out_ch, stride in structure:
            self.blocks.append(ResBlock(in_ch, out_ch))
            in_ch = out_ch

        # couche fully connected
        # fixer dynamiquement au premier forward
        self.fc = None
        self.input_dim = input_dim

    def forward_conv(self, x):
        x = self.init_conv(x)
        x = self.pool(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)

        # crée FC dynamiquement au premier forward si nécessaire
        if self.fc is None:
            fc_input_dim = x.size(1)
            self.fc = nn.Sequential(
                nn.Linear(fc_input_dim, 1000),
                nn.ReLU(),
                nn.Linear(1000, 10)
            ).to(self.device)

        return self.fc(x)

