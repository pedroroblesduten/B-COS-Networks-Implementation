import torch
import torch.nn as nn
import torch.nn.functional as F
from Bcos_modules import BcosConv2d, BcosLinear
from utils import FinalLayer, MyAdaptiveAvgPool2d
import numpy as np

class residualBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=3, s=1, p=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, ks, s, p)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(in_c, out_c, ks, s, p)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x + res

class baseResNet34(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.verbose = args.verbose
        #Number of residual blocks
        self.n_64 = 3
        self.n_128 = 3
        self.n_256 = 5
        self.n_512 = 2
        if args.dataset == 'CIFAR10':
            self.num_classes = 10
        self.logit_bias = np.log(.1/.9)
        self.logit_temperature = 1
        
        self.convin = nn.Conv2d(3, 64, 7, 2, 3)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.layers_64 = nn.ModuleList()
        for block in range(self.n_64):
            self.layers_64.append(residualBlock(64, 64))

        self.conv_1_128 = nn.Conv2d(64, 128, 4, 2, 2)
        self.conv_2_128 = nn.Conv2d(128, 128, 3, 1, 1)

        self.layers_128 = nn.ModuleList()
        for block in range(self.n_128):
            self.layers_128.append(residualBlock(128, 128))

        
        self.conv_1_256 = nn.Conv2d(128, 256, 4, 2, 2)
        self.conv_2_256 = nn.Conv2d(256, 256, 3, 1, 1)

        self.layers_256 = nn.ModuleList()
        for block in range(self.n_256):
            self.layers_256.append(residualBlock(256, 256))

        
        self.conv_1_512 = nn.Conv2d(256, 512, 4, 2, 2)
        self.conv_2_512 = nn.Conv2d(512, 512, 3, 1, 1)

        self.layers_512 = nn.ModuleList()
        for block in range(self.n_512):
            self.layers_512.append(residualBlock(512, 512))
        
        self.linear = nn.Linear(512*2*2, 10)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.verbose:
            print(f'INPUT SHAPE: {x.shape}')
        x = self.convin(x)
        if self.verbose:
            print(f'shape after convin: {x.shape}')
        x = self.avgpool(x)
        x = self.bn3(x)
        
        for l_64 in self.layers_64:
            x = l_64(x)

        x = self.conv_1_128(x)
        if self.verbose:
            print(f'shape bcos_conv1_128: {x.shape}')
        x = self.conv_2_128(x)

        for l_128 in self.layers_128:
            x = l_128(x)
        x = self.conv_1_256(x)
        x = self.conv_2_256(x)
        if self.verbose:
            print(f'shape bcos256 {x.shape}')

        for l_256 in self.layers_256:
            x = l_256(x)
        if self.verbose:
            print(f'shape layers256 {x.shape}')

        x = self.conv_1_512(x)
        x = self.conv_2_512(x)
        if self.verbose:
            print(f'shape bcos512{x.shape}')

        for l_512 in self.layers_512:
            x = l_512(x)
        #x = torch.flatten(x, 1)
        if self.verbose:
            print(f'shape layers512 {x.shape}')

        #x = self.fc(x)
        #if self.verbose:
        #    print(f'shape fc {x.shape}')
        
        x = x.view(x.shape[0], -1)
        if self.verbose:
            print(f'shape after flatten: {x.shape}')

        x = self.linear(x)
        #x = x.long()
        if self.verbose:
            print(f'OUTPUT SHAPE: {x.shape}')

        return x
