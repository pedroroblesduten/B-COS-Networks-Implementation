import torch
import torch.nn as nn
import torch.nn.functional as F
from Bcos_modules import BcosConv2d
from utils import FinalLayer, MyAdaptiveAvgPool2d

class residualBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=3, s=1, p=1):
        super().__init__()

        self.bcos_conv1 = BcosConv2d(in_c, out_c, ks, s, p)
        self.bcos_conv2 = BcosConv2d(in_c, out_c, ks, s, p)

    def forward(self, x):
        res = x
        x = self.bcos_conv1(x)
        x = self.bcos_conv2(x)

        return x + res

class resNet34(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_64 = 3
        self.n_128 = 3
        self.n_256 = 5
        self.n_512 = 2
        if args.dataset == 'CIFAR10':
            self.num_classes = 10
        self.logit_bias = np.log(.1/.9)
        self.logit_temperature = 1
        
        self.convin = BcosConv2d(3, 64, 7, 2, 3)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layers_64 = nn.ModuleList()
        for block in range(self.n_64):
            self.layers_64.append(residualBlock(64, 64))

        self.bcos_conv_1_128 = BcosConv2d(64, 128, 4, 2, 2)
        self.bcos_conv_2_128 = BcosConv2d(128, 128, 3, 1, 1)

        self.layers_128 = nn.ModuleList()
        for block in range(self.n_128):
            self.layers_128.append(residualBlock(128, 128))

        
        self.bcos_conv_1_256 = BcosConv2d(128, 256, 4, 2, 2)
        self.bcos_conv_2_256 = BcosConv2d(256, 256, 3, 1, 1)

        self.layers_256 = nn.ModuleList()
        for block in range(self.n_256):
            self.layers_256.append(residualBlock(256, 256))

        
        self.bcos_conv_1_512 = BcosConv2d(256, 512, 4, 2, 2)
        self.bcos_conv_2_512 = BcosConv2d(512, 512, 3, 1, 1)

        self.layers_512 = nn.ModuleList()
        for block in range(self.n_512):
            self.layers_512.append(residualBlock(512, 512))
        
        self.fc = BcosConv2d(512, self.num_classes)

        self.classifier = nn.Sequential([
            MyAdaptiveAvgPool2d((1, 1)),
            FinalLayer(bias=logit_bias, norm=logit_temperature)
        ])

        self.sequential_model = nn.Sequential(
            self.convin,
            self.avgpool,
            self.layers_64,
            self.layers_128,
            self.layers_256,
            self.layers_512,
            self.fc,
            self.classifier)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.convin(x)
        print(f'shape after convin: {x.shape}')
        x = self.avgpool(x)
        
        for l_64 in self.layers_64:
            x = l_64(x)

        x = self.bcos_conv_1_128(x)
        print(f'shape bcos_conv1_128: {x.shape}')
        x = self.bcos_conv_2_128(x)

        for l_128 in self.layers_128:
            x = l_128(x)
        x = self.bcos_conv_1_256(x)
        x = self.bcos_conv_2_256(x)
        print(f'shape bcos256 {x.shape}')

        for l_256 in self.layers_256:
            x = l_256(x)
        print(f'shape layers256 {x.shape}')

        x = self.bcos_conv_1_512(x)
        x = self.bcos_conv_2_512(x)
        print(f'shape bcos512{x.shape}')

        for l_512 in self.layers_512:
            x = l_512(x)
        #x = torch.flatten(x, 1)
        print(f'shape layers512 {x.shape}')

        x = self.fc(x)
        print(f'shape fc {x.shape}')

        return x

    def get_features(self, x):
        return self.sequential_model()[:-1](x)

    def get_layer_idx(self, idx):
        return int(np.ceil(len(self.sequential_model())*idx/10))   
