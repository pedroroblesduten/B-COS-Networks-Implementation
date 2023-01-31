import torch
import torch.nn as nn
import torch.nn.functional as F
from Bcos_modules import BcosConv2d

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
    def __init__(self):
        super().__init__()

        self.n_64 = 3
        self.n_128 = 3
        self.n_256 = 5
        self.n_512 = 2
        self.num_classes = 1000
        
        self.convin = BcosConv2d(3, 64, 7, 2, 64)
        self.avgpool = nn.AvgPool(kernel_size=3, stride=2, padding=1)

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
        
        self.fc = BcosConv2d(512 * residualBlock.expansion, self.num_classes)

        self.sequential_model = nn.Sequential(
            self.convin,
            self.avgpool,
            self.layers_64,
            self.layers_128,
            self.layers_256,
            self.layers_512,
            self.fc)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.convin(x)
        x = self.avgpool(x)
        x = self.layers_64(x)
        x = self.layers_128(x)
        x = self.layers_256(x)
        x = self.layers_512(x)

        return x

    def get_features(self, x):
        return self.sequential_model()[:-1](x)

    def get_layer_idx(self, idx):
        return int(np.ceil(len(self.sequential_model())*idx/10))








    







        


    