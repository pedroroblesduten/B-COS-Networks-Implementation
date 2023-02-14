import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# B-cos modifications in pytorch modules
# Follows the original implementantion https://github.com/moboehle/B-cos/blob/main/modules/bcosconv2d.py

#TODO: find possibles optimizations to reduce training time!!

class normConv2d(nn.Conv2d):
    
    # Computing |W_j|*a_j
    def forward(self, x):
        w_origial_shape = self.weight.shape

        #TODO: find a better way to normalize this, they reported that this increase training time
        w_hat = self.weight.view(w_origial_shape[0], -1)
        w_hat = w_hat/(w_hat.norm(p=2, dim=1, keepdim=True))
        w_hat = w_hat.view(w_origial_shape)

        return F.conv2d(x, w_hat,
                        self.bias, self.stride, self.padding, self.dilation, self.groups)

class BcosConv2d(nn.Module):

    def __init__(
            self,
            in_c, out_c,
            kernel_size=1,
            stride=1,
            padding=0,
            max_out=2,
            b=2,
            scale=None,
            scale_fact=100,
            **kwargs):
        super().__init__()

        ks = kernel_size
        self.stride = stride
        self.linear = normConv2d(in_c, out_c * max_out, ks, stride, padding, 1, 1, bias=False)
        self.outc = out_c * max_out
        self.b = b
        self.max_out = max_out
        self.inc = in_c
        self.kernel_size = ks
        self.kssq = ks**2 if not isinstance(ks, tuple) else np.prod(ks)
        self.padding = padding
        self.detach = False

        if scale is None:
            ks_scale = ks if not isinstance(ks, tuple) else np.sqrt(np.prod(ks))
            self.scale = (ks_scale * np.sqrt(self.inc)) / scale_fact
        else:
            self.scale = scale

    def explanationMode(self, detach=True):
        self.detach = detach

    def forward(self, x):
        out = self.linear(x)
        batch_size, c, h, w = out.shape

        #MaxOut:
        if self.max_out > 1:
            batch_size, c, h, w = out.shape
            out = out.view(batch_size, -1, self.max_out, h, w)
            out = out.max(dim=2, keepdim=False)[0]
                
        if self.b == 1:
            return out / self.scale

        norm = (F.avg_pool2d((x**2).sum(1, keepdim=True), self.kernel_size, padding=self.padding,
                stride=self.stride) * self.kssq + 1e-6).sqrt_()

        #Calculating cos similarity
        abs_cos = (out/norm).abs() + 1e-6

        if self.detach:
            abs_cos = abs_cos.detach()

        out = out*abs_cos.pow(self.b-1)

        return out/self.scale


class normLinear(nn.Linear):
    
    # Computing |W_j|*a_j
    def forward(self, x):
        w_origial_shape = self.weight.shape

        #TODO: find a better way to normalize this, they reported that this increase training time
        w_hat = self.weight.view(w_origial_shape[0], -1)
        w_hat = w_hat/(w_hat.norm(p=2, dim=1, keepdim=True))
        w_hat = w_hat.view(w_origial_shape)

        return F.linear(input = x,
                        weight = w_hat,
                        bias = self.bias)

class BcosLinear(nn.Module):

    def __init__(
            self,
            in_d, out_d,
            max_out=2,
            b=2,
            scale=None,
            scale_fact=100,
            **kwargs):
        super().__init__()

        self.linear = normLinear(in_d, out_d, bias=False)
        self.outc = out_d
        self.b = b
        self.inc = in_d
        self.detach = False

        #if scale is None:
        #    ks_scale = ks if not isinstance(ks, tuple) else np.sqrt(np.prod(ks))
        #    self.scale = (ks_scale * np.sqrt(self.inc)) / scale_fact
        #else:
            #self.scale = scale

    def explanationMode(self, detach=True):
        self.detach = detach

    def forward(self, x):
        out = self.linear(x)
        batch_size, d = out.shape
                
        if self.b == 1:
            return out / self.scale

        #norm = (F.avg_pool2d((x**2).sum(1, keepdim=True), self.kernel_size, padding=self.padding,
        #        stride=self.stride) * self.kssq + 1e-6).sqrt_()
        norm = torch.norm(out)
        
        #Calculating cos similarity
        abs_cos = (out/norm).abs() + 1e-6

        if self.detach:
            abs_cos = abs_cos.detach()

        out = out*abs_cos.pow(self.b-1)

        return out #/self.scale
