import torch
import torch.nn as nn
import torch.nn.functional as F
import math

     
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class CAConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, reduction=32):
        super(CAConv, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, padding=kernel_size // 2, stride=stride))

        self.k2 = nn.Sequential(
                    # nn.AvgPool2d(kernel_size=4, stride=4), 
                    nn.AvgPool2d(kernel_size=16, stride=16),
                    
                    nn.Conv2d(inp, oup, kernel_size=3, stride=1,
                                padding=2, dilation=2,
                                groups=1, bias=False),
                    nn.BatchNorm2d(inp),
                    )

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        

        y = torch.cat([x_h, x_w], dim=2)
        
        y = self.conv1(y)
        
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        # a_total = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))
        a_total = torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))

        out = identity * a_w * a_h

        return self.conv(out), a_total

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)



def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = CAConv(inplanes,planes,3,stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out, cancha = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)


 
        # out += identity
        out = out + identity + cancha
        out = self.relu(out)

        return out



# 16 --> 32  -->  48
class MF(nn.Module): 

    def __init__(self, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(newnet4, self).__init__()

        # 改变通道数
        self.convc1 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1) 
        self.convc2 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1) 

        self.bnc1 = nn.BatchNorm2d(32) 
        self.bnc2 = nn.BatchNorm2d(48) 

        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=3, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=3, out_channels=8, padding=(0, 3))
        
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8) 

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = BasicBlock(inplanes=16,planes=16,stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None)
        self.block2 = BasicBlock(inplanes=32,planes=32,stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None)
        self.block3 = BasicBlock(inplanes=48,planes=48,stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None)

        self.avgpool = nn.AdaptiveAvgPool2d((2, 3))


    def forward(self, x):
        
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa = F.relu(xa)

        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)

        x = torch.cat((xa, xb), 1)

        x = self.maxpool(x)
        x = self.block1(x)

        x = self.convc1(x)
        x = self.bnc1(x)
        x = self.relu(x)
        x = self.block2(x)

        x = self.convc2(x)
        x = self.bnc2(x)
        x = self.relu(x)
        x = self.block3(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        return x  # 32,288