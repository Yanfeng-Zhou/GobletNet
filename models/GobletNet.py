import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.distributions.uniform import Uniform
import numpy as np
BatchNorm2d = nn.BatchNorm2d
relu_inplace = True

BN_MOMENTUM = 0.1
# BN_MOMENTUM = 0.01

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
    def forward(self, x):
        x = self.down(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, if_relu=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.if_relu = if_relu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.bn2(out) + identity
        if self.if_relu:
            out = self.relu(out)
        return out

class DoubleBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample=None, if_relu=False):
        super(DoubleBasicBlock, self).__init__()

        self.DBB = nn.Sequential(
            BasicBlock(inplanes=inplanes, planes=planes, downsample=downsample),
            BasicBlock(inplanes=planes, planes=planes, if_relu=if_relu)
        )

    def forward(self, x):
        out = self.DBB(x)
        return out

class Fusion_Attention_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fusion_Attention_Module, self).__init__()

        self.Fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )

        self.Attention1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.Sigmoid()
        )
        self.Attention2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):

        x = torch.cat((x1, x2), dim=1)
        x = self.Fusion(x)

        x_attention_1 = self.Attention1(x)
        x_attention_2 = self.Attention2(x)

        x_output_1 = x1 * x_attention_1
        x_output_2 = x2 * x_attention_2

        x_output_1 = F.relu(x_output_1)
        x_output_2 = F.relu(x_output_2)

        return x_output_1, x_output_2

class GobletNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GobletNet, self).__init__()

        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024

        # encoder
        # l1
        self.b1_l1_1 = nn.Sequential(
            conv3x3(in_channels, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c, if_relu=False))
        self.b2_l1_1 = nn.Sequential(
            conv3x3(1, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c, if_relu=False))
        self.fa_l1 = Fusion_Attention_Module(l1c*2, l1c)
        self.b1_l1_2 = down_conv(l1c, l2c)
        self.b2_l1_2 = down_conv(l1c, l2c)

        # l2
        self.b1_l2_1 = DoubleBasicBlock(l2c, l2c)
        self.b2_l2_1 = DoubleBasicBlock(l2c, l2c)
        self.fa_l2 = Fusion_Attention_Module(l2c*2, l2c)
        self.b1_l2_2 = down_conv(l2c, l3c)
        self.b2_l2_2 = down_conv(l2c, l3c)

        # l3
        self.b1_l3_1 = DoubleBasicBlock(l3c, l3c)
        self.b2_l3_1 = DoubleBasicBlock(l3c, l3c)
        self.fa_l3 = Fusion_Attention_Module(l3c*2, l3c)
        self.b1_l3_2 = down_conv(l3c, l4c)
        self.b2_l3_2 = down_conv(l3c, l4c)

        # l4
        self.b1_l4_1 = DoubleBasicBlock(l4c, l4c)
        self.b2_l4_1 = DoubleBasicBlock(l4c, l4c)
        self.fa_l4 = Fusion_Attention_Module(l4c*2, l4c)
        self.b1_l4_2 = down_conv(l4c, l5c)
        self.b2_l4_2 = down_conv(l4c, l5c)

        # l5
        self.b1_l5_1 = BasicBlock(l5c, l5c, if_relu=True)
        self.b2_l5_1 = BasicBlock(l5c, l5c, if_relu=True)
        self.fa_l5 = BasicBlock(l5c*2, l5c, downsample=nn.Sequential(conv1x1(in_planes=l5c*2, out_planes=l5c), BatchNorm2d(l5c, momentum=BN_MOMENTUM)), if_relu=True)

        # decoder
        # d5
        self.d5_1 = up_conv(l5c, l4c)
        # d4
        self.d4_1 = DoubleBasicBlock(l4c*3, l4c, nn.Sequential(conv1x1(in_planes=l4c*3, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.d4_2 = up_conv(l4c, l3c)
        # d3
        self.d3_1 = DoubleBasicBlock(l3c*3, l3c, nn.Sequential(conv1x1(in_planes=l3c*3, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.d3_2 = up_conv(l3c, l2c)
        # d2
        self.d2_1 = DoubleBasicBlock(l2c*3, l2c, nn.Sequential(conv1x1(in_planes=l2c*3, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.d2_2 = up_conv(l2c, l1c)
        # d1
        self.d1_1 = DoubleBasicBlock(l1c*3, l1c, nn.Sequential(conv1x1(in_planes=l1c*3, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.d1_2 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABNSync):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABN):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input1, input2):

        x1_1 = self.b1_l1_1(input1)
        x2_1 = self.b2_l1_1(input2)
        x1_1_fa, x2_1_fa = self.fa_l1(x1_1, x2_1)

        x1_2 = self.b1_l1_2(x1_1_fa)
        x2_2 = self.b2_l1_2(x2_1_fa)
        x1_2 = self.b1_l2_1(x1_2)
        x2_2 = self.b2_l2_1(x2_2)
        x1_2_fa, x2_2_fa = self.fa_l2(x1_2, x2_2)

        x1_3 = self.b1_l2_2(x1_2_fa)
        x2_3 = self.b2_l2_2(x2_2_fa)
        x1_3 = self.b1_l3_1(x1_3)
        x2_3 = self.b2_l3_1(x2_3)
        x1_3_fa, x2_3_fa = self.fa_l3(x1_3, x2_3)

        x1_4 = self.b1_l3_2(x1_3_fa)
        x2_4 = self.b2_l3_2(x2_3_fa)
        x1_4 = self.b1_l4_1(x1_4)
        x2_4 = self.b2_l4_1(x2_4)
        x1_4_fa, x2_4_fa = self.fa_l4(x1_4, x2_4)

        x1_5 = self.b1_l4_2(x1_4_fa)
        x2_5 = self.b2_l4_2(x2_4_fa)
        x1_5 = self.b1_l5_1(x1_5)
        x2_5 = self.b2_l5_1(x2_5)
        x_5 = torch.cat((x1_5, x2_5), dim=1)
        x_5 = self.fa_l5(x_5)

        x_4 = self.d5_1(x_5)
        x_4 = torch.cat((x_4, x1_4_fa, x2_4_fa), dim=1)
        x_4 = self.d4_1(x_4)

        x_3 = self.d4_2(x_4)
        x_3 = torch.cat((x_3, x1_3_fa, x2_3_fa), dim=1)
        x_3 = self.d3_1(x_3)

        x_2 = self.d3_2(x_3)
        x_2 = torch.cat((x_2, x1_2_fa, x2_2_fa), dim=1)
        x_2 = self.d2_1(x_2)

        x_1 = self.d2_2(x_2)
        x_1 = torch.cat((x_1, x1_1_fa, x2_1_fa), dim=1)
        x_1 = self.d1_1(x_1)

        x_1 = self.d1_2(x_1)

        return x_1

# if __name__ == '__main__':
#     model = GobletNet(1, 2)
#     # total = sum([param.nelement() for param in model.parameters()])
#     # from thop import profile, clever_format
#
#     # input = torch.randn(1, 1, 128, 128)
#     # flops, params = profile(model, inputs=(input, input, ))
#     # macs, params = clever_format([flops, params], "%.3f")
#     # print(macs)
#     # print(params)
#     # print(total)
#     model.eval()
#     input1 = torch.rand(2,1,128,128)
#     input2 = torch.rand(2,1,128,128)
#     x = model(input1, input2)
#     output1 = x.data.cpu().numpy()
#     # print(output)
#     print(output1.shape)

