import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def clone(layer, n):
    return nn.ModuleList([deepcopy(layer) for _ in range(n)])


class LayerNorm(nn.Module):
    def __init__(self, channel_dim, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(channel_dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(channel_dim, 1, 1))
        self.eps = epsilon

    def forward(self, tensor):
        mean = tensor.mean(1, keepdim=True)
        std = tensor.std(1, keepdim=True)
        return self.a * (tensor - mean) / (std + self.eps) + self.b


class Preparor(nn.Module):
    def __init__(self, in_channels):
        super(Preparor, self).__init__()
        self.Conv_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=(1, 1))
        self.Pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm = LayerNorm(32)

    def forward(self, input_tensor):
        conved = self.Conv_1(input_tensor)
        conved = F.elu(self.norm(conved))
        padded = F.pad(conved, (1, 0, 1, 0))
        return self.Pool_1(padded)


class Sublayer_connection(nn.Module):
    def __init__(self, input_channel, output_channel, input_size, out_size):
        super(Sublayer_connection, self).__init__()
        self.channel = input_channel
        self.norm = LayerNorm(output_channel)
        self.cross_dim_res = None
        self.change = False
        if input_channel != output_channel and input_size != out_size:
            self.cross_dim_res = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=2)
            self.change = True
        elif input_channel != output_channel:
            self.cross_dim_res = nn.Conv2d(input_channel, output_channel, kernel_size=1)
            self.change = True

    def forward(self, x, sublayer):
        if not self.change:
            return F.elu(self.norm(x + sublayer(x)))
        else:
            transformed_res = self.cross_dim_res(x)
            sub = sublayer(x)
            return F.elu(self.norm(transformed_res + sub))


class Res_unit(nn.Module):
    def __init__(self, input_channel, out_channel, first_stride, second_stride, dialat=1):
        super(Res_unit, self).__init__()
        if first_stride == 2:
            self.first_padding = (1, 0, 1, 0)
            self.second_padding = (1, 1, 1, 1)
        else:
            self.first_padding = (1, 1, 1, 1)
            self.second_padding = (1, 1, 1, 1)
        if dialat != 1:
            self.first_padding = (2, 2, 2, 2)
            self.second_padding = (2, 2, 2, 2)
        self.dialat = dialat
        self.first_conv = nn.Conv2d(input_channel, out_channel, kernel_size=3, \
                                    stride=first_stride, dilation=dialat)

        self.second_conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, \
                                     stride=second_stride, dilation=dialat)
        self.norm = LayerNorm(out_channel)

    def forward(self, tensor):
        conved = self.first_conv(F.pad(tensor, (self.first_padding)))
        normed = F.elu(self.norm(conved))
        return self.second_conv(F.pad(normed, self.second_padding))


class Res_unit_b4(nn.Module):
    def __init__(self, input_channel, mid_out_channel, out_channel):
        super(Res_unit_b4, self).__init__()
        self.padding = (2, 2, 2, 2)
        self.first_conv = nn.Conv2d(input_channel, mid_out_channel, kernel_size=3, \
                                    dilation=2)
        self.second_conv = nn.Conv2d(mid_out_channel, out_channel, kernel_size=3, \
                                     dilation=2)

        self.norm = LayerNorm(mid_out_channel)

    def forward(self, tensor):
        conved = self.first_conv(F.pad(tensor, self.padding))
        normed = F.elu(self.norm(conved))
        return self.second_conv(F.pad(normed, self.padding))


class Block_4(nn.Module):
    def __init__(self):
        super(Block_4, self).__init__()
        self.sub_block_1 = Res_unit_b4(256, 256, 512)
        self.trans = Sublayer_connection(256, 512, 64, 64)
        self.sub_block_2 = clone(Res_unit_b4(512, 256, 512), 2)
        self.sub_connects = clone(Sublayer_connection(512, 512, 64, 64), 2)

    def forward(self, tensor):
        first = self.trans(tensor, self.sub_block_1)
        for i in range(len(self.sub_block_2)):
            first = self.sub_connects[i](first, self.sub_block_2[i])
        return first


class Block_1(nn.Module):
    def __init__(self):
        super(Block_1, self).__init__()
        self.sub_block_1 = Res_unit(32, 64, 2, 1)
        self.trans = Sublayer_connection(32, 64, 256, 128)
        self.sub_block_2 = clone(Res_unit(64, 64, first_stride=1, second_stride=1), 2)
        self.sub_connects = clone(Sublayer_connection(64, 64, 128, 128), 2)

    def forward(self, tensor):
        first_conved = self.trans(tensor, self.sub_block_1)
        for i in range(len(self.sub_block_2)):
            first_conved = self.sub_connects[i](first_conved, self.sub_block_2[i])
        return first_conved


class Block_2(nn.Module):
    def __init__(self):
        super(Block_2, self).__init__()
        self.sub_block1 = Res_unit(64, 128, first_stride=2, second_stride=1)
        self.sub_block2 = clone(Res_unit(128, 128, first_stride=1, second_stride=1), 2)
        self.trans = Sublayer_connection(64, 128, 128, 64)
        self.sub_connects = clone(Sublayer_connection(128, 128, 64, 64), 2)

    def forward(self, tensor):
        first = self.trans(tensor, self.sub_block1)
        for i in range(len(self.sub_block2)):
            first = self.sub_connects[i](first, self.sub_block2[i])
        return first


class Block_3(nn.Module):
    def __init__(self):
        super(Block_3, self).__init__()
        self.sub_block1 = Res_unit(128, 256, 1, 1, dialat=2)
        self.trans = Sublayer_connection(128, 256, 64, 64)
        self.sub_block2 = clone(Res_unit(256, 256, 1, 1, dialat=2), 5)
        self.sub_connects = clone(Sublayer_connection(256, 256, 64, 64), 5)

    def forward(self, tensor):
        first = self.trans(tensor, self.sub_block1)
        for i in range(len(self.sub_block2)):
            first = self.sub_connects[i](first, self.sub_block2[i])
        return first


class Block5_6res(nn.Module):
    def __init__(self, input_channel):
        super(Block5_6res, self).__init__()
        self.padding = (2, 2, 2, 2)
        self.first_conv = nn.Conv2d(input_channel, input_channel // 2, 1)
        self.second_conv = nn.Conv2d(input_channel // 2, input_channel, 3, dilation=2)
        self.third_conv = nn.Conv2d(input_channel, input_channel * 2, 1)
        self.norm1 = LayerNorm(input_channel // 2)
        self.norm2 = LayerNorm(input_channel)

    def forward(self, tensor):
        first = F.elu(self.norm1(self.first_conv(tensor)))
        second = F.elu(self.norm2(self.second_conv(F.pad(first, self.padding))))
        return self.third_conv(second)


class Block_5(nn.Module):
    def __init__(self):
        super(Block_5, self).__init__()
        self.sub_connect = Sublayer_connection(512, 1024, 64, 64)
        self.trans = Block5_6res(512)

    def forward(self, tensor):
        return self.sub_connect(tensor, self.trans)


class Block_6(nn.Module):
    def __init__(self):
        super(Block_6, self).__init__()
        self.sub_connect = Sublayer_connection(1024, 2048, 64, 64)
        self.trans = Block5_6res(1024)

    def forward(self, tensor):
        return self.sub_connect(tensor, self.trans)


class Expansive_path(nn.Module):
    def __init__(self):
        super(Expansive_path, self).__init__()
        self.first_conv = nn.Conv2d(32, 16, 3, padding=(1, 1))
        self.second_conv = nn.Conv2d(16, 8, 3, padding=(1, 1))
        self.norm1 = LayerNorm(16)
        self.norm2 = LayerNorm(8)
        # self.dropout1 = nn.Dropout2d()
        # self.dropout2 = nn.Dropout2d()

    def forward(self, tensor):
        droped = F.elu(self.norm1(self.first_conv(tensor)))
        return F.elu(self.norm2(self.second_conv(droped)))


class Output(nn.Module):
    def __init__(self, num_classes):
        super(Output, self).__init__()
        self.conv = nn.Conv2d(8, num_classes, 1)

    def forward(self, tensor):
        res = self.conv(tensor)
        return res


class CRN(nn.Module):
    def __init__(self, Pre, Block1, Block2, Block3, Block4, Block5, Block6, expansive, output):
        super(CRN, self).__init__()
        self.blocks = nn.ModuleList([Block1, Block2, Block3, Block4, Block5, Block6])
        self.Deconv_downsample = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=(1, 1))
        self.norm = LayerNorm(32)
        self.Deconv_Block1 = nn.ConvTranspose2d(64, 32, 8, 4, padding=(2, 2))
        self.Deconv = nn.ConvTranspose2d(2048, 32, 16, 8, padding=(4, 4))
        self.Pre = Pre
        self.expansive = expansive
        self.output_layer = output

    def forward(self, tensor):
        prepared = self.Pre(tensor)
        copy_pre = prepared
        block1_d = self.blocks[0](prepared)
        copy_block1_d = block1_d
        for i in range(1, len(self.blocks)):
            block1_d = self.blocks[i](block1_d)
        res_pre = self.Deconv_downsample(copy_pre)
        res_1_d = self.Deconv_Block1(copy_block1_d)
        res_fin = self.Deconv(block1_d)
        tens_for_expan = res_pre + res_1_d + res_fin
        tens_for_expan = F.elu(self.norm(tens_for_expan))
        expansed = self.expansive(tens_for_expan)
        return self.output_layer(expansed)

def create_model(in_channels, num_classes):
    Pre = Preparor(in_channels)
    Block1 = Block_1()
    Block2 = Block_2()
    Block3 = Block_3()
    Block4 = Block_4()
    Block5 = Block_5()
    Bolck6 = Block_6()
    expansive = Expansive_path()
    output = Output(num_classes)
    return CRN(Pre, Block1, Block2, Block3, Block4, Block5, Bolck6, expansive, output)

def DCR(in_channels, num_classes):
    model = create_model(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model


# if __name__ == '__main__':
#     model = DCR(1,10)
#     model.eval()
#     input = torch.rand(2,1,256,256)
#     output = model(input)
#     # print(output)
#     print(output.shape)

if __name__ == '__main__':
    model = DCR(1,2)

    total = sum([param.nelement() for param in model.parameters()])
    from thop import profile, clever_format

    input1 = torch.randn(1, 1, 128, 128)
    input2 = torch.randn(1, 1, 128, 128)
    flops, params = profile(model, inputs=(input1,))
    print(flops, params)
    macs, params = clever_format([flops, params], "%.3f")
    print(macs)
    print(params)
    print(total)

    # model.eval()
    # input1 = torch.rand(2,1,128,128)
    # input2 = torch.rand(2,1,128,128)
    # output = model(input1, input2)
    # output = output[0].data.cpu().numpy()
    # # print(output)
    # print(output.shape)