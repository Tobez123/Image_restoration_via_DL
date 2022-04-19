import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Resblock(nn.Module):
    def __init__(self, input_nc, norm):
        super(Resblock, self).__init__()
        rbk = [nn.ReflectionPad2d(1), nn.Conv2d(input_nc, input_nc, kernel_size=3), norm(input_nc), nn.ReLU(True)]
        rbk += [nn.ReflectionPad2d(1), nn.Conv2d(input_nc, input_nc, kernel_size=3), norm(input_nc)]
        self.rbk = nn.Sequential(*rbk)
    def forward(self, x):
        output = x + self.rbk(x)
        return output

class Align_module(nn.Module):
    def __init__(self,ndf=8):
        super(Align_module, self).__init__()
        self.conv1 = nn.Conv2d(3, ndf, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer_1_1_blk = Resblock(input_nc=ndf, norm=nn.InstanceNorm2d)
        self.layer_1_2_blk = Resblock(input_nc=ndf, norm=nn.InstanceNorm2d)
        self.layer_1_3_blk = Resblock(input_nc=ndf, norm=nn.InstanceNorm2d)
        self.layer_2_1_blk = Resblock(input_nc=ndf, norm=nn.InstanceNorm2d)
        self.layer_2_2_blk = Resblock(input_nc=ndf, norm=nn.InstanceNorm2d)
        self.layer_3_1_blk = Resblock(input_nc=ndf, norm=nn.InstanceNorm2d)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.downsample_1 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1, bias=True)
        self.downsample_2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, input):
        layer_1_feature = self.lrelu(self.conv_1(input))
        layer_2_feature = self.lrelu(self.downsample_1(layer_1_feature))
        layer_3_feature = self.lrelu(self.downsample_2(layer_2_feature))

        layer_3_feature = self.upsample(self.layer_3_1_blk(layer_3_feature))
        layer_2_feature = self.layer_2_1_blk(layer_2_feature) + layer_3_feature
        layer_2_feature = self.upsample(self.layer_2_2_blk(layer_2_feature))
        layer_1_feature = self.layer_1_2_blk(self.layer_1_1_blk(layer_1_feature)) + layer_2_feature
        output = self.layer_3_1_blk(layer_1_feature)
        return output
