"""Original per-forward random Linear + Tanh. Only device placement is portable."""
import torch
from torch import nn

class ConvLayer_dis(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_relu=True):
        super(ConvLayer_dis, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_relu = use_relu
        self.LeakyReLU=nn.LeakyReLU(0.2)
    def forward(self, x):

        out = self.conv2d(x)
        if self.use_relu is True:
            out = self.LeakyReLU(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels, tanh=True):
        super(Discriminator, self).__init__()
        fliter = [in_channels,16,32,64,128]
        kernel_size = 3
        stride = 2
        self.l1 = ConvLayer_dis(fliter[0], fliter[1], kernel_size, stride, use_relu=True)
        self.l2 = ConvLayer_dis(fliter[1], fliter[2], kernel_size, stride, use_relu=True)
        self.l3 = ConvLayer_dis(fliter[2], fliter[3], kernel_size, stride, use_relu=True)
        self.l4 = ConvLayer_dis(fliter[3], fliter[4], kernel_size, stride, use_relu=True)
        if not tanh:
            raise ValueError('Original-head version requires Tanh')
        self.tanh = nn.Tanh()


    def forward(self, x):

        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = out.view(out.size()[0], -1)
        linear = nn.Linear(out.size()[1], 1).to(out.device)
        out = self.tanh(linear(out))

        return out.squeeze()
