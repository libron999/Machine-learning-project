"""Direct IR+RGB -> RGB extension of the official triple-path ICAFusion."""
import torch
from torch import nn
import torch.nn.functional as F
from .attention import InterAtt, CompAtt
from .modules import MGDC, LEGM, HFP, FFCM, ResidualAdapter


class ConvLayer(nn.Module):
    def __init__(self, cin, cout, stride=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride)
        self.act = nn.PReLU() if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(F.pad(x, (1, 1, 1, 1), mode='reflect')))


class Generator(nn.Module):
    def __init__(self, base_channels=16, modules=None, residual_scale=0.01,
                 hfp_ratio=(0.25, 0.25)):
        super().__init__()
        c = base_channels
        if c < 4 or c % 4:
            raise ValueError('base_channels must be >=4 and divisible by 4')
        flags = dict(mgdc=False, legm=False, hfp=False, ffcm=False)
        if modules:
            if set(modules) - set(flags) or any(type(v) is not bool for v in modules.values()):
                raise ValueError('modules must contain boolean mgdc/legm/hfp/ffcm flags')
            flags.update(modules)
        self.conv1 = ConvLayer(4, c)
        self.ir_stem = ConvLayer(1, c)
        self.vis_stem = ConvLayer(3, c)
        # Preserve modality-shared conv3/5/6 and shared compensatory attention.
        self.conv3, self.conv4 = ConvLayer(c, c), ConvLayer(c, c)
        self.conv5, self.conv6 = ConvLayer(c, 2*c, 2), ConvLayer(2*c, 4*c, 2)
        self.inter_conv_1 = ConvLayer(4*c, 2*c, 2)
        self.inter_conv_2 = ConvLayer(8*c, 4*c, 2)
        self.inter = nn.ModuleList([InterAtt(2*c), InterAtt(4*c), InterAtt(8*c)])
        self.comp = nn.ModuleList([CompAtt(c), CompAtt(2*c), CompAtt(4*c)])
        self.conv7, self.conv8 = ConvLayer(24*c, 24*c), ConvLayer(24*c, 8*c)
        self.conv9, self.conv10 = ConvLayer(12*c, 12*c), ConvLayer(12*c, 4*c)
        self.conv11, self.conv12 = ConvLayer(6*c, 2*c), ConvLayer(2*c, 3, activation=False)
        def adapt(enabled, factory, channels):
            return ResidualAdapter(factory(channels), channels, residual_scale) if enabled else nn.Identity()
        # MGDC shared between IR/VIS, like the following shared original encoder.
        self.mgdc = adapt(flags['mgdc'], MGDC, c)
        self.legm = LEGM(2*c, residual_scale) if flags['legm'] else nn.Identity()
        self.hfp = adapt(flags['hfp'], lambda n: HFP(n, hfp_ratio), c)
        self.ffcm = adapt(flags['ffcm'], FFCM, 24*c)

    def forward(self, ir, vis):
        if ir.ndim != 4 or vis.ndim != 4 or ir.shape[1] != 1 or vis.shape[1] != 3:
            raise ValueError('Expected IR [B,1,H,W] and RGB [B,3,H,W]')
        if ir.shape[0] != vis.shape[0] or ir.shape[-2:] != vis.shape[-2:]:
            raise ValueError('IR and RGB batch/spatial sizes must match')
        h, w = ir.shape[-2:]
        if min(h, w) < 16:
            raise ValueError('Minimum input size is 16x16')
        # Pad bottom/right before encoding; avoids ambiguous odd-size skip alignment.
        pad = (0, (-w) % 4, 0, (-h) % 4)
        ir, vis = F.pad(ir, pad, mode='reflect'), F.pad(vis, pad, mode='reflect')
        concat = self.conv4(self.conv1(torch.cat((ir, vis), 1)))
        ir2 = self.mgdc(self.conv3(self.ir_stem(ir)))
        vis2 = self.mgdc(self.conv3(self.vis_stem(vis)))
        ir3, vis3 = self.conv5(ir2), self.legm(self.conv5(vis2))
        ir4, vis4 = self.conv6(ir3), self.conv6(vis3)
        f1 = self.inter[0](torch.cat((concat, ir2), 1), torch.cat((concat, vis2), 1))
        p1 = self.inter_conv_1(f1)
        f2 = self.inter[1](torch.cat((p1, ir3), 1), torch.cat((p1, vis3), 1))
        p2 = self.inter_conv_2(f2)
        f3 = self.inter[2](torch.cat((p2, ir4), 1), torch.cat((p2, vis4), 1))
        fused = self.ffcm(torch.cat((f3, self.comp[2](ir4), self.comp[2](vis4)), 1))
        de1 = self.conv8(self.conv7(F.interpolate(fused, size=ir3.shape[-2:], mode='nearest')))
        de1 = torch.cat((de1, self.comp[1](ir3), self.comp[1](vis3)), 1)
        de2 = self.conv10(self.conv9(F.interpolate(de1, size=ir2.shape[-2:], mode='nearest')))
        de2 = torch.cat((de2, self.hfp(self.comp[0](ir2)), self.hfp(self.comp[0](vis2))), 1)
        return self.conv12(self.conv11(de2)).tanh()[..., :h, :w]
