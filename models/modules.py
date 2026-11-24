"""Research adaptations; see SOURCES.md for provenance and exact deviations.

No MMCV, timm or custom CUDA extensions are required. Frequency operations run
in float32 even inside autocast. Model parameters must remain float32.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ResidualAdapter(nn.Module):
    def __init__(self, module, channels, scale=0.01):
        super().__init__()
        self.module = module
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), scale))

    def forward(self, x):
        return x + self.scale * self.module(x)


class ShiftPReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.before = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.after = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.act = nn.PReLU(channels)

    def forward(self, x):
        return self.act(x - self.before) + self.after


class MGDC(nn.Module):
    """Full-precision BHViT token_mixer adaptation, groups=4, dilation=1/3/5."""
    def __init__(self, channels, groups=4, dilations=(1, 3, 5)):
        super().__init__()
        if channels % groups:
            raise ValueError('MGDC channels must be divisible by groups')
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                                    groups=groups), ShiftPReLU(channels)) for d in dilations])
        self.norm = LayerNorm2d(channels)

    def forward(self, x):
        x = x + self.bias
        return self.norm(sum(branch(x) for branch in self.branches))


class FreMLP(nn.Module):
    """DarkIR amplitude processing with original phase reconstruction."""
    def __init__(self, channels):
        super().__init__()
        self.process = nn.Sequential(nn.Conv2d(channels, 2 * channels, 1),
                                     nn.LeakyReLU(0.1), nn.Conv2d(2 * channels, channels, 1))

    def forward(self, x):
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft2(x.float(), norm='backward')
            magnitude = self.process(spectrum.abs())
            phase = spectrum.angle()
            spectrum = torch.complex(magnitude * phase.cos(), magnitude * phase.sin())
            result = torch.fft.irfft2(spectrum, s=x.shape[-2:], norm='backward')
        return result.to(dtype)


class LEGM(nn.Module):
    """User PDF's Frequency_Domain wrapper; LEGM is its supplied alias."""
    def __init__(self, channels, scale=0.01):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.freq = FreMLP(channels)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), scale))

    def forward(self, x):
        return x + self.gamma * x * self.freq(self.norm(x))


def dct(x):
    """Orthonormal DCT-II along the last axis, via a 2N even extension."""
    n = x.shape[-1]
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    phase = torch.polar(torch.ones_like(k), -math.pi * k / (2 * n))
    spectrum = torch.fft.rfft(torch.cat((x, x.flip(-1)), -1), dim=-1)[..., :n]
    scale = torch.full_like(k, math.sqrt(2 / n))
    scale[0] = math.sqrt(1 / n)
    return (spectrum * phase).real * (scale / 2)


def idct(x):
    """Inverse of the orthonormal DCT-II above."""
    n = x.shape[-1]
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    scale = torch.full_like(k, math.sqrt(2 / n))
    scale[0] = math.sqrt(1 / n)
    phase = torch.polar(torch.ones_like(k), math.pi * k / (2 * n))
    spectrum = (2 * x / scale) * phase
    spectrum = torch.cat((spectrum, torch.zeros_like(spectrum[..., :1])), -1)
    return torch.fft.irfft(spectrum, n=2 * n, dim=-1)[..., :n]


def dct2(x):
    return dct(dct(x).transpose(-1, -2)).transpose(-1, -2)


def idct2(x):
    return idct(idct(x).transpose(-1, -2)).transpose(-1, -2)


class HFP(nn.Module):
    """HS-FPN high-frequency spatial/channel paths, without MMCV dependency."""
    def __init__(self, channels, ratio=(0.25, 0.25), patch=(8, 8)):
        super().__init__()
        if not all(0 < r < 1 for r in ratio):
            raise ValueError('HFP low-frequency removal ratio must be between 0 and 1')
        self.ratio, self.patch = ratio, patch
        groups = math.gcd(32, channels)
        self.channel1 = nn.Sequential(nn.Conv2d(channels, channels, 1, groups=groups,
                                               bias=False), nn.ReLU())
        self.channel2 = nn.Sequential(nn.Conv2d(channels, channels, 1, groups=groups,
                                               bias=False), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                                 nn.GroupNorm(groups, channels))

    def forward(self, x):
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            h, w = xf.shape[-2:]
            spectrum = dct2(xf)
            mask = torch.ones((h, w), device=x.device, dtype=xf.dtype)
            mask[:int(h * self.ratio[0]), :int(w * self.ratio[1])] = 0
            high = idct2(spectrum * mask)
            spatial = xf * high
            maximum = F.adaptive_max_pool2d(high, self.patch).relu().sum((2, 3), keepdim=True)
            average = F.adaptive_avg_pool2d(high, self.patch).relu().sum((2, 3), keepdim=True)
            weights = self.channel2(self.channel1(maximum) + self.channel1(average)).sigmoid()
            result = self.out(spatial + xf * weights)
        return result.to(dtype)


class FourierUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Sequential(nn.Conv2d(2 * channels, 2 * channels, 1, bias=False),
                                 nn.BatchNorm2d(2 * channels), nn.ReLU())

    def forward(self, x):
        b, c, h, w = x.shape
        spectrum = torch.fft.rfft2(x, norm='ortho')
        packed = torch.view_as_real(spectrum).permute(0, 1, 4, 2, 3).reshape(b, 2*c, h, w//2+1)
        packed = self.mix(packed).reshape(b, c, 2, h, w//2+1)
        spectrum = torch.view_as_complex(packed.permute(0, 1, 3, 4, 2).contiguous())
        return torch.fft.irfft2(spectrum, s=(h, w), norm='ortho')


class FFCM(nn.Module):
    """FADformer's Fused_Fourier_Conv_Mixer, no unused constructor parameters."""
    def __init__(self, channels):
        super().__init__()
        c = channels
        self.input = nn.Sequential(nn.Conv2d(c, 2*c, 1), nn.GELU())
        self.local3 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c,
                                             padding_mode='reflect'), nn.GELU())
        self.local5 = nn.Sequential(nn.Conv2d(c, c, 5, padding=2, groups=c,
                                             padding_mode='reflect'), nn.GELU())
        self.pre1 = nn.Sequential(nn.Conv2d(c, c, 1), nn.GELU())
        self.pre2 = nn.Sequential(nn.Conv2d(c, c, 1), nn.GELU())
        self.fourier = FourierUnit(2*c)
        self.post = nn.Sequential(nn.BatchNorm2d(2*c), nn.ReLU())
        self.reduce = nn.Sequential(nn.Conv2d(2*c, c, 1), nn.Conv2d(c, c, 3, padding=1,
                                    groups=c, padding_mode='reflect'), nn.GELU())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(1, c//4), 1),
                                nn.GELU(), nn.Conv2d(max(1, c//4), c, 1), nn.Sigmoid())

    def forward(self, x):
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            # Both local kernels use the FIRST split in the official code too.
            a, _ = self.input(x.float()).chunk(2, 1)
            local = torch.cat((self.pre1(self.local3(a)), self.pre2(self.local5(a))), 1)
            mixed = self.post(self.fourier(local) + local)
            result = self.reduce(mixed)
            result = result * self.ca(result)
        return result.to(dtype)
