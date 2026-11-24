"""ICAFusion channel/spatial interaction and compensation (Wang et al.)."""
import torch
from torch import nn


class CompAtt(nn.Module):
    def __init__(self, channels):
        super().__init__()
        def branch(pool):
            return nn.Sequential(pool, nn.Conv2d(channels, channels // 2, 1),
                                 nn.PReLU(), nn.Conv2d(channels // 2, channels, 1))
        self.ca_avg = branch(nn.AdaptiveAvgPool2d(1))
        self.ca_max = branch(nn.AdaptiveMaxPool2d(1))
        self.conv1 = nn.Conv2d(2 * channels, channels, 1)
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def channel_weight(self, x):
        return self.conv1(torch.cat((self.ca_avg(x), self.ca_max(x)), 1)).sigmoid()

    def spatial_weight(self, x):
        return self.conv(torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1)).sigmoid()

    def forward(self, x):
        x = x * self.channel_weight(x)
        return x * self.spatial_weight(x)


class InterAtt(CompAtt):
    def forward(self, ir, vis):
        # Softmax across MODALITIES, not channels or spatial locations.
        wc = torch.stack((self.channel_weight(ir), self.channel_weight(vis)), 0).softmax(0)
        ir, vis = ir * wc[0], vis * wc[1]
        ws = torch.stack((self.spatial_weight(ir), self.spatial_weight(vis)), 0).softmax(0)
        return torch.cat((ir * ws[0], vis * ws[1]), 1)
