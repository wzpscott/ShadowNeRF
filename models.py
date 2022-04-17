import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, x_dim, dir_dim, D=8, W=256, skips=[4]):
        super().__init__()
        self.x_dim = x_dim
        self.dir_dim = dir_dim
        self.W = W
        self.D = D
        self.skips = skips

        self.x_linears = nn.ModuleList(
            [nn.Linear(x_dim, W)] + [nn.Linear(W, W) if i not in skips else nn.Linear(W + x_dim, W) for i in range(D-1)])

        self.dir_linears = nn.ModuleList([nn.Linear(dir_dim + W, W//2)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)

    def forward(self, x, dir):
        h = x
        for i, l in enumerate(self.x_linears):
            h = self.x_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)

        h = torch.cat([feature, dir], -1)

        for i, l in enumerate(self.dir_linears):
            h = self.dir_linears[i](h)
            h = F.relu(h)
        rgb = self.rgb_linear(h)

        # output = torch.cat([alpha, rgb], dim=-1)
        output = {'alpha':alpha, 'rgb':rgb}
        return output