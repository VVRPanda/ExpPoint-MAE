import torch
import torch.nn as nn
import torch.nn.init as init
import math
__all__ = ["LinearLayer", "MaxPool1D"]

class LinearLayer(nn.Module):

    def __init__(self, in_channels, out_channels, fc=False, use_norm=True, use_relu=True):
        super().__init__()

        # we can skip the bias if we use batch normalization
        bias = not use_norm

        layers = [nn.Linear(in_channels, out_channels, bias=bias) if fc else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)]

        if use_norm:
            layers.append(nn.BatchNorm1d(out_channels))

        if use_relu:
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

        # initializing the weights of the module
        self.init_weights()

    def init_weights(self):
        init.kaiming_uniform_(self.net[0].weight, a=math.sqrt(2))
        # bias initialization?

    def forward(self, x):
        return self.net(x)


class MaxPool1D(nn.Module):
    def forward(self, x):
        return x.max(dim=-1)[0]
