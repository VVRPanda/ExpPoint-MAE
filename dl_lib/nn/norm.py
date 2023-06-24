import math
import torch
from torch import tensor
import torch.nn as nn

class RunningBatchNorm(nn.Module):

    def __init__(self, nf, mom=0.1, eps=1e-5):
        # nf: number of features
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones(nf, 1))
        self.adds = nn.Parameter(torch.zeros(nf, 1))
        self.register_buffer('sums', torch.zeros(1, nf, 1))
        self.register_buffer('sqrs', torch.zeros(1, nf, 1))
        self.register_buffer('batch', tensor(0.))
        self.register_buffer('count', tensor(0.))
        self.register_buffer('step', tensor(0.))
        self.register_buffer('dbias', tensor(0.))

    def update_stats(self, x):
        bs, nc, *_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = self.dims
        s = x.sum(dims, keepdim=True)
        ss = (x*x).sum(dims, keepdim=True)
        # torch.numel: returns the total number of elements
        #              in the input tensor
        c = self.count.new_tensor(x.numel()/nc)
        mom1 = 1 - (1-self.mom) / math.sqrt(bs-1)
        self.mom1 = self.dbias.new_tensor(mom1)
        # torch.lepr: linear interpolation of two tensors 
        #             args: input, end, weight 
        self.sums.lerp_(s, self.mom1)
        self.sqrs.lerp_(ss, self.mom1)
        self.count.lerp_(c, self.mom1)
        self.dbias = self.dbias * (1-self.mom1) + self.mom1
        self.batch += bs
        self.step += 1

    def forward(self, x):
        if self.training: self.update_stats(x)
        sums = self.sums
        sqrs = self.sqrs
        c = self.count
        if self.step < 100:
            sums = sums / self.dbias
            sqrs = sqrs / self.dbias
            c = c / self.dbias
        means = sums / c
        vars = (sqrs/c).sub_(means*means)
        if bool(self.batch < 20): vars.clamp_min_(0.01)
        x = (x-means).div_((vars.add_(self.eps)).sqrt())
        return x.mul_(self.mults).add_(self.adds)

    @property 
    def dims(self):
        # This funcition is to specify the dimensions where the features are stored
        # e.g. for RunningBatchNone2D will be: (0, 2, 3)
        raise NotImplementedError


class RunningBatchNorm1D(RunningBatchNorm):

    @property
    def dims(self):
        return (0, 2)

class RunningBatchNorm2D(RunningBatchNorm):

    @property
    def dims(self):
        return (0, 2, 3)