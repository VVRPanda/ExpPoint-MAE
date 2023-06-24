from torchvision.transforms import Compose
import numpy as np
import torch
import math
import random


__all__ = ['TwoCropsTransform', 'RandomPointKeep', 'UnitSphereNormalization', 'RandomShuffle', 'AnisotropicScale', \
            'RandomPointDropout', 'ToTensor', 'RandomRotate', 'FirstKPointsKeep']

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        if isinstance(base_transform, (list, tuple)):
            base_transform = Compose(base_transform)
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class RandomPointKeep:
    # NOT tested
    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, x):
        np.random.shuffle(x)
        return x[:self.num_points, :]

class FirstKPointsKeep:

    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, x):
        return x[:self.num_points, :]

class UnitSphereNormalization:

    def __call__(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


class RandomShuffle:
    def __call__(self, pc):
        np.random.shuffle(pc)
        return pc

class AnisotropicScale:
    def __call__(self, pc):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        pointcloud = pc
        translated_pointcloud = pointcloud * xyz1 + xyz2
        return translated_pointcloud.astype(np.float32)

class RandomPointDropout:

    def __init__(self, max_dropout_ratio = 0.875):
        self.max_dropout_ratio = max_dropout_ratio
        
    def __call__(self, pc):  
        dropout_ratio = np.random.random() * self.max_dropout_ratio # 0 ~ 0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]

        if len(drop_idx) > 0:
            pc[drop_idx,:] = pc[0,:] # set to the first point

        return pc

class ToTensor:

    def __call__(self, pc):
        pc = torch.from_numpy(pc)
        return pc



class RandomRotate:

    def __init__(self, theta, axis, torch_or_np = "torch"):
        # Args: 
        #   - theta      : rotation angle (rotation will be from [-theta, theta])
        #   - axis       : the rotation axis (0,1,2 for x,y,z)
        #   - torch_or_np: whether the input pointcloud is represented by a torch.Tensor or np.Array

        self.theta = math.pi * theta / 180.0 # transforming angle to rads from degs
        self.axis = axis
        assert torch_or_np in ["torch", "np", "numpy"]
        self.use_torch = True if torch_or_np == "torch" else False

    def __call__(self, pc):
        # pc : a set of points with shape Nx3

        degree = random.uniform(-self.theta, self.theta)
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        if self.use_torch:
            matrix = torch.tensor(matrix)
            pc = pc.unsqueeze(-1)
        else: 
            matrix = np.array(matrix)
            pc = pc[..., np.newaxis]

        return (matrix @ pc).squeeze(-1)
