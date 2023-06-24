from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import os

class CC3DPoints(Dataset):

    def __init__(self, path, split="train", transforms = []):

        assert split in ("train", "test")

        self.path = os.path.join(path, split)
        self.length = len(os.listdir(self.path))
        self.transforms = transforms

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        fname = os.path.join(self.path, f"{index}.xyz")

        pcloud = o3d.io.read_point_cloud(fname)
        # points = torch.from_numpy(np.asarray(pcloud.points)).float()
        points = np.asarray(pcloud.points)

        for t in self.transforms:
            points = t(points)

        return points