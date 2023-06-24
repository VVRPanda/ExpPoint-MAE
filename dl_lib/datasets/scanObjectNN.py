from torch.utils.data import Dataset
import h5py
import numpy as np
import os


class ScanObjectNN(Dataset):

    def __init__(self, path, subset, transforms=[], return_label=True):
        
        self.path = path
        self.subset = subset
        self.transforms = transforms
        self.return_label = return_label

        if self.subset == "train":  
            h5 = h5py.File(os.path.join(self.path, 'training_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == "test":
            h5 = h5py.File(os.path.join(self.path, 'test_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1]) #2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()

        for t in self.transforms:
            current_points = t(current_points)

        if not self.return_label:
            return current_points

        label = self.labels[idx]
        
        return current_points, np.array([label])

class ScanObjectNN_hardest(ScanObjectNN):
    
    def __init__(self, path, subset, transforms=[], return_label=True):
        
        self.path = path
        self.subset = subset
        self.transforms = transforms
        self.return_label = return_label


        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.path, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.path, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError


    