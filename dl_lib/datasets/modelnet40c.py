"""
Code from original ModelNet40-C repo --> https://github.com/jiachens/ModelNet40-C.git
"""
from torch.utils.data import Dataset
import os
import numpy as np

def load_data(data_path,corruption,severity):

    DATA_DIR = os.path.join(data_path, 'data_' + corruption + '_' +str(severity) + '.npy')
    # if corruption in ['occlusion']:
    #     LABEL_DIR = os.path.join(data_path, 'label_occlusion.npy')
    LABEL_DIR = os.path.join(data_path, 'label.npy')
    all_data = np.load(DATA_DIR)
    all_label = np.load(LABEL_DIR)
    return all_data, all_label

class ModelNet40C(Dataset):
    def __init__(self, path, corruption, severity):
        self.data_path = path
        self.corruption = corruption
        self.severity = severity

        self.data, self.label = load_data(self.data_path, self.corruption, self.severity)
        self.partition =  'test'

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        #return {'pc': pointcloud, 'label': label.item()}
        return pointcloud, label.item()

    def __len__(self):
        return self.data.shape[0]
