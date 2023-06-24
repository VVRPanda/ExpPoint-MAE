# Disclaimer: ModelNet40Sampled dataset is the version of the dataset used in dgcnn. 
#             The code is copied from the original dgcnn github repo. 
#             There are some minor changes to the code to receive a path for the dataset
# NOTES about the dataset and the augmentation procedure:
#           - Using a pre-sampled version of ModelNet40
#           - Does not apply any rotation to the input models that are pre-aligned 
#
# There is also a second version of the dataset, "ModelNet40SampledCustom". 
# This version receives custom transforms to apply to the input points.
# Note: the transforms should receive only the input pointcloud and not the labels.
# A series of such transforms as well as a custom collate function are also provided
# TODO: Move transforms and collate function to different folder.
#       They still remain in this file as they are dataset specific transforms  
#
# New Addition: ModelNet8k
#               This is a presampled version of modelnet used in Point-Bert and Point-MAE
#               (Point-Bert: https://github.com/lulutang0608/Point-BERT.git)
#               In this version you can either select the ModelNet40 or ModelNet10 version.
#               Modifications to the code have been made to make it suitable for this project

import os
import glob
import h5py 
import numpy as np
from torch.utils.data import Dataset
import pickle

def download(path=None):
    # adding the ability to use custom path
    if path is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
    else:
        DATA_DIR = path

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data(partition, path=None):
    download(path)
    # adding the ability to use custom path
    if path is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
    else: 
        DATA_DIR = path

    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40Sampled(Dataset):
    def __init__(self, path, num_points, partition='train'):
        assert partition in ['train', 'test'], "Partition should be either 'train' or 'test'"
        self.data, self.label = load_data(partition, path)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points] 
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40SampledCustom(ModelNet40Sampled):
    # A version of ModelNet40 sampled with custom transforms
    def __init__(self, path, num_points, partition='train', transforms=[]):
        super().__init__(path, num_points, partition)

        # making transforms a list to handle multiple transforms
        self.transforms = transforms if isinstance(transforms, (tuple, list)) else [transforms]
        # NOTE: transforms should operate only on the data, not the label

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        for t in self.transforms:
            pointcloud = t(pointcloud)
        return pointcloud, label       



class ModelNet8k(Dataset):
    # source: https://github.com/lulutang0608/Point-BERT.git
    def __init__(self, path, split='train', transforms=[], use_normals=False):

        assert (split == 'train' or split == 'test')
        self.path = path
        self.transforms = transforms if isinstance(transforms, list) else [transforms]
        self.use_normals = use_normals
        self.num_category = 40
        self.n_points = 8192

        # compressed file with data
        self.save_path = os.path.join(self.path, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.n_points))
        
        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        # return len(self.datapath)
        return len(self.list_of_points)

    def _get_item(self, idx):
        point_set, labels = self.list_of_points[idx], self.list_of_labels[idx]
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        labels = labels.astype(np.int64)
        return point_set, labels

    def __getitem__(self, idx):
        points, label = self._get_item(idx)
        
        for t in self.transforms:
            points = t(points)
        
        return points, label

