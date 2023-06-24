import os
from torch.utils.data import Dataset
from torch_geometric.io import read_obj
import torch
import numpy as np



class ShapeNet(Dataset):

    def __init__(self, root, category="all", transforms=[]):
        # storing the dataset root path
        self.root = root
        # storing the transforms
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

        # getting the available shapenet categories and mapping them to intigers
        self.categories = self.read_categories()
        self.create_label_map()

        # creating a list to store the items of the dataset
        self.items = []

        if category == "all": 
            for cat in self.categories:
                self.read_category(cat)
        # read a specific category from shapenet or a list of categories
        else:
            if not isinstance(category, (list, tuple)):
                category = [category]
            for cat in category:
                assert cat in self.categories, "Not a valid category"
                self.read_category(cat)


    def read_category(self, category):
        # get the path of the said category
        category_path = os.path.join(self.root, category)
        # get the subfolder names
        subfolders = os.listdir(category_path)
        subfolders = [os.path.join(category, subf) for subf in subfolders]
        self.items.extend(subfolders)
    
    def read_categories(self):
        # reading available categories from subdirectories
        # and storing the alphabetically
        categories = os.listdir(self.root)
        categories.sort()
        return categories

    def create_label_map(self):
        # mapping every category to an intiger 
        # also creating an inverse dir to map intigers to the original labels
        self.label_map = {}
        self.inv_label_map = {}
        for i, cat in enumerate(self.categories):
            self.label_map[cat] = i
            self.inv_label_map[str(i)] = cat

    # read the number of examples
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        category = item.split("/")[0] 

        # reading the model
        model_path = os.path.join(self.root, item, "model.obj")
        model = read_obj(model_path)

        # applying transforms to the model(mesh/pointcloud)
        for t in self.transforms:
            model = t(model)
        
        # adding the label information
        model.label = self.label_map[category]

        return model


class ShapeNet55(Dataset):
    """
    Version of ShapeNet used in Point-BERT (https://github.com/lulutang0608/Point-BERT/tree/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52)
    for pretraining the model.
    There are some minor modifications to the original code. 
    """

    def __init__(self, root, split, s_points=1024):

        assert split in ['train', 'test']

        # number of presampled points
        self.n_points = 8192
        # number of points to sample/use
        self.s_points = s_points

        pc_folder_name = 'shapenet_pc'
        split_folder_name = 'ShapeNet-55'
        split_file_name = f'{split}.txt'

        self.pc_path = os.path.join(root, pc_folder_name)
        self.split_file_path = os.path.join(root, split_folder_name, split_file_name)

        # reading split file names
        with open(self.split_file_path, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]

            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id'   : model_id,
                'file_path'  : line
            })

    
    def pc_norm(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        permutation = np.arange(self.n_points)
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        pc_path = os.path.join(self.pc_path, sample['file_path'])
        data = np.load(pc_path).astype(np.float32)
        data = self.random_sample(data, self.s_points)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()

        return data

    def __len__(self):
        return len(self.file_list)


class ShapeNet55Custom(Dataset):
    """
    Version of ShapeNet used in Point-BERT (https://github.com/lulutang0608/Point-BERT/tree/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52)
    for pretraining the model.
    There are some minor modifications to the original code. 
    """

    def __init__(self, root, split, s_points=1024, transforms=[]):

        assert split in ['train', 'test']

        # number of presampled points
        self.n_points = 8192
        # number of points to sample/use
        self.s_points = s_points
        # transforms to apply to the data
        if not isinstance(transforms, (list, tuple)):
            transforms = [transforms] # handle single transform
        self.transfroms = transforms

        pc_folder_name = 'shapenet_pc'
        split_folder_name = 'ShapeNet-55'
        split_file_name = f'{split}.txt'

        self.pc_path = os.path.join(root, pc_folder_name)
        self.split_file_path = os.path.join(root, split_folder_name, split_file_name)

        # reading split file names
        with open(self.split_file_path, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]

            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id'   : model_id,
                'file_path'  : line
            })

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        pc_path = os.path.join(self.pc_path, sample['file_path'])
        data = np.load(pc_path).astype(np.float32)
        # apply input transforms
        for t in self.transfroms:
            data = t(data)

        return data

    def __len__(self):
        return len(self.file_list)
