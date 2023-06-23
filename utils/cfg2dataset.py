from dl_lib.datasets.ready_datasets import get_ModelNet40, get_scanObjectNN
from dl_lib.datasets import ScanObjectNN, ScanObjectNN_hardest
from dl_lib.datasets import CC3DPoints, ShapeNet55Custom
from torch.utils.data import ConcatDataset, DataLoader

def get_cls_dataloader(cfg):

    dataset = cfg['dataset']['name']
    path    = cfg['dataset']['path']

    batch_size = cfg['training']['batch_size'] if 'batch_size' in cfg['training'] else 32

    if dataset == 'modelnet':
        train_loader, valid_loader = get_ModelNet40(path, 'normalized', batch_size=batch_size)

    elif dataset == 'scanObjectNN':
        train_loader, valid_loader = get_scanObjectNN(path, 'easy', batch_size=batch_size)
    
    elif dataset == 'scanObjectNN-hard':
        train_loader, valid_loader = get_scanObjectNN(path, 'hard', batch_size=batch_size)

    elif dataset == 'scanObjectNN_nobg':
        train_loader, valid_loader = get_scanObjectNN(path, 'easy', batch_size=batch_size)
    
    elif dataset == 'modelnet8k':
        train_loader, valid_loader = get_ModelNet40(path, '8k', batch_size=batch_size)
    
    elif dataset == 'modelnet8k_s1024':
        train_loader, valid_loader = get_ModelNet40(path, '8k_sampled_1k', batch_size=batch_size)

    else:
        raise NotImplementedError

    return train_loader, valid_loader


def get_pretrain_dataloader(cfg, transforms = []):

    batch_size = cfg['training']['batch_size']
    dataset_cfgs = cfg['dataset']

    if not isinstance(dataset_cfgs, list):
        dataset_cfgs = [dataset_cfgs]

    datasets = []

    for cfg in dataset_cfgs:

        if cfg['name'] == 'cc3d':
            datasets.append(
                CC3DPoints(cfg['path'], 'train', transforms=transforms)
            )
            datasets.append(
                CC3DPoints(cfg['path'], 'test', transforms=transforms)
            )

        if cfg['name'] == 'shapenet':
            datasets.append(
                ShapeNet55Custom(cfg['path'], 'train', transforms=transforms)
            )

        if cfg['name'] == 'scanObjectNN':
            datasets.append(
                ScanObjectNN(cfg['path'], 'train', transforms=transforms, return_label=False)
            )
        
        if cfg['name'] == 'scanObjectNN-hard':
            datasets.append(
                ScanObjectNN_hardest(cfg['path'], 'train', transforms=transforms, return_label=False)
            )

    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f'Total number of samples in the dataset: {len(dataset)}')
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    # print("Shuffle is false!!! Change it back")

    return train_loader
    

    