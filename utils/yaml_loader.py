from selectors import EpollSelector
import yaml
import os

# not used
def configure_yaml_paths(yaml_name, raw_path=False):
    if raw_path:
        path = raw_path
    else:
        path = os.getcwd()
        path = os.path.join(path, yaml_name + '.yaml')
    
    return path


def load_yaml_from_file(path):
    with open(path, 'r') as stream:
        cfg = yaml.safe_load(stream)
    return cfg


class MyYamlLoader:

    def __init__(self, cfg_name, path=None):

        self.path = path or os.getcwd()
        self.cfg_name = cfg_name
        self.configure_main_cfg()
        self.configure_datasets()
    
    def configure_main_cfg(self):
        cfg_path = os.path.join(self.path, 'cfgs', self.cfg_name + '.yaml')
        self.cfg = load_yaml_from_file(cfg_path)

    def configure_datasets(self):
        # getting the dataset names
        dataset_names = self.cfg['dataset']

        if not isinstance(self.cfg['dataset'], list):
            dataset_names = [dataset_names]

        dataset_cfgs = []
        for name in dataset_names:
            dataset_cfg_path = os.path.join(self.path, 'cfgs', 'datasets', name + '.yaml')
            dataset_cfg = load_yaml_from_file(dataset_cfg_path)
            # adding dataset name to the dataset cfg
            dataset_cfg['name'] = name
            dataset_cfgs.append(dataset_cfg)
        
        self.cfg['dataset'] = dataset_cfgs if len(dataset_cfgs) > 1 else dataset_cfgs[0]

    def __call__(self):
        from pprint import pprint
        pprint(self.cfg)


if __name__=="__main__":
    from pprint import pprint
    c = MyYamlLoader(cfg = 'finetune_example')
    c()
