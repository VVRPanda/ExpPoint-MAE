In this file we describe the structure of a configuration file. 

## Dataset configuration files
The first type of configuration datasets is to set up the available datasets. These files are placed the ```./cfg/datasets``` folder. 
The cfg file should be named under the name of the used dataset. For example the cfg for the ModelNet40 dataset is called **modelnet.yaml**.
Inside the yaml file we have to specify the path of the dataset and in case of a classification dataset the number of classes. 

### modelnet.yaml
```
path : ./data
num_classes: 40
```
### shapenet.yaml
```
path: ./data/Shapenet55-34
```

*Note* the depending on the dataloader used for the dataset, you may have to specify either the parent folder or the actual folder where the data are located. Check for instance the paths of the modelnet and shapenet examples above. This will be fixed in a future update. For now we have defined in each configuration file the directory that you have to assign. 

## Pretraining configuration files
This configuration files contains all the necessary information regarding the finetuning process. It should be placed either directly inside the ```./cfg``` directory or in a subdirectory as ```./cfg/pretrain```. 

The structure of the cfg is described as follows:
 - **dataset**: The dataset or datasets that will be used for pretraining. If using a single dataset, just fill the name of its cfg, without the ```.yaml``` subfix, or if using multiple datasets, a list with the names [dataset1, dataset2, ...].
 - **task**: The task that this configutation file is used for. This is just for readability purposes. 
 - **wandb**: This is to use *weights and biases* to monitor the process of your model. The *project* entry is the name of the project in the WandB site and the *name* entry is the name of the current run. 
 - **load_checkpoint**: Whether or not to load a checkpoint. If you don't want to load a checkpoint just set this to ```false```. In case you want to load a checkpoint add the name of the checkpoint (without a subfix). The checkpoint should be placed inside the ```./pretrained_checkpoints``` directory. 
 - **save_checkpoint**: Whether or not to save a checkpoint when the pretraining is done. Set to ```false``` if you do not want to save a checkpoint of set the name of the checkpoint (without subfix). The checkpoint will be stored inside the ```./pretrained_checkpoints``` directory. 
 - **network**: In this section the parameters of the model are defined. The network components used for pretraining are 
    1. Group devider: Devides the point cloud into a predefined number groups of a selected size. 
    2. Mask generator: Masks out a percentage of the pointcloud.
    3. MAE_encoder: This is main transformer network that we want to pretrain
    4. MAE_decoder: The decoder of the MAE pipeline
 - **training**: other training parameters such as number of epochs and the batch_size. You also have the option of monitoring the learning rate curve (in wandb) by setting the ```monitor_lr``` to ```true```.

```
dataset: [cc3d, shapenet]
task: pretrain

wandb:
  project: Point-MAE
  name: ShapenetCC3Dpretrain300

load_checkpoint: false
save_checkpoint: ShapenetCC3Dpretrain300

network:
  group_devider: 
    group_size: 32
    num_group: 64
  
  mask_generator:
    mask_ratio: 0.6
    mask_type: rand

  MAE_encoder: 
    embed_dim: 384
    depth: 12
    num_heads: 6
    drop_path_rate: 0.1
    feature_embed: true

  MAE_decoder:
    embed_dim: 384 
    depth: 4 
    num_heads: 6
    drop_path_rate: 0.1
    feature_embed: False

training: 
  num_epochs: 300
  batch_size: 128
  monitor_lr: true
```


## Finetuning configuration files

The finetuning configuration file follows a similar structure as the pretraining cfg. 
Here we focus on the differences of these files. 

- **dataset**: The finetuning is done in only one dataset, as we want to measure the accuracy of the model in a specific task. 
- **save_checkpoint**: The checkpoint is stored under the ```./finetuned_checkpoints``` directory. 
- **network**: At finetuning we only have to specify the parameters for the group devider and the transformer backbone. We must also specify the aggregation method. We can use any combination of the following that are classification token, max pooling and mean pooling. The feature vector dimension and the input size of the classification head are automatically adjusted based on our choice. 
- **training**: During finetuning there is the option to run some additional callbacks, such as freezing or unfreezing the backbone of the network or saving the checkpoint taht achieves the highest accuracy in the test set. 


```
dataset: modelnet
task: finetune

wandb:
  project: CLEAN_EXPERIMENTS_MODELNET40
  name: ModelNetExample

load_checkpoint : shapenet_cc3d_pretrain_bs128e300
save_checkpoint : ModelNetExample

network:
  group_devider: 
    group_size: 32
    num_group: 64

  MAE_encoder: 
    embed_dim: 384
    depth: 12
    num_heads: 6
    drop_path_rate: 0.1
    feature_embed: true
  
  # the feature dimension for the cls_head depends on the 
  # number of pooled feature vectors
  use_cls_token: true
  use_max_pooling: false
  use_mean_pooling: false

training: 
  num_epochs: 300
  freeze_backbone: true
  unfeeze_at_epoch: 240
  save_best: true
```
