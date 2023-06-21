# Explainable Transformer in Point Clouds
## A deep dive into explainable self-supervised transformers for point clouds

*Abstract* — In this paper we delve into the properties of trans-
formers, attained through self-supervision, in the point cloud
domain. Specifically, we evaluate the effectiveness of Masked
Autoencoding as a pretraining scheme, and explore Momentum
Contrast as an alternative. In our study we investigate the impact
of data quantity on the learned features, and uncover similarities
in the transformer’s behavior across domains. Through compre-
hensive visualizations, we observe that the transformer learns
to attend to semantically meaningful regions, indicating that
pretraining leads to a better understanding of the underlying
geometry. Moreover, we examine the finetuning process and its
effect on the learned representations. Based on that, we devise an
unfreezing strategy which consistently outperforms our baseline
without introducing any other modifications to the model or
the training pipeline, and achieve state-of-the-art results in the
classification task among transformer models.

*Index Terms* — Deep Learning, Transformers, Explainability,
Point Clouds, Self-Supervision
<br/><br/>
<div align="center">
    <img src = "./figures/pipelines.png", width = 666, aligh=center />
</div>
<br/><br/>

*The code will be made publically available after publication.*

## 1. Requirements

### - Step 1:

Install pytorch 1.12.0 and cuda 11.6 (recommended):
```
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
*We have also tested our code with pytorch 1.11.0 and cuda 11.3*

### - Step 2:

Install other requirements using requierements.txt

 - pip:

       pip install -r requirements.txt


### - Step 3:

Manual installation of the following dependancies:

1. Clone and install the chamfer distance from PointMAE original repo.
```
# create an extensions subfolder and cd to that folder
mkdir extensions && cd extensions
# initialize a github repository
git init
# add PointMAE repo as remote
git remote add -f origin https://github.com/Pang-Yatian/Point-MAE.git 
# set up the sparse checkout by running (requires git version > 1.7.0)
git config core.sparseCheckout true
# configure the folder that we want to checkout
echo "extensions/chamfer_dist" >> .git/info/sparse-checkout
# pull from the remote
git pull origin main
# restructure the folders to meet the project requirements
mv ./extensions/chamfer_dist ./ && rmdir extensions

# Install the module
cd chamfer_dist
python setup.py install --user

# Return to main repo 
cd .. && cd ..
```

2. Install PointNet++ utils

```
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```
3. Install GPU kNN
```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2. Datasets
In this work we use ModelNen40, ScanObjectNN, ShapeNet, and CC3D. See DATASET.md, indide the *docs* folder, for more details on how to download and set the datasets. 

*Do not forget to add the paths of the datasets in the configuration files.*

## 3. Examples 

 - Pretrain a transformer model on ShapeNet and CC3D - using the *cfgs/pretrain/ShapenetCC3DPretrain.yaml* configuration file. 
 ```
python pretrain.py --cfg_name "pretrain/ShapenetCC3DPretrain"
 ```
 - Finetune a transformer model on ModelNet40 using the checkpoint generated from the previous pretrain - using the *cfgs/finetune/ModelNetExample.yaml* configuration file. 
 ```
 python finetune.py --cfg_name "finetune/ModelNetExample"
 ```

For further details on how to create a custom configuration file see the CONFIG.md inside the *docs* folder.

