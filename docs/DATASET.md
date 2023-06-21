In this work the following datasets are being used: 

    1. ModelNet40
    2. ScanObjectNN
    3. ShapeNet
    4. CC3D
In this file we provide information on how to download and process the required datasets. 

*The datasets should not follow a specific stucture as you can just their paths in the respective configuration files. 
We choose this design so that you can share datasets among different projects on your device.* 

## ModelNet40

Modelnet40 should be automatically downloaded the first time that you try to access the data. 
In case you get the following error: 
```
ERROR: cannot verify shapenet.cs.stanford.edu's certificate, issued by ‘CN=InCommon RSA Server CA,OU=InCommon,O=Internet2,L=Ann Arbor,ST=MI,C=US’:
  Issued certificate has expired.
```
You can manually download the dataset using the this [link](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).
After the download is completed extract the downloaded *.zip* file and place it in the folder specified by the modelnet cfg.
The dataset *\*.h5* files should be located in the ```data_path_in_cfg/modelnet40_ply_hdf5_2048``` directory. 

## ScanObjectNN

You can download the data from the [official website](https://hkust-vgd.github.io/scanobjectnn/).

## ShapeNet
In order to download the ShapeNet dataset please follow the instructions provided by [Point-BERT](https://github.com/lulutang0608/Point-BERT.git).


## CC3D
In order to use the CC3D dataset you have to get permission from the dataset provider. Please follow the steps in their [official website](https://cvi2.uni.lu/cc3d-dataset/). As the data are provided in mesh format, we use **farthest point sampling** to sample 8192 points for each sample. 
When you get the perimission to access the data, feel free to contact us to provide you with the preprocessed version of the data. 

*Code on data preprocessing will be made available soon.*

