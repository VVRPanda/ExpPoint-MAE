## ModelNet40

- ModelNet40Sampled : version of modelnet40 used in dgcnn, PCT, pointMLP etc
                      Does not apply any rotation to the input models that are pre-aligned 

- ModelNet40SampledCustom: Same presampled version of the ModelNet40Sampled, but with the ability to add custom transforms


### Implemented transforms:

- VoxelizePointCloud (The is also a custom collate funciton "custom_collate_fn" to handle the voxels.)
- RandomShuffle
- AnisotropicScale
- RandomPointDropout
- ToTensor
- RandomRotate (along single axis)


### Predefined Versions of ModelNet

datasets -> ready_datasets -> get_ModelNet40

Versions:
- original
- rotated