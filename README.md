# 3D_Brain_Tumour_Segmentation_Project

### Orientation

- Download [BraTS2020 dataset from Kaggle](https://www.kaggle.com/datasets/overspleen/brats-2020-fixed-355) into the repo folder.
- workspace.ipynb is the main notebook currently

### Requirements

- torch, torchvision, torchviz, nilearn, numpy, pandas, matplotlib, skimage etc.
- Tested on Ubuntu-22.04 via WSL

### TO-DO

- [x] Additional Basic Exploratory Data Analysis + Visulaisation of BraTS 2020
- [ ] Dice Loss Implementation
- [x] Dataset Class
- [ ] Trainer Class
  - [ ] Loss Curve Tracking
  - [ ] Checkpoint-saving
- [ ] Baseline results after training
- [ ] Implementation of modern SOTA architecture like SwinUNETR
- [ ] List of results and comparisons with different SOTA approaches
- [ ] Trying/Modifying current approaches (Half-UNet to cut down on parameters etc.) 

### Useful References
- [**BraTS_2020 Dataset on Kaggle**](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) (More details on Dataset in Data Description section of [this paper](https://arxiv.org/pdf/2011.02881.pdf#:~:text=The%20BraTS%202020%20training%20dataset,and%201%20mm%20isotropic%20resolution.))
- [BraTS_2021 Dataset on Kaggle (Superset of BraTS_2020)](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)
- [**Full PyTorch Implementation 3D_UNet on BraTS_2020 Dataset**](https://www.kaggle.com/code/polomarco/brats20-3dunet-3dautoencoder)
- [**Tensorflow Implementation of 3D UNet on BraTS_2020 Dataset**](https://www.kaggle.com/code/maksudaislamlima/3d-unet-brats2020) (Still useful reference for designing our model/dataset/dataloader classes despite not being PyTorch)
- [**3D-UNet BraTS Pytorch Github**](https://github.com/pheonix-18/3D-Unet-BraTS-PyTorch)
- [U-Net on Wikipedia](https://en.wikipedia.org/wiki/U-Net)
- [Paper on 3D U-Net](https://arxiv.org/pdf/1606.06650.pdf)
- [Unofficial 3DUnet PyTorch implementation on Github](https://github.com/AghdamAmir/3D-UNet/blob/main/unet3d.py)
- [Vision Transformers on Wikipedia](https://en.wikipedia.org/wiki/Vision_transformer) (Harder to optimise and train; may require large datasets; but potentially better results than pure CNNs)
- [Explanation on Swin Transformer](https://towardsdatascience.com/a-comprehensive-guide-to-swin-transformer-64965f89d14c); a more efficient version of Vision Transformers
- [Heavily Abstracted notebook on training a SwinUNETR model with BraTS Dataset with Project Monai](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb)

### Performance Comparisons
- [**Performance of different UNET models on Synapse multi-organ CT**](https://paperswithcode.com/sota/medical-image-segmentation-on-synapse-multi?p=swin-unet-unet-like-pure-transformer-for)
- [Nvidia’s UNet Models used for the BraTS 2021 Dataset](https://developer.nvidia.com/blog/nvidia-data-scientists-take-top-spots-in-miccai-2021-brain-tumor-segmentation-challenge/)

### Architectures to Experiment With
- [SwinUNETR](https://arxiv.org/abs/2201.01266)
- [VT-UNet Github (UNet with Swin Transformers)](https://github.com/himashi92/VT-UNet)
- [nnU-Net Github](https://github.com/MIC-DKFZ/nnUNet) (Contains 3DUNet implementation and potential Dataloaders for BraTS dataset)
- [Half-UNET](https://www.frontiersin.org/articles/10.3389/fninf.2022.911679/full) + Variants
