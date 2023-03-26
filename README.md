# 3D_Brain_Tumour_Segmentation_Project

### Requirements

- torch, torchvision, torchviz, nilearn, numpy, pandas, matplotlib, skimage etc.
- Tested on torch==1.13.1
- Training and evaluation done on Ubuntu-22.04 via WSL
- Device specifications: Intel i5-13600k with 32GB of RAM and an RTX 3090 with 24GB VRAM.

### Instructions

1. Download [BraTS2020 dataset from Kaggle](https://www.kaggle.com/datasets/overspleen/brats-2020-fixed-355) into the repo folder.
2. Design/Modify your model in the `models` folder.
3. Import and Define your model in the 2nd cell of `Train_Notebook.ipynb`; Modify `train_logs_path` to `./Log/{your_model_name}` in the 3rd cell; results will be saved to this folder.
4. Run all cells in `Train_Notebook.ipynb` and wait till training is complete. (Delete unwanted model checkpoints in `./Log/{your_model_name` if applicable).
5. To evaluate all models, go to the 4th cell in `VizEval_Notebook.ipynb` and populate `modelDict` with the model name (which should be exactly the same as the name of its folder in `Logs`) and its corresponding instantiation.
6. Run all cells in `VizEval_Notebook.ipynb` to add additional evaluations/visualisations to the `Logs` folder and populate the `results` folder wiht statistics across all models.

### Other stuff

- `fold_data.csv` contains the information for mapping different patients (and their directory paths) to different training folds; generated from `Train_Notebook.ipynb`.
- `./utils` contain vital functions and classes such as `BratsDataset.py` (the dataset class),`Meter.py` (class for tracking results in the `Trainer` class) as well as other utility for visualisation/evaluations.

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
- [Heavily Abstracted notebook on training a SwinUNETR model with BraTS Dataset using Project MONAI](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb)

### Performance Comparisons
- [**Performance of different UNET models on Synapse multi-organ CT**](https://paperswithcode.com/sota/medical-image-segmentation-on-synapse-multi?p=swin-unet-unet-like-pure-transformer-for)
- [Nvidia’s UNet Models used for the BraTS 2021 Dataset](https://developer.nvidia.com/blog/nvidia-data-scientists-take-top-spots-in-miccai-2021-brain-tumor-segmentation-challenge/)

### Architectures to Experiment With
- [SwinUNETR](https://arxiv.org/abs/2201.01266)