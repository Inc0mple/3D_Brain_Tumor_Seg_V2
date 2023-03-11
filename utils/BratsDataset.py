import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import albumentations as A
from albumentations import Compose
import os


class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str = "test", do_resizing: bool = False):
        # Dataframe containing patient, path and fold mapping information
        self.df = df
        # "train" "valid" or "test". Determines whether to apply preprocessing
        self.phase = phase
        self.augmentations = self.get_augmentations(phase)
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        self.do_resizing = do_resizing

    def __len__(self):
        return self.df.shape[0]

    # Makes class accessible by square-bracket notations; determines behaviour upon square-bracket access
    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
        images = []
        # load all modalities(t1, t1ce, t2, flair)
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)  # .transpose(2, 0, 1)

            if self.do_resizing:
                img = self.resize(img)

            img = self.normalize(img)
            images.append(img)
        # stack all 4 modalities into single array as model input
        img = np.stack(images)
        # move axes of array to new position;
        # original shape - (4,240,240,155) -> (4,155,240,240) - new shape; facilitates input into 3DUnet model
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        mask_path = os.path.join(root_path, id_ + "_seg.nii")
        mask = self.load_img(mask_path)

        if self.do_resizing:
            mask = self.resize(mask)

        # Creates and stacks appropriate (possibly overlapping) masks as per BraTS challenge. More info in function.
        mask = self.preprocess_mask_labels(mask)

        mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
        mask = np.clip(mask, 0, 1)

        # Peform augmentations
        augmented = self.augmentations(image=img.astype(
            np.float32), mask=mask.astype(np.float32))
        img = augmented['image']
        mask = augmented['mask']

        # Returns dictionary with Id, image (x) and mask (y) in train/val phase, else return only id and image (x) in test phase
        return {
            "Id": id_,
            "image": img,
            "mask": mask,
        }
    # TO-DO: Implement possible augmentations here? Lower priority for now

    def get_augmentations(self, phase):
        list_transforms = []
        list_trfms = Compose(list_transforms)
        return list_trfms

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, data: np.ndarray):
        # data = resize(data, (224, 224, 128), preserve_range=True)
        data = self.crop_3d_array(data, (224, 224, 128))
        return data

    def preprocess_mask_labels(self, mask: np.ndarray):
        # In the BraTS challenge, the segmentation performance is evaluated on three partially overlapping sub-regions of tumors,
        # namely, whole tumor (WT), tumor core (TC), and enhancing tumor (ET).
        # The WT is the union of ED, NCR/NET, and ET, while the TC includes NCR/NET and ET.

        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask

    def crop_3d_array(self, arr, crop_shape):
        """
        Crop a 3D array to the specified shape.
        
        Parameters
        ----------
        arr : numpy.ndarray
            The 3D input array to be cropped.
        crop_shape : tuple
            The shape of the cropped array. Must be a 3-element tuple (depth, height, width).
            
        Returns
        -------
        numpy.ndarray
            The cropped array.
        """

        assert len(crop_shape) == 3, "crop_shape must be a 3-element tuple"
        assert crop_shape[0] <= arr.shape[0], "depth of crop_shape must be <= depth of arr"
        assert crop_shape[1] <= arr.shape[1], "height of crop_shape must be <= height of arr"
        assert crop_shape[2] <= arr.shape[2], "width of crop_shape must be <= width of arr"

        depth_diff = arr.shape[0] - crop_shape[0]
        height_diff = arr.shape[1] - crop_shape[1]
        width_diff = arr.shape[2] - crop_shape[2]

        if depth_diff % 2 == 0:
            depth_crop_start = depth_diff // 2
            depth_crop_end = arr.shape[0] - (depth_diff // 2)
        else:
            depth_crop_start = depth_diff // 2
            depth_crop_end = arr.shape[0] - (depth_diff // 2) - 1

        if height_diff % 2 == 0:
            height_crop_start = height_diff // 2
            height_crop_end = arr.shape[1] - (height_diff // 2)
        else:
            height_crop_start = height_diff // 2
            height_crop_end = arr.shape[1] - (height_diff // 2) - 1

        if width_diff % 2 == 0:
            width_crop_start = width_diff // 2
            width_crop_end = arr.shape[2] - (width_diff // 2)
        else:
            width_crop_start = width_diff // 2
            width_crop_end = arr.shape[2] - (width_diff // 2) - 1

        cropped_arr = arr[depth_crop_start:depth_crop_end,
                          height_crop_start:height_crop_end, width_crop_start:width_crop_end]

        return cropped_arr
