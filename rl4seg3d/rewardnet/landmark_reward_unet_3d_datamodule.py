import random
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import skimage
import torch
from lightning import LightningDataModule
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import torchio as tio

from vital.data.camus.config import Label
from vital.utils.image.us.measure import EchoMeasure


class RewardNet3DDataset(Dataset):
    """ Works with the output of 'utils.file_utils.save_batch_to_dataset """
    def __init__(self,
                 data_path,
                 test_frac=0.1,
                 test=False,
                 common_spacing=[0.37, 0.37, 1],
                 shape_divisible_by=(32, 32, 4),
                 max_window_len=4,
                 max_batch_size=2,
                 max_tensor_volume=5000000):
        super().__init__()
        self.data_path = data_path
        self.img_list = []
        self.common_spacing = common_spacing
        self.shape_divisible_by = shape_divisible_by
        self.max_window_len = max_window_len
        self.max_batch_size = max_batch_size
        self.max_tensor_volume = max_tensor_volume

        # only use /images/ folders to get number of individual entries
        for im_file in Path(f"{self.data_path}/img/").rglob("*.nii.gz"):
            self.img_list += [im_file.as_posix()]

        random.shuffle(self.img_list)

        # split according to test_frac
        test_len = int(test_frac * len(self.img_list))
        if test:
            self.img_list = self.img_list[-test_len:]
        else:
            self.img_list = self.img_list[:-test_len]

        print(f"LEN of reward net dataset: {self.__len__()}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_nifti = nib.load(self.img_list[idx])
        img = img_nifti.get_fdata()
        gt = nib.load(self.img_list[idx].replace("img", "gt").replace("_0000", "")).get_fdata()
        pred = nib.load(self.img_list[idx].replace("img", "segmentation").replace("_0000", "")).get_fdata()

        y = np.zeros_like(gt)
        for i in range(gt.shape[-1]):

            lv_points = np.asarray(
                EchoMeasure._endo_base(gt[..., i].T, lv_labels=Label.LV, myo_labels=Label.MYO))

            p = pred[..., i]

            lbl, num = ndimage.label(p != 0)
            # Count the number of elements per label
            count = np.bincount(lbl.flat)
            # Select the largest blob
            maxi = np.argmax(count[1:]) + 1
            # Remove the other blobs
            p[lbl != maxi] = 0


            p_points = np.asarray(
                EchoMeasure._endo_base(p.T, lv_labels=Label.LV, myo_labels=Label.MYO))
            a = np.zeros_like(p)
            b = np.zeros_like(p)

            lv_points = lv_points[np.argsort(lv_points[:, 1])]
            p_points = p_points[np.argsort(p_points[:, 1])]

            d0_sigma = (np.linalg.norm(lv_points[0] - p_points[0]) / a.shape[0] * 200)
            d1_sigma = (np.linalg.norm(lv_points[1] - p_points[1]) / b.shape[0] * 200)

            spacing = img_nifti.header['pixdim'][1:3]

            # larger than 5mm
            if (np.linalg.norm((lv_points[0] - p_points[0])*spacing)) > 4:
                rr, cc, val = draw.line_aa(p_points[0, 1], p_points[0, 0], lv_points[0, 1], lv_points[0, 0])
                a[rr, cc] = val
                a = gaussian_filter(a, sigma=d0_sigma)
                a = (a - np.min(a)) / (np.max(a) - np.min(a))
            if (np.linalg.norm((lv_points[1] - p_points[1])*spacing)) > 4:
                rr, cc, val = draw.line_aa(p_points[1, 1], p_points[1, 0], lv_points[1, 1], lv_points[1, 0])
                b[rr, cc] = val
                b = gaussian_filter(b, sigma=d1_sigma)
                b = (b - np.min(b)) / (np.max(b) - np.min(b))

            y[..., i] = np.maximum(a, b)

        # transforms and resampling
        if self.common_spacing is None:
            raise Exception("COMMON SPACING IS NONE!")
        transform = tio.Resample(self.common_spacing)
        resampled = transform(tio.ScalarImage(tensor=np.expand_dims(img, 0), affine=img_nifti.affine))

        croporpad = tio.CropOrPad(self.get_desired_size(resampled.shape[1:]))
        # croporpad_ones = tio.CropOrPad(self.get_desired_size(resampled.shape[1:]), padding_mode=1)
        resampled_cropped = croporpad(resampled)
        img = resampled_cropped.tensor.squeeze(0)
        y = croporpad(
            transform(tio.LabelMap(tensor=np.expand_dims(y, 0), affine=img_nifti.affine))).tensor
        pred = croporpad(
            transform(tio.LabelMap(tensor=np.expand_dims(pred, 0), affine=img_nifti.affine))).tensor.squeeze(0)

        # use partial time window, create as many batches as possible with it unless self.max_batch_size not set
        dynamic_batch_size = max(1, self.max_tensor_volume // (img.shape[0] * img.shape[1] * self.max_window_len))
        b_img = []
        b_pred = []
        b_y = []
        for i in range(dynamic_batch_size):
            start_idx = np.random.randint(low=0, high=max(img.shape[-1] - self.max_window_len, 1))
            b_img += [img[..., start_idx:start_idx + self.max_window_len]]
            b_pred += [pred[..., start_idx:start_idx + self.max_window_len]]
            b_y += [y[..., start_idx:start_idx + self.max_window_len]]
        img = torch.stack(b_img)
        pred = torch.stack(b_pred)
        y = torch.stack(b_y)

        x = torch.stack((img, pred), dim=1)

        return x.type(torch.float32), (1 - y).type(torch.float32)

    def get_desired_size(self, current_shape):
        # get desired closest divisible bigger shape
        x = int(np.ceil(current_shape[0] / self.shape_divisible_by[0]) * self.shape_divisible_by[0])
        y = int(np.ceil(current_shape[1] / self.shape_divisible_by[1]) * self.shape_divisible_by[1])
        z = int(max(np.floor(current_shape[2] / self.shape_divisible_by[2]), 1) * self.shape_divisible_by[2])
        return x, y, z


class RewardNet3DDataModule(LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, data_path, *args, **kwargs):
        super().__init__()
        self.data_path = data_path

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally.
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def prepare_data_per_node(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here

        if stage == "fit" or stage is None:
            train_set_full = RewardNet3DDataset(self.data_path)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = RewardNet3DDataset(self.data_path, test=True)

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=1, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=1, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, num_workers=8)


if __name__ == "__main__":

    dl = RewardNet3DDataModule('/data/landmarks_cardinal-icardio/')
    dl.setup()
    count = 0
    for batch in iter(dl.train_dataloader()):
       print(batch[0].shape)
       print(batch[1].shape)

       from matplotlib import pyplot as plt
       plt.figure()
       plt.imshow(batch[0][0, 0, 0, :, :, 0].cpu().numpy().T, cmap='gray')

       plt.figure()
       plt.imshow(batch[0][0, 0, 1, :, :, 0].cpu().numpy().T, cmap='gray')
       plt.imshow(1 - batch[1][0, 0, 0, :, :, 0].cpu().numpy().T, alpha=0.35, cmap='jet')
       plt.show()

       if count < 5:
           count += 1
       else:
           break
