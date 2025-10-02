import os
from random import shuffle
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from lightning import LightningDataModule
from monai.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def get_img_subpath(row):
    """
    Format string for path of image in file structure
    :param row: dataframe row with all columns filled in
    :return: string containing path to image file
    """
    return f"{row['study']}/{str(row['view']).lower()}/{row['dicom_uuid']}_0000.nii.gz"


class RL3dDataset(Dataset):
    def __init__(self,
                 df,
                 data_path,
                 approx_gt_path = None,
                 allow_real_gt=True,
                 available_gt=None,
                 common_spacing=None,
                 max_window_len=None,
                 use_dataset_fraction=1.0,
                 max_batch_size=None,
                 max_tensor_volume=5000000,
                 shape_divisible_by=(32, 32, 4),
                 test=False,
                 val=False,
                 *args, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.df = df
        self.test = test
        self.val = val
        self.approx_gt_path = approx_gt_path
        self.allow_real_gt = allow_real_gt
        # pass down list of available ground truths
        self.use_gt = available_gt.fillna(False).to_numpy() if available_gt is not None else \
            np.asarray([False for _ in range(len(self.df))])
        print(f"Number of ground truths available: {self.use_gt.sum()}")

        self.max_tensor_volume = max_tensor_volume
        self.shape_divisible_by = shape_divisible_by
        self.max_window_len = max_window_len
        self.max_batch_size = max_batch_size
        if self.max_batch_size and self.max_batch_size > 10:
            print("WARNING: max_batch_size set to a large number, "
                  "behavior is set to use largest batch possible "
                  "if max_batch_size is larger than max calculated length")
        self.common_spacing = common_spacing

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        # Get paths and open images
        sub_path = get_img_subpath(self.df.iloc[idx])
        img_nifti = nib.load(self.data_path + '/img/' + sub_path)
        img = img_nifti.get_fdata()
        if int(img.max()) > 1:
            img = img / 255
        mask = nib.load(self.data_path + '/segmentation/' + sub_path.replace("_0000", "")).get_fdata()
        original_shape = np.asarray(list(img.shape))

        # integrate this into dataset
        if self.use_gt[idx]:
            approx_gt_path = self.approx_gt_path + '/approx_gt/' + sub_path
            approx_gt = nib.load(approx_gt_path).get_fdata()
        else:
            approx_gt = np.zeros_like(mask)

        # img = img[..., 24:28]
        # mask = mask[..., 24:28]
        # approx_gt = approx_gt[..., 24:28]

        # limit size of tensor so it can fit on GPU, choose random slice
        if not self.test and self.max_window_len is None:
            if img.shape[0] * img.shape[1] * img.shape[2] > self.max_tensor_volume:
                time_len = int(self.max_tensor_volume // (img.shape[0] * img.shape[1]))
                start_idx = np.random.randint(low=0, high=max(img.shape[-1] - time_len, 1))
                img = img[..., start_idx:start_idx + time_len]
                mask = mask[..., start_idx:start_idx + time_len]
                approx_gt = approx_gt[..., start_idx:start_idx + time_len]

        # transforms and resampling
        if self.common_spacing is None:
            raise Exception("COMMON SPACING IS NONE!")
        transform = tio.Resample(self.common_spacing)
        resampled = transform(tio.ScalarImage(tensor=np.expand_dims(img, 0), affine=img_nifti.affine))

        croporpad = tio.CropOrPad(self.get_desired_size(resampled.shape[1:]))
        resampled_cropped = croporpad(resampled)
        resampled_affine = resampled_cropped.affine
        img = resampled_cropped.tensor
        mask = croporpad(transform(tio.LabelMap(tensor=np.expand_dims(mask, 0),
                                                affine=img_nifti.affine))).tensor.squeeze(0)
        approx_gt = croporpad(transform(tio.LabelMap(tensor=np.expand_dims(approx_gt, 0),
                                                     affine=img_nifti.affine))).tensor.squeeze(0)

        if not self.test:
            if self.max_window_len:
                # use partial time window, create as many batches as possible with it unless self.max_batch_size not set
                # for validation, use entire sequence split up to batches
                if self.val:
                    max_time_len = img.shape[-1]
                else:
                    max_time_len = int(self.max_tensor_volume // (img.shape[1] * img.shape[2]))
                dynamic_batch_size = max(1, max_time_len // self.max_window_len) \
                    if not self.max_batch_size or not (self.max_batch_size > 0 and
                                                       (self.max_batch_size * self.max_window_len) < max_time_len) \
                    else self.max_batch_size
                b_img = []
                b_mask = []
                b_approx_gt = []
                for i in range(dynamic_batch_size):
                    if self.val:
                        b_img += [img[..., (i*self.max_window_len):(i*self.max_window_len+self.max_window_len)]]
                        b_mask += [mask[..., (i*self.max_window_len):(i*self.max_window_len+self.max_window_len)]]
                        b_approx_gt += [approx_gt[..., (i*self.max_window_len):(i*self.max_window_len+self.max_window_len)]]
                    else:
                        start_idx = np.random.randint(low=0, high=max(img.shape[-1] - self.max_window_len, 1))
                        b_img += [img[..., start_idx:start_idx + self.max_window_len]]
                        b_mask += [mask[..., start_idx:start_idx + self.max_window_len]]
                        b_approx_gt += [approx_gt[..., start_idx:start_idx + self.max_window_len]]
                img = torch.stack(b_img)
                mask = torch.stack(b_mask)
                approx_gt = torch.stack(b_approx_gt)
            else:
                # use entire available time window
                # must unsqueeze to accommodate code in train/val step
                img = img.unsqueeze(0)
                mask = mask.unsqueeze(0)
                approx_gt = approx_gt.unsqueeze(0)

        return {'img': img.type(torch.float32),
                'gt': mask.type(torch.LongTensor) if self.allow_real_gt or self.test else torch.zeros_like(torch.tensor(mask).type(torch.LongTensor)),
                'approx_gt': approx_gt.type(torch.LongTensor),
                'use_gt': torch.tensor([self.use_gt[idx] for _ in range(img.shape[0])]),
                'image_meta_dict': {'case_identifier': self.df.iloc[idx]['dicom_uuid'],
                                    'original_shape': original_shape,
                                    'original_spacing': img_nifti.header['pixdim'][1:4],
                                    'original_affine': img_nifti.affine,
                                    'resampled_affine': resampled_affine,
                                    }
                }

    def get_desired_size(self, current_shape):
        # get desired closest divisible bigger shape
        x = int(np.ceil(current_shape[0] / self.shape_divisible_by[0]) * self.shape_divisible_by[0])
        y = int(np.ceil(current_shape[1] / self.shape_divisible_by[1]) * self.shape_divisible_by[1])
        if not self.test:
            # use floor to avoid zero padded frames
            z = int(max(np.floor(current_shape[2] / self.shape_divisible_by[2]), 1) * self.shape_divisible_by[2])
        else:
            z = current_shape[2]
        return x, y, z


class RL3dDataModule(LightningDataModule):
    """Data module for nnUnet pipeline."""

    def __init__(
            self,
            data_dir: str = "data/",
            dataset_name: str = "",
            csv_file: str = "subset.csv",
            splits_column: str = None,
            batch_size: int = 1,
            seed: int = 0,
            common_spacing: tuple[float, ...] = None,
            max_image_area: int = 600000,
            max_window_len: int = None,
            max_batch_size: int = None,
            max_tensor_volume: int = 5000000,
            shape_divisible_by: tuple[int, ...] = (32, 32, 4),
            subset_frac: float = 1.0,
            approx_gt_dir=None,
            supervised=False,
            gt_column=None,
            gt_frac=None,
            num_workers: int = os.cpu_count() - 1,
            pin_memory: bool = True,
            dataset=RL3dDataset,
            *args,
            **kwargs
    ):
        """Initialize class instance.

        Args:
            data_dir: Path to the data directory.
            dataset_name: Name of dataset to be used.
            batch_size: Batch size to be used for training and validation.
            num_workers: Number of subprocesses to use for data loading.
            pin_memory: Whether to pin memory to GPU.

        Raises:
            NotImplementedError: If the patch shape is not 2D nor 3D.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.args = args
        self.kwargs = kwargs
        self.save_hyperparameters(logger=False)

        self.data_path = self.hparams.data_dir + '/' + self.hparams.dataset_name
        # open dataframe for dataset
        self.df = pd.read_csv(self.data_path + '/' + self.hparams.csv_file, index_col=0)

        self.data_train: Optional[torch.utils.Dataset] = None
        self.data_val: Optional[torch.utils.Dataset] = None
        self.data_test: Optional[torch.utils.Dataset] = None

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally.
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def prepare_data_per_node(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        More detailed steps:
        1. Split the dataset into train, validation (and test) folds if it was not done.
        2. Use the specified fold for training. Create random 80:10:10 or 80:20 split if requested
           fold is larger than the length of saved splits.
        3. Set variables: `self.data_train`, `self.data_val`, `self.data_test`, `self.data_predict`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # keep only valid entries in dataframe
        # self.df = self.df[self.df['valid_segmentation'] == True]

        # Calculate common spacing if not given
        if self.hparams.common_spacing is None:
            max_num = 100
            common_spacing = self.calculate_common_spacing(num_samples=max_num)
            print(f"ESTIMATED COMMON AVERAGE SPACING WITH {max_num} SAMPLES: {common_spacing}")
        else:
            common_spacing = np.asarray(self.hparams.common_spacing)

        if self.hparams.gt_frac:
            self.hparams.gt_column = 'default_gt'
            self.df[self.hparams.gt_column] = self.df.get(self.hparams.gt_column, False)
            use_gt = np.zeros(len(self.df))
            use_gt[:int(self.hparams.gt_frac * len(self.df))] = 1
            np.random.shuffle(use_gt)
            self.df[self.hparams.gt_column] = use_gt.astype(bool)

        # Do splits
        if self.hparams.splits_column and self.hparams.splits_column in self.df.columns:
            # splits are already defined in csv file
            print(f"Using split from column: {self.hparams.splits_column}")
            self.train_idx = self.df.index[self.df[self.hparams.splits_column].isin(["train"])].tolist()
            self.val_idx = self.df.index[self.df[self.hparams.splits_column] == 'val'].tolist()
            self.test_idx = self.df.index[self.df[self.hparams.splits_column] == 'test'].tolist()
            self.pred_idx = self.df.index[(self.df[self.hparams.splits_column] == 'pred')].tolist()
        else:
            # create new splits, save if column name is given
            print(f"Creating new splits!")
            self.train_idx, val_and_test_idx = train_test_split(self.df.index.to_list(),
                                                                train_size=0.8,
                                                                random_state=self.hparams.seed)
            self.val_idx, self.test_idx = train_test_split(val_and_test_idx,
                                                           test_size=0.5,
                                                           random_state=self.hparams.seed)
            self.pred_idx = []
            if self.hparams.splits_column:
                print(f"Saving new split to column: {self.hparams.splits_column}")
                self.df.loc[self.train_idx, self.hparams.splits_column] = 'train'
                self.df.loc[self.val_idx, self.hparams.splits_column] = 'val'
                self.df.loc[self.test_idx, self.hparams.splits_column] = 'test'
                self.df.to_csv(self.data_path + '/' + self.hparams.csv_file)

        if self.hparams.subset_frac and type(self.hparams.subset_frac) == float:
            train_num = int(self.hparams.subset_frac * len(self.train_idx))
            self.train_idx = self.train_idx[:train_num]
            val_num = int(self.hparams.subset_frac * len(self.val_idx))
            self.val_idx = self.val_idx[:val_num]
            test_num = int(self.hparams.subset_frac * len(self.test_idx))
            self.test_idx = self.test_idx[:test_num]
            pred_num = int(self.hparams.subset_frac * len(self.pred_idx))
            self.pred_idx = self.pred_idx[-pred_num:]
        elif self.hparams.subset_frac and type(self.hparams.subset_frac) == int:
            self.train_idx = self.train_idx[:min(self.hparams.subset_frac, len(self.train_idx))]
            self.val_idx = self.val_idx[:min(self.hparams.subset_frac, len(self.val_idx))]
            self.test_idx = self.test_idx[:min(self.hparams.subset_frac, len(self.test_idx))]
            self.pred_idx = self.pred_idx[:min(self.hparams.subset_frac, len(self.pred_idx))]

        if self.hparams.max_image_area:
            if not {'H', 'W'}.issubset(self.df.columns):
                raise "MISSING KEY FOR MAX_IMAGE_AREA!"
            self.df["expected_area"] = ((self.df["H"] * (self.df["dx"] * 10) / self.hparams.common_spacing[0]) *
                                        (self.df["W"] * (self.df["dy"] * 10) / self.hparams.common_spacing[1]))
            print(f"Removing {len(self.df[(self.df[self.hparams.splits_column].isin(['pred', 'train'])) & (self.df['expected_area'] > self.hparams.max_image_area)])} "
                  f"entries with expected resampled area larger than {self.hparams.max_image_area} (before: {len(self.df['expected_area'])})")
            self.df = self.df[((self.df[self.hparams.splits_column].isin(['pred', 'train'])) &
                              (self.df["expected_area"] <= self.hparams.max_image_area)) |
                              (~self.df[self.hparams.splits_column].isin(['pred', 'train']))]
            # remake train_idx and pred_idx
            self.train_idx = self.df.index[self.df[self.hparams.splits_column] == 'train'].tolist()
            self.pred_idx = self.df.index[(self.df[self.hparams.splits_column] == 'pred')].tolist()

        if stage == "fit" or stage is None:
            self.data_train = self.hparams.dataset(self.df.loc[self.train_idx],
                                                     data_path=self.data_path,
                                                     common_spacing=common_spacing,
                                                     max_window_len=self.hparams.max_window_len,
                                                     max_batch_size=self.hparams.max_batch_size,
                                                     max_tensor_volume=self.hparams.max_tensor_volume,
                                                     shape_divisible_by=list(self.hparams.shape_divisible_by),
                                                     available_gt=self.df.loc[self.train_idx].get(self.hparams.gt_column,
                                                                                                  None),
                                                     approx_gt_path=self.hparams.approx_gt_dir,
                                                     allow_real_gt=self.hparams.supervised,
                                                     *self.args,
                                                     **self.kwargs,
                                                     )
            print(f"LEN OF TRAIN SET: {len(self.data_train)}")
            self.data_val = self.hparams.dataset(self.df.loc[self.val_idx],
                                                   data_path=self.data_path,
                                                   common_spacing=common_spacing,
                                                   max_window_len=self.hparams.max_window_len,
                                                   max_batch_size=-1,
                                                   val=True,
                                                   max_tensor_volume=self.hparams.max_tensor_volume,
                                                   shape_divisible_by=list(self.hparams.shape_divisible_by),
                                                   available_gt=None,
                                                   *self.args,
                                                   **self.kwargs,
                                                   )
            print(f"LEN OF VAL SET: {len(self.data_val)}")
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = self.hparams.dataset(self.df.loc[self.test_idx],
                                                    data_path=self.data_path,
                                                    test=True,
                                                    common_spacing=common_spacing,
                                                    shape_divisible_by=list(self.hparams.shape_divisible_by),
                                                    available_gt=None,
                                                    *self.args,
                                                    **self.kwargs,
                                                    )
            print(f"LEN OF TEST SET: {len(self.data_test)}")
        if stage == "predict":
            self.data_pred = self.hparams.dataset(self.df.loc[self.pred_idx],
                                                  data_path=self.data_path,
                                                  common_spacing=common_spacing,
                                                  shape_divisible_by=list(self.hparams.shape_divisible_by),
                                                  max_window_len=self.hparams.max_window_len,
                                                  max_batch_size=self.hparams.max_batch_size,
                                                  max_tensor_volume=self.hparams.max_tensor_volume,
                                                  available_gt=None,
                                                  *self.args,
                                                  **self.kwargs,
                                                  )

            print(f"LEN OF PRED SET: {len(self.data_pred)}")

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            dataset=self.data_train,
            batch_size=1,
            num_workers=max(self.hparams.num_workers, 1),
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=max(self.hparams.num_workers, 1),
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            dataset=self.data_pred,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def calculate_common_spacing(self, num_samples=100):
        spacings = np.zeros(3)
        idx = self.df.reset_index().index.to_list()
        shuffle(idx)
        idx = idx[:max(num_samples, len(idx))]

        for i in idx:
            sub_path = get_img_subpath(self.df.iloc[i])
            img_nifti = nib.load(self.data_path + '/img/' + sub_path)
            spacings += img_nifti.header['pixdim'][1:4]

        return spacings / len(idx)

    def add_to_train(self, id):
        self.df.loc[(self.df['dicom_uuid'] == id), self.trainer.datamodule.hparams.splits_column] = 'train'

    def add_to_gt(self, id):
        self.df.loc[(self.df['dicom_uuid'] == id), self.trainer.datamodule.hparams.gt_column] = True

    def update_dataframe(self):
        self.df.to_csv(self.data_path + '/' + self.hparams.csv_file)

    def get_approx_gt_subpath(self, id):
        return get_img_subpath(self.df.loc[self.df['dicom_uuid'] == id].iloc[0])


if __name__ == "__main__":
    import pyrootutils
    import matplotlib
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    # root = pyrootutils.setup_root(__file__, pythonpath=True)

    dl = RL3dDataModule('/data/cardinal_landmarks/', #(root / 'data/').as_posix(),
                                   common_spacing=(0.37, 0.37, 1.0),
                                   max_window_len=4,
                                   max_batch_size=None,
                                   dataset_name='',
                                   splits_column='lm_split',
                                   csv_file='cardinal.csv',
                                   num_workers=1,
                                   batch_size=1,
                                   use_dataset_fraction=0.1,
                                   )
    dl.setup()
    for batch in iter(dl.train_dataloader()):
        bimg = batch['image'].squeeze(0)
        blabel = batch['label'].squeeze(0)
        print(bimg.shape)
        print(blabel.shape)
        plt.figure()
        plt.imshow(bimg[0, 0, :, :, 1].T)

        plt.figure()
        plt.imshow(blabel[0, 0, :, :, 1].T)
        plt.show()
