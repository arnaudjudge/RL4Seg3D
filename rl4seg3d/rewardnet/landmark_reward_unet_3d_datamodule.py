import json
import random
import re
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import skimage.draw as draw
import torch
from lightning import LightningDataModule
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import torchio as tio

from vital.data.camus.config import Label
from vital.utils.image.us.measure import EchoMeasure

VIEW_MAP = {"a2c": 0, "a3c": 1, "a4c": 2}

class RewardNet3DDataset(Dataset):
    """ Works with the output of 'utils.file_utils.save_batch_to_dataset """
    def __init__(self,
                 data_path,
                 num_views=0,
                 view_mapping_file=None,
                 test_frac=0.1,
                 test=False,
                 common_spacing=[0.37, 0.37, 1],
                 shape_divisible_by=(32, 32, 4),
                 max_window_len=4,
                 max_batch_size=2,
                 max_tensor_volume=5000000):
        super().__init__()
        self.data_path = data_path
        self.num_views = num_views
        self.img_list = []
        self.common_spacing = common_spacing
        self.shape_divisible_by = shape_divisible_by
        self.max_window_len = max_window_len
        self.max_batch_size = max_batch_size
        self.max_tensor_volume = max_tensor_volume
        self.view_mapping = self._load_view_mapping(view_mapping_file)

        # /pred/ is split into one subfolder per prediction type (base_model,
        # model_noise, ...), so the same case appears once per type. img/ and gt/
        # are flat and shared across types, keyed by basename. Keep only preds that
        # have both a matching img and gt (segmentation .nii.gz or landmark .npy);
        # the rest can't produce a reward target.
        skipped = 0
        for im_file in Path(f"{self.data_path}/pred/").rglob("*.nii.gz"):
            pred_path = im_file.as_posix()
            if self._gt_path(pred_path) is not None and self._sibling_path(pred_path, "img").exists():
                self.img_list += [pred_path]
            else:
                skipped += 1
        if skipped:
            print(f"[RewardNet3DDataset] skipped {skipped} pred files with no matching img/gt")

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

    def _load_view_mapping(self, view_mapping_file):
        """Load a {filename_stem: view_str} JSON into a {stem: view_id} dict.

        If ``view_mapping_file`` is None, auto-detect ``view_mapping.json`` in the
        dataset root. View strings are normalized via VIEW_MAP (case-insensitive);
        unknown labels are skipped. Returns {} when no mapping is available, in
        which case __getitem__ falls back to parsing the view from the filename.
        """
        path = view_mapping_file
        if path is None:
            default = Path(self.data_path) / "view_mapping.json"
            path = default if default.exists() else None
        if path is None:
            return {}

        with open(path) as f:
            raw = json.load(f)

        mapping = {}
        unknown = set()
        for stem, view in raw.items():
            view_id = VIEW_MAP.get(str(view).strip().lower())
            if view_id is None:
                unknown.add(str(view))
                continue
            mapping[stem] = view_id
        if unknown:
            print(f"[RewardNet3DDataset] ignored unknown view labels in {path}: {sorted(unknown)}")
        print(f"[RewardNet3DDataset] loaded {len(mapping)} view mappings from {path}")
        return mapping

    def _sibling_path(self, pred_path, folder):
        """Map a pred file to its shared img/gt counterpart.

        pred files live under pred/<type>/<name>, while img/ and gt/ are flat and
        shared across prediction types, so we drop the subfolder and key by basename
        (also stripping the nnU-Net _0000 channel suffix the targets don't carry).
        """
        name = Path(pred_path).name.replace("_0000", "")
        return Path(self.data_path) / folder / name

    def _gt_path(self, pred_path):
        """Existing gt file for a pred, or None.

        gt is stored either as a segmentation map (.nii.gz) or as a precomputed
        (T, 2, 2) numpy array of LV endo-base points (.npy), keyed by basename.
        """
        nii = self._sibling_path(pred_path, "gt")
        if nii.exists():
            return nii
        npy = nii.with_name(nii.name[:-len(".nii.gz")] + ".npy")
        return npy if npy.exists() else None

    def _load_lv_points(self, pred_path):
        """Per-frame LV endo-base points as an int array of shape (T, 2, 2).

        From a .npy file the points are stored as (row, col); the seg path produces
        them via ``_endo_base(gt.T)`` which returns (col, row), so we reverse the
        last axis to match that convention. From a .nii.gz segmentation they are
        extracted per frame.
        """
        gt_path = self._gt_path(pred_path)
        if gt_path.suffix == ".npy":
            return np.rint(np.load(gt_path)[..., ::-1]).astype(int)
        gt = nib.load(gt_path).get_fdata()
        pts = [np.asarray(EchoMeasure._endo_base(gt[..., i].T, lv_labels=Label.LV, myo_labels=Label.MYO))
               for i in range(gt.shape[-1])]
        return np.stack(pts)

    def _resolve_view_id(self, img_path):
        """View id for a sample, from the mapping (keyed by filename stem) with a
        filename-parsing fallback (e.g. ``0001_A4C_bmode`` -> a4c)."""
        name = Path(img_path).name
        stem = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
        # nnU-Net appends a _0000 channel suffix on some datasets; the mapping is keyed without it.
        for key in (stem, stem.replace("_0000", "")):
            if key in self.view_mapping:
                return self.view_mapping[key]
        for token in re.split(r"[_\-]", stem.lower()):
            if token in VIEW_MAP:
                return VIEW_MAP[token]
        return 0

    def __getitem__(self, idx):
        pred_path = self.img_list[idx]
        img_nifti = nib.load(pred_path)
        pred = img_nifti.get_fdata()
        img = nib.load(self._sibling_path(pred_path, "img")).get_fdata()
        lv_points_per_frame = self._load_lv_points(pred_path)

        y = np.zeros_like(pred)
        for i in range(pred.shape[-1]):

            lv_points = lv_points_per_frame[i]

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

        view_id = self._resolve_view_id(self.img_list[idx])

        return (x.type(torch.float32),
                (1 - y).type(torch.float32),
                torch.tensor(view_id, dtype=torch.long))

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

    def __init__(self, data_path, num_views=0, view_mapping_file=None, *args, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.num_views = num_views
        self.view_mapping_file = view_mapping_file

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
            train_set_full = RewardNet3DDataset(self.data_path, num_views=self.num_views,
                                                view_mapping_file=self.view_mapping_file)
            train_set_size = int(len(train_set_full) * 0.95)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = RewardNet3DDataset(self.data_path, num_views=self.num_views,
                                           view_mapping_file=self.view_mapping_file, test=True)

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=1, num_workers=20)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=1, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, num_workers=12)


if __name__ == "__main__":

    dl = RewardNet3DDataModule('/data/landmarks_dataset_234ch/')
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
