import os
from pathlib import Path
from typing import Tuple

import hydra
import nibabel as nib
import pydicom
import numpy as np
import torch
import torchio as tio
from dotenv import load_dotenv
from lightning import LightningModule
from lightning import Trainer
from monai import transforms
from monai.data import DataLoader, ArrayDataset, MetaTensor
from monai.transforms import MapTransform
from monai.transforms import ToTensord
from omegaconf import DictConfig

from patchless_nnunet import utils, setup_root
from rl4seg3d.utils.preprocessing import apply_eq_adapthist, rescale

log = utils.get_pylogger(__name__)


def load_image(input_path):
    if (input_path.suffix == ".nii" or "".join(input_path.suffixes[-2:]) == ".nii.gz"):
        nifti_img = nib.load(input_path)
        data = nifti_img.get_fdata()
        aff = nifti_img.affine
        spacing = nifti_img.header['pixdim'][1:4].tolist()
    elif input_path.suffix == ".dcm":
        dcm = pydicom.dcmread(input_path)
        arr = dcm.pixel_array
        if len(arr.shape) > 3:
            arr = arr.mean(-1)
        data = arr

        spacing = None
        if 'PixelSpacing' in dcm:
            spacing = [float(x) for x in dcm.PixelSpacing]
        elif 'ImagerPixelSpacing' in dcm:
            spacing = [float(x) for x in dcm.ImagerPixelSpacing]
        elif 'SequenceOfUltrasoundRegions' in dcm:
            seq = dcm.SequenceOfUltrasoundRegions[0]
            if hasattr(seq, 'PhysicalDeltaX') and hasattr(seq, 'PhysicalDeltaY'):
                spacing = [abs(float(seq.PhysicalDeltaX)) * 10, abs(float(seq.PhysicalDeltaY)) * 10, 1.0]
        else:
            spacing = [1.0, 1.0]  # default fallback if no calibration info

        data = data.transpose((2, 1, 0))
        aff = np.diag([spacing[1], spacing[0], 1, 0])
    else:
        raise Exception(f"Tried to load file with invalid type: {input_path}")

    return data, aff, spacing


class PatchlessPreprocess(MapTransform):
    """Load and preprocess data path given in dictionary keys.

    Dictionary must contain the following key(s): "image" and/or "label".
    """

    def __init__(
            self, keys, common_spacing, inference_dir, save_to_gif
    ) -> None:
        """Initialize class instance.

        Args:
            keys: Keys of the corresponding items to be transformed.
            common_spacing: Common spacing to resample the data.
        """
        super().__init__(keys)
        self.keys = keys
        self.common_spacing = np.array(common_spacing)
        self.inference_dir = inference_dir
        self.save_to_gif = save_to_gif

    def __call__(self, data: dict[str, str]):
        # load data
        d = dict(data)
        image = d["image"]

        image_meta_dict = {
            "case_identifier": os.path.basename(image._meta["filename_or_obj"]),
            "original_shape": np.array(image.shape[1:]),
            "original_spacing": np.array(image._meta['spacing']),
            "inference_save_dir": self.inference_dir,
            "save_to_gif": self.save_to_gif,
        }
        original_affine = np.array(image._meta["original_affine"].tolist())
        image_meta_dict["original_affine"] = original_affine

        image = image.cpu().detach().numpy()
        # transforms and resampling
        if self.common_spacing is None:
            raise Exception("COMMON SPACING IS NONE!")
        transform = tio.Resample(self.common_spacing)
        resampled = transform(tio.ScalarImage(tensor=image, affine=original_affine))

        croporpad = tio.CropOrPad(self.get_desired_size(resampled.shape[1:]))
        resampled_cropped = croporpad(resampled)
        resampled_affine = resampled_cropped.affine

        d["image"] = resampled_cropped.numpy().astype(np.float32)

        image_meta_dict['resampled_affine'] = resampled_affine

        d["image_meta_dict"] = image_meta_dict
        return d

    def get_desired_size(self, current_shape, divisible_by=(32, 32, 4)):
        # get desired closest divisible bigger shape
        x = int(np.ceil(current_shape[0] / divisible_by[0]) * divisible_by[0])
        y = int(np.ceil(current_shape[1] / divisible_by[1]) * divisible_by[1])
        z = current_shape[2]
        return x, y, z


class RL4Seg3DPredictor:
    @classmethod
    def main(cls) -> None:
        """Runs the requested experiment."""
        # Set up the environment
        cls.pre_run_routine()

        # Run the system with config loaded by @hydra.main
        cls.run_system()

    @classmethod
    def pre_run_routine(cls) -> None:
        """Sets-up the environment before running the training/testing."""
        # Load environment variables from `.env` file if it exists
        # Load before hydra main to allow for setting environment variables with ${oc.env:ENV_NAME}
        setup_root()

    @staticmethod
    def get_array_dataset(input_path, apply_eq_hist=False):
        tensor_list = []
        # find all nifti files in input_path
        # open and get relevant information
        # add to list of data
        input_path = Path(input_path)

        if (input_path.is_file() and
                (input_path.suffix == ".dcm" or
                 input_path.suffix == ".nii" or
                 "".join(input_path.suffixes[-2:]) == ".nii.gz")):
            input_files = [input_path]
        elif input_path.is_dir():
            input_files = (list(input_path.rglob('*.dcm')) +
                           list(input_path.rglob("*.nii")) +
                           list(input_path.rglob("*.nii.gz")))
        else:
            raise ValueError(f"Invalid input path: {input_path}")

        for input_file_p in input_files:
            data, aff, spacing = load_image(input_file_p)

            data = data[None,]  # add batch/channel dim
            data = rescale(data)
            if apply_eq_hist:
                data = apply_eq_adapthist(data)

            meta = {
                "filename_or_obj": input_file_p.stem.split('.')[0].removesuffix("_0000"),
                "spacing": spacing,
                "original_affine": aff,
            }
            tensor_list.append({'image': MetaTensor(torch.tensor(data, dtype=torch.float32), meta=meta)})
        return tensor_list

    @staticmethod
    @hydra.main(version_base="1.3", config_path="config", config_name="predict3d")
    @utils.task_wrapper
    def run_system(cfg: DictConfig) -> Tuple[dict, dict]:
        """Predict unseen cases with a given checkpoint.

        Currently, this method only supports inference for nnUNet models.

        This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
        failure. Useful for multiruns, saving info about the crash, etc.

        Args:
            cfg (DictConfig): Configuration composed by Hydra.

        Returns:
            Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.

        Raises:
            ValueError: If the checkpoint path is not provided.
        """
        if not cfg.ckpt_path:
            raise ValueError("ckpt_path must not be empty!")

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

        object_dict = {
            "cfg": cfg,
            "model": model,
            "trainer": trainer,
        }

        preprocessed = PatchlessPreprocess(keys='image',
                                           common_spacing=cfg.common_spacing,
                                           inference_dir=cfg.output_path,
                                           save_to_gif=cfg.save_as_gif)
        tf = transforms.compose.Compose([preprocessed, ToTensord(keys="image", track_meta=True)])

        numpy_arr_data = RL4Seg3DPredictor.get_array_dataset(cfg.input_path, cfg.apply_eq_hist)
        dataset = ArrayDataset(img=numpy_arr_data, img_transform=tf)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            shuffle=False,
        )

        log.info("Starting predicting!")
        log.info(f"Using checkpoint: {cfg.ckpt_path}")
        if torch.load(cfg.ckpt_path).get("pytorch-lightning_version", None):
            trainer.predict(model=model, dataloaders=dataloader, ckpt_path=cfg.ckpt_path)
        else:
            model.net.load_state_dict(torch.load(cfg.ckpt_path))
            trainer.predict(model=model, dataloaders=dataloader)

        metric_dict = trainer.callback_metrics

        return metric_dict, object_dict


def main():
    """Run the script."""
    load_dotenv()

    RL4Seg3DPredictor.main()


if __name__ == '__main__':
    RL4Seg3DPredictor.main()
