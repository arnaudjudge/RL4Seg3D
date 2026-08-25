import os
import json
from pathlib import Path
from typing import Tuple
import re

import hydra
import nibabel as nib
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchio as tio
from dotenv import load_dotenv
from lightning import LightningModule
from lightning import Trainer
from monai import transforms
from monai.data import DataLoader, MetaTensor
from monai.transforms import MapTransform
from monai.transforms import ToTensord
from omegaconf import DictConfig

from patchless_nnunet import utils, setup_root
from rl4seg3d.utils.preprocessing import apply_eq_adapthist, rescale


log = utils.get_pylogger(__name__)

# Must match the training mapping (rl4seg3d/datamodules/RL_3d_datamodule.py).
VIEW_MAP = {"a2c": 0, "a3c": 1, "a4c": 2}


def load_view_mapping(path):
    """Load a {case_identifier: view_str} JSON mapping, or return {} if path is None."""
    if not path:
        return {}
    with open(path) as f:
        mapping = json.load(f)
    return {str(k): str(v) for k, v in mapping.items()}


def case_id_from_path(path):
    """Case identifier used to name this case's outputs.

    Filename with every suffix and any trailing nnU-Net modality tag removed. Used both to
    name outputs and to test whether they already exist, so the two can never disagree.
    """
    return Path(path).name.split('.')[0].removesuffix("_0000")


def load_case_list(path, column=None, query=None):
    """Read a work list and return the set of case identifiers it names, or None if no path.

    Accepts a text file (one entry per line, blank lines and '#' comments ignored) or a csv.
    Entries may be bare case ids, filenames or relative paths: all are reduced to a case id,
    so the same list works whatever produced it.

    `column` and `query` apply to csv input only. `query` is a pandas expression selecting
    rows before ids are read -- e.g. "expected_area <= 550000" to keep only the cases that
    fit a smaller GPU -- and `column` names the id column (inferred when left None).
    """
    if not path:
        return None

    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if query:
            before = len(df)
            df = df.query(query)
            log.info(f"case_list_query {query!r} kept {len(df)}/{before} rows of {path.name}")
        if column is None:
            for candidate in ("dicom_uuid", "case_identifier", "case_id", "relative_path"):
                if candidate in df.columns:
                    column = candidate
                    break
            else:
                raise ValueError(
                    f"Could not infer the case id column of {path}; set case_list_column. "
                    f"Available columns: {list(df.columns)}"
                )
        elif column not in df.columns:
            raise ValueError(
                f"case_list_column {column!r} is not in {path} (available: {list(df.columns)})"
            )
        log.info(f"Reading case ids from column {column!r} of {path}")
        entries = df[column].dropna().astype(str).tolist()
    else:
        if query:
            log.warning(f"case_list_query is ignored for non-csv case_list {path}")
        with open(path) as f:
            entries = [line.strip() for line in f]
        entries = [e for e in entries if e and not e.startswith("#")]

    case_ids = {case_id_from_path(e) for e in entries}
    log.info(f"Work list {path} names {len(case_ids)} unique case(s)")
    return case_ids


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
            self, keys, common_spacing, inference_dir, save_to_gif, save_rewards=False
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
        self.save_rewards = save_rewards

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
            "save_rewards": self.save_rewards,
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


class LazyEchoDataset(Dataset):
    """Loads and preprocesses files on-demand instead of all at once."""

    def __init__(self, input_path, transform=None, apply_eq_hist=False, file_match_regex=".*",
                 view_mapping=None, case_ids=None, shard_id=0, num_shards=1,
                 skip_existing=False, output_path=None, expect_reward_maps=False):
        if num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        if not 0 <= shard_id < num_shards:
            raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")

        self.transform = transform
        self.apply_eq_hist = apply_eq_hist
        self.view_mapping = view_mapping or {}
        self.case_ids = case_ids
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.skip_existing = skip_existing
        self.expect_reward_maps = expect_reward_maps
        # One directory listing instead of a stat() per case: at tens of thousands of cases on a
        # parallel filesystem the difference is minutes per job.
        self.existing_outputs = set()
        if skip_existing:
            if not output_path:
                raise ValueError("skip_existing requires output_path")
            out_dir = Path(output_path)
            if out_dir.is_dir():
                self.existing_outputs = set(os.listdir(out_dir))
        self.input_files = self._collect_files(input_path, file_match_regex)

    def _outputs_exist(self, case_id):
        """True only when every output this run is configured to write is already present.

        Requiring all of them means a case interrupted between the mask write and the reward
        map write is redone rather than skipped forever as a partial result.
        """
        needed = [f"{case_id}.nii.gz"]
        if self.expect_reward_maps:
            needed.append(f"{case_id}_merged_all_rewards.nii.gz")
        return all(name in self.existing_outputs for name in needed)

    def _collect_files(self, input_path, file_match_regex):
        input_path = Path(input_path)
        if input_path.is_file() and input_path.suffix in (".dcm", ".nii") or \
                "".join(input_path.suffixes[-2:]) == ".nii.gz":
            files = [input_path]
        elif input_path.is_dir():
            files = (list(input_path.rglob('*.dcm')) +
                     list(input_path.rglob("*.nii")) +
                     list(input_path.rglob("*.nii.gz")))
        else:
            raise ValueError(f"Invalid input path: {input_path}")

        # Sort before anything slices the list. rglob order is filesystem-dependent, so sharding
        # is only reproducible -- and only guaranteed to partition the work rather than overlap
        # or drop cases -- if every process agrees on the ordering.
        files = sorted(files, key=lambda f: f.as_posix())
        n_found = len(files)

        files = [f for f in files if re.search(file_match_regex, f.as_posix())]
        n_regex = len(files)

        if self.case_ids is not None:
            files = [f for f in files if case_id_from_path(f) in self.case_ids]
        n_listed = len(files)

        # Skipping happens before sharding so that re-running to fill holes spreads the
        # remaining cases over every shard, instead of leaving most jobs with nothing to do
        # while a few carry all the leftovers.
        if self.skip_existing:
            files = [f for f in files if not self._outputs_exist(case_id_from_path(f))]
        n_todo = len(files)

        files = files[self.shard_id::self.num_shards]

        log.info(
            f"Work list: {n_found} file(s) found -> {n_regex} after file_filter_regex "
            f"-> {n_listed} after case_list -> {n_todo} not yet predicted "
            f"-> {len(files)} in shard {self.shard_id}/{self.num_shards}"
        )
        if self.case_ids is not None and n_listed < len(self.case_ids):
            log.warning(
                f"case_list names {len(self.case_ids)} case(s) but only {n_listed} were found "
                f"under {input_path} (after file_filter_regex)"
            )
        return files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_p = self.input_files[idx]
        data, aff, spacing = load_image(input_file_p)

        if data.shape[-1] < 4:
            print(f"Sequence too short with {data.shape[-1]} frames, skipping.")
            # Return None and filter in collate, or raise to skip
            return None

        data = data[None,]
        data = rescale(data)
        if self.apply_eq_hist:
            data = apply_eq_adapthist(data)
        # TEMP: hardcoded spatial Gaussian blur for one run — DELETE AFTER.
        # from scipy.ndimage import gaussian_filter
        # _blur_sigma = 1.5  # in-plane blur strength (pixels); 0 = no change
        # data = gaussian_filter(data, sigma=(0, _blur_sigma, _blur_sigma, 0))  # (C, H, W, T)

        case_id = case_id_from_path(input_file_p)
        meta = {
            "filename_or_obj": case_id,
            "spacing": spacing,
            "original_affine": aff,
        }
        sample = {'image': MetaTensor(torch.tensor(data, dtype=torch.float32), meta=meta)}

        if self.view_mapping:
            view_str = self.view_mapping.get(case_id)
            view_id = VIEW_MAP.get(str(view_str).lower()) if view_str is not None else None
            if view_id is None:
                print(f"WARNING: no usable view for {case_id} (got {view_str!r}); "
                      f"prediction will be unconditioned.")
            else:
                sample['view_as_id'] = torch.tensor(view_id, dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return sample


def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None


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
    def get_array_dataset(input_path, apply_eq_hist=False, file_match_regex=".*"):
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
            if not re.search(file_match_regex, input_file_p.as_posix()):
                continue
            print(input_file_p)
            data, aff, spacing = load_image(input_file_p)

            if data.shape[-1] < 4:
                print(f"Sequence too short with {data.shape[-1]} frames, cannot be processed by this model!")
                continue

            data = data[None,] # add batch/channel dim
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

        save_rewards = cfg.get("save_rewards", False)
        preprocessed = PatchlessPreprocess(keys='image',
                                           common_spacing=cfg.common_spacing,
                                           inference_dir=cfg.output_path,
                                           save_to_gif=cfg.save_as_gif,
                                           save_rewards=save_rewards)
        tf = transforms.compose.Compose([preprocessed, ToTensord(keys="image", track_meta=True)])

        # numpy_arr_data = RL4Seg3DPredictor.get_array_dataset(cfg.input_path, cfg.apply_eq_hist, cfg.file_filter_regex)
        view_mapping = load_view_mapping(cfg.get("view_mapping", None))
        case_ids = load_case_list(
            cfg.get("case_list", None),
            column=cfg.get("case_list_column", None),
            query=cfg.get("case_list_query", None),
        )
        dataset = LazyEchoDataset(
            input_path=cfg.input_path,
            transform=tf,
            apply_eq_hist=cfg.apply_eq_hist,
            file_match_regex=cfg.file_filter_regex,
            view_mapping=view_mapping,
            case_ids=case_ids,
            shard_id=cfg.get("shard_id", 0),
            num_shards=cfg.get("num_shards", 1),
            # `overwrite_existing` was declared in the config but never read by anything. It now
            # drives resume behaviour, which is what makes a killed or timed-out job cheap to
            # re-submit: the re-run picks up only what is missing.
            skip_existing=not cfg.get("overwrite_existing", True),
            output_path=cfg.output_path,
            expect_reward_maps=save_rewards,
        )

        if len(dataset) == 0:
            log.info("Nothing left to predict for this shard; exiting before loading the model.")
            # An empty shard is a success, not a failure: across an array sweep most tasks end
            # here on a re-run, and they must exit zero for a non-zero exit to stay meaningful.
            return {}, {}

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

        object_dict = {
            "cfg": cfg,
            "model": model,
            "trainer": trainer,
        }

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            shuffle=False,
            collate_fn=collate_skip_none,  # handles short-sequence Nones
        )

        log.info("Starting predicting!")
        log.info(f"Using checkpoint: {cfg.ckpt_path}")
        # Checkpoint holds only the segmentation net (saved by Supervised3DOptimizer).
        # It may be a raw net state_dict, or a Lightning checkpoint whose weights are under
        # 'state_dict' with a leading 'net.' prefix. Either way, load into actor.actor.net.
        ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        sd = {(k[len("net."):] if k.startswith("net.") else k): v for k, v in sd.items()}
        model.actor.actor.net.load_state_dict(sd)

        # TEMP: inject small Gaussian noise into net weights before predicting — DELETE AFTER.
        # _noise = 0.5  # relative scale (fraction of each param's RMS magnitude)
        # with torch.no_grad():
        #     for p in model.actor.actor.net.parameters():
        #         rms = p.data.pow(2).mean().sqrt()
        #         p.add_(torch.randn_like(p) * _noise * rms)

        # return_predictions=False stops Lightning retaining every predict_step result for the
        # whole epoch. Nothing needs them (each case is written to disk as it is produced), and
        # over tens of thousands of cases the accumulated list is what exhausts host RAM.
        trainer.predict(model=model, dataloaders=dataloader, return_predictions=False)

        # Two empty dicts, holding no references to the model or the data: @utils.task_wrapper
        # unpacks this as `metric_dict, object_dict`, and returning None instead makes the
        # process raise a TypeError once every prediction is already written -- which on a job
        # array shows up as all tasks having failed.
        return {}, {}


def main():
    """Run the script."""
    load_dotenv()

    RL4Seg3DPredictor.main()


if __name__ == '__main__':
    RL4Seg3DPredictor.main()