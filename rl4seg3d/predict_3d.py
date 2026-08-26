import os
import json
import string
import time
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


# csv columns searched, in order, when a view mapping is read from a csv
_VIEW_ID_COLUMNS = ("dicom_uuid", "case_identifier", "case_id")
_VIEW_COLUMNS = ("view", "view_str")


def load_view_mapping(path):
    """Load a {case_identifier: view_str} mapping, or return {} if path is None.

    Accepts a JSON object, or a csv carrying the same information in two columns (the dataset
    csv already does, so no separate file has to be generated and kept in sync with it).
    """
    if not path:
        return {}

    path = Path(path)
    if path.suffix.lower() != ".csv":
        with open(path) as f:
            mapping = json.load(f)
        return {str(k): str(v) for k, v in mapping.items()}

    df = pd.read_csv(path, low_memory=False)

    def pick(candidates, kind):
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(
            f"No {kind} column in {path}; looked for {list(candidates)}. "
            f"Available columns: {list(df.columns)}"
        )

    id_col, view_col = pick(_VIEW_ID_COLUMNS, "case id"), pick(_VIEW_COLUMNS, "view")
    # df[view_col] rather than df.view: attribute access resolves to Series.view
    sub = df[[id_col, view_col]].dropna()
    mapping = dict(zip(sub[id_col].astype(str), sub[view_col].astype(str).str.lower()))
    log.info(f"Read {len(mapping)} view(s) from {path} "
             f"(columns {id_col!r} -> {view_col!r})")
    return mapping


CASE_CSV_DIRNAME = "csv"


def case_csv_dir(output_path):
    """Flat directory of one csv per case, at the root of the output tree.

    Flat rather than mirrored on purpose: "which cases are taken" is then a single directory
    listing instead of one stat per case, and the file is also the claim that stops two
    concurrently running tasks from predicting the same case.
    """
    return Path(output_path) / CASE_CSV_DIRNAME


def claim_case(csv_dir, case_id):
    """Atomically claim a case by creating its csv. True if this process won it.

    O_CREAT|O_EXCL is atomic (on lustre a single MDS operation), so exactly one of any number
    of racing tasks gets each case -- which is what makes overlapping submissions safe without
    coordinating shard assignment between them.

    The file is created EMPTY and filled only once the case's outputs are written, so an
    abandoned claim is identifiable as a zero-length file rather than needing a timestamp or a
    lease: `find <output>/csv -empty -delete` releases them.
    """
    csv_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(csv_dir / f"{case_id}.csv", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def case_id_from_path(path):
    """Case identifier used to name this case's outputs.

    Filename with every suffix and any trailing nnU-Net modality tag removed. Used both to
    name outputs and to test whether they already exist, so the two can never disagree.
    """
    return Path(path).name.split('.')[0].removesuffix("_0000")


def load_case_list(path, column=None, query=None, order_by=None, order_desc=True,
                   path_template=None):
    """Read a work list, returning (case_ids, relative_paths) in priority order.

    Both are None when `path` is None. `relative_paths` is None unless `path_template` is given,
    in which case it holds one path per case, built by formatting the template against that csv
    row -- letting the dataset construct paths directly instead of walking the input tree.

    Accepts a text file (one entry per line, blank lines and '#' comments ignored) or a csv.
    Entries may be bare case ids, filenames or relative paths: all are reduced to a case id,
    so the same list works whatever produced it.

    `column`, `query` and `order_by` apply to csv input only. `query` is a pandas expression
    selecting rows before ids are read -- e.g. "expected_area <= 550000" to keep only the cases
    that fit a smaller GPU -- and `column` names the id column (inferred when left None).

    `order_by` names a column to prioritise by, so that truncating the run with `limit` keeps
    the cases worth having: e.g. "IQ_expval" with `order_desc` to take the best image quality
    first. Cases with no value in that column are placed FIRST, on the grounds that an unscored
    case is not evidence of a bad one. Order is stable, so equal keys keep their csv order.

    `path_template` is a str.format template over the csv columns, e.g.
    "{dataset}/img/{relative_path}.nii.gz", resolved relative to `input_path`.
    """
    if not path:
        return None, None

    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
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
        if order_by:
            if order_by not in df.columns:
                raise ValueError(
                    f"case_order_by {order_by!r} is not in {path} "
                    f"(available: {list(df.columns)})"
                )
            key = pd.to_numeric(df[order_by], errors="coerce")
            # missing values first (ascending=False on the bool), then by key; mergesort keeps
            # equal keys in csv order so the sequence is reproducible across processes
            df = df.assign(_missing=key.isna(), _key=key).sort_values(
                ["_missing", "_key"], ascending=[False, not order_desc], kind="mergesort")
            log.info(f"Ordered {path.name} by {order_by!r} "
                     f"({'highest' if order_desc else 'lowest'} first, "
                     f"{int(key.isna().sum())} case(s) with no value placed first)")

        log.info(f"Reading case ids from column {column!r} of {path}")
        sub = df[df[column].notna()]

        rel_paths = None
        if path_template:
            tpl_cols = [f for _, f, _, _ in string.Formatter().parse(path_template) if f]
            unknown = sorted(c for c in tpl_cols if c not in sub.columns)
            if unknown:
                raise ValueError(
                    f"case_path_template references column(s) {unknown} not in {path} "
                    f"(available: {list(sub.columns)})"
                )
            # Rows missing any column the template needs cannot name a file at all -- str.format
            # would turn the NaN into the literal "nan" and the case would be reported as merely
            # absent from disk, which is a different and recoverable condition. Drop them here so
            # the two are not conflated and a resumed run stops retrying them forever.
            n_before = len(sub)
            sub = sub.dropna(subset=tpl_cols)
            n_dropped = n_before - len(sub)
            if n_dropped:
                log.warning(
                    f"Dropped {n_dropped} row(s) with no value in {tpl_cols} -- they cannot "
                    f"name a file and would never produce output"
                )
            rel_paths = [path_template.format(**row)
                         for row in sub.to_dict(orient="records")]
            log.info(f"Building paths from {path_template!r} "
                     f"(no input tree scan); e.g. {rel_paths[0] if rel_paths else '-'}")

        entries = sub[column].astype(str).tolist()
        return _dedupe_work_list(entries, rel_paths, path)
    else:
        for name, val in (("case_list_query", query), ("case_order_by", order_by)):
            if val:
                log.warning(f"{name} is ignored for non-csv case_list {path}")
        if path_template:
            log.warning(f"case_path_template is ignored for non-csv case_list {path}")
        with open(path) as f:
            entries = [line.strip() for line in f]
        entries = [e for e in entries if e and not e.startswith("#")]

    return _dedupe_work_list(entries, None, path)


def _dedupe_work_list(entries, rel_paths, path):
    """Reduce entries to case ids, keeping first occurrence so priority order survives."""
    case_ids, kept_paths, seen = [], [] if rel_paths is not None else None, set()
    for i, e in enumerate(entries):
        cid = case_id_from_path(e)
        if cid in seen:
            continue
        seen.add(cid)
        case_ids.append(cid)
        if kept_paths is not None:
            kept_paths.append(rel_paths[i])
    log.info(f"Work list {path} names {len(case_ids)} unique case(s)")
    return case_ids, kept_paths


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
            self, keys, common_spacing, inference_dir, save_to_gif, save_rewards=False,
            case_csv_dir=None
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
        self.case_csv_dir = case_csv_dir

    def __call__(self, data: dict[str, str]):
        # load data
        d = dict(data)
        image = d["image"]

        # When the dataset supplies a relative directory, outputs mirror the input tree; the
        # writers already mkdir their target, so nothing else has to change.
        rel_dir = image._meta.get("rel_dir", "")
        save_dir = os.path.join(self.inference_dir, rel_dir) if rel_dir else self.inference_dir

        image_meta_dict = {
            "case_identifier": os.path.basename(image._meta["filename_or_obj"]),
            "original_shape": np.array(image.shape[1:]),
            "original_spacing": np.array(image._meta['spacing']),
            "inference_save_dir": save_dir,
            "save_to_gif": self.save_to_gif,
            "save_rewards": self.save_rewards,
            # written last, so its presence marks the case complete
            "case_csv_path": (os.path.join(str(self.case_csv_dir),
                                           f"{os.path.basename(image._meta['filename_or_obj'])}.csv")
                              if self.case_csv_dir else ""),
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
                 view_mapping=None, case_ids=None, case_paths=None,
                 use_case_list_order=False, limit=None,
                 shard_id=0, num_shards=1, mirror_input_structure=False,
                 skip_existing=False, output_path=None, expect_reward_maps=False,
                 claim_cases=False, mirror_skip_dirs=()):
        if num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        if not 0 <= shard_id < num_shards:
            raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")

        self.transform = transform
        self.apply_eq_hist = apply_eq_hist
        self.view_mapping = view_mapping or {}
        # membership as a set; rank only when the work list carries a meaningful order
        self.case_ids = set(case_ids) if case_ids is not None else None
        self.case_rank = ({cid: i for i, cid in enumerate(case_ids)}
                          if case_ids is not None and use_case_list_order else None)
        self.case_paths = case_paths
        self.limit = limit
        self.input_root = Path(input_path)
        self.mirror_input_structure = mirror_input_structure
        self.mirror_skip_dirs = {str(d) for d in (mirror_skip_dirs or ())}
        self.claim_cases = claim_cases
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.skip_existing = skip_existing
        self.expect_reward_maps = expect_reward_maps
        # One directory listing instead of a stat() per case: at tens of thousands of cases on a
        # parallel filesystem the difference is minutes per job.
        # A case is "taken" if its csv exists -- claimed by a running task or already finished.
        # One listing of the flat csv directory, rather than a walk of the output tree or a stat
        # per case.
        self.case_csv_dir = case_csv_dir(output_path) if output_path else None
        self.existing_outputs = set()
        if skip_existing:
            if not output_path:
                raise ValueError("skip_existing requires output_path")
            if self.case_csv_dir.is_dir():
                self.existing_outputs = set(os.listdir(self.case_csv_dir))
        self.input_files = self._collect_files(input_path, file_match_regex)

    def _outputs_exist(self, case_id):
        """True if this case is already claimed or finished, i.e. its csv exists.

        The csv is written last, after the mask and reward maps, so its presence means the case
        is done; its presence while empty means a task holds (or held) the claim.
        """
        return f"{case_id}.csv" in self.existing_outputs

    def _collect_from_work_list(self, input_path, file_match_regex):
        """Build paths straight from the work list, without touching the input tree.

        Cases are visited in the work list's priority order and stat()ed one at a time, so a
        run capped by `limit` costs that many stat calls rather than a full traversal -- the
        walk's cost scales with the dataset, this scales with what was actually asked for.

        Already-predicted cases are filtered here rather than afterwards, so `limit` counts
        cases still to do and early-stop survives: `limit=1` hands back the next case not yet
        predicted, having stat()ed only as far as it had to.

        Rows whose file is absent are skipped and counted: a csv naming cases that are not on
        disk is a known condition of this dataset, not an error.
        """
        kept, missing, n_regex_skipped, n_done = [], 0, 0, 0
        for rel in self.case_paths:
            if self.limit is not None and len(kept) >= self.limit:
                break
            f = input_path / rel
            if not re.search(file_match_regex, f.as_posix()):
                n_regex_skipped += 1
                continue
            if not f.is_file():
                missing += 1
                continue
            if self.skip_existing and self._outputs_exist(case_id_from_path(f)):
                n_done += 1
                continue
            kept.append(f)

        log.info(
            f"Work list -> {len(kept)} file(s) to predict from {len(self.case_paths)} row(s) "
            f"({missing} with no file on disk"
            + (f", {n_done} already predicted" if n_done else "")
            + (f", {n_regex_skipped} filtered by file_filter_regex" if n_regex_skipped else "")
            + (f", stopped at limit={self.limit}" if self.limit is not None else "") + ")"
        )
        return kept

    def _collect_files(self, input_path, file_match_regex):
        input_path = Path(input_path)
        if self.case_paths is not None and input_path.is_dir():
            # resume filtering and `limit` are both applied inside the loop above
            files = self._collect_from_work_list(input_path, file_match_regex)
            n_todo = len(files)
            files = files[self.shard_id::self.num_shards]
            log.info(f"{n_todo} not yet predicted -> {len(files)} in shard "
                     f"{self.shard_id}/{self.num_shards}")
            return files

        if input_path.is_file() and input_path.suffix in (".dcm", ".nii") or \
                "".join(input_path.suffixes[-2:]) == ".nii.gz":
            files = [input_path]
        elif input_path.is_dir():
            # One walk, not one rglob per extension: on a parallel filesystem with tens of
            # thousands of files this is the longest step before anything is logged, so it is
            # announced first and traverses the tree once instead of three times.
            log.info(f"Scanning {input_path} for .dcm/.nii/.nii.gz files "
                     f"(slow on first run over a large tree)...")
            start = time.time()
            files = []
            for dirpath, _, filenames in os.walk(input_path):
                d = Path(dirpath)
                files += [d / fn for fn in filenames
                          if fn.endswith((".dcm", ".nii", ".nii.gz"))]
            log.info(f"Scan found {len(files)} file(s) in {time.time() - start:.1f}s")
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

        # Adopt the work list's priority order, so `limit` truncates to the cases worth having
        # rather than to an arbitrary slice of the filesystem ordering.
        if self.case_rank is not None:
            files = sorted(files, key=lambda f: self.case_rank[case_id_from_path(f)])

        # Skipping happens before sharding so that re-running to fill holes spreads the
        # remaining cases over every shard, instead of leaving most jobs with nothing to do
        # while a few carry all the leftovers.
        if self.skip_existing:
            files = [f for f in files if not self._outputs_exist(case_id_from_path(f))]
        n_todo = len(files)

        # `limit` counts cases still to do, so limit=N always yields N unpredicted cases (or
        # everything left, if fewer) rather than N that may already be finished.
        if self.limit is not None:
            files = files[:self.limit]
        n_limited = len(files)

        files = files[self.shard_id::self.num_shards]

        log.info(
            f"Work list: {n_found} file(s) found -> {n_regex} after file_filter_regex "
            f"-> {n_listed} after case_list -> {n_todo} not yet predicted "
            f"-> {n_limited} after limit "
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
        case_id = case_id_from_path(input_file_p)

        # Claimed here rather than when the work list was built, so a task that dies holds at
        # most what its dataloader had prefetched. Losing the race means another task already
        # has this case: return None and let collate_skip_none drop it.
        if self.claim_cases and not claim_case(self.case_csv_dir, case_id):
            print(f"{case_id} claimed by another task, skipping.")
            return None

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

        meta = {
            "filename_or_obj": case_id,
            "spacing": spacing,
            "original_affine": aff,
        }
        if self.mirror_input_structure:
            try:
                rel = input_file_p.parent.relative_to(self.input_root)
                # Components named in mirror_skip_dirs are dropped: the input tree's "img" level
                # only distinguishes images from other modalities on the input side and carries
                # no meaning among predictions.
                meta["rel_dir"] = "/".join(pt for pt in rel.parts
                                           if pt not in self.mirror_skip_dirs)
            except ValueError:
                # input_path pointed at a single file, so there is no tree to mirror
                meta["rel_dir"] = ""
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
                                           save_rewards=save_rewards,
                                           case_csv_dir=(case_csv_dir(cfg.output_path)
                                                         if cfg.get("write_case_csv", True)
                                                         else None))
        tf = transforms.compose.Compose([preprocessed, ToTensord(keys="image", track_meta=True)])

        # numpy_arr_data = RL4Seg3DPredictor.get_array_dataset(cfg.input_path, cfg.apply_eq_hist, cfg.file_filter_regex)
        view_mapping = load_view_mapping(cfg.get("view_mapping", None))
        case_order_by = cfg.get("case_order_by", None)
        case_ids, case_paths = load_case_list(
            cfg.get("case_list", None),
            column=cfg.get("case_list_column", None),
            query=cfg.get("case_list_query", None),
            order_by=case_order_by,
            order_desc=cfg.get("case_order_desc", True),
            path_template=cfg.get("case_path_template", None),
        )
        dataset = LazyEchoDataset(
            input_path=cfg.input_path,
            transform=tf,
            apply_eq_hist=cfg.apply_eq_hist,
            file_match_regex=cfg.file_filter_regex,
            view_mapping=view_mapping,
            case_ids=case_ids,
            case_paths=case_paths,
            use_case_list_order=bool(case_order_by),
            mirror_input_structure=cfg.get("mirror_input_structure", False),
            mirror_skip_dirs=cfg.get("mirror_skip_dirs", ()),
            limit=cfg.get("limit", None),
            shard_id=cfg.get("shard_id", 0),
            num_shards=cfg.get("num_shards", 1),
            # `overwrite_existing` was declared in the config but never read by anything. It now
            # drives resume behaviour, which is what makes a killed or timed-out job cheap to
            # re-submit: the re-run picks up only what is missing.
            skip_existing=not cfg.get("overwrite_existing", True),
            output_path=cfg.output_path,
            expect_reward_maps=save_rewards,
            claim_cases=cfg.get("write_case_csv", True),
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