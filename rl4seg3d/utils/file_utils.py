import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def reward_stats(reward_map, prefix):
    """Summarise a reward map (1, H, W, T) as sequence mean and worst-frame mean.

    `min` here is the minimum *over frames* of each frame's spatial average -- i.e. the score
    of the worst frame in the sequence. A per-pixel global min is useless: some pixel in the
    volume always sits at ~0, so it reads ~1e-19 on every case regardless of quality.

    Accepts numpy arrays or torch tensors (on any device) so the same definition is used
    online during prediction and offline in scripts/score_predictions_offline.py -- the two
    must not drift, or scores stop being comparable across runs.
    """
    a = reward_map[0]
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.asarray(a, dtype=np.float32)
    per_frame_mean = a.mean(axis=(0, 1))  # (T,) spatial average of each frame
    return {
        f"{prefix}_mean": float(a.mean()),
        f"{prefix}_min_frame_mean": float(per_frame_mean.min()),
        f"{prefix}_worst_frame": int(per_frame_mean.argmin()),
    }


def append_csv_row(path, row):
    """Append one row to a csv, writing the header only when the file is created.

    Intended for one file per process: a sharded prediction sweep gives each task its own
    path, so no two writers ever touch the same file and no locking is needed. Appending per
    case (rather than dumping at the end) means an interrupted job keeps everything it already
    scored, and a resumed run appends only the cases it actually redoes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Empty counts as new: a per-case csv is created empty as a claim before it is filled, so
    # testing existence alone left every one of them without a header row.
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def get_img_subpath(row, suffix='', extension='.nii.gz'):
    """
    Format string for partial path of image in file structure
    :param row: dataframe row with all columns filled in
    :param suffix: suffix before file extension
    :param extension: file extension
    :return: string containing path to image file
    """
    return f"{row['study']}/{row['view'].lower()}/{row['dicom_uuid']}" + suffix + extension


def save_to_reward_dataset(save_dir, filename, image, gt, action, spacing=None):
    # make sure directories exist
    Path(f"{save_dir}/images").mkdir(parents=True, exist_ok=True)
    Path(f"{save_dir}/gt").mkdir(parents=True, exist_ok=True)
    Path(f"{save_dir}/pred").mkdir(parents=True, exist_ok=True)

    # prepare. Arrays are (1, H, W, frames); when `spacing` (mm) is given, stamp it on the
    # in-plane H/W axes so the reward dataset records the common voxel spacing. Default
    # (None) keeps the legacy unit-spacing affine so the main RL pipeline is unaffected.
    s = 1 if spacing is None else float(spacing)
    affine = np.diag(np.asarray([1, s, s, 0]))
    hdr = nib.Nifti1Header()

    # save three files
    nifti_img = nib.Nifti1Image(image, affine, hdr)
    nifti_img.to_filename(f"{save_dir}/images/{filename}")

    nifti_gt = nib.Nifti1Image(gt, affine, hdr)
    nifti_gt.to_filename(f"{save_dir}/gt/{filename}")

    nifti_pred = nib.Nifti1Image(action, affine, hdr)
    nifti_pred.to_filename(f"{save_dir}/pred/{filename}")



def save_reward_stack(reward_maps, affine, out_path):
    """Write per-net reward maps as one 4D nifti, net axis first.

    Layout is `(n_nets, W, H, T)` on whatever grid `reward_maps` are already on, so
    `nib.load(out_path).get_fdata()[i]` lines up voxel-for-voxel with the matching
    segmentation. This is what the SynthEcho tracking pipeline reads
    (`synthecho/utils/file_utils.load_subject` -> `reward_0`, `reward_1`, ...), and the
    same layout RL4Echo's `clean_rewards.py` built by stacking per-net `_{i}_reward` files.

    Stored as uint8 with `scl_slope = 1/255` rather than float32: these are sigmoids in
    [0, 1], so 1/255 steps sit well below the nets' own noise, and it cuts a 100-case
    sample from ~24 GB to ~0.5 GB. nibabel applies the slope in `get_fdata()`, so readers
    still see floats in [0, 1]. Worst-case per-voxel error is 0.002.

    Args:
        reward_maps: sequence of (W, H, T) float arrays, one per reward net, in net order.
        affine: affine to stamp on the file (readers key off the data layout, not this).
        out_path: destination `.nii.gz`.
    """
    stack = np.stack([np.asarray(r) for r in reward_maps], axis=0)
    stack = np.rint(np.clip(stack, 0, 1) * 255).astype(np.uint8)
    img = nib.Nifti1Image(stack, affine)
    img.header.set_slope_inter(1 / 255, 0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.to_filename(str(out_path))
