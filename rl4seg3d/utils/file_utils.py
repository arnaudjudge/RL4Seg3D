from pathlib import Path

import nibabel as nib
import numpy as np


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

