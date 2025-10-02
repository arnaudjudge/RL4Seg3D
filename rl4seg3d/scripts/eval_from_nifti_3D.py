import os
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt, animation
from matplotlib.colors import LinearSegmentedColormap
from rl4seg3d.utils.test_metrics import full_test_metrics
from scipy import ndimage
from torchio import Resize, LabelMap
from tqdm.contrib.concurrent import process_map

warnings.simplefilter(action='ignore', category=UserWarning)


def as_batch(action):
    y_pred_np_as_batch = action.transpose((2, 0, 1))
    return y_pred_np_as_batch


def clean_blobs(action):
    for i in range(action.shape[-1]):
        try:
            lbl, num = ndimage.label(action[..., i].round() != 0)
            # Count the number of elements per label
            count = np.bincount(lbl.flat)
            # Select the largest blob
            maxi = np.argmax(count[1:]) + 1
            # Remove the other blobs
            action[..., i][lbl != maxi] = 0
        except:
            print("WARNING EMPTY SEGMENTATION FRAME")
    return action


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        if key == "dicom_uuid":
            continue
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def results_to_excel(results, method_name, excel_path):
    # Convert list of dicts to a DataFrame
    df = pd.DataFrame(results).set_index("dicom_uuid")
    df.index = df.index.astype(str)  # ensure consistent string ids

    # If Excel exists, load and update each sheet
    if os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            for metric in df.columns:
                try:
                    existing = pd.read_excel(excel_path, sheet_name=metric.replace("/", "-"), index_col=0)
                except Exception:
                    existing = pd.DataFrame()

                existing[method_name] = df[metric]
                existing.to_excel(writer, sheet_name=metric.replace("/", "-"))
    else:
        # First write: create one sheet per metric
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for metric in df.columns:
                single_metric_df = df[[metric]].rename(columns={metric: method_name})
                single_metric_df.to_excel(writer, sheet_name=metric.replace("/", "-"))

def do(p, plot=False, split_mask=False):
    #
    # LOAD IMAGES
    #
    # print(p)
    pred_nifti = nib.load(p)
    pred = clean_blobs(pred_nifti.get_fdata())

    if split_mask:
        pred2_nifti = nib.load(p.as_posix().replace("/1/", "/2/"))
        pred2 = clean_blobs(pred2_nifti.get_fdata())
        pred = pred + pred2 * 2
        pred[pred > 2] = 2
        # pred = pred * 2
        new_pred = nib.Nifti1Image(pred, pred_nifti.affine, pred_nifti.header)
        Path(p.as_posix().replace("/1/", "/merged/")).parent.mkdir(exist_ok=True)
        nib.save(new_pred, p.as_posix().replace("/1/", "/merged/"))

    gt_p = next(Path(GT_PATH).rglob(f"*{p.name}"))  # only one exists
    gt = nib.load(gt_p).get_fdata()

    img_p = gt_p.as_posix().replace(".nii.gz", "_0000.nii.gz").replace('segmentation', 'img')
    img_nifti = nib.load(img_p)
    img = img_nifti.get_fdata()

    #
    # METRICS
    #

    # resize if needed
    if pred.shape != gt.shape:
        resize_up = Resize((gt.shape[0], gt.shape[1], gt.shape[2]))
        pred = LabelMap(tensor=pred[None,], affine=np.diag(pred_nifti.header["pixdim"][1:3].tolist() + [1, 0]))
        pred = resize_up(pred).numpy().squeeze(0)

    pred_b = as_batch(pred)
    gt_b = as_batch(gt)

    # get voxel spacing
    voxel_spacing = np.asarray([img_nifti.header["pixdim"][1:3]]).repeat(repeats=len(pred_b), axis=0)

    # compute metrics here
    logs = full_test_metrics(pred_b, gt_b, voxel_spacing, device="cpu", verbose=False)
    logs.update({'endo_epi-Dice': np.mean([logs["test/dice/LV"], logs["test/dice/epi"]]),
                 "endo_epi-HD": np.mean([logs["test/hd/LV"], logs["test/hd/epi"]]),
                 "dicom_uuid": img_p.split("/")[-1].split("_0000")[0]})
    logs = {k: float(v.item()) if hasattr(v, 'item') else v for k, v in logs.items()}

    #
    # FIGURES
    #
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        bk = []
        for ax in axes:
            bk += [ax.imshow(img[..., 0].T, animated=True, cmap='gray')]

        custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
        im1 = axes[0].imshow(pred[..., 0].T, animated=True,
                             cmap=custom_cmap,
                             alpha=0.35,
                             interpolation='none')
        axes[0].set_title("Evaluated segmentation")
        axes[0].axis("off")
        im2 = axes[1].imshow(gt[..., 0].T, animated=True,
                             cmap=custom_cmap,
                             alpha=0.35,
                             interpolation='none')
        axes[1].set_title(f"GT")
        axes[1].axis("off")

        def update(i):
            im1.set_array(pred[..., i].T)
            im2.set_array(gt[..., i].T)
            axes[0].set_title(i)
            for b in bk:
                b.set_array(img[..., i].T)
            return bk[0], bk[1], im1, im2

        animation_fig = animation.FuncAnimation(fig, update, frames=img.shape[-1], interval=100, blit=True,
                                                repeat_delay=10, )
        animation_fig.save(f"{GIF_PATH}/{p.stem.split('.')[0]}.gif")
        # plt.show()
        plt.close()
    return logs


if __name__ == "__main__":
    GT_PATH = "<GT_PATH>"
    SET_PATH = "<COMPARED_PATH>"

    GIF_PATH = None
    if GIF_PATH:
        Path(GIF_PATH).mkdir(exist_ok=True)

    paths = [p for p in Path(SET_PATH).rglob('*.nii.gz')]

    all_logs = process_map(do, paths, max_workers=12, chunksize=1)

    # output to csv on sequence wise basis
    # results_to_excel(all_logs, "RL4Seg3D_TTO", "results2.xlsx")

    #
    # AGGREGATION AND OUTPUT
    #
    final_results = dict_mean(all_logs)
    print(f"\n\n***\nAGGREGATED RESULTS : {len(all_logs)} sequences\n")
    for k, v in final_results.items():
        if torch.is_tensor(v):
            v = v.item()
        print(k, f"{v}")
