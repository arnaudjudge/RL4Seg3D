import time
from typing import Tuple

import numpy as np
import torch
from medpy.metric import dc, hd

from rl4seg3d.utils.Metrics import is_anatomically_valid, mitral_valve_distance
from rl4seg3d.utils.temporal_metrics import check_temporal_validity
from vital.data.camus.config import Label
from vital.data.config import LabelEnum


def dice(pred: np.ndarray, target: np.ndarray, labels: Tuple[LabelEnum], exclude_bg: bool = True,
         all_classes: bool = False):
    """Compute dice for one sample.

    Args:
        pred: prediction array in categorical form (H, W)
        target: target array in categorical form (H, W)
        labels: List of labels for which to compute the dice.
        exclude_bg: If true, background dice is not considered.

    Returns:
        mean dice
    """
    dices = []
    if len(labels) > 2:
        for label in labels:
            if exclude_bg and label == 0:
                pass
            else:
                pred_mask, gt_mask = np.isin(pred, label), np.isin(target, label)
                dices.append(dc(pred_mask, gt_mask))
        if all_classes:
            dice_dict = {f"dice/{label.name}": dice for label, dice in zip(labels[1:], dices)}
            dice_dict['Dice'] = np.array(dices).mean()
            return dice_dict
        else:
            return np.array(dices).mean()
    else:
        if all_classes:
            return {'Dice': dc(pred.squeeze(), target.squeeze())}
        else:
            return dc(pred.squeeze(), target.squeeze())


def hausdorff(pred: np.ndarray, target: np.ndarray, labels: Tuple[LabelEnum], exclude_bg: bool = True,
         all_classes: bool = False, voxel_spacing: Tuple[float] = None):
    """Compute hausdorff for one sample.

    Args:
        pred: prediction array in categorical form (H, W)
        target: target array in categorical form (H, W)
        labels: List of labels for which to compute the metric.
        exclude_bg: If true, background dice is not considered.

    Returns:
        hausdorff
    """
    hd_dict = {}
    for i in range(len(pred)):
        hausdorffs = []
        for label in labels:
            if exclude_bg and label == 0:
                pass
            else:
                pred_mask, gt_mask = np.isin(pred[i], label), np.isin(target[i], label)
                if pred_mask.sum() == 0:
                    pred_mask[pred_mask.shape[0] // 2, pred_mask.shape[1] // 2] = True
                    print('empty mask')
                hausdorffs.append(hd(pred_mask, gt_mask, voxel_spacing[i] if voxel_spacing is not None else None))
        if all_classes:
            for label, haus in zip(labels[1:], hausdorffs):
                hd_dict[f"hd/{label.name}"] = hd_dict.get(f"hd/{label.name}", 0) + haus / len(pred)
            hd_dict['Hausdorff'] = hd_dict.get('Hausdorff', 0) + np.array(hausdorffs).mean() / len(pred)
        else:
            hd_dict['Hausdorff'] = hd_dict.get('Hausdorff', 0) + np.array(hausdorffs).mean() / len(pred)
    return hd_dict


def full_test_metrics(y_pred_as_batch, gt_as_batch, voxel_spacing, device, prefix='test', verbose=True):
    start_time = time.time()
    test_dice = dice(y_pred_as_batch, gt_as_batch, labels=(Label.BG, Label.LV, Label.MYO),
                     exclude_bg=True, all_classes=True)
    test_dice_epi = dice((y_pred_as_batch != 0).astype(np.uint8), (gt_as_batch != 0).astype(np.uint8),
                         labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False)
    if verbose:
        print(f"Dice took {round(time.time() - start_time, 4)} (s).")

    start_time = time.time()
    test_hd = hausdorff(y_pred_as_batch, gt_as_batch, labels=(Label.BG, Label.LV, Label.MYO),
                        exclude_bg=True, all_classes=True, voxel_spacing=voxel_spacing)
    test_hd_epi = hausdorff((y_pred_as_batch != 0).astype(np.uint8), (gt_as_batch != 0).astype(np.uint8),
                            labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False,
                            voxel_spacing=voxel_spacing)['Hausdorff']
    if verbose:
        print(f"HD took {round(time.time() - start_time, 4)} (s).")

    start_time = time.time()
    anat_errors = is_anatomically_valid(y_pred_as_batch)
    if verbose:
        print(f"AV took {round(time.time() - start_time, 4)} (s).")

    start_time = time.time()
    temporal_valid, num_temporal_errors = check_temporal_validity(y_pred_as_batch.transpose((0, 2, 1)),
                                             voxel_spacing[0])
    if verbose:
        print(f"TV took {round(time.time() - start_time, 4)} (s).")

    start_time = time.time()
    lm_metrics = mitral_valve_distance(y_pred_as_batch, gt_as_batch, voxel_spacing[0])
    if verbose:
        print(f"LM dist took {round(time.time() - start_time, 4)} (s).")

    logs = {
        f"{prefix}/anat_valid": torch.tensor(int(all(anat_errors)), device=device),
        f"{prefix}/anat_valid_frames": torch.tensor(anat_errors, device=device).mean(),
        f"{prefix}/temporal_valid": torch.tensor(temporal_valid, device=device),
        f"{prefix}/temporal_errors": torch.tensor(num_temporal_errors, device=device),
        f"{prefix}/dice/epi": torch.tensor(test_dice_epi, device=device),
        f"{prefix}/hd/epi": torch.tensor(test_hd_epi, device=device),
    }
    logs.update({f"{prefix}/{k}": v for k, v in test_dice.items()})
    logs.update({f"{prefix}/{k}": v for k, v in test_hd.items()})
    logs.update({f"{prefix}/LM/{k}": v for k, v in lm_metrics.items()})

    return logs

