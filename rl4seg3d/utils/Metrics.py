import os
import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from itertools import repeat

from rl4seg3d.utils.cardiac_cycle_utils import estimate_num_cycles
from vital.data.camus.config import Label
from vital.metrics.camus.anatomical.utils import check_segmentation_validity
from vital.metrics.evaluate.attribute import compute_temporal_consistency_metric, check_temporal_consistency_errors
from vital.metrics.train.functional import differentiable_dice_score
from vital.utils.image.us.measure import EchoMeasure

warnings.simplefilter(action='ignore', category=FutureWarning)


def accuracy(pred, imgs, gt):
    actions = torch.round(pred)
    assert actions.shape == gt.shape, \
        print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
    simple = (actions == gt).float()
    return simple.mean(dim=(1, 2), keepdim=True)


def dice_score(output, target):
    classes = target.unique()
    out = torch.zeros(len(output), device=output.device)
    for i in range(len(output)):
        d = 0
        for c in classes:
            target_c = (target[i] == c)
            output_c = (output[i] == c)
            intersection = torch.sum(target_c * output_c)
            d += (2. * intersection) / (torch.sum(target_c) + torch.sum(output_c))
        out[i] = d / len(classes)
    return out


def is_anatomically_valid(output, voxelspacing=(1.0, 1.0)):
    out = torch.zeros(len(output))
    for i in range(len(output)):
        try:
            out[i] = int(check_segmentation_validity(output[i].T, voxelspacing, [0, 1, 2]))
        except:
            out[i] = 0
    return out


def check_frame_anatomical_validity(frame, voxelspacing, labels):
    try:
        return int(check_segmentation_validity(frame, voxelspacing, labels))
    except:
        return 0


def is_anatomically_valid_multiproc(output, voxel_spacing=(1.0, 1.0), num_proc=os.cpu_count() - 1):
    segmentations = [output[i].T for i in range(len(output))]
    with Pool(processes=num_proc) as pool:
        out = list(
            pool.starmap(
                check_frame_anatomical_validity,
                zip(
                    segmentations,
                    repeat(voxel_spacing),
                    repeat([0, 1, 2])
                )
            )
        )
    return out


def mitral_valve_distance(segmentation, gt, spacing, mistake_distances=[5, 7.5], return_mean=True):
    mae = []
    mse = []
    mistakes = dict((f"mistake_per_cycle_{d}mm", 0) for d in mistake_distances)
    mistakes.update(dict((f"mistake_per_cycle_{d}mm_L", 0) for d in mistake_distances))
    mistakes.update(dict((f"mistake_per_cycle_{d}mm_R", 0) for d in mistake_distances))

    lv_area = EchoMeasure.structure_area(gt, labels=1)
    n_cardiac_cycles, _, _ = estimate_num_cycles(lv_area)

    for i in range(len(gt)):
        try:
            lv_points = np.asarray(
                EchoMeasure._endo_base(gt[i].T, lv_labels=Label.LV, myo_labels=Label.MYO))
            p_points = np.asarray(
                EchoMeasure._endo_base(segmentation[i].T, lv_labels=Label.LV, myo_labels=Label.MYO))
            mae_values = [np.linalg.norm(lv_points[0] - p_points[0]), np.linalg.norm(lv_points[1] - p_points[1])]
            mae += [mae_values]
            mse += [((lv_points - p_points) ** 2).mean()]

            for dist in mistake_distances:
                if abs(spacing[0] - spacing[1]) > 0.001:  # account for very close but not exactly same
                    raise ValueError("Spacing not isometric, not currently handled")
                num_pixels = dist / spacing[1]

                # left
                if mae_values[0] > num_pixels:
                    mistakes[f"mistake_per_cycle_{dist}mm_L"] += 1
                # right
                if mae_values[1] > num_pixels:
                    mistakes[f"mistake_per_cycle_{dist}mm_R"] += 1
                # either
                if (mae_values > num_pixels).any():
                    mistakes[f"mistake_per_cycle_{dist}mm"] += 1

            # plt.figure()
            # plt.imshow(gt[i].T)
            # plt.imshow(segmentation[i].T, alpha=0.35)
            # plt.title(mae_values)
            # plt.scatter(p_points[0, 1], p_points[0, 0], c='r')
            # plt.scatter(p_points[1, 1], p_points[1, 0], c='y')
            # plt.scatter(lv_points[0, 1], lv_points[0, 0], marker='x', c='g')
            # plt.scatter(lv_points[1, 1], lv_points[1, 0], marker='x', c='b')
            # plt.show()

        except Exception as e:
            print(f"LM exception: {e}")
            mae += [[segmentation.shape[-1], segmentation.shape[-1]]]
            mse += [segmentation.shape[-1] ** 2]
            for k in mistakes.keys():
                mistakes[k] += 1

    # normalize by number of cycles
    for k in mistakes.keys():
        mistakes[k] /= n_cardiac_cycles

    mae = np.asarray(mae)
    if return_mean:
        metrics = {"mse": np.asarray(mse).mean(), "mae_L": mae[..., 0].mean(), "mae_R": mae[..., 1].mean(),
                   "mae": mae.mean()}
    else:
        metrics = {"mse": np.asarray(mse), "mae_L": mae[..., 0], "mae_R": mae[..., 1], "mae": mae}
    metrics.update(mistakes)
    return metrics
