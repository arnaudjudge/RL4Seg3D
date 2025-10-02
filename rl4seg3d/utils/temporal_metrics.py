import os
from itertools import repeat
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from medpy.metric import hd

from vital.data.camus.config import Label
from vital.metrics.evaluate.attribute import compute_temporal_consistency_metric, check_temporal_consistency_errors
from vital.utils.image.us.measure import EchoMeasure

attr_thresholds = {
    'lv_area': 0.2,
    'lv_base_width': 0.45,
    'lv_length': 0.25,
    'myo_area': 0.4,
    'epi_center_x': 0.3,
    'epi_center_y': 0.2,
    'hd_frames_myo': 7.0, # 5.5,
    'hd_frames_epi': 7.5, #6.0,
}


def temporal_hd(tpls, voxel_spacing, label=Label.MYO):
    curr, bckw, forw = tpls
    if label:
        return (hd(curr == label, forw == label, voxel_spacing) +
                hd(curr == label, bckw == label, voxel_spacing)) / 2
    return (hd(curr, forw, voxel_spacing) + hd(curr, bckw, voxel_spacing)) / 2


def get_temporal_hd_metric(pred_as_batch, voxel_spacing, label=None, num_threads=1):
    prev_neigh = pred_as_batch[:-2]  # Previous neighbors of non-edge instants
    next_neigh = pred_as_batch[2:]  # Next neighbors of non-edge instants
    tuples = [(pred_as_batch[1:-1][i], prev_neigh[i], next_neigh[i]) for i in range(len(pred_as_batch[1:-1]))]

    hds = []
    for t in tuples:
        hds += [temporal_hd(t, voxel_spacing, label)]

    return [0] + hds + [0]


def get_temporal_consistencies(segmentation_3d, voxelspacing=(0.37, 0.37), skip_measurement_metrics=False):
    measures_1d = {}
    # calculate measures
    # if exception, make sure threshold is triggered
    try:
        measures_1d["lv_area"] = EchoMeasure.structure_area(segmentation_3d, labels=1,
                                                            voxelarea=voxelspacing[0]*voxelspacing[1])
    except :
        print("lv_area extraction failed")
        measures_1d["lv_area"] = np.resize([1, 0], len(segmentation_3d))

    try:
        measures_1d["myo_area"] = EchoMeasure.structure_area(segmentation_3d, labels=2,
                                                             voxelarea=voxelspacing[0]*voxelspacing[1])
    except:
        print("myo_area extraction failed")
        measures_1d["myo_area"] = np.resize([1, 0], len(segmentation_3d))

    try:
        measures_1d['epi_center_x'] = EchoMeasure.structure_center(segmentation_3d, labels=[1, 2], axis=0)
    except:
        print("epi_center_x extraction failed")
        measures_1d["epi_center_x"] = np.resize([1, 0], len(segmentation_3d))

    try:
        measures_1d['epi_center_y'] = EchoMeasure.structure_center(segmentation_3d, labels=[1, 2], axis=1)
    except:
        print("epi_center_y extraction failed")
        measures_1d["epi_center_y"] = np.resize([1, 0], len(segmentation_3d))

    if not skip_measurement_metrics:
        try:
            measures_1d["lv_base_width"] = EchoMeasure.lv_base_width(segmentation_3d, lv_labels=1, myo_labels=2,
                                                                     voxelspacing=voxelspacing)
        except:
            print("lv_base_width extraction failed")
            measures_1d["lv_base_width"] = np.resize([1, 0], len(segmentation_3d))

        try:
            measures_1d["lv_length"] = EchoMeasure.lv_length(segmentation_3d, lv_labels=1, myo_labels=2,
                                                             voxelspacing=voxelspacing)
        except:
            print("lv_length extraction failed")
            measures_1d["lv_length"] = np.resize([1, 0], len(segmentation_3d))

    t_consistencies = {}
    for attr in measures_1d.keys():
        thresh = attr_thresholds[attr]
        t_consistencies[attr] = check_temporal_consistency_errors(thresh, measures_1d[attr],
                                                                  bounds=(measures_1d[attr].min()*0.99,
                                                                          measures_1d[attr].max()*1.01))
    if not skip_measurement_metrics:
        try:
            measures_1d["hd_frames_myo"] = np.asarray(
                get_temporal_hd_metric(segmentation_3d, voxelspacing, label=Label.MYO))
        except:
            print("hd_frames_myo extraction failed")
            measures_1d["hd_frames_myo"] = np.resize([1, 0], len(segmentation_3d))
        t_consistencies["hd_frames_myo"] = measures_1d["hd_frames_myo"] > attr_thresholds["hd_frames_myo"]

        try:
            measures_1d["hd_frames_epi"] = np.asarray(get_temporal_hd_metric(segmentation_3d, voxelspacing))
        except:
            print("hd_frames_epi extraction failed")
            measures_1d["hd_frames_epi"] = np.resize([1, 0], len(segmentation_3d))
        t_consistencies["hd_frames_epi"] = measures_1d["hd_frames_epi"] > attr_thresholds["hd_frames_epi"]

    return t_consistencies, measures_1d


def check_temporal_validity(segmentation_3d, voxelspacing=(0.37, 0.37), relaxed_factor=None, plot=False, verbose=False):
    total_errors = []
    frames = []
    temp_constistencies, measures_1d = get_temporal_consistencies(segmentation_3d, voxelspacing)
    for attr in temp_constistencies.keys():
        thresh = attr_thresholds[attr]
        t_consistency = temp_constistencies[attr]
        total_errors += [t_consistency.sum()]
        frames += [t_consistency]
        
        if plot:
            temp_constistencies[attr] = compute_temporal_consistency_metric(measures_1d[attr])
            idx = [i for i in range(len(t_consistency)) if t_consistency[i]]
            idxall = range(len(t_consistency))
            prev_neigh = measures_1d[attr][:-2]  # Previous neighbors of non-edge instants
            next_neigh = measures_1d[attr][2:]  # Next neighbors of non-edge instants
            neigh_inter_diff = ((prev_neigh + next_neigh) / 2)
            if t_consistency.sum() > 0:
                plt.figure()
                plt.plot(measures_1d[attr])
                plt.plot(measures_1d[attr], 'o')
                plt.plot(idx, measures_1d[attr][idx], 'x')
                plt.plot(idxall[1:-1], neigh_inter_diff)
                plt.title(attr)

                plt.figure()
                plt.plot(temp_constistencies[attr])
                plt.title(attr)
        if verbose:
            temp_constistencies[attr] = compute_temporal_consistency_metric(measures_1d[attr])
            idx = [i for i in range(len(t_consistency)) if t_consistency[i]]
            print(idx)
            print(f"{attr}: {t_consistency.sum()} - THRESH :{thresh}")
            # print(f"{attr} - {['%.4f' % tc for tc in measures_1d[attr]]}")
            # if t_consistency.sum() > 0:
            print(f"{attr} - {['%.4f' % tc for tc in temp_constistencies[attr] if abs(tc) > thresh]}")
    if plot:
        plt.show()
    # allow for one metric to have one error in it if relaxed.
    # print(total_errors)
    # print(sum([e != 0 for e in total_errors]))
    # count number of frames with errors in them
    frame_errors = (np.asarray(frames).sum(axis=0) != 0)
    return sum([e for e in total_errors]) <= 1 if relaxed_factor else sum([e != 0 for e in total_errors]) == 0, \
        frame_errors.sum()


