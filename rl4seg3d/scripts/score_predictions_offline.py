#!/usr/bin/env python
"""Score already-saved predictions offline: reward-net stats + anatomical/temporal validity.

`inference_predict_step` computes reward maps and the anat/temporal validity flags but
records neither -- the flags only gate the TTO branch, and are evaluated on the *pre*-TTO
prediction. This script recovers both from predictions already on disk, so you don't have
to repeat a TTA+TTO run to get numbers. No actor, no TTA, no TTO: just the reward nets'
forward pass plus numpy metrics on the saved masks.

Preprocessing mirrors predict_3d.PatchlessPreprocess (rescale -> resample to common_spacing
-> crop/pad to a divisible shape) so the reward nets see the same input distribution they
saw during inference. Predictions are resampled with nearest-neighbour (LabelMap).

Usage:
    python rl4seg3d/scripts/score_predictions_offline.py \
        --img-dir  /path/to/input/images \
        --pred-dir /data/maple_norm/temp_outputs \
        --view-mapping /path/to/view_mapping.json \
        --out /data/maple_norm/temp_outputs/scores.csv
"""
import argparse
import json
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchio as tio
from scipy import ndimage

from rl4seg3d.utils.file_utils import reward_stats
from rl4seg3d.predict_3d import VIEW_MAP, load_image
from rl4seg3d.utils.Metrics import is_anatomically_valid
from rl4seg3d.utils.preprocessing import rescale
from rl4seg3d.utils.temporal_metrics import check_temporal_validity


def build_reward_func(config_name, overrides):
    """Instantiate cfg.model.reward from the hydra config (same nets as the run)."""
    import hydra
    from hydra import compose, initialize_config_dir

    config_dir = str((Path(__file__).resolve().parents[1] / "config"))
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    reward = hydra.utils.instantiate(cfg.model.reward)
    return reward, cfg


def desired_size(current_shape, divisible_by=(32, 32, 4)):
    """Same divisible-shape rule as predict_3d.PatchlessPreprocess.get_desired_size."""
    x = int(np.ceil(current_shape[0] / divisible_by[0]) * divisible_by[0])
    y = int(np.ceil(current_shape[1] / divisible_by[1]) * divisible_by[1])
    return x, y, current_shape[2]


def preprocess_pair(img_path, pred_path, common_spacing):
    """Return (image, prediction) as (1, H, W, T) float arrays on the inference grid."""
    data, aff, _ = load_image(Path(img_path))
    data = rescale(data[None, ]).astype(np.float32)  # eq-hist already applied upstream

    pred_nii, pred_aff, _ = load_image(Path(pred_path))
    pred_nii = pred_nii[None, ].astype(np.float32)

    resample = tio.Resample(np.array(common_spacing))
    img_r = resample(tio.ScalarImage(tensor=data, affine=aff))
    pred_r = resample(tio.LabelMap(tensor=pred_nii, affine=pred_aff))

    croporpad = tio.CropOrPad(desired_size(img_r.shape[1:]))
    img_r = croporpad(img_r).numpy().astype(np.float32)
    # crop/pad the prediction to the image's grid, not its own
    pred_r = tio.CropOrPad(img_r.shape[1:])(pred_r).numpy().astype(np.float32)
    return img_r, pred_r


def largest_blob_per_frame(pred_tchw):
    """Keep only the largest connected foreground blob per frame, as inference does."""
    out = pred_tchw.copy()
    for i in range(len(out)):
        lbl, _ = ndimage.label(out[i] != 0)
        count = np.bincount(lbl.flat)
        if len(count) <= 1:
            continue
        maxi = np.argmax(count[1:]) + 1
        out[i][lbl != maxi] = 0
    return out


def reward_maps(reward_func, pred, imgs, cond, temps):
    """Reward maps with per-net temperature: sigmoid(logit / t).

    `RewardUnets3D.predict_full_sequence` hardcodes sigmoid(logit) with no temperature,
    unlike `__call__` which scales the first (anatomical) net by `temp_factor`. This
    reproduces predict_full_sequence's stacking/inferer setup but applies `temps` per net,
    so the scores match the calibrated reward rather than the raw one.
    """
    stack = torch.stack((imgs.squeeze(1), pred), dim=1)
    reward_func.patch_size = list([stack.shape[-3], stack.shape[-2], 4])
    reward_func.inferer.roi_size = reward_func.patch_size
    logits = reward_func.predict(stack, cond=cond)
    return [torch.sigmoid(p / t).squeeze(1) for p, t in zip(logits, temps)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-dir", required=True, help="directory of input images (recursed)")
    p.add_argument("--pred-dir", required=True, help="directory of saved prediction niftis")
    p.add_argument("--view-mapping", default=None, help="{case_id: view} json for FiLM cond")
    p.add_argument("--out", required=True, help="output csv path")
    p.add_argument("--common-spacing", type=float, nargs=3, default=[0.37, 0.37, 1])
    p.add_argument("--config-name", default="predict3d", help="hydra config to read nets from")
    p.add_argument("--override", action="append", default=[],
                   help="hydra override, repeatable (e.g. model/reward=lm_only_film)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="score only the first N cases")
    p.add_argument("--temp", type=float, default=4.5,
                   help="temperature for --temp-net: sigmoid(logit / temp). 1 = no scaling")
    p.add_argument("--temp-net", default="anatomical",
                   help="net the temperature applies to (others stay at 1), matching "
                        "RewardUnets3D.__call__ which only scales the anatomical net")
    args = p.parse_args()

    reward_func, _ = build_reward_func(args.config_name, args.override)
    net_names = list(getattr(reward_func, "nets", {}).keys())
    for net in reward_func.get_nets():
        net.to(args.device).eval()
    reward_func.prepare_for_full_sequence()
    temps = [args.temp if n == args.temp_net else 1.0 for n in net_names]
    if args.temp != 1.0 and args.temp_net not in net_names:
        raise SystemExit(f"--temp-net {args.temp_net!r} not among nets {net_names}")
    print(f"reward nets: {net_names}   temperatures: {temps}")

    view_mapping = {}
    if args.view_mapping:
        with open(args.view_mapping) as f:
            view_mapping = {str(k): str(v) for k, v in json.load(f).items()}

    img_dir, pred_dir = Path(args.img_dir), Path(args.pred_dir)
    images = sorted(list(img_dir.rglob("*.nii")) + list(img_dir.rglob("*.nii.gz")))
    if args.limit:
        images = images[:args.limit]

    rows = []
    for n, img_path in enumerate(images, 1):
        case_id = img_path.stem.split(".")[0].removesuffix("_0000")
        pred_path = pred_dir / f"{case_id}.nii.gz"
        if not pred_path.exists():
            print(f"[{n}/{len(images)}] {case_id}: no prediction, skipped")
            continue

        img_r, pred_r = preprocess_pair(img_path, pred_path, args.common_spacing)

        # (T, H, W) view for the numpy metrics, matching inference_predict_step
        pred_tchw = largest_blob_per_frame(pred_r[0].transpose((2, 0, 1)))
        voxel_spacing = np.asarray(args.common_spacing[:2])
        anat = is_anatomically_valid(pred_tchw).numpy()
        temporal_valid, _ = check_temporal_validity(pred_tchw.transpose((0, 2, 1)), voxel_spacing)

        cond = None
        view_str = view_mapping.get(case_id)
        view_id = VIEW_MAP.get(str(view_str).lower()) if view_str is not None else None
        if view_id is not None:
            cond = torch.tensor([view_id], dtype=torch.long, device=args.device)
        elif view_mapping:
            print(f"  WARNING: no usable view for {case_id} ({view_str!r}); unconditioned")

        img_t = torch.tensor(img_r[None, ], device=args.device)               # (1,1,H,W,T)
        pred_t = torch.tensor(pred_r[0][None, ], device=args.device)          # (1,H,W,T)
        with torch.no_grad():
            rew = reward_maps(reward_func, pred_t, img_t, cond, temps)
        merged = reduce(torch.minimum, rew) if len(rew) > 1 else rew[0]

        row = {
            "case_id": case_id,
            "view": view_str,
            "n_frames": pred_tchw.shape[0],
            "anat_valid_frac": float(anat.mean()),
            "anat_valid_all": bool(anat.all()),
            "temporal_valid": bool(temporal_valid),
            # the exact gate inference used to decide whether to run TTO
            "validated": bool(anat.all()) and bool(temporal_valid),
            **reward_stats(merged, "merged"),
        }
        for name, r in zip(net_names, rew):
            row.update(reward_stats(r, name))
        rows.append(row)
        print(f"[{n}/{len(images)}] {case_id}: validated={row['validated']} "
              f"merged_mean={row['merged_mean']:.4f} "
              f"merged_min_frame_mean={row['merged_min_frame_mean']:.4f} "
              f"(frame {row['merged_worst_frame']})")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows -> {args.out}")
    if len(df):
        stat_cols = [c for c in df.columns if c.endswith(("_mean", "_frac"))]
        print(df[stat_cols].describe().T.to_string())
        print(f"\nvalidated: {int(df.validated.sum())}/{len(df)}  "
              f"(anat_all {int(df.anat_valid_all.sum())}, temporal {int(df.temporal_valid.sum())})")


if __name__ == "__main__":
    main()
