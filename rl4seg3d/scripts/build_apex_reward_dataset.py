"""Build an apex reward-net dataset by corrupting the apex of clean reference segs.

Writes ``images/ gt/ pred/`` triples (via ``save_to_reward_dataset``) so the existing
anatomical datamodule (``config/datamodule/rewardnet_diff_3d.yaml`` ->
``RewardNet3DDataModule``) reads it unchanged and computes ``y = (gt == pred)`` as a
localized apex-error target.

The clean seg is written as ``gt`` and the apex-corrupted seg as ``pred``. Output
filenames end in ``_<view>.nii.gz`` so the datamodule's view parser
(``reward_unet_3d_datamodule.py:49``) recovers the FiLM view id.

Train afterwards with the existing runner (only the data path changes)::

    python runner.py --config-name=reward_3d_runner.yaml \
        datamodule.data_path=<save_dir> \
        model.save_model_path=./apex_rewardnet.ckpt

Example::

    python -m rl4seg3d.scripts.build_apex_reward_dataset \
        --seg-dir  /data/clean_segs \
        --img-dir  /data/images \
        --save-dir /data/apex_rewardDS \
        --clips-per-video 10

Each triple is a random 4-frame clip (matching the reward dataset's (1,H,W,4) shape),
with one consistent apex deformation applied across the clip.
"""
import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torchio as tio

from rl4seg3d.rewardnet.apex_corruption import corrupt_apex_volume, sample_mode_and_mag
from rl4seg3d.utils.file_utils import save_to_reward_dataset

_VIEW_RE = re.compile(r"(a2c|a3c|a4c)", re.IGNORECASE)

# Match the RL datamodule (RL_3d_datamodule.py): resample to a common in-plane spacing and
# crop/pad the spatial dims to a multiple of `shape_divisible_by` (temporal handled by clips).
COMMON_SPACING = [0.37, 0.37, 1]
SHAPE_DIVISIBLE_BY = (32, 32, 4)


def _view_tag(path: Path, default: str) -> str:
    """Recover the a2c/a3c/a4c view tag from a file path, or fall back to `default`."""
    m = _VIEW_RE.search(path.as_posix())
    return m.group(1).lower() if m else default


def _resample(img, seg, affine):
    """Resample image (linear) and seg (nearest) to COMMON_SPACING and crop/pad the
    spatial dims to a multiple of SHAPE_DIVISIBLE_BY -- identical to RL_3d_datamodule.

    Returns (img_resampled (H, W, F), seg_resampled (H, W, F) int16) on the 0.37mm grid.
    """
    transform = tio.Resample(COMMON_SPACING)
    img_r = transform(tio.ScalarImage(tensor=np.expand_dims(img, 0), affine=affine))
    seg_r = transform(tio.LabelMap(tensor=np.expand_dims(seg, 0), affine=affine))

    h, w = img_r.shape[1], img_r.shape[2]
    desired = (int(np.ceil(h / SHAPE_DIVISIBLE_BY[0]) * SHAPE_DIVISIBLE_BY[0]),
               int(np.ceil(w / SHAPE_DIVISIBLE_BY[1]) * SHAPE_DIVISIBLE_BY[1]),
               img_r.shape[3])                        # keep all frames; clips handle the temporal axis
    crop = tio.CropOrPad(desired)
    img_c = crop(img_r).tensor.squeeze(0).numpy()
    seg_c = crop(seg_r).tensor.squeeze(0).numpy().astype(np.int16)
    return img_c, seg_c


def _allowed_from_csv(csv_path, path_col, split_col, split_val):
    """Set of seg relative-paths (posix, no extension) selected from a CSV split column."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    sel = df[df[split_col] == split_val]
    return {str(p).strip("/") for p in sel[path_col]}


def build(seg_dir, img_dir, save_dir, clips_per_video=10, clip_len=4, mode="random",
          default_view="a4c", seed=0, allowed=None, clean_frac=0.2, clip_index_offset=0):
    """Each output triple is a random `clip_len`-frame slice of a video, with one
    consistent apex deformation applied across the slice (matching the reward dataset's
    (1, H, W, clip_len) convention). `clips_per_video` slices are drawn per clean video.

    A fraction `clean_frac` of each video's clips are saved UNCORRUPTED (pred == gt), so
    the reward target y = (gt == pred) is 1 everywhere -- positive anchors that teach the
    net a correct apex is good, not just that any apex region is bad.

    If `allowed` is given (set of seg relative-paths without extension, e.g. from a CSV
    split), only those videos are processed -- everything else under `seg_dir` is skipped.
    """
    rng = np.random.default_rng(seed)
    seg_dir, img_dir = Path(seg_dir), Path(img_dir)
    seg_paths = sorted(seg_dir.rglob("*.nii.gz"))
    if not seg_paths:
        raise SystemExit(f"no .nii.gz segmentations found under {seg_dir}")

    n_written = n_skipped = 0
    for seg_path in seg_paths:
        rel_no_ext = seg_path.relative_to(seg_dir).as_posix()[:-len(".nii.gz")]
        if allowed is not None and rel_no_ext not in allowed:
            continue                                       # not in the requested CSV split
        img_path = img_dir / seg_path.relative_to(seg_dir)
        if not img_path.exists():
            print(f"[skip] no matching image for {seg_path.name}")
            n_skipped += 1
            continue

        seg_nii = nib.load(seg_path)
        img_nii = nib.load(img_path)
        # resample to the common 0.37mm grid (same as RL_3d_datamodule) BEFORE clipping
        img_r, seg_r = _resample(img_nii.get_fdata(),
                                 seg_nii.get_fdata().astype(np.int16),
                                 img_nii.affine)
        spacing = (COMMON_SPACING[0], COMMON_SPACING[1])   # (0.37, 0.37) after resample
        view = _view_tag(seg_path, default_view)
        base = seg_path.name[:-len(".nii.gz")]

        n_frames = seg_r.shape[-1]
        L = min(clip_len, n_frames)
        max_start = n_frames - L
        n_clean = int(round(clips_per_video * clean_frac))   # first n_clean clips are positives
        for k in range(clips_per_video):
            start = int(rng.integers(0, max_start + 1))       # random clip_len-frame window
            seg_clip = seg_r[..., start:start + L]
            img_clip = img_r[..., start:start + L]

            if k < n_clean:
                # clean positive: pred == gt -> reward target is 1 everywhere (good apex).
                pred_clip, tag = seg_clip, "clean"
            else:
                chosen_mode, mag = sample_mode_and_mag(rng, mode=mode)  # per-clip mode+mag
                corrupt, n_changed = corrupt_apex_volume(seg_clip, spacing, mode=chosen_mode,
                                                         mag_mm=mag, rng=rng)
                if n_changed == 0:
                    continue                                  # corruption was a no-op -> skip
                pred_clip, tag = corrupt, chosen_mode

            # (1, H, W, clip_len) leading-channel convention for the reward datamodule
            gt = seg_clip[None]
            image = img_clip[None]
            pred = pred_clip[None]
            # trailing token before .nii.gz MUST be the view for the datamodule parser.
            # clip_index_offset keeps appended passes from colliding with earlier ones.
            filename = f"{base}_{tag}_s{start}_{k + clip_index_offset}_{view}.nii.gz"
            save_to_reward_dataset(save_dir, filename, image, gt, pred, spacing=COMMON_SPACING[0])
            n_written += 1

    print(f"done: wrote {n_written} triples, skipped {n_skipped} cases -> {save_dir}")
    return n_written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-dir", required=True, help="dir of clean reference segmentations (.nii.gz)")
    ap.add_argument("--img-dir", required=True, help="dir of matching images (same relative paths)")
    ap.add_argument("--save-dir", required=True, help="output reward-dataset dir (images/ gt/ pred/)")
    ap.add_argument("--clips-per-video", type=int, default=10, help="random clips drawn per clean video")
    ap.add_argument("--clip-len", type=int, default=4, help="frames per clip (reward dataset uses 4)")
    ap.add_argument("--clean-frac", type=float, default=0.2,
                    help="fraction of clips saved uncorrupted (pred==gt) as good-apex positives")
    ap.add_argument("--clip-index-offset", type=int, default=0,
                    help="offset added to the clip index in filenames (avoids collisions when appending)")
    ap.add_argument("--mode", default="random", help="corruption mode or 'random'")
    ap.add_argument("--default-view", default="a4c", help="fallback view tag when unparsable")
    ap.add_argument("--seed", type=int, default=0)
    # optional CSV split filter (e.g. only the icardio validation dicoms)
    ap.add_argument("--filter-csv", default=None, help="CSV whose split column selects which videos to use")
    ap.add_argument("--filter-col", default="split_official", help="CSV column holding the split label")
    ap.add_argument("--filter-val", default="val", help="value in --filter-col to keep")
    ap.add_argument("--filter-path-col", default="relative_path",
                    help="CSV column with the seg relative path (study/view/dicom, no extension)")
    args = ap.parse_args()

    allowed = None
    if args.filter_csv:
        allowed = _allowed_from_csv(args.filter_csv, args.filter_path_col, args.filter_col, args.filter_val)
        print(f"filter: {len(allowed)} videos from {args.filter_csv} where {args.filter_col}=={args.filter_val!r}")
    build(args.seg_dir, args.img_dir, args.save_dir, clips_per_video=args.clips_per_video,
          clip_len=args.clip_len, mode=args.mode, default_view=args.default_view,
          seed=args.seed, allowed=allowed, clean_frac=args.clean_frac,
          clip_index_offset=args.clip_index_offset)


if __name__ == "__main__":
    main()
