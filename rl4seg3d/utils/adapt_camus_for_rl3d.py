"""Adapt a CAMUS500-style NIfTI dataset to the layout expected by RL3dDataModule.

Produces a parallel dataset root with:
- `img/<study>/<a2c|a3c|a4c>/...` as symlinks to the original view directories.
- `segmentation/...` mirrored the same way if present.
- A CSV copy with the `view` column remapped 2CH/3CH/4CH -> a2c/a3c/a4c.

Idempotent — safe to re-run.
"""
import argparse
import os
from pathlib import Path

import pandas as pd

VIEW_REMAP = {"2CH": "a2c", "3CH": "a3c", "4CH": "a4c"}


def mirror_view_dirs(src_root: Path, dst_root: Path) -> tuple[int, int]:
    """For each <study>/<VIEW>/ under src_root, create symlink dst_root/<study>/<new_view>.
    Returns (linked, skipped)."""
    linked = skipped = 0
    if not src_root.exists():
        return linked, skipped
    for study_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        for view_dir in study_dir.iterdir():
            if not view_dir.is_dir():
                continue
            new_name = VIEW_REMAP.get(view_dir.name, view_dir.name.lower())
            dst_study = dst_root / study_dir.name
            dst_study.mkdir(parents=True, exist_ok=True)
            dst_view = dst_study / new_name
            if dst_view.exists() or dst_view.is_symlink():
                skipped += 1
                continue
            os.symlink(view_dir.resolve(), dst_view)
            linked += 1
    return linked, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_root", type=Path,
                        help="Source dataset root (containing img/ and optionally segmentation/).")
    parser.add_argument("csv", type=Path, help="Source CSV with original views.")
    parser.add_argument("dst_root", type=Path, help="Destination dataset root (will be created).")
    parser.add_argument("--csv-name", default=None,
                        help="Filename for the remapped CSV under dst_root (default: same as source).")
    args = parser.parse_args()

    args.dst_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, dtype={"study": str, "view": str, "dicom_uuid": str})
    before = df["view"].value_counts().to_dict()
    df["view"] = df["view"].replace(VIEW_REMAP)
    after = df["view"].value_counts().to_dict()
    csv_out = args.dst_root / (args.csv_name or args.csv.name)
    df.to_csv(csv_out, index=False)
    print(f"CSV view counts before: {before} -> after: {after} (wrote {csv_out})")

    for sub in ("img", "segmentation"):
        src = args.src_root / sub
        dst = args.dst_root / sub
        l, s = mirror_view_dirs(src, dst)
        if src.exists():
            print(f"{sub}/: linked {l} view dirs, skipped {s} (already present)")
        else:
            print(f"{sub}/: not present in source, skipped")


if __name__ == "__main__":
    main()
