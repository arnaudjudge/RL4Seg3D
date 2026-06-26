import argparse
from pathlib import Path

import nibabel as nib
import pandas as pd
from tqdm import tqdm

from rl4seg3d.utils.file_utils import get_img_subpath


def resolve_nifti_path(nifti_root: Path, row) -> Path | None:
    sub_lower = get_img_subpath(row)
    study, view, uuid = str(row["study"]), str(row["view"]), str(row["dicom_uuid"])
    sub_orig = f"{study}/{view}/{uuid}.nii.gz"
    candidates = []
    for sub in {sub_lower, sub_orig}:
        candidates.append(nifti_root / sub)
        candidates.append(nifti_root / sub.replace(".nii.gz", "_0000.nii.gz"))
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Fill H, W, T columns of a dataframe from NIfTI files.")
    parser.add_argument("dataframe", type=Path, help="Path to the input dataframe (csv).")
    parser.add_argument("nifti_root", type=Path, help="Directory containing study/view/<uuid>.nii.gz files.")
    parser.add_argument("--output", type=Path, default=None, help="Output csv path (defaults to overwriting input).")
    parser.add_argument("--spacing-tol", type=float, default=1e-3,
                        help="Absolute tolerance (mm) for dx/dy mismatch between df and NIfTI.")
    parser.add_argument("--common-spacing", type=float, nargs=2, default=(0.37, 0.37),
                        metavar=("SX", "SY"), help="Common spacing (mm) used for expected_area computation.")
    args = parser.parse_args()

    df = pd.read_csv(args.dataframe, dtype={"study": str, "view": str, "dicom_uuid": str})

    for col in ("H", "W", "T", "dx", "dy"):
        if col not in df.columns:
            df[col] = pd.NA

    missing = []
    spacing_mismatches = []
    filled_spacing = 0
    for i in tqdm(range(len(df)), desc="Filling dataframe"):
        row = df.iloc[i]
        uuid = str(row["dicom_uuid"])
        path = resolve_nifti_path(args.nifti_root, row)
        if path is None:
            missing.append(uuid)
            continue

        img = nib.load(path.as_posix())
        h, w, t = img.shape[0], img.shape[1], img.shape[2]
        df.at[df.index[i], "H"] = h
        df.at[df.index[i], "W"] = w
        df.at[df.index[i], "T"] = t

        # df.dx / df.dy are in cm; NIfTI pixdim is in mm — compare after *10.
        pixdim = img.header["pixdim"]
        for axis, nifti_val in (("dx", float(pixdim[1])), ("dy", float(pixdim[2]))):
            df_val = row.get(axis)
            if pd.isna(df_val):
                df.at[df.index[i], axis] = nifti_val / 10
                filled_spacing += 1
                continue
            if abs(float(df_val) * 10 - nifti_val) > args.spacing_tol:
                spacing_mismatches.append((uuid, axis, float(df_val) * 10, nifti_val))

    sx, sy = args.common_spacing
    df["expected_area"] = ((df["H"] * (df["dx"] * 10) / sx) * (df["W"] * (df["dy"] * 10) / sy))

    out = args.output if args.output is not None else args.dataframe
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df) - len(missing)}/{len(df)} rows filled, {len(missing)} missing, "
          f"{filled_spacing} dx/dy values backfilled from NIfTI)")
    if missing:
        print(f"Missing dicom_uuids (first 10): {missing[:10]}")
    if spacing_mismatches:
        print(f"Spacing mismatches: {len(spacing_mismatches)} (first 10):")
        for uuid, axis, df_mm, nifti_mm in spacing_mismatches[:10]:
            print(f"  {uuid}: {axis} df={df_mm:.4f}mm vs nifti={nifti_mm:.4f}mm")


if __name__ == "__main__":
    main()
