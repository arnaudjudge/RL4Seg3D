from pathlib import Path

import torch
import numpy as np
import nibabel as nib
import argparse

from tqdm import tqdm


def adjust_image(img_nii):
    """
    Normalize image and rescale to target spacing (0.37 mm in-plane).
    """
    img = img_nii.get_fdata().astype(np.float32)
    img /= img.max() if img.max() > 0 else 1.0  # normalize safely

    spacing = img_nii.header.get_zooms()[:3]
    H, W, T = img.shape

    # --- Rescale in-plane spacing to 0.37 mm if needed ---
    target_spacing = (0.37, 0.37)
    scale_h = spacing[0] / target_spacing[0]
    scale_w = spacing[1] / target_spacing[1]
    if abs(scale_h - 1) > 0.05 or abs(scale_w - 1) > 0.05:
        print(f"Warning: Rescaling from spacing {spacing[:2]} → {target_spacing}")
        new_H = int(round(H * scale_h))
        new_W = int(round(W * scale_w))

        # Simple nearest-neighbor resize (no scipy dependency)
        yy = np.clip((np.linspace(0, H - 1, new_H)).astype(int), 0, H - 1)
        xx = np.clip((np.linspace(0, W - 1, new_W)).astype(int), 0, W - 1)
        img = img[yy][:, xx]

    return img, spacing, (H, W, T)


def restore_image(processed_img, original_shape, original_spacing):
    """
    Undo spacing adjustment to approximate original in-plane resolution.

    processed_img: np.ndarray of shape (H', W', T')
    original_nii: nibabel image loaded before adjust_image()
    original_shape: tuple from original_nii.shape
    original_spacing: tuple from original_nii.header.get_zooms()[:3]
    """
    H0, W0, T0 = original_shape
    spacing0 = original_spacing[:2]
    H, W, T = processed_img.shape

    # Resize back to original in-plane spacing
    current_spacing = (0.37, 0.37)
    scale_h = current_spacing[0] / spacing0[0]
    scale_w = current_spacing[1] / spacing0[1]
    if abs(scale_h - 1) > 0.05 or abs(scale_w - 1) > 0.05:
        yy = np.clip((np.linspace(0, H - 1, H0)).astype(int), 0, H - 1)
        xx = np.clip((np.linspace(0, W - 1, W0)).astype(int), 0, W - 1)
        # Apply indexing vectorized over T
        processed_img = processed_img[np.ix_(yy, xx, np.arange(T))]

    return processed_img

def process_single_file(input_path, output_dir, model, tta=True):
    img_nii = nib.load(input_path)
    img, original_spacing, original_shape = adjust_image(img_nii)

    H, W, T = img.shape
    if T > min(H, W):
        print(f"Warning: {input_path} — Temporal dimension might not be last.")

    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, H, W, T)
    with torch.no_grad():
        out = model(img_tensor, tta)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    in_name = Path(input_path).stem.replace('.nii', '')  # handle .nii and .nii.gz

    # Save outputs
    segmentation = restore_image(out[0].cpu().numpy(), original_shape, original_spacing)
    nib.save(nib.Nifti1Image(segmentation.astype(np.uint8), affine=img_nii.affine),
             output_dir / f"{in_name}_segmentation.nii.gz")

    rew_fusion = restore_image(out[1].cpu().numpy(), original_shape, original_spacing)
    nib.save(nib.Nifti1Image(rew_fusion, affine=img_nii.affine),
             output_dir / f"{in_name}_reward_fusion.nii.gz")

    rew_anat = restore_image(out[2][0].cpu().numpy(), original_shape, original_spacing)
    nib.save(nib.Nifti1Image(rew_anat, affine=img_nii.affine),
             output_dir / f"{in_name}_reward_anat.nii.gz")

    rew_lm = restore_image(out[2][0].cpu().numpy(), original_shape, original_spacing)
    nib.save(nib.Nifti1Image(rew_lm, affine=img_nii.affine),
             output_dir / f"{in_name}_reward_LM.nii.gz")

    print(f"Done processing: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Run lightweight TorchScript RL4Seg model on a NIfTI image or folder")
    parser.add_argument("--input", "-i", required=True, help="Path to input NIfTI image or folder")
    parser.add_argument("--output", "-o", required=True, help="Path to save output NIfTI files")
    parser.add_argument("--ckpt", "-c", default='./data/checkpoints/rl4seg3d_torchscript_TTA.pt',
                        help="Path to TorchScript checkpoint. Default is ./data/checkpoints/rl4seg3d_torchscript_TTA.pt")
    parser.add_argument("--no_tta", "-t", action="store_false",
                        help="Turn off TTA (faster inference, but reduced segmentation quality)")
    args = parser.parse_args()

    model = torch.jit.load(args.ckpt, map_location='cuda')
    model.eval()

    input_path = Path(args.input)
    if input_path.is_dir():
        nii_files = sorted(list(input_path.rglob("*.nii")) + list(input_path.rglob("*.nii.gz")))
        if not nii_files:
            print(f"No NIfTI files found recursively in {input_path}")
            return
        print(f"Found {len(nii_files)} NIfTI files in {input_path}"
              f"\nProcessing WITH{'OUT' if not args.no_tta else ''} Test-time augmentation (TTA)")
        for file in tqdm(nii_files, desc="Processing files", ncols=80):
            try:
                process_single_file(file, args.output, model, args.no_tta) # beware of negation logic for no_tta arg
            except Exception as e:
                print(f"Failed on {file}: {e}")
    else:
        print(f"Processing WITH{'OUT' if not args.no_tta else ''} Test-time augmentation (TTA)")
        process_single_file(args.input, args.output, model, args.no_tta)

    print(f"\nOutputs saved to {args.output}")

if __name__ == "__main__":
    main()
