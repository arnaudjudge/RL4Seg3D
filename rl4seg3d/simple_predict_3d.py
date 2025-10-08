from pathlib import Path

import torch
import numpy as np
import nibabel as nib
import argparse


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


def main():
    parser = argparse.ArgumentParser(description="Run lightweight TorchScript RL4Seg model on a NIfTI image")
    parser.add_argument("--input", "-i", required=True, help="Path to input NIfTI image")
    parser.add_argument("--output", "-o", required=True, help="Path to save output NIfTI")
    parser.add_argument("--ckpt", "-c", required=True, help="Path to TorchScript checkpoint")
    args = parser.parse_args()

    # Load input image
    img_nii = nib.load(args.input)
    img, original_spacing, original_shape = adjust_image(img_nii)

    H, W, T = img.shape
    if T > min(H, W):
        print("Warning: Temporal dimension might not be last. Image may be transposed incorrectly.")
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, H, W, T)

    # Load TorchScript module onto cuda
    scripted_module = torch.jit.load(args.ckpt, map_location='cuda')

    # Get predictions
    with torch.no_grad():
        out = scripted_module(img_tensor)

    Path(args.output).mkdir(exist_ok=True, parents=True)
    in_name = Path(args.input).name
    # Convert to numpy and save to nifti
    segmentation = restore_image(out[0].cpu().numpy(), original_shape, original_spacing)
    nib.save(nib.Nifti1Image(segmentation, affine=img_nii.affine, dtype='uint8'), f"{args.output}/{in_name.replace('.nii.gz', '_segmentation.nii.gz')}")
    rew_fusion = restore_image(out[1].cpu().numpy(), original_shape, original_spacing)
    nib.save(nib.Nifti1Image(rew_fusion, affine=img_nii.affine), f"{args.output}/{in_name.replace('.nii.gz', '_reward_fusion.nii.gz')}")
    rew_anat = restore_image(out[2][0].cpu().numpy(), original_shape, original_spacing)
    nib.save(nib.Nifti1Image(rew_anat, affine=img_nii.affine), f"{args.output}/{in_name.replace('.nii.gz', '_reward_anat.nii.gz')}")
    rew_lm = restore_image(out[2][0].cpu().numpy(), original_shape, original_spacing)
    nib.save(nib.Nifti1Image(rew_lm, affine=img_nii.affine), f"{args.output}/{in_name.replace('.nii.gz', 'reward_LM.nii.gz')}")

    print(f"Saved output to {args.output}")

if __name__ == "__main__":
    main()