from pathlib import Path

import torch
import numpy as np
import nibabel as nib
import argparse

from tqdm import tqdm

# View strings the FiLM-conditioned models accept. A model exported before view
# conditioning takes no view at all; both are handled, see call_model().
KNOWN_VIEWS = ("a2c", "a3c", "a4c")


def find_view(path):
    """Pick the view out of a case's path, the way the rest of the project does.

    icardio-style trees put it in a directory (``.../st-X/a4c/di-Y.nii.gz``); failing that
    the filename is checked. Returns None when nothing matches, which is a hard error for a
    conditioned model -- guessing a view would silently segment the case as the wrong
    chamber rather than fail.
    """
    path = Path(path)
    for part in reversed(path.parts[:-1]):
        if part.lower() in KNOWN_VIEWS:
            return part.lower()
    name = path.name.lower()
    for view in KNOWN_VIEWS:
        if view in name:
            return view
    return None


def resize_in_plane(img, new_H, new_W, order):
    """Resize the in-plane axes of an (H, W, T) volume, frame by frame.

    ``order='linear'`` for intensities and reward maps, ``'nearest'`` for label maps (a
    linear interpolation between labels 0 and 2 would invent a label 1 boundary that the
    model never predicted).
    """
    tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[:, None]
    mode = "bilinear" if order == "linear" else "nearest"
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    out = torch.nn.functional.interpolate(
        tensor.float(), size=(new_H, new_W), mode=mode, **kwargs
    )
    return out[:, 0].permute(1, 2, 0).numpy()


def equalize(img):
    """Per-frame CLAHE, the contrast normalisation the training images were built with.

    The models are trained on the ``maple_norm`` dataset, which is
    ``equalize_adapthist(frame, clip_limit=0.01)`` applied frame by frame
    (rl4seg3d/scripts/normalise_images.py). Images that have not been through it are out of
    distribution, so this belongs on any raw input -- but applying it twice is just as wrong,
    which is why it is a flag rather than the default.
    """
    import skimage.exposure as exp
    out = np.empty_like(img)
    for i in range(img.shape[-1]):
        out[..., i] = exp.equalize_adapthist(img[..., i], clip_limit=0.01)
    return out


def adjust_image(img_nii, equalize_hist=False):
    """
    Normalize image and rescale to target spacing (0.37 mm in-plane).
    """
    img = img_nii.get_fdata().astype(np.float32)
    img /= img.max() if img.max() > 0 else 1.0  # normalize safely
    if equalize_hist:
        # Before the resample, matching the order in predict_3d.py's dataset.
        img = equalize(img).astype(np.float32)

    spacing = img_nii.header.get_zooms()[:3]
    H, W, T = img.shape

    # --- Rescale in-plane spacing to 0.37 mm ---
    target_spacing = (0.37, 0.37)
    scale_h = spacing[0] / target_spacing[0]
    scale_w = spacing[1] / target_spacing[1]
    new_H = int(round(H * scale_h))
    new_W = int(round(W * scale_w))

    # Every case is resampled, as predict_3d.py's tio.Resample is. Bilinear: the model was
    # trained and validated on torchio's linearly resampled volumes.
    if (new_H, new_W) != (H, W):
        print(f"Rescaling from spacing {spacing[:2]} -> {target_spacing} "
              f"({H}x{W} -> {new_H}x{new_W})")
        img = resize_in_plane(img, new_H, new_W, "linear")

    return img, spacing, (H, W, T)


def restore_image(processed_img, original_shape, order="linear"):
    """
    Undo the spacing adjustment, putting an output back on the input's own grid.

    Driven by the shape the model actually produced rather than by recomputing the spacing
    test: whatever adjust_image did or skipped, resizing exactly when the in-plane shape
    differs is self-consistent, and the two can no longer disagree about whether a resample
    happened.

    processed_img: np.ndarray of shape (H', W', T')
    original_shape: tuple from original_nii.shape
    order: 'nearest' for label maps, 'linear' for images and reward maps
    """
    H0, W0, _ = original_shape
    if processed_img.shape[:2] == (H0, W0):
        return processed_img
    return resize_in_plane(processed_img, H0, W0, order)


def resolve_device(requested=None):
    """Pick the device to run on: what was asked for, else the best available.

    Auto-detection is cuda then cpu. Any device may still be named explicitly, and an
    explicit choice is honoured as given rather than silently downgraded, so a machine that
    was meant to use its GPU fails loudly instead of quietly taking 25x longer on the CPU.
    """
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def call_model(model, img_tensor, view, tta):
    """Run the model, whether or not it takes a view.

    Models exported before view conditioning have the signature ``(image, tta)``; the FiLM
    ones take ``(image, view, tta)``. Which it is has already been settled by whether the
    model exposes view_names(), and view is None exactly when it does not.

    Deliberately no try/except around the conditioned call: catching RuntimeError to detect
    an unconditioned model also catches every failure raised from inside the model, and then
    reports it as "this model takes no view argument" while re-running and failing a second
    time on the signature. The real error is what should surface.
    """
    if view is None:
        return model(img_tensor, tta)
    return model(img_tensor, view, tta)


def reward_names(model, count):
    """Names for the reward maps, from the model when it carries them."""
    if hasattr(model, "reward_map_names"):
        names = list(model.reward_map_names())
        if len(names) == count:
            return names
    # Older packages carry no names; fall back to the historical order.
    return (["anat", "LM", "apex"] + [f"reward_{i}" for i in range(count)])[:count]


def process_single_file(input_path, output_dir, model, tta=True, view=None,
                        equalize_hist=False, device="cuda"):
    img_nii = nib.load(input_path)
    img, original_spacing, original_shape = adjust_image(img_nii, equalize_hist=equalize_hist)

    H, W, T = img.shape
    if T > min(H, W):
        print(f"Warning: {input_path} — Temporal dimension might not be last.")

    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W, T)
    with torch.no_grad():
        out = call_model(model, img_tensor, view, tta)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    in_name = Path(input_path).stem.replace('.nii', '')  # handle .nii and .nii.gz

    def save(array, suffix, order):
        restored = restore_image(array, original_shape, order=order)
        if order == "nearest":
            restored = restored.astype(np.uint8)
        nib.save(nib.Nifti1Image(restored, affine=img_nii.affine),
                 output_dir / f"{in_name}_{suffix}.nii.gz")

    # A segmentation-only package returns the mask alone; the full one returns
    # (mask, fused reward, per-net rewards). Both are accepted so the same command works
    # whichever file was passed to --ckpt.
    if isinstance(out, torch.Tensor):
        seg, fused, per_net = out, None, None
    else:
        seg, fused, per_net = out[0], out[1], out[2]

    # Nearest for the label map, linear for the reward maps: resampling labels linearly
    # would fabricate boundary classes.
    save(seg.cpu().numpy(), "segmentation", "nearest")

    if per_net is not None:
        save(fused.cpu().numpy(), "reward_fusion", "linear")
        # per_net is (N, H, W, T), one map per reward net, in reward_map_names() order.
        per_net = per_net.cpu().numpy()
        for name, reward_map in zip(reward_names(model, len(per_net)), per_net):
            save(reward_map, f"reward_{name}", "linear")

    print(f"Done processing: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Run lightweight TorchScript RL4Seg model on a NIfTI image or folder")
    parser.add_argument("--input", "-i", required=True, help="Path to input NIfTI image or folder")
    parser.add_argument("--output", "-o", required=True, help="Path to save output NIfTI files")
    parser.add_argument("--ckpt", "-c", default='./data/checkpoints/rl4seg3d_torchscript_TTA.pt',
                        help="Path to TorchScript checkpoint. Default is ./data/checkpoints/rl4seg3d_torchscript_TTA.pt")
    parser.add_argument("--no_tta", "-t", action="store_false",
                        help="Turn off TTA (faster inference, but reduced segmentation quality)")
    parser.add_argument("--view", "-v", default=None, choices=KNOWN_VIEWS,
                        help="Echo view for the FiLM view conditioning. Detected from each "
                             "file's path when omitted; required by view-conditioned models.")
    parser.add_argument("--equalize", "-e", action="store_true",
                        help="Apply per-frame CLAHE (equalize_adapthist, clip_limit=0.01) "
                             "before inference. The training images were normalised this way, "
                             "so raw images need it -- but images from an already-normalised "
                             "dataset must NOT be equalized twice.")
    parser.add_argument("--device", "-d", default=None,
                        help="Device to run on (cuda, cuda:1, mps, cpu). Defaults to cuda "
                             "when one is visible, else cpu. NOTE: TTA is 25 full "
                             "inference passes, so it is impractically slow on cpu -- pair "
                             "--device cpu with --no_tta.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Running on {device}")
    model = torch.jit.load(args.ckpt, map_location=device)
    model.eval()

    conditioned = hasattr(model, "view_names")
    if conditioned:
        print(f"View-conditioned model; accepted views: {list(model.view_names())}")

    def view_for(path):
        """--view wins; otherwise read it off the path, and refuse to guess."""
        if args.view:
            return args.view
        if not conditioned:
            return None
        view = find_view(path)
        if view is None:
            raise ValueError(
                f"No view (one of {KNOWN_VIEWS}) found in the path of {path}, and this model "
                f"is view-conditioned. Pass --view explicitly."
            )
        return view

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
                # beware of negation logic for no_tta arg
                process_single_file(file, args.output, model, args.no_tta, view_for(file),
                                    args.equalize, device)
            except Exception as e:
                print(f"Failed on {file}: {e}")
    else:
        print(f"Processing WITH{'OUT' if not args.no_tta else ''} Test-time augmentation (TTA)")
        process_single_file(args.input, args.output, model, args.no_tta,
                            view_for(args.input), args.equalize, device)

    print(f"\nOutputs saved to {args.output}")

if __name__ == "__main__":
    main()
