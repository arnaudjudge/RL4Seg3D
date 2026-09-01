#!/usr/bin/env python
"""Check a packaged TorchScript model against the research pipeline it was built from.

The packaged module reimplements the inference path in pure torch so it can be scripted.
This script runs the ORIGINAL code -- ``RLmodule_3D.predict`` / ``tta_predict`` and
``RewardUnets3D.predict_full_sequence``, with MONAI's sliding-window inferer underneath --
on the same input and asserts the two agree.

Four things are checked, cheapest first:

1. the window origins against ``patchless_nnunet.compute_steps_for_sliding_window``
2. the gaussian blending weights against ``monai.data.compute_importance_map``
3. segmentation and reward maps, TTA off
4. segmentation and reward maps, TTA on (25 passes, so this is the slow one)

Plus a check that an input whose H/W are not multiples of 32 -- which the module pads
internally and the pipeline pads in preprocessing -- gives the same answer either way.

    python rl4seg3d/scripts/validate_torchscript_package.py \
        --package data/checkpoints/rl4seg3d_torchscript_FILM.pt \
        --ckpt best25kcheated-31-08-26.ckpt

Random volumes are the default input: the claim under test is a numerical identity between
two implementations, and it either holds for arbitrary input or does not hold. Pass
``--input`` to run real NIfTI sequences through it as well.
"""
import argparse
import math
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data.utils import compute_importance_map
from patchless_nnunet.utils.inferers import (
    SlidingWindowInferer,
    compute_steps_for_sliding_window,
)

from rl4seg3d.RLmodule_3D import RLmodule3D
from rl4seg3d.packaging.torchscript_module import _importance_map, _window_starts
from rl4seg3d.reward.reward_nets_3d import RewardUnets3D
from rl4seg3d.supervised.conditioned_unet_film import FiLMUNet

import export_torchscript_3d as exporter


def check_window_starts():
    """Our window origins vs the function the pipeline uses, over every plausible length."""
    worst = []
    for length in range(4, 200):
        theirs = compute_steps_for_sliding_window((1, 1, 4), (1, 1, length), 0.5)[2]
        ours = _window_starts(length, 4, 0.5)
        if theirs != ours:
            worst.append((length, theirs, ours))
    return worst


def check_importance_map():
    """Our blending weights vs MONAI's, including the floor it applies to non-positive ones."""
    worst = 0.0
    for h, w in [(32, 32), (352, 288), (96, 80)]:
        theirs = compute_importance_map((h, w, 4), mode="gaussian", sigma_scale=0.125)
        theirs = torch.clamp(theirs, min=max(theirs[theirs != 0].min().item(), 1e-3))
        ours = _importance_map(h, w, 4, 0.125, torch.float32, torch.device("cpu"))
        worst = max(worst, (theirs - ours).abs().max().item())
    return worst


def build_reference(ckpt_path, reward_names):
    """The genuine pipeline objects, with the least scaffolding that makes them run.

    ``RLmodule_3D.predict``/``tta_predict`` only reach for ``self.actor.actor.net``, so the
    actor is stubbed down to that; everything those methods actually do -- the border pad,
    the inferer, the conditioning closure -- is the shipped code, unmodified.
    """
    actor_sd, reward_sds = exporter.load_checkpoint_nets(ckpt_path, reward_names)

    actor_net = FiLMUNet(**exporter.net_kwargs())
    actor_net.load_state_dict(actor_sd, strict=True)
    actor_net.eval()

    reward_net = FiLMUNet(**exporter.net_kwargs(**exporter.REWARD_NET_OVERRIDES))
    ordered = [n for n in reward_names if n in reward_sds]
    # Paths of None: RewardUnets3D builds one copy of the skeleton per entry and warns that
    # it loaded nothing, which is exactly what we want before filling them in by hand -- the
    # weights live inside the RL checkpoint, not in files of their own.
    reward_func = RewardUnets3D(reward_net, {name: None for name in ordered})
    for name in ordered:
        reward_func.nets[name].load_state_dict(reward_sds[name], strict=True)
        reward_func.nets[name].eval()
    if not ordered:
        # A segmentation-only package validated against an actor-only checkpoint: there is
        # nothing to score, and RewardUnets3D cannot size its inferer without a net.
        print("No reward nets in the checkpoint: validating the segmentation alone.")

    module = RLmodule3D(
        actor=types.SimpleNamespace(actor=types.SimpleNamespace(net=actor_net)),
        reward=reward_func,
        corrector=None,
        num_views=len(exporter.VIEWS),
    )
    module.eval()
    if ordered:
        reward_func.prepare_for_full_sequence(1)
    return module, ordered


def reference_outputs(module, image, view, tta):
    """One case through the original inference path, returning (labels, per-net rewards)."""
    view_id = torch.tensor([exporter.VIEWS[view]], dtype=torch.long)
    module._current_cond = view_id
    module.patch_size = [image.shape[-3], image.shape[-2], 4]
    module.inferer = SlidingWindowInferer(
        roi_size=module.patch_size, sw_batch_size=1, overlap=0.5,
        mode="gaussian", cache_roi_weight_map=True,
    )
    if module.reward_func.nets:
        module.reward_func.inferer = SlidingWindowInferer(
            roi_size=module.patch_size, sw_batch_size=1, overlap=0.5,
            mode="gaussian", cache_roi_weight_map=True,
        )

    with torch.no_grad():
        if tta:
            seg = module.tta_predict(image)
        else:
            seg = module.predict(image).argmax(dim=1)
        rewards = (module.reward_func.predict_full_sequence(seg, image, None, cond=view_id)
                   if module.reward_func.nets else [])
    return seg, rewards


def pad_to_multiple(image, block=32):
    """torchio CropOrPad's split: the larger half of an odd pad goes before the image."""
    h, w = image.shape[2], image.shape[3]
    diff_h = int(math.ceil(h / block)) * block - h
    diff_w = int(math.ceil(w / block)) * block - w
    before_h, before_w = diff_h - diff_h // 2, diff_w - diff_w // 2
    padded = F.pad(image, (0, 0, before_w, diff_w - before_w, before_h, diff_h - before_h))
    return padded, before_h, before_w


def run_package(package, image, view, tta):
    """Normalise both build shapes to (segmentation, reward maps or None).

    A --no-rewards package returns the mask alone; the full one returns the 3-tuple.
    """
    with torch.no_grad():
        out = package(image, view, tta)
    if isinstance(out, torch.Tensor):
        return out, None, None
    return out[0], out[1], out[2]


def compare(module, package, image, view, tta, label, results):
    ref_seg, ref_rewards = reference_outputs(module, image, view, tta)
    seg, merged, maps = run_package(package, image, view, tta)

    disagreeing = int((ref_seg[0] != seg).sum().item())
    total = int(seg.numel())
    ok = disagreeing == 0
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print(f"        segmentation: {disagreeing}/{total} voxels differ")

    if maps is not None:
        reward_diff = max(
            (ref[0] - got).abs().max().item() for ref, got in zip(ref_rewards, maps)
        )
        ref_merged = ref_rewards[0][0]
        for r in ref_rewards[1:]:
            ref_merged = torch.minimum(ref_merged, r[0])
        merged_diff = (ref_merged - merged).abs().max().item()
        ok = ok and reward_diff == 0.0 and merged_diff == 0.0
        print(f"        reward maps:  max|diff| = {reward_diff:.3e}   fused: {merged_diff:.3e}")

    results.append(ok)
    return ok


def load_nifti(path):
    import nibabel as nib
    from rl4seg3d.utils.preprocessing import rescale
    data = nib.load(path).get_fdata().astype(np.float32)
    return torch.from_numpy(rescale(data[None, None]))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", required=True, help="the exported .pt to validate")
    parser.add_argument("--ckpt", required=True, help="the checkpoint it was exported from")
    parser.add_argument("--reward-names", default=",".join(exporter.DEFAULT_REWARD_NAMES))
    parser.add_argument("--input", default=None,
                        help="optional NIfTI file or directory to validate on as well")
    parser.add_argument("--limit", type=int, default=2, help="how many --input files to use")
    parser.add_argument("--no-tta", action="store_true",
                        help="skip the TTA comparison (25x cheaper, but leaves TTA unchecked)")
    args = parser.parse_args()

    torch.manual_seed(0)
    results = []

    print("1. window origins vs compute_steps_for_sliding_window")
    mismatches = check_window_starts()
    results.append(not mismatches)
    print(f"  {'PASS' if not mismatches else 'FAIL'}  lengths 4..199"
          + (f", {len(mismatches)} mismatch(es): {mismatches[:3]}" if mismatches else ""))

    print("2. blending weights vs monai.compute_importance_map")
    diff = check_importance_map()
    results.append(diff == 0.0)
    print(f"  {'PASS' if diff == 0.0 else 'FAIL'}  max|diff| = {diff:.3e}")

    package = torch.jit.load(args.package, map_location="cpu").eval()
    reward_names = [n.strip() for n in args.reward_names.split(",") if n.strip()]
    module, ordered = build_reference(args.ckpt, reward_names)
    if hasattr(package, "reward_map_names"):
        packaged_names = list(package.reward_map_names())
        results.append(packaged_names == ordered)
        print(f"3. reward map order: package {packaged_names} vs checkpoint {ordered} -> "
              f"{'PASS' if packaged_names == ordered else 'FAIL'}")
    else:
        print("3. reward map order: skipped, this is a segmentation-only package")

    print("4. segmentation and reward maps vs the pipeline")
    cases = [(torch.rand(1, 1, 64, 64, 8), "a4c"), (torch.rand(1, 1, 96, 64, 11), "a2c")]
    for image, view in cases:
        compare(module, package, image, view, False,
                f"random {tuple(image.shape[2:])} view={view} tta=off", results)
    if not args.no_tta:
        image, view = cases[0]
        compare(module, package, image, view, True,
                f"random {tuple(image.shape[2:])} view={view} tta=ON", results)

    print("5. in-plane padding (module pads internally, pipeline pads in preprocessing)")
    raw = torch.rand(1, 1, 50, 44, 8)
    padded, off_h, off_w = pad_to_multiple(raw)
    ref_seg, ref_rewards = reference_outputs(module, padded, "a3c", False)
    seg, _, maps = run_package(package, raw, "a3c", False)
    cropped = ref_seg[0, off_h:off_h + 50, off_w:off_w + 44, :]
    ok = int((cropped != seg).sum().item()) == 0
    if maps is not None:
        cropped_reward = ref_rewards[0][0, off_h:off_h + 50, off_w:off_w + 44, :]
        ok = ok and (cropped_reward - maps[0]).abs().max().item() == 0.0
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  50x44 padded to 64x64 and cropped back")

    if args.input:
        print("6. real sequences")
        path = Path(args.input)
        files = sorted(list(path.rglob("*.nii")) + list(path.rglob("*.nii.gz"))) \
            if path.is_dir() else [path]
        for f in files[:args.limit]:
            image = load_nifti(f)
            compare(module, package, image, "a4c", not args.no_tta,
                    f"{f.name} {tuple(image.shape[2:])}", results)

    passed = sum(1 for r in results if r)
    print(f"\n{passed}/{len(results)} checks passed")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
