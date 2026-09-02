#!/usr/bin/env python
"""Package an RL4Seg3D checkpoint as a self-contained TorchScript model.

The result is one ``.pt`` that ``torch.jit.load``s with no RL4Seg3D, hydra, MONAI or
torchio import on the far side, and reproduces the inference path of ``predict_3d.py``:
temporal sliding window, border padding, test-time augmentation, and the reward maps.

    forward(image, view, tta=True) -> (segmentation, fused_reward, per_net_rewards)
        image             float tensor (1, 1, H, W, T), intensities rescaled to [0, 1] and
                          resampled to the common 0.37mm in-plane spacing
        view              'a2c' | 'a3c' | 'a4c'  (the FiLM view conditioning)
        tta               25-pass test-time augmentation; ~25x slower, better masks
        segmentation      long  (H, W, T), labels 0/1/2
        fused_reward      float (H, W, T), elementwise min over the reward nets
        per_net_rewards   float (N, H, W, T), order given by reward_map_names()

``--no-rewards`` builds a segmentation-only model whose forward returns just the mask.

A full RL checkpoint carries the actor and every reward net, so one file is all it needs:

    python export_torchscript_3d.py --ckpt best25kcheated-31-08-26.ckpt \
        --out data/checkpoints/rl4seg3d_torchscript_FILM.pt

Reward nets can also be supplied separately, which is what an actor-only checkpoint (a
``*_net.ckpt``) requires:

    python export_torchscript_3d.py --ckpt best25kcheated-31-08-26_net.ckpt \
        --reward anatomical=anat_epoch72.ckpt \
        --reward landmarks=landmarks_long.ckpt \
        --reward apex=apex_rewardnet.ckpt \
        --out data/checkpoints/rl4seg3d_torchscript_FILM.pt

Export runs on CPU on purpose: the saved graph is portable (load with ``map_location``),
and CPU convolutions avoid the TF32 rounding that would blur the eager-vs-scripted
equivalence check below.
"""
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rl4seg3d.packaging.scriptable_unet import ScriptableFiLMUNet
from rl4seg3d.packaging.torchscript_module import (
    FullPackage,
    SegmentationPackage,
    WindowedNet,
)
from rl4seg3d.supervised.conditioned_unet_film import FiLMUNet

# Architecture, read from the config the checkpoints were trained under so the two cannot
# drift apart. The reward nets share it except for their input/output channels: they take
# (image, segmentation) and emit a single logit per voxel.
NET_CONFIG = Path(__file__).parent / "rl4seg3d/config/actor/actor/net/film.yaml"
REWARD_NET_OVERRIDES = dict(in_channels=2, num_classes=1)

# Must match VIEW_MAP in predict_3d.py and RL_3d_datamodule.py -- these indices address the
# rows of the trained view embedding, so a permutation here silently mis-conditions.
VIEWS = {"a2c": 0, "a3c": 1, "a4c": 2}

# Reward nets in the order RewardUnets3D loads them (config/reward/rewardunets_3d.yaml),
# which is the order they are stored under in a full checkpoint.
DEFAULT_REWARD_NAMES = ["anatomical", "landmarks", "apex"]

ACTOR_PREFIXES = ["actor.actor.net.", "net."]


def net_kwargs(**overrides):
    cfg = OmegaConf.to_container(OmegaConf.load(NET_CONFIG), resolve=True)
    cfg.pop("_target_", None)
    cfg.update(overrides)
    return cfg


def _strip_prefix(state_dict, prefix):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_checkpoint_nets(path, reward_names):
    """Return ``(actor_state_dict, {name: reward_state_dict})`` from any checkpoint layout.

    A Lightning RL checkpoint nests the actor under ``actor.actor.net.`` and the reward nets
    under ``rewardnet_<i>.``; a converted ``*_net.ckpt`` is the bare actor state_dict with no
    prefix and no reward nets at all. Both are accepted, so the same command works whichever
    file is at hand.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    actor = {}
    for prefix in ACTOR_PREFIXES:
        actor = _strip_prefix(state_dict, prefix)
        if actor:
            print(f"Actor: stripped prefix {prefix!r} ({len(actor)} tensors)")
            break
    if not actor:
        # Already a bare net state_dict.
        actor = dict(state_dict)
        print(f"Actor: checkpoint is a bare net state_dict ({len(actor)} tensors)")

    rewards = {}
    for i, name in enumerate(reward_names):
        sd = _strip_prefix(state_dict, f"rewardnet_{i}.")
        if sd:
            rewards[name] = sd
            print(f"Reward net {i} -> {name!r} ({len(sd)} tensors)")
    return actor, rewards


def build_net(state_dict, **overrides):
    net = ScriptableFiLMUNet(**net_kwargs(**overrides))
    net.load_state_dict(state_dict, strict=True)
    return net.eval()


def check_matches_eager(scripted_net, state_dict, in_channels, num_classes, num_views):
    """Assert the scripted subclass still computes what the training class computes.

    ScriptableFiLMUNet only rewrites how the blocks are called, but that rewrite is the one
    thing standing between the shipped weights and the published results, so it is checked
    against FiLMUNet itself rather than trusted.
    """
    reference = FiLMUNet(**net_kwargs(in_channels=in_channels, num_classes=num_classes))
    reference.load_state_dict(state_dict, strict=True)
    reference.eval()

    x = torch.rand(1, in_channels, 64, 64, 4)
    worst = 0.0
    with torch.no_grad():
        for v in range(num_views):
            view = torch.tensor([v], dtype=torch.long)
            worst = max(worst, (scripted_net(x, view) - reference(x, view)).abs().max().item())
    return worst


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", required=True,
                        help="RL checkpoint, or a converted actor-only *_net.ckpt")
    parser.add_argument("--out", required=True, help="path to write the TorchScript model to")
    parser.add_argument("--reward", action="append", default=[], metavar="NAME=PATH",
                        help="reward net to add, repeatable. Overrides one read from --ckpt. "
                             "Required when --ckpt holds no reward nets.")
    parser.add_argument("--reward-names", default=",".join(DEFAULT_REWARD_NAMES),
                        help="names for rewardnet_0..N-1 inside a full checkpoint, in order")
    parser.add_argument("--no-clean-blobs", action="store_true",
                        help="Skip the per-frame largest-connected-component cleanup. It is "
                             "on by default because predict_3d.py applies it before scoring, "
                             "so the reward maps describe the cleaned mask either way.")
    parser.add_argument("--no-rewards", action="store_true",
                        help="build a segmentation-only model (forward returns just the mask)")
    parser.add_argument("--half", action="store_true",
                        help="store weights as float16, halving the file at some precision cost")
    parser.add_argument("--skip-checks", action="store_true",
                        help="skip the eager-equivalence and view-sensitivity checks")
    args = parser.parse_args()

    reward_names = [n.strip() for n in args.reward_names.split(",") if n.strip()]
    actor_sd, reward_sds = load_checkpoint_nets(args.ckpt, reward_names)

    for spec in args.reward:
        if "=" not in spec:
            parser.error(f"--reward expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        reward_sds[name] = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
        print(f"Reward net {name!r} <- {path}")

    if not args.no_rewards and not reward_sds:
        parser.error(
            f"{args.ckpt} holds no reward nets and none were given with --reward. "
            "Pass --no-rewards for a segmentation-only model, or supply them explicitly."
        )

    num_views = net_kwargs()["num_views"]
    if len(VIEWS) != num_views:
        raise ValueError(f"{len(VIEWS)} view names but the net has {num_views} embedding rows")

    actor = build_net(actor_sd)
    segmenter = WindowedNet(actor, net_kwargs()["num_classes"])

    clean_blobs = not args.no_clean_blobs
    print(f"Largest-blob cleanup: {'on' if clean_blobs else 'off'}")

    if args.no_rewards:
        model = SegmentationPackage(segmenter, dict(VIEWS), clean_blobs).eval()
        preserved = ["view_names", "cleans_blobs"]
    else:
        # Order the maps the way the checkpoint/config orders the nets, and put any
        # separately supplied net that is not in that list at the end.
        ordered = [n for n in reward_names if n in reward_sds]
        ordered += [n for n in reward_sds if n not in ordered]
        reward_nets = [
            WindowedNet(build_net(reward_sds[n], **REWARD_NET_OVERRIDES), 1) for n in ordered
        ]
        print(f"Reward maps, in output order: {ordered}")
        model = FullPackage(segmenter, reward_nets, ordered, dict(VIEWS), clean_blobs).eval()
        preserved = ["view_names", "reward_map_names", "cleans_blobs"]

    if not args.skip_checks:
        diff = check_matches_eager(actor, actor_sd, 1, net_kwargs()["num_classes"], num_views)
        print(f"Scripted vs eager FiLMUNet, max|diff| = {diff:.3e}")
        assert diff == 0.0, "the scriptable subclass does not reproduce FiLMUNet exactly"

    scripted = torch.jit.script(model)
    if args.half:
        scripted = scripted.half()
    scripted = torch.jit.freeze(scripted.eval(), preserved_attrs=preserved)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(args.out)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"\nSaved -> {args.out}  ({size_mb:.1f} MB)")

    # Run the check on the SAVED file rather than the object in memory: it is the artifact
    # that ships, and torch.jit.load places it on the target device properly. Moving a frozen
    # module with .to() does not -- freezing bakes the weights in as constants that stay
    # where they were, and calling it with input from another device kills the process
    # outright, with no exception to catch.
    device = "cpu"
    if args.half and not args.skip_checks:
        # float16 convolutions on CPU are slow enough to be unusable.
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("Skipping the runtime check: --half needs a GPU to run in tolerable time "
                  "and none is visible. Re-run the check where there is one.")
            args.skip_checks = True

    if not args.skip_checks:
        dtype = torch.float16 if args.half else torch.float32
        loaded = torch.jit.load(args.out, map_location=device).eval()
        image = torch.rand(1, 1, 96, 80, 7, dtype=dtype, device=device)
        with torch.no_grad():
            a = loaded(image, "a2c", False)
            b = loaded(image, "a4c", False)
        seg_a = a if args.no_rewards else a[0]
        seg_b = b if args.no_rewards else b[0]
        changed = int((seg_a != seg_b).sum().item())
        print(f"Reloaded from disk on {device}; output shape {tuple(seg_a.shape)} from input "
              f"{tuple(image.shape[2:])} (non-multiples of 32 padded internally)")
        print(f"View conditioning is live: {changed} voxels differ between a2c and a4c")
        assert changed > 0, "the view argument does not change the output -- conditioning is dead"
    print("Validate against the research pipeline with:\n"
          f"  python rl4seg3d/scripts/validate_torchscript_package.py --package {args.out} "
          f"--ckpt {args.ckpt}")


if __name__ == "__main__":
    main()
