from copy import deepcopy
from typing import Union, List, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from torch import Tensor

from patchless_nnunet.utils.inferers import SlidingWindowInferer
from rl4seg3d.actors.Actors_3d import _zero_missing_cond_projectors
from rl4seg3d.reward.generic_reward import Reward

"""
Reward functions must each have pred, img, gt as input parameters
"""


def _reward_net_forward(net, stack, cond):
    """Call reward net with cond when supported, otherwise plain forward.

    Supports legacy plain-UNet reward checkpoints alongside ConditionedUNet ones.
    """
    if cond is not None and hasattr(net, "cond_projectors"):
        return net(stack, cond)
    return net(stack)


class RewardUnets3D(Reward):
    def __init__(self, net, state_dict_paths, temp_factor=1):
        self.nets = {}
        for name, path in state_dict_paths.items():
            n = deepcopy(net)
            if path and Path(path).exists():
                # strict=False so legacy plain-UNet checkpoints load into a
                # ConditionedUNet skeleton; any cond_projector keys that aren't in
                # the legacy checkpoint get zeroed so the reward output is
                # bit-identical to the original plain-UNet behavior at load time.
                missing, unexpected = n.load_state_dict(torch.load(path), strict=False)
                _zero_missing_cond_projectors(n, missing)
                if unexpected:
                    print(f"[RewardUnets3D] {name}: unexpected keys when loading {path}: {unexpected}")
                non_cond_missing = [k for k in missing if "cond_projector" not in k]
                if non_cond_missing:
                    print(f"[RewardUnets3D] {name}: missing keys when loading {path}: {non_cond_missing}")
            else:
                print(f"BEWARE! You don't have a valid path for this reward net: {name}\n"
                      "Ignore if using full checkpoint file")
            n.eval()
            self.nets.update({name: n})
        self.temp_factor = temp_factor

    @torch.no_grad()
    def __call__(self, pred, imgs, gt, cond=None):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)
        r = []
        for net in self.get_nets():
            out = _reward_net_forward(net, stack, cond)
            # assert not torch.isnan(o).any(), "NaNs in RewardNet out"
            t = self.temp_factor if len(r) < 1 else 1 # ONLY TEMPERATURE SCALE ANATOMICAL (0)
            rew = torch.sigmoid(out / t).squeeze(1)
            for i in range(rew.shape[0]):
                for j in range(rew.shape[-1]):
                    rew[i, ..., j] = rew[i, ..., j] - rew[i, ..., j].min()
                    rew[i, ..., j] = rew[i, ..., j] / rew[i, ..., j].max()
            r += [rew]
        if len(r) > 1:
            r = [torch.minimum(r[0], r[1])]
        return r

    def get_nets(self):
        return list(self.nets.values())

    def get_reward_index(self, reward_name):
        return list(self.nets.keys()).index(reward_name)

    @torch.no_grad()
    def predict_full_sequence(self, pred, imgs, gt, cond=None):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)

        self.patch_size = list([stack.shape[-3], stack.shape[-2], 4])
        self.inferer.roi_size = self.patch_size

        return [torch.sigmoid(p).squeeze(1) for p in self.predict(stack, cond=cond)]

    def prepare_for_full_sequence(self, batch_size=1) -> None:  # noqa: D102
        sw_batch_size = batch_size

        self.inferer = SlidingWindowInferer(
            roi_size=self.get_nets()[0].patch_size,
            sw_batch_size=sw_batch_size,
            overlap=0.5,
            mode='gaussian',
            cache_roi_weight_map=True,
        )

    def predict(
        self, image: Union[Tensor, MetaTensor],
        cond: Optional[Tensor] = None,
    ) -> List[Union[Tensor, MetaTensor]]:
        """Predict 2D/3D images with sliding window inference.

        Args:
            image: Image to predict.
            cond: optional view conditioning tensor (B, cond_dim).

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            NotImplementedError: If the patch shape is not 2D nor 3D.
            ValueError: If 3D patch is requested to predict 2D images.
        """
        if len(image.shape) == 5:
            if np.asarray([len(self.get_nets()[i].patch_size) == 3 for i in range(len(self.get_nets()))]).all():
                # Pad the last dimension to avoid 3D segmentation border artifacts
                pad_len = 6 if image.shape[-1] > 6 else image.shape[-1] - 1
                image = F.pad(image, (pad_len, pad_len, 0, 0, 0, 0), mode="reflect")
                pred = self.predict_3D_3Dconv_tiled(image, cond=cond)
                # Inverse the padding after prediction
                return [p[..., pad_len:-pad_len] for p in pred]
            else:
                raise ValueError("Check your patch size. You dummy.")
        if len(image.shape) == 4:
            raise ValueError("No 2D images here. You dummy.")

    def predict_3D_3Dconv_tiled(
        self, image: Union[Tensor, MetaTensor],
        cond: Optional[Tensor] = None,
    ) -> List[Union[Tensor, MetaTensor]]:
        """Predict 3D image with 3D model.

        Args:
            image: Image to predict.
            cond: optional view conditioning tensor (B, cond_dim).

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            ValueError: If image is not 3D.
        """
        if not len(image.shape) == 5:
            raise ValueError("image must be (b, c, w, h, d)")

        return self.sliding_window_inference(image, cond=cond)

    def sliding_window_inference(
        self, image: Union[Tensor, MetaTensor],
        cond: Optional[Tensor] = None,
    ) -> List[Union[Tensor, MetaTensor]]:
        """Inference using sliding window, broadcasting cond to every patch batch."""
        results = []
        for n in self.get_nets():
            if cond is not None and hasattr(n, "cond_projectors"):
                def _net(x, _n=n, _c=cond):
                    return _n(x, _c.expand(x.shape[0], -1))
                results.append(self.inferer(inputs=image, network=_net))
            else:
                results.append(self.inferer(inputs=image, network=n))
        return results
