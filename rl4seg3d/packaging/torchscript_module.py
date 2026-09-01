"""The scripted modules that get serialised into the shipped ``.pt``.

Everything here is written to compile under ``torch.jit.script`` and to reproduce
``RLmodule_3D``'s inference path exactly, so a packaged model and the research pipeline
give the same answer for the same input:

* temporal sliding window -- ``roi = [H, W, 4]``, 0.5 overlap, gaussian blending. The
  window step and the importance map replicate ``patchless_nnunet``'s
  ``compute_steps_for_sliding_window`` and MONAI's ``compute_importance_map`` bit for bit
  (asserted by ``rl4seg3d/scripts/validate_torchscript_package.py``).
* the reflect pad of 6 frames on either side that ``RLmodule_3D.predict`` applies to keep
  border artefacts out of the first and last frames.
* the 25-pass test-time augmentation of ``RLmodule_3D.tta_predict``.
* reward maps as raw per-net sigmoids fused with an elementwise minimum -- matching
  ``RewardUnets3D.predict_full_sequence``, which (unlike ``RewardUnets3D.__call__``,
  used during training) applies neither the per-frame renorm nor the temperature.

What is NOT in here, and so is not in the packaged model:

* test-time optimization. It needs autograd and mutates the actor's weights per case, so
  it cannot be a TorchScript graph at all.
* the largest-connected-component cleanup that ``inference_predict_step`` runs before the
  reward nets. The shipped packages have never included it either, so leaving it out keeps
  parity with them; ``torchscript_predict_3d.py`` is where to add it if that changes.
* resampling to the common 0.37mm spacing, which stays with the caller -- see
  ``torchscript_predict_3d.py``.

The in-plane pad to a multiple of 32 IS here, because the caller has no way to know the
network's downsampling factor.
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms.functional import adjust_contrast, rotate

# Frames per window, and the fraction of it the window advances by. Both fixed by the
# inference path in RLmodule_3D (patch_size = [H, W, 4], SlidingWindowInferer overlap=0.5).
_WINDOW_FRAMES = 4
_OVERLAP = 0.5
# MONAI's gaussian blending default: sigma = _SIGMA_SCALE * window length, per axis.
_SIGMA_SCALE = 0.125
# In-plane downsampling factor of the net (five stride-2 stages).
_DIVISIBLE_BY = 32
# Frames reflected onto each end before inference, from RLmodule_3D.predict.
_BORDER_PAD_FRAMES = 6

_TTA_CONTRAST = [1.1, 0.9, 1.25, 0.75]
_TTA_TRANSLATIONS = [40, 60, 80, 120]
_TTA_ROTATIONS = [5.0, 10.0, -5.0, -10.0]

# TorchScript resolves neither module-level constants nor closures inside a compiled method,
# so every value above reaches the graph as a function argument or a module attribute set in
# __init__. The names here stay the single place they are defined.


def _gaussian_1d(length: int, sigma_scale: float, device: torch.device) -> Tensor:
    """MONAI's per-axis gaussian window: exp(-x^2 / 2 sigma^2), sigma = 0.125 * length."""
    sigma = float(length) * sigma_scale
    x = torch.arange(length, dtype=torch.float32, device=device) - (float(length) - 1.0) / 2.0
    return torch.exp(x * x / (-2.0 * sigma * sigma))


def _importance_map(
    h: int, w: int, d: int, sigma_scale: float, dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Separable 3D gaussian blending weights, floored the way MONAI floors them.

    The floor matters: where the in-plane gaussian has decayed to nothing (the window spans
    the full image, so the corners are far from its centre) it is the clamp, not the
    gaussian, that sets the weight -- and the clamped region blends by plain averaging.
    """
    imp = (
        _gaussian_1d(h, sigma_scale, device).reshape(-1, 1, 1)
        * _gaussian_1d(w, sigma_scale, device).reshape(1, -1, 1)
        * _gaussian_1d(d, sigma_scale, device).reshape(1, 1, -1)
    )
    floor = float(imp.min().item())
    if floor < 1e-3:
        floor = 1e-3
    return imp.clamp_min(floor).to(dtype)


def _window_starts(length: int, window: int, overlap: float) -> List[int]:
    """nnU-Net-style window origins: as few windows as the overlap allows, evenly spread.

    Not a fixed stride -- the step is stretched so the last window ends exactly on the last
    frame, which is why a 23-frame sequence tiles without a ragged final window.
    """
    if length <= window:
        return [0]
    target_step = float(window) * overlap
    num = int(math.ceil((float(length) - float(window)) / target_step)) + 1
    if num <= 1:
        return [0]
    step = (float(length) - float(window)) / float(num - 1)
    starts = torch.jit.annotate(List[int], [])
    for i in range(num):
        starts.append(int(round(step * float(i))))
    return starts


class WindowedNet(nn.Module):
    """One net plus the border pad and gaussian-blended temporal sliding window around it.

    A module rather than a function because TorchScript cannot pass a submodule as a value:
    the window loop has to live next to the net it calls.
    """

    def __init__(self, net: nn.Module, out_channels: int) -> None:
        super().__init__()
        self.net = net
        self.out_channels = out_channels
        self.window = _WINDOW_FRAMES
        self.border_pad = _BORDER_PAD_FRAMES
        self.overlap = _OVERLAP
        self.sigma_scale = _SIGMA_SCALE

    def forward(self, x: Tensor, view_id: Tensor) -> Tensor:
        n_frames = x.shape[4]
        pad = self.border_pad if n_frames > self.border_pad else n_frames - 1
        if pad > 0:
            x = F.pad(x, (pad, pad, 0, 0, 0, 0), mode="reflect")
        y = self._slide(x, view_id)
        if pad > 0:
            y = y[:, :, :, :, pad:y.shape[4] - pad]
        return y

    def _slide(self, x: Tensor, view_id: Tensor) -> Tensor:
        window = self.window
        depth = x.shape[4]
        # A sequence too short to hold one window: zero-extend, then drop the extension.
        # The datamodule refuses anything under 4 frames, so this only guards a caller
        # feeding the module directly.
        tail = 0
        if depth < window:
            tail = window - depth
            x = F.pad(x, (0, tail, 0, 0, 0, 0))
            depth = window

        h = x.shape[2]
        w = x.shape[3]
        imp = _importance_map(h, w, window, self.sigma_scale, x.dtype, x.device)
        out = torch.zeros(
            [x.shape[0], self.out_channels, h, w, depth], dtype=x.dtype, device=x.device
        )
        weight = torch.zeros([1, 1, h, w, depth], dtype=x.dtype, device=x.device)

        for start in _window_starts(depth, window, self.overlap):
            patch = x[:, :, :, :, start:start + window]
            pred = self.net(patch, view_id.expand(patch.shape[0]))
            out[:, :, :, :, start:start + window] += pred * imp
            weight[:, :, :, :, start:start + window] += imp

        y = out / weight
        if tail > 0:
            y = y[:, :, :, :, :depth - tail]
        return y


class _PackageBase(nn.Module):
    """Shared machinery: view lookup, in-plane padding, and the segmentation passes."""

    def __init__(self, segmenter: WindowedNet, views: Dict[str, int]) -> None:
        super().__init__()
        self.segmenter = segmenter
        self.views = views
        self.divisible_by = _DIVISIBLE_BY
        self.tta_contrast = _TTA_CONTRAST
        self.tta_translations = _TTA_TRANSLATIONS
        self.tta_rotations = _TTA_ROTATIONS

    @torch.jit.export
    def view_names(self) -> List[str]:
        """Accepted view strings, so a caller can validate before running a whole study."""
        names = torch.jit.annotate(List[str], [])
        for name in sorted(self.views.keys()):
            names.append(name)
        return names

    def _view_id(self, view: str, device: torch.device) -> Tensor:
        key = view.lower()
        if key not in self.views:
            raise ValueError("unknown view '" + view + "'; call view_names() for the valid set")
        return torch.tensor([self.views[key]], dtype=torch.long, device=device)

    def _pad_in_plane(self, image: Tensor) -> Tuple[Tensor, int, int]:
        """Zero-pad H and W up to a multiple of 32, splitting as torchio's CropOrPad does.

        CropOrPad puts the larger half of an odd pad BEFORE the image; matching that keeps
        the packaged model aligned with predict_3d.py rather than off by one voxel.
        """
        h = image.shape[2]
        w = image.shape[3]
        block = self.divisible_by
        target_h = int(math.ceil(float(h) / float(block))) * block
        target_w = int(math.ceil(float(w) / float(block))) * block
        diff_h = target_h - h
        diff_w = target_w - w
        if diff_h == 0 and diff_w == 0:
            return image, 0, 0
        before_h = diff_h - diff_h // 2
        before_w = diff_w - diff_w // 2
        padded = F.pad(
            image, (0, 0, before_w, diff_w - before_w, before_h, diff_h - before_h)
        )
        return padded, before_h, before_w

    def _probabilities(self, image: Tensor, view_id: Tensor) -> Tensor:
        return F.softmax(self.segmenter(image, view_id), dim=1)

    def _tta_probabilities(self, image: Tensor, view_id: Tensor) -> Tensor:
        """Mean of 25 augmented passes: identity, 4 contrasts, 16 shifts, 4 rotations.

        Mirrors RLmodule_3D.tta_predict pass for pass, including that the shifted and
        rotated predictions are mapped back before averaging.
        """
        preds = self._probabilities(image, view_id)

        for factor in self.tta_contrast:
            adjusted = adjust_contrast(image.permute((4, 0, 1, 2, 3)), factor)
            preds = preds + self._probabilities(adjusted.permute((1, 2, 3, 4, 0)), view_id)

        for shift in self.tta_translations:
            preds = preds + _shift_x_right(
                self._probabilities(_shift_x_left(image, shift), view_id), shift)
            preds = preds + _shift_x_left(
                self._probabilities(_shift_x_right(image, shift), view_id), shift)
            preds = preds + _shift_y_down(
                self._probabilities(_shift_y_up(image, shift), view_id), shift)
            preds = preds + _shift_y_up(
                self._probabilities(_shift_y_down(image, shift), view_id), shift)

        n_frames = image.shape[4]
        for angle in self.tta_rotations:
            rotated = torch.zeros_like(image)
            for i in range(n_frames):
                rotated[0, :, :, :, i] = rotate(image[0, :, :, :, i], angle)
            rot_pred = self._probabilities(rotated, view_id)
            for i in range(n_frames):
                rot_pred[0, :, :, :, i] = rotate(rot_pred[0, :, :, :, i], -angle)
            preds = preds + rot_pred

        total = float(
            len(self.tta_contrast) + len(self.tta_translations) * 4 + len(self.tta_rotations) + 1
        )
        return preds / total

    def _segment(self, image: Tensor, view_id: Tensor, tta: bool) -> Tensor:
        if tta:
            probs = self._tta_probabilities(image, view_id)
        else:
            probs = self._probabilities(image, view_id)
        return probs.argmax(dim=1)


def _shift_x_left(img: Tensor, amount: int) -> Tensor:
    return F.pad(img, (0, 0, 0, 0, amount, 0))[:, :, :img.shape[2], :, :]


def _shift_x_right(img: Tensor, amount: int) -> Tensor:
    return F.pad(img, (0, 0, 0, 0, 0, amount))[:, :, amount:, :, :]


def _shift_y_up(img: Tensor, amount: int) -> Tensor:
    return F.pad(img, (0, 0, amount, 0, 0, 0))[:, :, :, :img.shape[3], :]


def _shift_y_down(img: Tensor, amount: int) -> Tensor:
    return F.pad(img, (0, 0, 0, amount, 0, 0))[:, :, :, amount:, :]


class SegmentationPackage(_PackageBase):
    """Segmentation only. ``forward(image, view, tta) -> labels (H, W, T)``."""

    def forward(self, image: Tensor, view: str, tta: bool = True) -> Tensor:
        view_id = self._view_id(view, image.device)
        padded, off_h, off_w = self._pad_in_plane(image)
        seg = self._segment(padded, view_id, tta)
        h = image.shape[2]
        w = image.shape[3]
        return seg[0, off_h:off_h + h, off_w:off_w + w, :]


class FullPackage(_PackageBase):
    """Segmentation and reward maps.

    ``forward(image, view, tta) -> (labels, fused_reward, per_net_rewards)`` with shapes
    ``(H, W, T)``, ``(H, W, T)`` and ``(N, H, W, T)`` -- the return contract the shipped
    packages and ``torchscript_predict_3d.py`` already use, with the view string added.
    ``reward_map_names()`` gives the order of the N maps.
    """

    def __init__(
        self,
        segmenter: WindowedNet,
        reward_nets: List[WindowedNet],
        reward_names: List[str],
        views: Dict[str, int],
    ) -> None:
        super().__init__(segmenter, views)
        self.rewards = nn.ModuleList(reward_nets)
        self.reward_names = reward_names

    @torch.jit.export
    def reward_map_names(self) -> List[str]:
        """Names of the reward maps, in the order of the returned stack's first axis."""
        return self.reward_names

    def forward(self, image: Tensor, view: str, tta: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
        view_id = self._view_id(view, image.device)
        padded, off_h, off_w = self._pad_in_plane(image)

        seg = self._segment(padded, view_id, tta)
        # The reward nets score the segmentation against the image it came from, as a
        # 2-channel volume: raw labels, not one-hot.
        stack = torch.stack((padded.squeeze(1), seg.to(padded.dtype)), dim=1)

        maps = torch.jit.annotate(List[Tensor], [])
        for net in self.rewards:
            maps.append(torch.sigmoid(net(stack, view_id)).squeeze(1))

        # A voxel scores high only where every net agrees it is good.
        merged = maps[0]
        for i in range(1, len(maps)):
            merged = torch.minimum(merged, maps[i])
        stacked = torch.stack(maps, dim=0)

        h = image.shape[2]
        w = image.shape[3]
        return (
            seg[0, off_h:off_h + h, off_w:off_w + w, :],
            merged[0, off_h:off_h + h, off_w:off_w + w, :],
            stacked[:, 0, off_h:off_h + h, off_w:off_w + w, :],
        )
