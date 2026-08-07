"""Logit-space losses for the reward net, selectable from config for A/B comparison.

Why these exist
---------------
``Reward3DOptimizer`` historically applied ``nn.MSELoss`` to the **post-sigmoid** output.
For a voxel that really is an error (target 0), the gradient w.r.t. the logit is then
``2(p - y) * p(1 - p)``, whose ``p(1 - p)`` factor collapses exactly where the net
confidently endorses a real mistake::

    reward p    MSE-after-sigmoid    BCE-with-logits    ratio
    0.9                   0.162              0.900     0.18
    0.99                  0.0196             0.990     0.020
    0.999                 0.0020             0.999     0.002

So the more wrong the net is, the less it learns -- measured on the 16k reward nets, half
of all gross errors sit above 0.9, i.e. in the region where MSE supplies almost no signal.
BCE-with-logits has its *largest* gradient there, which is the point of switching.

Class convention (important)
----------------------------
The target is ``y = (gt == pred)``: **1 = the candidate agrees with the reference**
(~96% of voxels), **0 = error**. The minority class we care about is therefore y=0, which
inverts the usual convention -- which is why these classes expose ``error_weight`` on the
y=0 term instead of torch's ``pos_weight`` (that would up-weight the already-dominant
agreement class).

Contract
--------
Every loss here consumes RAW LOGITS and sets ``expects_logits = True``;
``Reward3DOptimizer`` reads that flag and skips the sigmoid. ``nn.MSELoss`` and
``apex_loss.ApexWeightedMSELoss`` consume post-sigmoid probabilities and don't set it, so
both paths coexist and can be swapped from config alone.

Calibration note: ``BCEWithLogits`` is a proper scoring rule, so its probabilities stay
calibrated -- relevant because the reward magnitude feeds PPO's advantage directly.
``FocalBCEWithLogits`` is *not* proper and will distort calibration; pair it with the
module's ``do_temp_scale`` recalibration if you use it.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _error_weights(target: torch.Tensor, error_weight: float) -> torch.Tensor:
    """Per-voxel weight: ``error_weight`` on error voxels (target 0), 1.0 on agreement."""
    return torch.where(target > 0.5, torch.ones_like(target),
                       torch.full_like(target, error_weight))


class BCEWithLogits(nn.Module):
    """Binary cross-entropy on logits, optionally up-weighting error voxels.

    Args:
        error_weight: multiplier on the loss of error voxels (target 0). 1.0 is plain BCE;
            raising it counteracts the ~96%/4% agreement/error imbalance. Note this is
            *not* torch's ``pos_weight``, which would weight the majority class here.
    """

    expects_logits = True

    def __init__(self, error_weight: float = 1.0):
        super().__init__()
        self.error_weight = float(error_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if self.error_weight != 1.0:
            bce = bce * _error_weights(target, self.error_weight)
        return bce.mean()


class FocalBCEWithLogits(nn.Module):
    """Focal BCE on logits: down-weights easy voxels by ``(1 - p_t) ** gamma``.

    Attacks the same imbalance as ``error_weight`` but without needing to pick a class --
    the modulation keys off how *hard* each voxel is, which is where the reward net's
    remaining errors live. Not a proper scoring rule (see module docstring).

    Args:
        gamma: focusing strength; 0 reduces to BCE, 2.0 is the usual default.
        error_weight: additional fixed multiplier on error voxels, applied on top.
    """

    expects_logits = True

    def __init__(self, gamma: float = 2.0, error_weight: float = 1.0):
        super().__init__()
        self.gamma = float(gamma)
        self.error_weight = float(error_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)      # prob assigned to the truth
        loss = ((1.0 - p_t) ** self.gamma) * bce
        if self.error_weight != 1.0:
            loss = loss * _error_weights(target, self.error_weight)
        return loss.mean()


class RegionWeightedBCEWithLogits(nn.Module):
    """Logit-space counterpart of ``apex_loss.ApexWeightedMSELoss``.

    Splits each sample into the error neighborhood (the error region dilated by ``margin``)
    and the background, takes the *mean* BCE within each, and combines them with fixed
    weights -- so a tiny error region contributes comparably to a huge correct background
    no matter how few error voxels there are. Same region logic as the apex loss, but
    without the vanishing-gradient problem of MSE-after-sigmoid.

    Args:
        margin: dilation radius (voxels) around the error region.
        fg_weight: weight on the mean BCE inside the error neighborhood.
        bg_weight: weight on the mean BCE over the background (keeps the reward calibrated
            away from errors; set 0 to train on the neighborhood only).
        err_thresh: target below this counts as an error voxel.
        eps: numerical floor for empty regions.
    """

    expects_logits = True

    def __init__(self, margin: int = 6, fg_weight: float = 1.0, bg_weight: float = 0.25,
                 err_thresh: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.margin = int(margin)
        self.fg_weight = float(fg_weight)
        self.bg_weight = float(bg_weight)
        self.err_thresh = float(err_thresh)
        self.eps = float(eps)

    def _neighborhood(self, err: torch.Tensor) -> torch.Tensor:
        """Dilate the error mask by `margin` voxels via a max-pool (3-D over H, W, T)."""
        if self.margin <= 0:
            return err
        k = 2 * self.margin + 1
        return (F.max_pool3d(err, kernel_size=k, stride=1, padding=self.margin) > 0.5).float()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = (target < self.err_thresh).float()
        neigh = self._neighborhood(err)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        dims = tuple(range(1, target.dim()))                  # all non-batch dims
        fg = (bce * neigh).sum(dims) / (neigh.sum(dims) + self.eps)
        bg = (bce * (1.0 - neigh)).sum(dims) / ((1.0 - neigh).sum(dims) + self.eps)
        return (self.fg_weight * fg + self.bg_weight * bg).mean()
