"""Error-focused weighted MSE for localized (apex) reward targets.

The apex reward target ``y = (gt == pred)`` is ~1 almost everywhere and dips to 0 only in
the small apex-error region. A plain mean MSE is therefore dominated by the trivially
correct background and barely learns the apex signal -- the same flat-map problem that
makes the landmark reward net noisy under the pipeline's per-slice min-max renorm.

This loss splits each sample into the apex neighborhood (the disagreement region dilated
by a margin) and the background, computes MSE within each, and combines them with fixed
weights. Because each term is a *mean* over its region, the tiny apex region contributes
comparably to the huge background regardless of how few error voxels there are.

Drop-in for ``Reward3DOptimizer``'s ``loss`` (called as ``loss(y_pred, y)`` with both in
[0, 1] and shape ``(B, 1, H, W, T)``); default behavior of the anatomical net is untouched
because that net keeps using ``nn.MSELoss``.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ApexWeightedMSELoss(nn.Module):
    def __init__(self, margin: int = 6, fg_weight: float = 1.0, bg_weight: float = 0.25,
                 err_thresh: float = 0.5, eps: float = 1e-6):
        """
        Args:
            margin: dilation radius (voxels) around the disagreement region defining the
                "apex neighborhood" the loss focuses on.
            fg_weight: weight on the mean MSE inside the apex neighborhood.
            bg_weight: weight on the mean MSE over the background (keeps reward calibrated
                away from the apex; set 0 to train on the apex neighborhood only).
            err_thresh: target below this counts as an error voxel (target is 0/1).
            eps: numerical floor for empty regions.
        """
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

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        err = (y < self.err_thresh).float()
        neigh = self._neighborhood(err)
        se = (y_pred - y) ** 2

        dims = tuple(range(1, y.dim()))                       # all non-batch dims
        fg = (se * neigh).sum(dims) / (neigh.sum(dims) + self.eps)
        bg = (se * (1.0 - neigh)).sum(dims) / ((1.0 - neigh).sum(dims) + self.eps)

        per_sample = self.fg_weight * fg + self.bg_weight * bg
        return per_sample.mean()
