import copy

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _net_supports_cond(net):
    """True iff ``net`` accepts a view conditioning tensor in its forward().

    Covers both conditioning variants: FiLMUNet (``view_embed``, expects integer
    view indices) and the additive/AdaIN ConditionedUNet (``cond_projectors``,
    expects a one-hot float tensor). Plain UNet checkpoints have neither.
    """
    return hasattr(net, "cond_projectors") or hasattr(net, "view_embed")


def _forward_with_optional_cond(net, x, cond):
    """Call net(x, cond) iff net supports view conditioning, else net(x).

    Keeps backward compatibility with plain UNet checkpoints (no cond_projectors/
    view_embed) while routing the view tensor through a ConditionedUNet/FiLMUNet.
    """
    if cond is not None and _net_supports_cond(net):
        return net(x, cond)
    return net(x)


def _zero_missing_cond_projectors(net, missing_keys):
    """Zero out any cond_projector params that were absent from a loaded state_dict.

    Additive ConditionedUNet projectors are random-initialised by default, so a plain-UNet
    checkpoint loaded with strict=False would add noise to the output. Zeroing the missing
    projectors makes the conditioned net behave identically to the original plain net until
    the projectors are trained.
    """
    if not missing_keys:
        return
    projector_names = {k.rsplit(".", 1)[0] for k in missing_keys if "cond_projector" in k}
    for name in projector_names:
        mod = net
        for part in name.split("."):
            mod = getattr(mod, part, None)
            if mod is None:
                break
        if mod is None:
            continue
        for p in mod.parameters(recurse=False):
            torch.nn.init.zeros_(p)


class Unet3DActorCategorical(nn.Module):
    def __init__(self, net, pretrain_ckpt=None, ref_ckpt=None):
        super().__init__()
        self.net = net
        self.old_net = copy.deepcopy(self.net)

        if pretrain_ckpt:
            # if starting from pretrained model, keep version of
            missing, _ = self.net.load_state_dict(torch.load(pretrain_ckpt), strict=False)
            _zero_missing_cond_projectors(self.net, missing)

            if ref_ckpt:
                # copy to have version of initial pretrained net
                missing_ref, _ = self.old_net.load_state_dict(torch.load(ref_ckpt), strict=False)
                _zero_missing_cond_projectors(self.old_net, missing_ref)
                # will never be updated
                self.old_net.requires_grad_(False)

    def forward(self, x, cond=None):
        with torch.cuda.amp.autocast(enabled=False):  # force float32
            logits = torch.softmax(_forward_with_optional_cond(self.net, x, cond).float(), dim=1)
        dist = Categorical(probs=logits.permute(0, 2, 3, 4, 1))

        if hasattr(self, "old_net"):
            old_logits = torch.softmax(_forward_with_optional_cond(self.old_net, x, cond), dim=1)
            old_dist = Categorical(probs=old_logits.permute(0, 2, 3, 4, 1))
        else:
            old_dist = None
        return logits, dist, old_dist


class Unet3DCritic(nn.Module):

    def __init__(self, net, pretrain_ckpt=None):
        super().__init__()
        self.net = net

        if pretrain_ckpt:
            missing, _ = self.net.load_state_dict(torch.load(pretrain_ckpt), strict=False)
            _zero_missing_cond_projectors(self.net, missing)

    def forward(self, x, cond=None):
        y = _forward_with_optional_cond(self.net, x, cond)
        if self.net.deep_supervision and self.net.training:
           return [torch.sigmoid(y_) for y_ in y]
        return torch.sigmoid(y)
