import copy

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Unet3DActorCategorical(nn.Module):
    def __init__(self, net, pretrain_ckpt=None, ref_ckpt=None):
        super().__init__()
        self.net = net
        self.old_net = copy.deepcopy(self.net)

        if pretrain_ckpt:
            # if starting from pretrained model, keep version of
            self.net.load_state_dict(torch.load(pretrain_ckpt))

            if ref_ckpt:
                # copy to have version of initial pretrained net
                self.old_net.load_state_dict(torch.load(ref_ckpt))
                # will never be updated
                self.old_net.requires_grad_(False)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):  # force float32
            logits = torch.softmax(self.net(x).float(), dim=1)
        dist = Categorical(probs=logits.permute(0, 2, 3, 4, 1))

        if hasattr(self, "old_net"):
            old_logits = torch.softmax(self.old_net(x), dim=1)
            old_dist = Categorical(probs=old_logits.permute(0, 2, 3, 4, 1))
        else:
            old_dist = None
        return logits, dist, old_dist


class Unet3DCritic(nn.Module):

    def __init__(self, net, pretrain_ckpt=None):
        super().__init__()
        self.net = net

        if pretrain_ckpt:
            self.net.load_state_dict(torch.load(pretrain_ckpt))

    def forward(self, x):
        y = self.net(x)
        if self.net.deep_supervision and self.net.training:
           return [torch.sigmoid(y_) for y_ in y]
        return torch.sigmoid(y)
