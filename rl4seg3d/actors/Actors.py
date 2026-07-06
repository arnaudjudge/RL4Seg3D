import copy

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from vital.models.segmentation.unet import UNet


class UnetActorBinary(nn.Module):

    def __init__(self, input_shape=(1, 256, 256), output_shape=(1, 256, 256), pretrain_ckpt=None, ref_ckpt=None):
        super().__init__()
        if output_shape[0] > 1:
            raise Exception("Wrong input shape, you are using binary actor with multi-class output shape")

        self.net = UNet(input_shape=input_shape, output_shape=output_shape)
        self.old_net = UNet(input_shape=input_shape, output_shape=output_shape)

        if pretrain_ckpt:
            # if starting from pretrained model, keep version of
            self.net.load_state_dict(torch.load(pretrain_ckpt))

            if ref_ckpt:
                # copy to have version of initial pretrained net
                self.old_net.load_state_dict(torch.load(ref_ckpt))
                # will never be updated
                self.old_net.requires_grad_(False)

    def forward(self, x, cond=None):
        # cond is unused for the 2D Bernoulli actor (UNet has no cond_projectors),
        # accepted only so the call signature matches the conditioned 3D actor.
        logits = torch.sigmoid(self.net(x)).squeeze(1)
        dist = Bernoulli(probs=logits)

        if hasattr(self, "old_net"):
            old_logits = torch.sigmoid(self.old_net(x)).squeeze(1)
            old_dist = Bernoulli(probs=old_logits)
        else:
            old_dist = None
        return logits, dist, old_dist


class UnetActorCategorical(nn.Module):
    def __init__(self, input_shape=(1, 256, 256), output_shape=(3, 256, 256), pretrain_ckpt=None, ref_ckpt=None):
        super().__init__()
        self.net = UNet(input_shape=input_shape, output_shape=output_shape)
        self.old_net = UNet(input_shape=input_shape, output_shape=output_shape)

        if pretrain_ckpt:
            # if starting from pretrained model, keep version of
            self.net.load_state_dict(torch.load(pretrain_ckpt))

            if ref_ckpt:
                # copy to have version of initial pretrained net
                self.old_net.load_state_dict(torch.load(ref_ckpt))
                # will never be updated
                self.old_net.requires_grad_(False)

    def forward(self, x, cond=None):
        # cond is unused for the 2D categorical actor (UNet has no cond_projectors).
        logits = torch.softmax(self.net(x), dim=1)
        dist = Categorical(probs=logits.permute(0, 2, 3, 1))

        if hasattr(self, "old_net"):
            old_logits = torch.softmax(self.old_net(x), dim=1)
            old_dist = Categorical(probs=old_logits.permute(0, 2, 3, 1))
        else:
            old_dist = None
        return logits, dist, old_dist


class UnetCritic(nn.Module):

    def __init__(self, input_shape=(1, 256, 256), output_shape=(1, 256, 256), pretrain_ckpt=None):
        super().__init__()
        self.net = UNet(input_shape=input_shape, output_shape=output_shape)

        if pretrain_ckpt:
            self.net.load_state_dict(torch.load(pretrain_ckpt))

    def forward(self, x, cond=None):
        # cond is unused for the 2D UnetCritic; accepted for signature parity.
        return torch.sigmoid(self.net(x))


class Actor(nn.Module):
    """
        Simple policy gradient actor
    """
    def __init__(self,
                 actor,
                 critic,
                 eps_greedy_term=0.0,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 proj_lr_mult=0.1):
        super().__init__()

        self.actor = actor
        self.critic = critic if issubclass(critic.__class__, nn.Module) else None

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        # View-conditioning projection layers (FiLM/AdaIN `.proj.`, additive `*cond_projector*`,
        # FiLM `view_embed`) are freshly (zero-)initialised on top of a pretrained net, so they
        # train at a much lower LR to avoid destabilising the pretrained weights.
        self.proj_lr_mult = proj_lr_mult

        self.eps_greedy_term = eps_greedy_term

    @staticmethod
    def _is_proj_param(name):
        return ("view_embed" in name) or ("proj" in name)

    def _net_param_groups(self, net, base_lr):
        """Split a net's params into base and view-conditioning-projection groups.

        The projection group gets ``base_lr * proj_lr_mult``. Falls back to a single
        group when the net has no projection params (plain UNet).
        """
        proj, base = [], []
        for name, p in net.named_parameters():
            (proj if self._is_proj_param(name) else base).append(p)
        if not proj:
            return [{"params": base, "lr": base_lr}]
        return [
            {"params": base, "lr": base_lr},
            {"params": proj, "lr": base_lr * self.proj_lr_mult},
        ]

    def get_optimizers(self):
        actor_opt = torch.optim.Adam(self._net_param_groups(self.actor.net, self.actor_lr))
        if self.critic is None:
            return actor_opt
        critic_opt = torch.optim.Adam(self._net_param_groups(self.critic.net, self.critic_lr))
        return actor_opt, critic_opt

    def act(self, imgs, sample=True, cond=None):
        """
            Get actions from actor based on batch of images
        Args:
            imgs: batch of images
            sample: bool, use sample from distribution or deterministic method
            cond: optional view conditioning tensor (B, cond_dim), ignored when actor net isn't conditioned

        Returns:
            Actions
        """
        logits, distribution, _ = self.actor(imgs, cond)

        if sample:
            actions = distribution.sample()

            if logits.shape != actions.shape:
                logits = torch.argmax(logits, dim=1)
            random = torch.rand(logits.shape).to(actions.device)
            actions = torch.where(random >= self.eps_greedy_term, actions, torch.round(logits))
        else:
            if len(logits.shape) > 3:
                # categorical, softmax output
                actions = torch.argmax(logits, dim=1)
            else:
                # bernoulli, sigmoid output
                actions = torch.round(logits)

        return actions

    def evaluate(self, imgs, actions, cond=None):
        """
            Evaluate images with both actor and critic
            In this default case, the critic is null, therefore is not considered
        Args:
            imgs: (state) images to evaluate
            actions: segmentation taken over images
            cond: optional view conditioning tensor (B, cond_dim)

        Returns:
            actions (sampled), logits from actor predictions, log_probs,
            entropy placeholder, placeholder value function estimate from critic
        """
        logits, distribution, old_distribution = self.actor(imgs, cond)
        log_probs = distribution.log_prob(actions)

        if old_distribution:
            old_log_probs = old_distribution.log_prob(actions).detach()
        else:
            old_log_probs = log_probs.detach()

        actions = distribution.sample()

        return actions, logits, log_probs, torch.zeros(len(actions)), torch.zeros(len(actions)), old_log_probs
