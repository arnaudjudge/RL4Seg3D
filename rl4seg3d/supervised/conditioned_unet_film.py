"""FiLM-conditioned UNet with learned view embedding.

Why not additive (conditioned_unet.py):
  A spatially-uniform bias added between blocks is fully cancelled by the next IN layer:
  IN((f + c)) = IN(f) because IN removes the channel mean, which is the only thing a uniform
  bias affects.  The effective conditioning in the additive model comes only from the last
  decoder injection (no subsequent IN to cancel it), and a single unconstrained linear
  projection right before the output head produces the squiggly boundary artefacts.

Why not AdaIN (conditioned_unet_adain.py):
  AdaIN uses IN(affine=True) alongside conditioning-predicted scale/shift, creating two
  competing affine transforms in the same layer.  In practice the static affine dominates
  (it has a non-zero-init head-start) and the conditioning projectors stay near zero.

FiLM fixes both:
  1. IN(affine=False): conditioning projectors are the SOLE affine transform — no competition.
  2. Conditioning injected AFTER IN but BEFORE the nonlinear activation: subsequent IN layers
     cannot fully undo the effect because LeakyReLU(IN(x) + β(cond)) ≠ LeakyReLU(IN(x)) + k
     (the nonlinearity changes which neurons fire, and that pattern persists).
  3. nn.Embedding instead of one-hot: richer per-view representation, same overhead.

Optimizer changes needed in supervised_3d_optimizer.py:
  - _get_cond: return batch['view_as_id'].long().to(self.device)  (shape B, not B×num_views)
  - forward / sliding_window_inference: use hasattr(self.net, 'view_embed') instead of
    hasattr(self.net, 'cond_projectors')
  - training_step expand: cond.expand(x.shape[0]) instead of cond.expand(x.shape[0], -1)
"""

import torch
from torch import nn, Tensor

from patchless_nnunet.models.components.unet import UNet
from patchless_nnunet.models.components.unet_related.layers import (
    get_conv,
    get_transp_conv,
    UpsampleBlock,
)

_INJECT_MODES = {"all", "bottleneck", "encoder", "decoder"}


class _FiLMLayer(nn.Module):
    """conv → IN(affine=False) → (1 + γ(cond)) · x + β(cond) → LeakyReLU.

    Zero-init projector → starts as plain IN at initialisation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        cond_dim: int,
        **kwargs,
    ) -> None:
        super().__init__()
        dim = kwargs["dim"]
        norm_cls = nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d
        self.conv = get_conv(in_channels, out_channels, kernel_size, stride, dim)
        self.norm = norm_cls(out_channels, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * out_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.lrelu = nn.LeakyReLU(kwargs["negative_slope"], inplace=True)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        if cond is not None:
            gamma, beta = self.proj(cond).chunk(2, dim=-1)
            B = x.shape[0]
            s = (1,) * (x.dim() - 2)
            x = x * (1.0 + gamma.view(B, -1, *s)) + beta.view(B, -1, *s)
        return self.lrelu(x)


class _FiLMConvBlock(nn.Module):
    """Two _FiLMLayers — conditioned drop-in for ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        cond_dim: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = _FiLMLayer(in_channels, out_channels, kernel_size, stride, cond_dim, **kwargs)
        self.conv2 = _FiLMLayer(out_channels, out_channels, kernel_size, 1, cond_dim, **kwargs)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        return self.conv2(self.conv1(x, cond), cond)


class _FiLMUpsampleBlock(nn.Module):
    """TransposedConv + cat(skip) + _FiLMConvBlock — conditioned drop-in for UpsampleBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        cond_dim: int,
        **kwargs,
    ) -> None:
        super().__init__()
        dim = kwargs["dim"]
        self.transp_conv = get_transp_conv(in_channels, out_channels, stride, stride, dim)
        self.conv_block = _FiLMConvBlock(
            2 * out_channels, out_channels, kernel_size, 1, cond_dim, **kwargs
        )

    def forward(self, x: Tensor, skip: Tensor, cond: Tensor | None = None) -> Tensor:
        out = self.transp_conv(x)
        out = torch.cat((out, skip), dim=1)
        return self.conv_block(out, cond)


class FiLMUNet(UNet):
    """UNet with FiLM view-conditioning and a learned view embedding.

    ``inject`` controls which stages receive conditioning:
    - ``"all"``        encoder + bottleneck + decoder  (default)
    - ``"bottleneck"`` bottleneck only
    - ``"encoder"``    encoder + bottleneck
    - ``"decoder"``    bottleneck + decoder

    ``forward(x, view_id)`` takes a LongTensor of view indices (shape B,).
    The embedding lives inside the network so the file is self-contained.
    """

    def __init__(
        self,
        num_views: int,
        embed_dim: int = 16,
        inject: str = "all",
        **kwargs,
    ) -> None:
        if inject not in _INJECT_MODES:
            raise ValueError(f"inject must be one of {_INJECT_MODES}, got {inject!r}")
        # Must be set before super().__init__() — get_conv_block is called during it.
        self._embed_dim = embed_dim
        self._inject = inject
        self._build_call = 0
        self._n_stages = len(kwargs["strides"])
        super().__init__(**kwargs)
        # Initialised after super() so initialize_weights (Conv-only) does not touch it.
        self.view_embed = nn.Embedding(num_views, embed_dim)

    # ------------------------------------------------------------------
    # Build-time helpers (called only during __init__ via super())
    # ------------------------------------------------------------------

    def _stage_for_call(self, is_upsample: bool) -> str:
        """Map the current get_conv_block call index to a stage name.

        UNet.__init__ calls get_conv_block in this order:
          1          → input_block          (encoder)
          2 .. n-2   → downsamples          (encoder)
          n-1        → bottleneck
          n .. 2n-2  → upsamples            (decoder)
        where n = len(strides).
        """
        self._build_call += 1
        if is_upsample:
            return "decoder"
        if self._build_call == self._n_stages:
            return "bottleneck"
        return "encoder"

    def _use_film(self, stage: str) -> bool:
        inj = self._inject
        if inj == "all":
            return True
        if inj == "bottleneck":
            return stage == "bottleneck"
        if inj == "encoder":
            return stage in ("encoder", "bottleneck")
        return stage in ("bottleneck", "decoder")  # "decoder"

    def get_conv_block(
        self,
        conv_block,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        drop_block: bool = False,
    ) -> nn.Module:
        is_up = conv_block is UpsampleBlock
        stage = self._stage_for_call(is_up)

        if not self._use_film(stage):
            return super().get_conv_block(
                conv_block, in_channels, out_channels, kernel_size, stride, drop_block
            )

        shared = dict(
            cond_dim=self._embed_dim,
            dim=self.dim,
            norm=self.norm,
            negative_slope=self.negative_slope,
            attention=self.attention,
        )
        if is_up:
            return _FiLMUpsampleBlock(in_channels, out_channels, kernel_size, stride, **shared)
        return _FiLMConvBlock(
            in_channels, out_channels, kernel_size, stride, drop_block=drop_block, **shared
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @staticmethod
    def _fwd(block: nn.Module, *args, cond: Tensor | None = None) -> Tensor:
        """Call block with cond if it is a FiLM block, otherwise without."""
        if isinstance(block, (_FiLMConvBlock, _FiLMUpsampleBlock)):
            return block(*args, cond=cond)
        return block(*args)

    def forward(self, input_data: Tensor, view_id: Tensor | None = None) -> Tensor:
        cond = self.view_embed(view_id) if view_id is not None else None

        out = self._fwd(self.input_block, input_data, cond=cond)
        encoder_outputs = [out]
        for downsample in self.downsamples:
            out = self._fwd(downsample, out, cond=cond)
            encoder_outputs.append(out)

        out = self._fwd(self.bottleneck, out, cond=cond)

        num_enc = len(encoder_outputs)
        for idx, upsample in enumerate(self.upsamples):
            skip = encoder_outputs[num_enc - 1 - idx]
            out = self._fwd(upsample, out, skip, cond=cond)

        return self.output_block(out)
