import torch
from torch import nn, Tensor

from patchless_nnunet.models.components.unet import UNet
from patchless_nnunet.models.components.unet_related.layers import (
    get_conv,
    get_transp_conv,
    UpsampleBlock,
)


class AdaIN(nn.Module):
    """Instance normalisation followed by conditioning-predicted scale and shift.

    Applied as:  x  →  InstanceNorm(x)  →  (1 + γ(cond)) * x  +  β(cond)

    Because the shift is injected AFTER normalisation it is never cancelled by
    the norm — unlike plain additive conditioning inserted between blocks.

    The projector is zero-initialised so γ=0, β=0 at the start of training and
    the module is identical to a regular InstanceNorm with learned affine.
    """

    def __init__(self, num_features: int, cond_dim: int, dim: int = 3) -> None:
        super().__init__()
        norm_cls = nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d
        self.norm = norm_cls(num_features, affine=True)
        self.proj = nn.Linear(cond_dim, 2 * num_features)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, cond: Tensor = None) -> Tensor:
        x = self.norm(x)
        if cond is None:
            return x
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        B = x.shape[0]
        spatial = (1,) * (x.dim() - 2)
        return x * (1 + gamma.view(B, -1, *spatial)) + beta.view(B, -1, *spatial)


class ConditionedConvLayer(nn.Module):
    """Conv → AdaIN → LeakyReLU  (conditioned drop-in for ConvLayer)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride,
                 cond_dim: int, **kwargs) -> None:
        super().__init__()
        dim = kwargs["dim"]
        self.conv = get_conv(in_channels, out_channels, kernel_size, stride, dim)
        self.norm = AdaIN(out_channels, cond_dim, dim)
        self.lrelu = nn.LeakyReLU(kwargs["negative_slope"], inplace=True)

    def forward(self, x: Tensor, cond: Tensor = None) -> Tensor:
        return self.lrelu(self.norm(self.conv(x), cond))


class ConditionedConvBlock(nn.Module):
    """Two ConditionedConvLayers  (conditioned drop-in for ConvBlock)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride,
                 cond_dim: int, **kwargs) -> None:
        super().__init__()
        self.conv1 = ConditionedConvLayer(
            in_channels, out_channels, kernel_size, stride, cond_dim, **kwargs
        )
        self.conv2 = ConditionedConvLayer(
            out_channels, out_channels, kernel_size, 1, cond_dim, **kwargs
        )

    def forward(self, x: Tensor, cond: Tensor = None) -> Tensor:
        return self.conv2(self.conv1(x, cond), cond)


class ConditionedUpsampleBlock(nn.Module):
    """TransposedConv + cat(skip) + ConditionedConvBlock  (conditioned UpsampleBlock)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride,
                 cond_dim: int, **kwargs) -> None:
        super().__init__()
        dim = kwargs["dim"]
        self.transp_conv = get_transp_conv(in_channels, out_channels, stride, stride, dim)
        self.conv_block = ConditionedConvBlock(
            2 * out_channels, out_channels, kernel_size, 1, cond_dim, **kwargs
        )

    def forward(self, x: Tensor, skip: Tensor, cond: Tensor = None) -> Tensor:
        out = self.transp_conv(x)
        out = torch.cat((out, skip), dim=1)
        return self.conv_block(out, cond)


class ConditionedUNetAdaIN(UNet):
    """UNet where every InstanceNorm is replaced by AdaIN conditioned on the view label.

    Overrides get_conv_block so all encoder, bottleneck, and decoder blocks use
    AdaIN internally. The patchless_nnunet submodule is never modified.

    At initialisation the conditioning projectors are all zeros, so the network
    is identical to the baseline UNet and can be warm-started from any checkpoint.
    """

    def __init__(self, cond_dim: int, **kwargs) -> None:
        # Must be set before super().__init__() because get_conv_block is called during it.
        self._cond_dim = cond_dim
        super().__init__(**kwargs)

    def get_conv_block(self, conv_block, in_channels: int, out_channels: int,
                       kernel_size, stride, drop_block: bool = False) -> nn.Module:
        shared = dict(
            dim=self.dim,
            norm=self.norm,
            negative_slope=self.negative_slope,
            attention=self.attention,
            cond_dim=self._cond_dim,
        )
        if conv_block is UpsampleBlock:
            return ConditionedUpsampleBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                **shared,
            )
        return ConditionedConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            drop_block=drop_block,
            **shared,
        )

    def forward(self, input_data: Tensor, cond: Tensor = None) -> Tensor:  # noqa: D102
        out = self.input_block(input_data, cond)

        encoder_outputs = [out]
        for downsample in self.downsamples:
            out = downsample(out, cond)
            encoder_outputs.append(out)

        out = self.bottleneck(out, cond)

        num_encoders = len(encoder_outputs)
        for idx, upsample in enumerate(self.upsamples):
            skip = encoder_outputs[num_encoders - 1 - idx]
            out = upsample(out, skip, cond)

        return self.output_block(out)
