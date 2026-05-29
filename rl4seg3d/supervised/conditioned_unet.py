from torch import nn, Tensor

from patchless_nnunet.models.components.unet import UNet

# Valid injection modes:
#   "all"         — every encoder stage, bottleneck, every decoder stage (original behaviour)
#   "bottleneck"  — bottleneck only
#   "encoder"     — encoder stages + bottleneck, no decoder
#   "decoder"     — bottleneck + decoder stages, no encoder
_INJECT_MODES = {"all", "bottleneck", "encoder", "decoder"}


class ConditionedUNet(UNet):
    """UNet with additive view conditioning broadcast-added to spatial activations.

    ``inject`` controls which blocks receive conditioning:
    - ``"all"``        every encoder stage, bottleneck, every decoder stage
    - ``"bottleneck"`` bottleneck only
    - ``"encoder"``    encoder stages + bottleneck
    - ``"decoder"``    bottleneck + decoder stages
    """

    def __init__(self, cond_dim: int, inject: str = "all", **kwargs):
        super().__init__(**kwargs)
        if inject not in _INJECT_MODES:
            raise ValueError(f"inject must be one of {_INJECT_MODES}, got {inject!r}")
        self.inject = inject

        use_enc = inject in {"all", "encoder"}
        use_dec = inject in {"all", "decoder"}

        # Encoder projectors (filters[0] = input_block, filters[1:-1] = downsamples)
        self.cond_projectors = nn.ModuleList([
            nn.Linear(cond_dim, f) if use_enc else None
            for f in self.filters[:-1]
        ])
        # Bottleneck projector (always present)
        self.bottleneck_cond_projector = nn.Linear(cond_dim, self.filters[-1])
        # Decoder projectors (mirrors encoder in reverse, excluding bottleneck)
        self.decoder_cond_projectors = nn.ModuleList([
            nn.Linear(cond_dim, f) if use_dec else None
            for f in reversed(self.filters[:-1])
        ])

    def _add_cond(self, out: Tensor, projector, cond: Tensor) -> Tensor:
        if cond is None or projector is None:
            return out
        B = out.shape[0]
        ones = (1,) * self.dim
        return out + projector(cond).view(B, -1, *ones)

    def forward(self, input_data: Tensor, cond: Tensor = None) -> Tensor:  # noqa: D102
        out = self.input_block(input_data)
        out = self._add_cond(out, self.cond_projectors[0], cond)

        encoder_outputs = [out]
        for i, downsample in enumerate(self.downsamples):
            out = downsample(out)
            out = self._add_cond(out, self.cond_projectors[i + 1], cond)
            encoder_outputs.append(out)

        out = self.bottleneck(out)
        out = self._add_cond(out, self.bottleneck_cond_projector, cond)

        num_encoders = len(encoder_outputs)
        for idx, upsample in enumerate(self.upsamples):
            skip = encoder_outputs[num_encoders - 1 - idx]
            out = upsample(out, skip)
            out = self._add_cond(out, self.decoder_cond_projectors[idx], cond)

        return self.output_block(out)
