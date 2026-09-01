"""A ``torch.jit.script``-able forward for FiLMUNet, used only at export time.

``FiLMUNet.forward`` dispatches through ``_fwd``, which receives the block as an argument
and picks the call signature with ``isinstance``. TorchScript allows neither: modules
cannot be passed as values, and ``*args`` is rejected outright.

Removing ``_fwd`` from the training class itself would mean giving the non-FiLM blocks a
matching ``forward(x, cond)`` signature -- i.e. wrapping them -- which adds a level to the
module tree and so renames every parameter of an ``inject != "all"`` checkpoint. This
subclass calls the blocks directly instead, which is only correct when every block is a
FiLM block, hence the ``inject="all"`` requirement.

Nothing else changes: the same state_dict loads, and the output is bit-identical to the
parent class (asserted at export time by ``export_torchscript_3d.py``).
"""

from torch import Tensor

from rl4seg3d.supervised.conditioned_unet_film import FiLMUNet


class ScriptableFiLMUNet(FiLMUNet):
    """FiLMUNet whose forward compiles under TorchScript. Requires ``inject="all"``."""

    def __init__(self, **kwargs) -> None:
        if kwargs.get("inject", "all") != "all":
            raise ValueError(
                "ScriptableFiLMUNet requires inject='all': with any other mode some blocks "
                "are plain ConvBlocks that do not accept a cond argument, and the isinstance "
                "dispatch that would pick their signature is exactly what cannot be scripted."
            )
        super().__init__(**kwargs)

    def forward(self, input_data: Tensor, view_id: Tensor | None = None) -> Tensor:
        cond = self.view_embed(view_id) if view_id is not None else None

        out = self.input_block(input_data, cond)
        encoder_outputs = [out]
        for downsample in self.downsamples:
            out = downsample(out, cond)
            encoder_outputs.append(out)

        out = self.bottleneck(out, cond)

        # `enumerate` over a ModuleList is not scriptable; count by hand instead.
        num_enc = len(encoder_outputs)
        idx = 0
        for upsample in self.upsamples:
            out = upsample(out, encoder_outputs[num_enc - 1 - idx], cond)
            idx += 1

        return self.output_block(out)
