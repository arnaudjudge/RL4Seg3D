from typing import Tuple, List

import hydra
import torch
from lightning import LightningModule
from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.nn.functional import softmax
from torchvision.transforms.functional import adjust_contrast, rotate
from torch.nn.functional import pad


OmegaConf.register_new_resolver(
    "get_class_name", lambda name: name.split('.')[-1]
)


class RLReward3DInferenceWrapper(torch.nn.Module):
    def __init__(self, reward_nets):
        super().__init__()
        self.reward_net0 = reward_nets[0] # ANAT
        self.reward_net1 = reward_nets[1] # LM

    # must be implemented with net_id to accommodate torchscript, could this be improved?
    def temporal_sliding_window(
            self,
            x: torch.Tensor,
            net_id: int,
            window_size: int = 4,
            overlap: int = 2,
    ) -> torch.Tensor:
        B, C, H, W, T = x.shape

        # Warmup to get output channels
        dummy_slice = x[..., 0:window_size]
        if net_id == 0:
            out_channels = self.reward_net0(dummy_slice).shape[1]
        else:
            out_channels = self.reward_net1(dummy_slice).shape[1]

        output = torch.zeros(B, out_channels, H, W, T, device=x.device, dtype=x.dtype)
        weight_map = torch.zeros_like(output)

        step = window_size - overlap
        start = 0
        while start < T:
            end = min(start + window_size, T)
            slice_x = x[..., start:end]

            # pad last slice with last frame
            t_len = slice_x.shape[-1]
            if t_len < window_size:
                pad_size = window_size - t_len
                last_pad = slice_x[..., -1:].expand(B, C, H, W, pad_size)
                slice_x = torch.cat((slice_x, last_pad), dim=-1)

            if net_id == 0:
                pred = self.reward_net0(slice_x)
            else:
                pred = self.reward_net1(slice_x)

            valid_len = min(window_size, T - start)
            output[..., start:start + valid_len] += pred[..., :valid_len]
            weight_map[..., start:start + valid_len] += 1.0

            start += step

        output = output / weight_map.clamp(min=1.0)
        return output

    def compute_bounds(self, size: int, pad: int):
        if pad >= 0:  # pad equally both sides
            pre = pad // 2
            post = pad - pre
            return pre, post, 0, size
        else:  # crop equally both sides
            crop = -pad
            pre = crop // 2
            post = crop - pre
            return 0, 0, pre, size - post

    def adjust_to_multiple(self, x, div: Tuple[int, int] = (32, 32)):
        """
        Crop or pad spatial dims (H, W) so they are multiples of div.
        Returns adjusted tensor and pad tensor for undoing later.
        """
        H, W = x.shape[-3:-1]
        target_H = int(round(H / div[0]) * div[0])
        target_W = int(round(W / div[1]) * div[1])

        pad_H = target_H - H
        pad_W = target_W - W

        pad_H0, pad_H1, crop_H0, crop_H1 = self.compute_bounds(H, pad_H)
        pad_W0, pad_W1, crop_W0, crop_W1 = self.compute_bounds(W, pad_W)

        # Crop first
        x = x[..., crop_H0:crop_H1, crop_W0:crop_W1, :]

        # Then pad if needed
        pad = (0, 0, pad_W0, pad_W1, pad_H0, pad_H1)  # Only spatial pads
        x = F.pad(x, pad, mode="replicate")

        return x, torch.tensor(pad)

    def undo_adjust(self, x, pad):
        pad_list: List[int] = pad.tolist()
        pad_T0, pad_T1, pad_W0, pad_W1, pad_H0, pad_H1 = pad_list
        H, W, T = x.shape[-3:]
        return x[..., pad_H0:H - pad_H1, pad_W0:W - pad_W1, :]

    def forward(self, x, y):
        x, pad = self.adjust_to_multiple(x)
        y, pad = self.adjust_to_multiple(y)
        with torch.no_grad():
            r0 = torch.sigmoid(self.temporal_sliding_window(torch.stack((x, y), dim=1), 0))
            r1 = torch.sigmoid(self.temporal_sliding_window(torch.stack((x, y), dim=1), 1))
        r0 = self.undo_adjust(r0, pad).squeeze(0)
        r1 = self.undo_adjust(r1, pad).squeeze(0)
        return r0, r1, torch.minimum(r0, r1).squeeze(0)


if __name__ == "__main__":
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from dotenv import load_dotenv
    load_dotenv()

    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path='config')
    sub_cfg = compose(config_name=f"model/ppo_3d.yaml")
    model: LightningModule = hydra.utils.instantiate(sub_cfg.model)
    model.load_state_dict(
        torch.load('/home/local/USHERBROOKE/juda2901/dev/RL4Seg3D/data/other_ckpt/rl4seg3d_ANAT+LM_policy_rewards_state_dict_only.ckpt'),
        strict=False
    )

    wrapper = RLReward3DInferenceWrapper(model.reward_func.get_nets()).cuda()
    example_img = torch.rand((1, 487, 480, 15), device='cuda') # B, C, H, W, T
    example_seg = torch.rand((1, 487, 480, 15), device='cuda')  # B, C, H, W, T

    print(wrapper(example_img, example_seg)[0].shape)

    script = torch.jit.script(wrapper)
    script = torch.jit.optimize_for_inference(script)
    torch.jit.save(script, "../data/checkpoints/rl4seg3d_REWARD_torchscript.pt")

    print(script(example_img, example_seg)[0].shape)


