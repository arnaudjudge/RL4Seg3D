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


class RLModule3DInferenceWrapper(torch.nn.Module):
    def __init__(self, model, reward_nets):
        super().__init__()
        self.model = model.eval().cuda() # POLICY
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
            out_channels = self.model(dummy_slice).shape[1]
        elif net_id == 1:
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

            # call the correct network directly
            if net_id == 0:
                pred = self.model(slice_x)
            elif net_id == 1:
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

    def x_translate_left(self, img, amount:int = 20):
        return pad(img, (0, 0, 0, 0, amount, 0), mode="constant")[:, :, :-amount, :, :]

    def x_translate_right(self, img, amount:int = 20):
        return pad(img, (0, 0, 0, 0, 0, amount), mode="constant")[:, :, amount:, :, :]

    def y_translate_up(self, img, amount:int = 20):
        return pad(img, (0, 0, amount, 0, 0, 0), mode="constant")[:, :, :, :-amount, :]

    def y_translate_down(self, img, amount:int = 20):
        return pad(img, (0, 0, 0, amount, 0, 0), mode="constant")[:, :, :, amount:, :]

    def tta_predict(self, x):
        preds = softmax(self.temporal_sliding_window(x, net_id=0), dim=1)
        factors = [1.1, 0.9, 1.25, 0.75]
        translations = [40, 60, 80, 120]
        rotations = [5, 10, -5, -10]

        for factor in factors:
            preds += softmax(self.temporal_sliding_window(adjust_contrast(
                x.permute((4, 0, 1, 2, 3)), factor).permute((1, 2, 3, 4, 0)), net_id=0), dim=1)

        for translation in translations:
            preds += self.x_translate_right(softmax(self.temporal_sliding_window(self.x_translate_left(x, translation), net_id=0), dim=1),
                                            translation)
            preds += self.x_translate_left(softmax(self.temporal_sliding_window(self.x_translate_right(x, translation), net_id=0), dim=1),
                                           translation)
            preds += self.y_translate_down(softmax(self.temporal_sliding_window(self.y_translate_up(x, translation), net_id=0), dim=1),
                                           translation)
            preds += self.y_translate_up(softmax(self.temporal_sliding_window(self.y_translate_down(x, translation), net_id=0), dim=1),
                                         translation)

        # TODO: optimize this for compute time
        for rotation in rotations:
            rotated = torch.zeros_like(x)
            for i in range(x.shape[-1]):
                rotated[0, :, :, :, i] = rotate(x[0, :, :, :, i], angle=float(rotation))
            rot_pred = softmax(self.temporal_sliding_window(rotated, net_id=0), dim=1)
            for i in range(x.shape[-1]):
                rot_pred[0, :, :, :, i] = rotate(rot_pred[0, :, :, :, i], angle=-float(rotation))
            preds += rot_pred

        preds /= len(factors) + len(translations) * 4 + len(rotations) + 1
        return preds.argmax(dim=1)

    def forward(self, x, tta:bool = True):
        x, pad = self.adjust_to_multiple(x)
        with torch.no_grad():
            y = self.tta_predict(x) if tta else self.temporal_sliding_window(x, 0).argmax(dim=1)
            r0 = torch.sigmoid(self.temporal_sliding_window(torch.stack((x.squeeze(1), y), dim=1), 1))
            r1 = torch.sigmoid(self.temporal_sliding_window(torch.stack((x.squeeze(1), y), dim=1), 2))
        y = self.undo_adjust(y, pad)
        r0 = self.undo_adjust(r0, pad).squeeze(0)
        r1 = self.undo_adjust(r1, pad).squeeze(0)
        return y.squeeze(0), torch.minimum(r0, r1).squeeze(0), torch.concat((r0, r1), dim=0)


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
        torch.load('../data/checkpoints/rl4seg3d_ANAT+LM_policy_rewards_state_dict_only.ckpt'),
        strict=False
    )

    wrapper = RLModule3DInferenceWrapper(model.actor.actor.net, model.reward_func.get_nets()).cuda()
    example_input = torch.rand((1, 1, 487, 480, 15), device='cuda') # B, C, H, W, T

    print(wrapper(example_input)[0].shape)

    script = torch.jit.script(wrapper)
    script = torch.jit.optimize_for_inference(script)
    torch.jit.save(script, "../data/checkpoints/rl4seg3d_torchscript_TTA.pt")

    print(script(example_input)[0].shape) # With TTA

    print(script(example_input, tta=False)[0].shape) # Without TTA

