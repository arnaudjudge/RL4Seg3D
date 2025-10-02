from typing import Union

import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from rl4seg3d.reward.generic_reward import Reward
from torch import Tensor

from patchless_nnunet.utils.inferers import SlidingWindowInferer
from patchless_nnunet.utils.softmax import softmax_helper
from vital.models.segmentation.unet import UNet

"""
Reward functions must each have pred, img, gt as input parameters
"""


class RewardUnet(Reward):
    def __init__(self, state_dict_path, temp_factor=1):
        self.net = UNet(input_shape=(2, 256, 256), output_shape=(1, 256, 256))
        self.net.load_state_dict(torch.load(state_dict_path))
        self.temp_factor = temp_factor
        if torch.cuda.is_available():
            self.net.cuda()

    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)
        return torch.sigmoid(self.net(stack)/self.temp_factor).squeeze(1)


class RewardUnet3D(Reward):
    def __init__(self, net, state_dict_path, temp_factor=1):
        self.net = net
        self.net.load_state_dict(torch.load(state_dict_path))
        self.net.eval()
        self.temp_factor = temp_factor

    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)
        # return as list for code suiting multireward

        with torch.cuda.amp.autocast(enabled=False):  # force float32
            out = self.net(stack)
        rew = torch.sigmoid(out / self.temp_factor).squeeze(1)

        # normalize
        for i in range(rew.shape[0]):
            for j in range(rew.shape[-1]):
                rew[i, ..., j] = rew[i, ..., j] - rew[i, ..., j].min()
                rew[i, ..., j] = rew[i, ..., j] / rew[i, ..., j].max()
        return [rew]

    @torch.no_grad()
    def predict_full_sequence(self, pred, imgs, gt):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)

        self.patch_size = list([stack.shape[-3], stack.shape[-2], 4])
        self.inferer.roi_size = self.patch_size
        with torch.cuda.amp.autocast(enabled=False):  # force float32
            return [torch.sigmoid(self.predict(stack, apply_softmax=False)).squeeze(1)]

    def prepare_for_full_sequence(self, batch_size=1) -> None:  # noqa: D102
        sw_batch_size = batch_size

        self.inferer = SlidingWindowInferer(
            roi_size=self.net.patch_size,
            sw_batch_size=sw_batch_size,
            overlap=0.5,
            mode='gaussian',
            cache_roi_weight_map=True,
        )

    def predict(
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 2D/3D images with sliding window inference.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            NotImplementedError: If the patch shape is not 2D nor 3D.
            ValueError: If 3D patch is requested to predict 2D images.
        """
        if len(image.shape) == 5:
            if len(self.net.patch_size) == 3:
                # Pad the last dimension to avoid 3D segmentation border artifacts
                pad_len = 6 if image.shape[-1] > 6 else image.shape[-1] - 1
                image = F.pad(image, (pad_len, pad_len, 0, 0, 0, 0), mode="reflect")
                pred = self.predict_3D_3Dconv_tiled(image, apply_softmax)
                # Inverse the padding after prediction
                return pred[..., pad_len:-pad_len]
            else:
                raise ValueError("Check your patch size. You dummy.")
        if len(image.shape) == 4:
            raise ValueError("No 2D images here. You dummy.")

    def predict_3D_3Dconv_tiled(
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 3D image with 3D model.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            ValueError: If image is not 3D.
        """
        if not len(image.shape) == 5:
            raise ValueError("image must be (b, c, w, h, d)")

        if apply_softmax:
            return softmax_helper(self.sliding_window_inference(image))
        else:
            return self.sliding_window_inference(image)

    def sliding_window_inference(
        self, image: Union[Tensor, MetaTensor],
    ) -> Union[Tensor, MetaTensor]:
        """Inference using sliding window.

        Args:
            image: Image to predict.

        Returns:
            Predicted logits.
        """
        return self.inferer(
            inputs=image,
            network=self.net,
        )


class AccuracyMap(Reward):
    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        actions = torch.round(pred)
        assert actions.shape == gt.shape, \
            print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
        simple = (actions == gt).float()
        simple = simple.mean(dim=(1, 2))
        mean_matrix = torch.zeros_like(pred, dtype=torch.float)
        for i in range(len(pred)):
            mean_matrix[i, ...] = simple[i]
        return mean_matrix


class Accuracy(Reward):
    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        actions = torch.round(pred)
        assert actions.shape == gt.shape, \
            print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
        simple = (actions == gt).float()
        return simple.mean(dim=(1, 2, 3), keepdim=True)


class PixelWiseAccuracy(Reward):
    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        actions = torch.round(pred)
        assert actions.shape == gt.shape, \
            print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
        return [(actions == gt).float()]

