import copy
import functools
import os
import random
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchio as tio
from einops import rearrange
from lightning import LightningModule
from monai.data import MetaTensor
from scipy import ndimage
from torch import Tensor
from torch.nn.functional import pad
from torchvision.transforms.functional import adjust_contrast, rotate

from patchless_nnunet.utils.inferers import SlidingWindowInferer
from patchless_nnunet.utils.softmax import softmax_helper
from rl4seg3d.utils.Metrics import accuracy, dice_score
from rl4seg3d.utils.correctors import AEMorphoCorrector
from rl4seg3d.utils.file_utils import append_csv_row, reward_stats, save_reward_stack, save_to_reward_dataset
from rl4seg3d.utils.logging_helper import log_sequence, log_video
from rl4seg3d.utils.tensor_utils import convert_to_numpy
from rl4seg3d.utils.test_metrics import full_test_metrics
from rl4seg3d.utils.Metrics import is_anatomically_valid
from rl4seg3d.utils.temporal_metrics import check_temporal_validity
from rl4seg3d.utils.viz_utils import save_reward_gif
from vital.metrics.camus.anatomical.utils import check_segmentation_validity

try:
    # submitit raises this inside the process when the walltime signal fires. It says nothing
    # about the case, so it must not be mistaken for one failing.
    from submitit.core.utils import UncompletedJobError as _JobEndedError
except Exception:  # submitit absent (local runs)
    class _JobEndedError(Exception):
        """Never raised; placeholder so the isinstance check below is always valid."""


# Reward-net subsets to fuse and report alongside the full merge, in the csv written per case.
# Each is scored exactly as `merged` is; a subset whose nets are not all loaded is skipped.
_FUSION_SUBSETS = [("anatomical", "landmarks")]


class RLmodule3D(LightningModule):

    def __init__(self, actor, reward,
                 corrector=None,
                 actor_save_path=None,
                 critic_save_path=None,
                 save_uncertainty_path=None,
                 predict_save_dir=None,
                 predict_do_model_perturb=True,
                 predict_do_img_perturb=True,
                 predict_do_corrections=True,
                 predict_do_temporal_glitches=True,
                 ae_comp_threshold=0.935,
                 save_on_test=True,
                 vae_on_test=False,
                 worst_frame_thresholds=None, #{"anatomical": 0.985},
                 save_csv_after_predict=None,
                 val_batch_size=4,
                 tta=True,
                 tto='off',
                 temp_files_path='.',
                 inference=False,
                 num_views: int = 0,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(logger=False, ignore=["actor", "reward", "corrector"])

        self.actor = actor
        self.reward_func = reward
        if hasattr(self.reward_func, 'net'):
            self.register_module('rewardnet', self.reward_func.net)
        elif hasattr(self.reward_func, 'nets'):
            if isinstance(self.reward_func.nets, list):  # backward compatibility
                for i, n in enumerate(self.reward_func.nets):
                    self.register_module(f'rewardnet_{i}', n)
            elif isinstance(self.reward_func.nets, dict):
                for i, n in enumerate(self.reward_func.nets.values()):
                    self.register_module(f'rewardnet_{i}', n)

        self.pred_corrector = corrector
        if isinstance(self.pred_corrector, AEMorphoCorrector):
            self.register_module('correctorAE', self.pred_corrector.ae_corrector.temporal_regularization.autoencoder)

        self.predicted_rows = []

        self.temp_files_path = Path(temp_files_path)
        if not self.temp_files_path.exists() and self.trainer.global_rank == 0:
            self.temp_files_path.mkdir(parents=True, exist_ok=True)

        # for test time overfitting
        self.initial_test_params = None
        # cond cached for sliding-window inference closures (test/predict)
        self._current_cond = None

    def configure_optimizers(self):
        return self.actor.get_optimizers()

    def _policy_net(self):
        """Best-effort handle on the underlying policy network, for capability checks.

        Used only to decide the conditioning *format* (FiLM indices vs one-hot); returns
        None if the actor doesn't expose a ``.net`` (then we fall back to one-hot).
        """
        return getattr(getattr(self.actor, 'actor', None), 'net', None)

    def _get_cond(self, batch) -> Tensor | None:
        """View conditioning tensor from the batch, or None.

        FiLMUNet (``view_embed``) expects an integer LongTensor of shape (B,); the
        additive/AdaIN ConditionedUNet (``cond_projectors``) expects a one-hot float
        tensor of shape (B, num_views). Each RL sample emits a single ``view_as_id``
        scalar, so the returned tensor has batch dim 1; expand to the per-step batch
        at the call site via ``_expand_cond``.
        """
        if self.hparams.num_views <= 0 or batch is None or 'view_as_id' not in batch:
            return None
        view_id = batch['view_as_id'].long().view(-1).to(self.device)
        net = self._policy_net()
        if net is not None and hasattr(net, 'view_embed'):
            return view_id
        return F.one_hot(view_id, num_classes=self.hparams.num_views).float()

    @torch.no_grad()  # no grad since tensors are reused in PPO's for loop
    def rollout(self, imgs: torch.tensor, gt: torch.tensor, use_gt: torch.tensor = None,
                sample: bool = True, cond: Tensor = None):
        """
            Rollout the policy over a batch of images and ground truth pairs
        Args:
            imgs: batch of images
            gt: batch of ground truth segmentation maps
            use_gt: replace policy result with ground truth (bool mask of len of batch)
            sample: whether to sample actions from distribution or determinist
            cond: optional view conditioning tensor (B, cond_dim)

        Returns:
            Actions (used for rewards, log_pobs, etc), sampled_actions (mainly for display), log_probs, rewards
        """
        actor_cond = self._expand_cond(cond, imgs.shape[0])
        actions = self.actor.act(imgs, sample=sample, cond=actor_cond)
        rewards = self.reward_func(actions, imgs, gt, cond=actor_cond)

        if use_gt is not None:
            actions[use_gt, ...] = gt[use_gt, ...]

        _, _, log_probs, _, _, _ = self.actor.evaluate(imgs, actions, cond=actor_cond)
        return actions, log_probs, rewards

    @staticmethod
    def _expand_cond(cond: Tensor | None, batch_size: int) -> Tensor | None:
        if cond is None:
            return None
        if cond.shape[0] == batch_size:
            return cond
        # FiLM cond is a 1-D LongTensor (B,); additive cond is 2-D (B, num_views).
        if cond.dim() == 1:
            return cond.expand(batch_size)
        return cond.expand(batch_size, -1)

    def training_step(self, batch: dict[str, Tensor], nb_batch):
        """
            Defines training steo, calculates loss based on the minibatch recieved.
            Optionally backprop through networks with self.automatic_optimization = False flag,
            otherwise return loss dict and Lighting does it automatically
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics or None
        """
        raise NotImplementedError

    def compute_policy_loss(self, batch, sample=True, **kwargs):
        """
            Compute unsupervised loss to maximise reward using policy gradient method.
        Args:
            batch: batch of images, actions, log_probs, rewards and groudn truth
            sample: whether to sample from distribution or deterministic approach (mainly for val, test steps)

        Returns:
            mean loss(es) for the batch, metrics dictionary
        """
        raise NotImplementedError

    def ttoptimize(self, batch_image, num_iter=10, **kwargs):
        """
            Run a few iterations of optimization to overfit on one test sequence in unsupervised
        Args:
            batch: batch of images
            num_iter: number of iterations of the optimization loop
        Returns:
            None, actor is modified, ready for inference on batch_image alone
        """
        raise NotImplementedError

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        """
            Defines validation step (using sampling to show model confidence)
            Computes actions from current policy and calculates loss, rewards (and other metrics)
            Logs images and segmentations to logger
        Args:
            batch: batch of images and ground truth
            batch_idx: index of batch

        Returns:
            Dict of logs
        """
        b_imgs, b_gts, b_use_gts = batch['img'].squeeze(0), batch['gt'].squeeze(0), batch['use_gt'].squeeze(0)
        base_cond = self._get_cond(batch)

        logs = {'val/loss': [],
                "val/reward": [],
                "val/acc": [],
                "val/dice": []
                }
        for i in range(0, b_imgs.shape[0], self.hparams.val_batch_size):
            b_img = b_imgs[i:i+self.hparams.val_batch_size]
            b_gt = b_gts[i:i+self.hparams.val_batch_size]
            b_use_gt = b_use_gts[i:i+self.hparams.val_batch_size]
            step_cond = self._expand_cond(base_cond, b_img.shape[0])

            prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt, sample=False, cond=step_cond)
            prev_rewards = torch.mean(torch.stack(prev_rewards, dim=0), dim=0)

            loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions, prev_rewards,
                                                                        prev_log_probs, b_gt, b_use_gt),
                                                                       cond=step_cond)

            acc = accuracy(prev_actions, b_img, b_gt)
            dice = dice_score(prev_actions, b_gt)

            logs["val/loss"] += [loss]
            logs["val/reward"] += [torch.mean(prev_rewards.type(torch.float))]
            logs["val/acc"] += [acc.mean()]
            logs["val/dice"] += [dice.mean()]

            _, _, _, _, v, _ = self.actor.evaluate(b_img, prev_actions, cond=step_cond)
            # log images
            if self.trainer.global_rank == 0:
                idx = random.randint(0, len(b_img) - 1)  # which image to log
                log_sequence(self.logger, img=b_img[idx], title='Image', number=batch_idx, epoch=self.current_epoch)
                log_sequence(self.logger, img=b_gt[idx].unsqueeze(0), title='GroundTruth', number=batch_idx,
                             epoch=self.current_epoch)
                log_sequence(self.logger, img=prev_actions[idx].unsqueeze(0), title='Prediction', number=batch_idx,
                          img_text=prev_rewards[idx].mean(), epoch=self.current_epoch)
                log_sequence(self.logger, img=v[idx].unsqueeze(0), title='VFunction', number=batch_idx,
                             img_text=v[idx].mean(), epoch=self.current_epoch)
                if prev_rewards.shape == prev_actions.shape:
                    log_sequence(self.logger, img=prev_rewards[idx].unsqueeze(0), title='RewardMap', number=batch_idx,
                                 epoch=self.current_epoch)

        logs = {k: torch.tensor(v, device=self.device).mean() for k, v in logs.items()}
        self.log_dict(logs, on_epoch=True, sync_dist=True)
        return logs

    def test_step(self, batch: dict[str, Tensor], batch_idx: int):
        """
            Defines test step (uses deterministic method to show real results)
            Computes actions from current policy and calculates loss, rewards (and other metrics)
            Logs images and segmentations to logger
        Args:
            batch: batch of images and ground truth
            batch_idx: index of batch

        Returns:
            Dict of logs
        """
        b_img, b_gt, meta_dict = batch['img'], batch['gt'], batch['image_meta_dict']
        self._current_cond = self._get_cond(batch)

        # Same reasoning as inference_predict_step: start every case from the checkpoint, never
        # from the previous case's TTO result.
        if self.hparams.tto in ('force', 'on'):
            self._reset_to_initial_params()

        self.patch_size = list([b_img.shape[-3], b_img.shape[-2], 4])
        self.inferer.roi_size = self.patch_size

        start_time = time.time()
        prev_actions = self.tta_predict(b_img) if self.hparams.tta else self.predict(b_img).argmax(dim=1)
        print(f"\nFirst Prediction took {round(time.time() - start_time, 4)} (s).")

        start_time = time.time()
        y_pred_np_as_batch = prev_actions.cpu().numpy().squeeze(0).transpose((2, 0, 1))
        b_gt_np_as_batch = b_gt.cpu().numpy().squeeze(0).transpose((2, 0, 1))

        for i in range(len(y_pred_np_as_batch)):
            lbl, num = ndimage.measurements.label(y_pred_np_as_batch[i] != 0)
            # Count the number of elements per label
            count = np.bincount(lbl.flat)
            # Select the largest blob
            maxi = np.argmax(count[1:]) + 1
            # Remove the other blobs
            y_pred_np_as_batch[i][lbl != maxi] = 0

        # should be still valid to use resampled spacing for metrics here
        voxel_spacing = np.asarray([[abs(meta_dict['resampled_affine'][0, 0, 0].cpu().numpy()),
                                     abs(meta_dict['resampled_affine'][0, 1, 1].cpu().numpy())]]).repeat(
                                    repeats=len(y_pred_np_as_batch), axis=0)
        print(f"Cleaning took {round(time.time() - start_time, 4)} (s).")

        anat_errors = is_anatomically_valid(y_pred_np_as_batch)
        temporal_valid, _ = check_temporal_validity(y_pred_np_as_batch.transpose((0, 2, 1)),
                                                                      voxel_spacing[0])
        validated = int(all(anat_errors)) and temporal_valid
        if self.hparams.tto == 'force' or (not validated and self.hparams.tto == 'on'):
            self._reset_to_initial_params()

            start_time = time.time()
            self.ttoptimize(b_img)
            print(f"\nTTOverfit took {round(time.time() - start_time, 4)} (s).")

            start_time = time.time()
            prev_actions = self.tta_predict(b_img) if self.hparams.tta else self.predict(b_img).argmax(dim=1)
            print(f"\nPost-TTO Prediction took {round(time.time() - start_time, 4)} (s).")

            start_time = time.time()
            y_pred_np_as_batch = prev_actions.cpu().numpy().squeeze(0).transpose((2, 0, 1))
            b_gt_np_as_batch = b_gt.cpu().numpy().squeeze(0).transpose((2, 0, 1))

            for i in range(len(y_pred_np_as_batch)):
                lbl, num = ndimage.measurements.label(y_pred_np_as_batch[i] != 0)
                # Count the number of elements per label
                count = np.bincount(lbl.flat)
                # Select the largest blob
                maxi = np.argmax(count[1:]) + 1
                # Remove the other blobs
                y_pred_np_as_batch[i][lbl != maxi] = 0

            # should be still valid to use resampled spacing for metrics here
            voxel_spacing = np.asarray([[abs(meta_dict['resampled_affine'][0, 0, 0].cpu().numpy()),
                                         abs(meta_dict['resampled_affine'][0, 1, 1].cpu().numpy())]]).repeat(
                                        repeats=len(y_pred_np_as_batch), axis=0)
            print(f"Cleaning took {round(time.time() - start_time, 4)} (s).")

        prev_rewards = torch.stack(self.reward_func.predict_full_sequence(prev_actions, b_img, b_gt,
                                                                          cond=self._current_cond), dim=0)
        prev_rewards_mean = torch.mean(prev_rewards, dim=0)

        logs = full_test_metrics(y_pred_np_as_batch, b_gt_np_as_batch, voxel_spacing, self.device)
        logs.update({"test/reward": torch.mean(prev_rewards_mean.type(torch.float))})

        if self.hparams.vae_on_test:
            start_time = time.time()
            corrected, corrected_validity, ae_comp, _ = self.pred_corrector.correct_single_seq(
                b_img.squeeze(0), prev_actions.squeeze(0), voxel_spacing)
            # actions_unsampled_clean = actions_unsampled_clean[None,]
            corrected = corrected.transpose((2, 0, 1))
            vae_logs = full_test_metrics(corrected, b_gt_np_as_batch, voxel_spacing, self.device, prefix="test_vae", verbose=False)
            vae_logs.update({"test_vae/vae_comp": ae_comp})
            logs.update(vae_logs)
            print(f"VAE took {round(time.time() - start_time, 4)} (s).")

        if self.hparams.worst_frame_thresholds:
            # skip if rewards are too low according to thresholds
            reward_indices = [self.reward_func.get_reward_index(key) for key in self.hparams.worst_frame_thresholds.keys()]
            reward_frame_mins = np.take(prev_rewards.cpu().numpy().mean(axis=(1, 2, 3)).min(axis=1), reward_indices, 0)
            thresholds = np.asarray(list(self.hparams.worst_frame_thresholds.values()))
            validated = (reward_frame_mins > thresholds).all()
            if validated:
                fname = meta_dict.get('case_identifier')[0]
                print(f"{self.trainer.datamodule.get_approx_gt_subpath(fname).rsplit('/', 1)[0]}/{fname} - "
                      f"Min frame reward higher than threshold: "
                      f"{reward_frame_mins} vs  thresh:{thresholds}")
                logs.update({f'{k.replace("test", "test_validated")}': v for k, v in logs.items()})
                self.log("test_validated/count", 1)
            else:
                self.log("test_validated/count", 0)

        start_time = time.time()
        # for logging v
        # Use only first 4 for visualization, avoids having to implement sliding window inference for critic
        _, _, _, _, v, _ = self.actor.evaluate(b_img[..., :4], prev_actions[..., :4],
                                                cond=self._expand_cond(self._current_cond, b_img.shape[0]))

        if self.trainer.global_rank == 0 and batch_idx % 1 == 0:
            log_video(self.logger, img=b_gt, background=b_img.squeeze(0), title='test_GroundTruth', number=batch_idx,
                         epoch=self.current_epoch)
            log_video(self.logger, img=prev_actions, background=b_img.squeeze(0), title='test_Prediction',
                         number=batch_idx, epoch=self.current_epoch)
            if v.shape == prev_actions[..., :4].shape:
                log_sequence(self.logger, img=v, title='test_v_function', number=batch_idx,
                          img_text=v.mean(), epoch=self.current_epoch)
            log_video(self.logger, img=prev_rewards_mean, title='test_RewardMap',
                      number=batch_idx, epoch=self.current_epoch)
            if self.hparams.vae_on_test:
                log_video(self.logger, img=corrected.transpose((1, 2, 0))[None,], background=b_img.squeeze(0),
                          title='test_VAE_corrected', number=batch_idx, epoch=self.current_epoch)

        self.log_dict(logs, sync_dist=True)
        print(f"Logging took {round(time.time() - start_time, 4)} (s).")

        if self.hparams.save_on_test:
            #prev_actions = prev_actions.squeeze(0).cpu().detach().numpy()
            prev_actions = y_pred_np_as_batch.transpose((1, 2, 0))
            original_shape = meta_dict.get("original_shape").cpu().detach().numpy()[0]

            fname = meta_dict.get("case_identifier")[0]
            spacing = meta_dict.get("original_spacing").cpu().detach().numpy()[0]
            resampled_affine = meta_dict.get("resampled_affine").cpu().detach().numpy()[0]
            save_dir = os.path.join(self.trainer.default_root_dir, f"testing_raw/{self.trainer.datamodule.get_approx_gt_subpath(fname).rsplit('/', 1)[0]}/")

            final_preds = np.expand_dims(prev_actions, 0)
            transform = tio.Resample(spacing)
            croporpad = tio.CropOrPad(original_shape)
            final_preds = croporpad(transform(tio.LabelMap(tensor=final_preds, affine=resampled_affine))).numpy()[0]

            self.save_mask(final_preds, fname, spacing.astype(np.float64), save_dir)

        return logs

    def on_test_start(self) -> None:  # noqa: D102
        super().on_test_start()

        if self.trainer.world_size > 1:
            print(f"\nWorld size is {self.trainer.world_size}, default to skip worse frame threshold and vae_on_test")
            self.hparams.worst_frame_thresholds = None
            self.hparams.vae_on_test = False

        if self.trainer.datamodule is None:
            sw_batch_size = 2
        else:
            sw_batch_size = self.trainer.datamodule.hparams.batch_size

        self.inferer = SlidingWindowInferer(
            roi_size=self.actor.actor.net.patch_size,
            sw_batch_size=sw_batch_size,
            overlap=0.5,
            mode='gaussian',
            cache_roi_weight_map=True,
        )

        self.reward_func.prepare_for_full_sequence(self.trainer.datamodule.hparams.batch_size)
        self.initial_test_params = copy.deepcopy(self.actor.actor.net.state_dict())


    def predict(
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 2D/3D images with sliding window inference.

        Uses ``self._current_cond`` (set by test/predict steps) for view conditioning
        when the actor net supports it.

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
            if len(self.actor.actor.net.patch_size) == 3:
                # Pad the last dimension to avoid 3D segmentation border artifacts
                pad_len = 6 if image.shape[-1] > 6 else image.shape[-1] - 1
                image = pad(image, (pad_len, pad_len, 0, 0, 0, 0), mode="reflect")
                pred = self.predict_3D_3Dconv_tiled(image, apply_softmax)
                # Inverse the padding after prediction
                return pred[..., pad_len:-pad_len]
            else:
                raise ValueError("Check your patch size. You dummy.")
        if len(image.shape) == 4:
            raise ValueError("No 2D images here. You dummy.")

    def tta_predict(
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict with test time augmentation.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over number of flips.
        """
        preds = self.predict(image, apply_softmax)
        factors = [1.1, 0.9, 1.25, 0.75]
        translations = [40, 60, 80, 120]
        rotations = [5, 10, -5, -10]

        for factor in factors:
            preds += self.predict(adjust_contrast(
                image.permute((4, 0, 1, 2, 3)), factor).permute((1, 2, 3, 4, 0)), apply_softmax)

        def x_translate_left(img, amount=20):
            return pad(img, (0, 0, 0, 0, amount, 0), mode="constant")[:, :, :-amount, :, :]
        def x_translate_right(img, amount=20):
            return pad(img, (0, 0, 0, 0, 0, amount), mode="constant")[:, :, amount:, :, :]
        def y_translate_up(img, amount=20):
            return pad(img, (0, 0, amount, 0, 0, 0), mode="constant")[:, :, :, :-amount, :]
        def y_translate_down(img, amount=20):
            return pad(img, (0, 0, 0, amount, 0, 0), mode="constant")[:, :, :, amount:, :]

        for translation in translations:
            preds += x_translate_right(self.predict(x_translate_left(image, translation), apply_softmax),
                                       translation)
            preds += x_translate_left(self.predict(x_translate_right(image, translation), apply_softmax),
                                      translation)
            preds += y_translate_down(self.predict(y_translate_up(image, translation), apply_softmax),
                                      translation)
            preds += y_translate_up(self.predict(y_translate_down(image, translation), apply_softmax),
                                    translation)

        # TODO: optimize this for compute time
        for rotation in rotations:
            rotated = torch.zeros_like(image)
            for i in range(image.shape[-1]):
                rotated[0, :, :, :, i] = rotate(image[0, :, :, :, i], angle=rotation)
            rot_pred = self.predict(rotated, apply_softmax)
            for i in range(image.shape[-1]):
                rot_pred[0, :, :, :, i] = rotate(rot_pred[0, :, :, :, i], angle=-rotation)
            preds += rot_pred

        preds /= len(factors) + len(translations) * 4 + len(rotations) + 1
        return preds.argmax(dim=1)

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
        self, image: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor]:
        """Inference using sliding window, broadcasting ``self._current_cond`` to each patch batch."""
        cond = self._current_cond
        net = self.actor.actor.net
        if cond is not None and hasattr(net, "view_embed"):
            # FiLMUNet: LongTensor of view indices, shape (B,)
            def _net(x, _n=net, _c=cond):
                return _n(x, _c.expand(x.shape[0]))
            return self.inferer(inputs=image, network=_net)
        if cond is not None and hasattr(net, "cond_projectors"):
            # additive/AdaIN ConditionedUNet: one-hot float, shape (B, num_views)
            def _net(x, _n=net, _c=cond):
                return _n(x, _c.expand(x.shape[0], -1))
            return self.inferer(inputs=image, network=_net)
        return self.inferer(inputs=image, network=net)

    def save_mask(
        self, preds: np.ndarray, fname: str, spacing: np.ndarray, save_dir: Union[str, Path],
            type: type[np.generic] | type[float] = np.uint8,
    ) -> None:
        """Save segmentation mask to the given save directory.

        Args:
            preds: Predicted segmentation mask.
            fname: Filename to save.
            spacing: Spacing to save the segmentation mask.
            save_dir: Directory to save the segmentation mask.
        """
        print(f"Saving segmentation for {fname}... in {save_dir}")

        os.makedirs(save_dir, exist_ok=True)

        preds = preds.astype(type)
        itk_image = sitk.GetImageFromArray(rearrange(preds, "w h d ->  d h w"))
        itk_image.SetSpacing(spacing)
        sitk.WriteImage(itk_image, os.path.join(save_dir, str(fname) + ".nii.gz"))

    def on_test_end(self) -> None:
        actor_save_path = self.hparams.actor_save_path if self.hparams.actor_save_path else \
            f"{self.trainer.log_dir}/{self.trainer.logger.version}/actor.ckpt"
        actor_save_path = Path(actor_save_path)
        actor_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.actor.net.state_dict(), actor_save_path)
        print(f"actor saved at: {actor_save_path}")

        critic_save_path = self.hparams.critic_save_path if self.hparams.critic_save_path else \
            f"{self.trainer.log_dir}/{self.trainer.logger.version}/critic.ckpt"
        critic_save_path = Path(critic_save_path)
        critic_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.critic.net.state_dict(), critic_save_path)
        print(f"critic saved at: {critic_save_path}")

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self.hparams.inference:
            # One bad case must not take the rest of the task with it. Without this a single OOM
            # aborts every case still queued behind it in the same process, and on a re-run the
            # same case aborts whichever task picks it up again -- so a handful of oversized
            # cases can stall a sweep indefinitely.
            try:
                self.inference_predict_step(batch, batch_idx)
            except Exception as e:
                self._record_case_failure(batch, e)
            return None
        # must be batch size 1 as images have varied sizes
        b_img, meta_dict = batch['img'].squeeze(0), batch['image_meta_dict']
        self._current_cond = self._get_cond(batch)
        id = meta_dict['case_identifier'][0]
        view = meta_dict['view'][0]
        voxel_spacing = np.asarray([abs(meta_dict['resampled_affine'][0, 0, 0].cpu().numpy()),
                                     abs(meta_dict['resampled_affine'][0, 1, 1].cpu().numpy())])

        # could use full sequence later and split into subsecquions here
        actions, _, _ = self.rollout(b_img, torch.zeros_like(b_img).squeeze(1), sample=True,
                                     cond=self._current_cond)
        actions_unsampled, _, _ = self.rollout(b_img, torch.zeros_like(b_img).squeeze(1), sample=False,
                                               cond=self._current_cond)

        is_a3c = view.lower() == 'a3c'

        if is_a3c:
            # corrector not trained on A3C — skip it and use unsampled prediction directly as pseudo-GT
            actions_unsampled_clean = actions_unsampled.cpu().numpy()
            corrected_validity = False
            ae_comp = 1.0
        else:
            corrected, corrected_validity, ae_comp, actions_unsampled_clean = self.pred_corrector.correct_single_seq(b_img.squeeze(0), actions_unsampled.squeeze(0), voxel_spacing)
            actions_unsampled_clean = actions_unsampled_clean[None,]
            corrected = corrected[None,]

        initial_params = copy.deepcopy(self.actor.actor.net.state_dict())
        itr = 0

        self.trainer.datamodule.add_to_train(id)

        action_uns_anatomical_validity = [
            check_segmentation_validity(actions_unsampled_clean[0, ..., i].T, voxel_spacing, [0, 1, 2])
            for i in range(actions_unsampled_clean.shape[-1])]

        if ae_comp > self.hparams.ae_comp_threshold and all(action_uns_anatomical_validity):
            self.trainer.datamodule.add_to_gt(id)

            if self.hparams.predict_do_model_perturb:
                # level 0 uses fixed mult=0.1; subsequent levels are scaled to hit target degradations
                # rate is updated from each level's observation to track the local nonlinearity
                target_degradations = [0.05, 0.12, 0.22]  # 1-dice targets for levels 1, 2, 3
                multiplier = 0.1
                degrad_rate = None  # estimated as (1-dice)/mult, updated every level

                for j in range(4):
                    if j > 0 and degrad_rate is not None:
                        multiplier = min(target_degradations[j - 1] / degrad_rate, 20.0)

                    # get random seed based on time to maximise randomness of noise and subsequent predictions
                    # explore as much space around policy as possible
                    time_seed = int(round(datetime.now().timestamp())) + j
                    torch.manual_seed(time_seed)

                    # load initial params so noise is not compounded
                    self.actor.actor.net.load_state_dict(initial_params, strict=True)

                    # add noise to params
                    with torch.no_grad():
                        for param in self.actor.actor.net.parameters():
                            param.add_(torch.randn(param.size()).to(next(self.actor.actor.net.parameters()).device) * multiplier * param.data.norm() / param.data.numel() ** 0.5)

                    # make prediction
                    deformed_action, *_ = self.actor.actor(b_img,
                                                            self._expand_cond(self._current_cond, b_img.shape[0]))
                    if len(deformed_action.shape) > 4:
                        deformed_action = deformed_action.argmax(dim=1)
                    else:
                        deformed_action = torch.round(deformed_action)

                    if deformed_action.sum() == 0:
                        if j == 0:
                            degrad_rate = 1.0
                        continue

                    d = dice_score(deformed_action, torch.tensor(actions_unsampled_clean, device=deformed_action.device))
                    d_val = d.mean().item()
                    print(f"[weights mult={multiplier:.3f}] dice vs clean: {d_val:.3f}")

                    degrad_rate = max((1 - d_val) / multiplier, 1e-4)

                    filename = f"{batch_idx}_{itr}_{time_seed}_{self.trainer.global_rank}_weights_{view}.nii.gz"
                    save_to_reward_dataset(self.hparams.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img.squeeze(0)),
                                           convert_to_numpy(actions_unsampled_clean),
                                           convert_to_numpy(deformed_action))
                self.actor.actor.net.load_state_dict(initial_params)
            if self.hparams.predict_do_img_perturb:
                contrast_factors = [0.4, 0.05]  # NOTE: near-zero effect observed — model appears contrast-invariant
                for factor in contrast_factors:
                    in_img = copy.deepcopy(b_img)
                    in_img = adjust_contrast(in_img.permute((4, 0, 1, 2, 3)), factor).permute((1, 2, 3, 4, 0))

                    # make prediction
                    contr_action, *_ = self.actor.actor(in_img,
                                                         self._expand_cond(self._current_cond, in_img.shape[0]))
                    if len(contr_action.shape) > 4:
                        contr_action = contr_action.argmax(dim=1)
                    else:
                        contr_action = torch.round(contr_action)

                    if contr_action.sum() == 0:
                        continue

                    d = dice_score(contr_action, torch.tensor(actions_unsampled_clean, device=contr_action.device))
                    print(f"[contrast factor={factor:.2f}] dice vs clean: {d.mean():.3f}")

                    time_seed = int(round(datetime.now().timestamp())) + int(factor*10)
                    filename = f"{batch_idx}_{itr}_{time_seed}_{self.trainer.global_rank}_contrast_{view}.nii.gz"
                    save_to_reward_dataset(self.hparams.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img.squeeze(0)),
                                           convert_to_numpy(actions_unsampled_clean),
                                           convert_to_numpy(contr_action))

                gaussian_blurs = [0.3, 1.0, 1.5]
                for blur in gaussian_blurs:
                    in_img = b_img.clone()
                    in_img += torch.randn(in_img.size()).to(next(self.actor.actor.net.parameters()).device) * blur
                    in_img /= in_img.max()

                    # make prediction
                    blurred_action, *_ = self.actor.actor(in_img,
                                                           self._expand_cond(self._current_cond, in_img.shape[0]))
                    if len(blurred_action.shape) > 4:
                        blurred_action = blurred_action.argmax(dim=1)
                    else:
                        blurred_action = torch.round(blurred_action)

                    if blurred_action.sum() == 0:
                        continue

                    d = dice_score(blurred_action, torch.tensor(actions_unsampled_clean, device=blurred_action.device))
                    print(f"[blur std={blur:.2f}] dice vs clean: {d.mean():.3f}")

                    time_seed = int(round(datetime.now().timestamp())) + int(blur*100)
                    filename = f"{batch_idx}_{itr}_{time_seed}_{self.trainer.global_rank}_blur_{view}.nii.gz"
                    save_to_reward_dataset(self.hparams.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img.squeeze(0)),
                                           convert_to_numpy(actions_unsampled_clean),
                                           convert_to_numpy(blurred_action))

        else:
            print(f"INVALID - {view} -  {ae_comp}")
            if corrected_validity:
                if self.hparams.predict_do_corrections:
                    filename = f"{batch_idx}_{itr}_{int(round(datetime.now().timestamp()))}_{self.trainer.global_rank}_correction_{view}.nii.gz"
                    save_to_reward_dataset(self.hparams.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img.squeeze(0)),
                                           convert_to_numpy(corrected),
                                           convert_to_numpy(actions))

        self.trainer.datamodule.update_dataframe()
        # make sure initial params are back at end of step
        self.actor.actor.net.load_state_dict(initial_params)
        self.predicted_rows += [self.trainer.datamodule.df.loc[self.trainer.datamodule.df['dicom_uuid'] == id]]

    def on_predict_start(self) -> None:  # noqa: D102
        if self.hparams.inference:
            super().on_predict_start()
            if self.trainer.datamodule is None:
                sw_batch_size = 2
            else:
                sw_batch_size = self.trainer.datamodule.hparams.batch_size

            self.inferer = SlidingWindowInferer(
                roi_size=self.actor.actor.net.patch_size,
                sw_batch_size=sw_batch_size,
                overlap=0.5,
                mode='gaussian',
                cache_roi_weight_map=True,
            )

            self.reward_func.prepare_for_full_sequence()

            print(f"\n\nPredict step parameters: \n"
                  f"    - Sliding window len: {4}\n"
                  f"    - Sliding window overlap: {0.5}\n"
                  f"    - Sliding window importance map: {'gaussian'}\n")
            self.initial_test_params = copy.deepcopy(self.actor.actor.net.state_dict())

    def _reset_to_initial_params(self):
        """Restore the checkpoint's weights, undoing any test-time optimisation.

        TTO overfits the net to one case. Without this, the next case is predicted with the
        previous case's overfit weights -- and only cases that pass validity are affected, since
        a case that runs TTO restores before optimising. That makes the contamination invisible
        exactly where nothing appears to have happened.
        """
        if self.initial_test_params is not None:
            self.actor.actor.net.load_state_dict(self.initial_test_params, strict=False)

    def inference_predict_step(self, batch: dict[str, Tensor], batch_idx: int):
        img, properties_dict = batch["image"], batch["image_meta_dict"]
        self._current_cond = self._get_cond(batch)

        # Every case starts from the checkpoint, never from a previous case's TTO result.
        if self.hparams.tto in ('force', 'on'):
            self._reset_to_initial_params()

        self.patch_size = list([img.shape[-3], img.shape[-2], 4])
        self.inferer.roi_size = self.patch_size

        start_time = time.time()
        preds = self.tta_predict(img) if self.hparams.tta else self.predict(img).argmax(dim=1)
        print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")

        # check if valid or else do TTO if is tto=True
        if self.hparams.tto in ['force', 'on']:
            start_time = time.time()
            y_pred_np_as_batch = preds.squeeze(0).cpu().numpy().transpose((2, 0, 1))

            for i in range(len(y_pred_np_as_batch)):
                lbl, num = ndimage.measurements.label(y_pred_np_as_batch[i] != 0)
                # Count the number of elements per label
                count = np.bincount(lbl.flat)
                # Select the largest blob
                maxi = np.argmax(count[1:]) + 1
                # Remove the other blobs
                y_pred_np_as_batch[i][lbl != maxi] = 0

            # should be still valid to use resampled spacing for metrics here
            voxel_spacing = np.asarray([[0.37, 0.37]]).repeat(repeats=len(y_pred_np_as_batch), axis=0)
            print(f"Cleaning took {round(time.time() - start_time, 4)} (s).")

            anat_errors = is_anatomically_valid(y_pred_np_as_batch)
            temporal_valid, _ = check_temporal_validity(y_pred_np_as_batch.transpose((0, 2, 1)),
                                                        voxel_spacing[0])
            validated = int(all(anat_errors)) and temporal_valid
            if self.hparams.tto == 'force' or (not validated and self.hparams.tto == 'on'):
                self._reset_to_initial_params()

                start_time = time.time()
                self.ttoptimize(img)
                print(f"\nTTOverfit took {round(time.time() - start_time, 4)} (s).")

                start_time = time.time()
                preds = self.tta_predict(img) if self.hparams.tta else self.predict(img).argmax(dim=1)
                print(f"\nPost-TTO Prediction took {round(time.time() - start_time, 4)} (s).")

        # remove extra blobs if any
        y_pred_np_as_batch = preds[0].cpu().numpy().transpose((2, 0, 1))
        for i in range(len(y_pred_np_as_batch)):
            lbl, num = ndimage.measurements.label(y_pred_np_as_batch[i] != 0)
            # Count the number of elements per label
            count = np.bincount(lbl.flat)
            # Select the largest blob
            maxi = np.argmax(count[1:]) + 1
            # Remove the other blobs
            y_pred_np_as_batch[i][lbl != maxi] = 0
        preds = torch.tensor(y_pred_np_as_batch.transpose((1, 2, 0))[None,], device=self.device)

        # get reward maps
        rew = self.reward_func.predict_full_sequence(preds, img, None, cond=self._current_cond)
        # fuse ALL nets, not just the first two (matches RewardUnets3D.__call__): a pixel is
        # high-reward only where every net agrees.
        merged = functools.reduce(torch.minimum, rew) if len(rew) > 1 else rew[0]

        preds = preds.cpu().detach().numpy()
        merged = merged.cpu().detach().numpy()
        rew = [r.cpu().detach().numpy() for r in rew]

        # save stuff
        original_shape = properties_dict.get("original_shape").cpu().detach().numpy()[0]
        fname = properties_dict.get("case_identifier")[0]
        spacing = properties_dict.get("original_spacing").cpu().detach().numpy()[0]
        resampled_affine = properties_dict.get("resampled_affine").cpu().detach().numpy()[0]
        affine = properties_dict.get('original_affine').cpu().detach().numpy()[0]

        transform = tio.Resample(spacing)
        croporpad = tio.CropOrPad(original_shape)

        inference_save_dir = properties_dict.get("inference_save_dir", None)
        save_dir = inference_save_dir[0] if inference_save_dir else os.path.join(self.trainer.default_root_dir, "inference_raw")

        self.save_mask(croporpad(transform(tio.LabelMap(tensor=preds, affine=resampled_affine))).numpy()[0],
                       fname, spacing, save_dir)

        if properties_dict.get("save_rewards", None):
            # Resampled back onto the original image grid so the maps line up with the
            # saved mask. Net order follows cfg.model.reward.state_dict_paths; format and
            # rationale live in save_reward_stack.
            reward_path = os.path.join(save_dir, f"{fname}_merged_all_rewards.nii.gz")
            print(f"Saving reward maps ({list(getattr(self.reward_func, 'nets', {}).keys())}) "
                  f"for {fname}... in {save_dir}")
            save_reward_stack(
                [croporpad(transform(tio.ScalarImage(tensor=r, affine=resampled_affine))).numpy()[0]
                 for r in rew],
                affine, reward_path,
            )

        save_gif = properties_dict.get("save_to_gif", None)
        if save_gif:
            gif_path = os.path.join(save_dir, str(fname) + ".gif")
            print(f"Saving gif to {gif_path}")
            img_b = img.cpu().numpy().squeeze(0).squeeze(0).transpose((2, 0, 1))
            # one figure: image+segmentation panel, then each net's map (and their fused
            # min, which is only worth its own panel when there is more than one net)
            reward_panels = [r[0].transpose((2, 0, 1)) for r in rew]
            panel_names = list(getattr(self.reward_func, "nets", {}).keys())
            if len(rew) > 1:
                reward_panels = [merged[0].transpose((2, 0, 1))] + reward_panels
                panel_names = ["reward (min)"] + panel_names
            save_reward_gif(img_b, reward_panels, gif_path, overlay=y_pred_np_as_batch,
                            names=panel_names)

        case_csv_path = properties_dict.get("case_csv_path", [""])[0]
        if case_csv_path:
            self._write_inference_row(case_csv_path, fname, batch, y_pred_np_as_batch, rew,
                                      merged, resampled_affine)

        #return preds, merged, rew

    def _record_case_failure(self, batch, exc):
        """Log a failed case and decide whether it should be attempted again.

        Two kinds of failure say nothing about the case and must stay eligible for a later run,
        so their claim is left EMPTY (released with
        `find <output>/csv -type f -empty -delete`):

          * a CUDA OOM, which is a statement about the GPU, and
          * submitit ending the job on walltime, which is a statement about run_time_min. That
            one is also re-raised: the process is being torn down, so there is nothing to be
            gained by continuing to the next case.

        Any other exception is treated as a property of the case -- an unreadable file, an
        unexpected shape -- and its claim is FILLED, so it is not retried blindly at every tier
        of an escalating re-run.

        Either way a row lands in <output>/failures/<job>.csv. The file is per process, so no
        two writers ever share one and concurrent submissions need no coordination.
        """
        properties_dict = batch.get("image_meta_dict", {})
        fname = (properties_dict.get("case_identifier") or ["<unknown>"])[0]
        csv_path = (properties_dict.get("case_csv_path") or [""])[0]
        job_ended = isinstance(exc, _JobEndedError)
        retryable = job_ended or isinstance(exc, torch.cuda.OutOfMemoryError)

        detail = str(exc).replace("\n", " ")[:300]
        print(f"FAILED {fname}: {type(exc).__name__}: {detail}"
              f" ({'will be retried once its claim is released' if retryable else 'not retried'})")
        traceback.print_exc()

        if csv_path:
            out_root = Path(csv_path).parent.parent
            # unique per process: array task if we are under slurm, pid otherwise
            job = "-".join(filter(None, [os.environ.get("SLURM_JOB_ID"),
                                         os.environ.get("SLURM_ARRAY_TASK_ID")])) \
                  or f"pid{os.getpid()}"
            append_csv_row(out_root / "failures" / f"{job}.csv", {
                "case_id": fname,
                "error_type": type(exc).__name__,
                "error": detail,
                "retryable": retryable,
            })
            if not retryable:
                # fills the claim, so the case is not offered again
                append_csv_row(csv_path, {"case_id": fname, "failed": True,
                                          "error_type": type(exc).__name__, "error": detail})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if job_ended:
            raise exc

    def _write_inference_row(self, csv_path, fname, batch, pred_tchw, rew, merged,
                             resampled_affine):
        """Write this case's validity flags and reward summaries to the case's own csv.

        One file per case, claimed empty before the case ran and filled here once its outputs
        are on disk: no two writers ever touch it, so concurrent tasks -- or whole overlapping
        submissions -- need no coordination, and a non-empty file means the case is complete.

        Column definitions match scripts/score_predictions_offline.py, so rows produced online
        here stay comparable with the scores computed offline from saved masks.
        """
        voxel_spacing = np.asarray([abs(resampled_affine[0, 0]), abs(resampled_affine[1, 1])])
        anat = is_anatomically_valid(pred_tchw).numpy()
        temporal_valid, _ = check_temporal_validity(pred_tchw.transpose((0, 2, 1)), voxel_spacing)

        view_as_id = batch.get('view_as_id', None)
        row = {
            "case_id": fname,
            "view_id": int(view_as_id[0]) if view_as_id is not None else None,
            "n_frames": int(pred_tchw.shape[0]),
            "anat_valid_frac": float(anat.mean()),
            "anat_valid_all": bool(anat.all()),
            "temporal_valid": bool(temporal_valid),
            # the same gate inference uses to decide whether TTO is needed
            "validated": bool(anat.all()) and bool(temporal_valid),
            **reward_stats(merged, "merged"),
        }
        # net order follows the reward function's nets, exactly as the gif panels do
        by_name = dict(zip(getattr(self.reward_func, "nets", {}).keys(), rew))
        for name, r in by_name.items():
            row.update(reward_stats(r, name))

        # Named subsets fused the same way the full merge is (elementwise minimum: a pixel is
        # high-reward only where every net in the subset agrees). Lets a run be scored against a
        # narrower gate than the one it was produced with -- e.g. anatomical+landmarks, without
        # apex -- without re-predicting. Column names match scripts/... fusion ablation output.
        for subset in _FUSION_SUBSETS:
            if all(n in by_name for n in subset):
                fused = functools.reduce(np.minimum, (by_name[n] for n in subset))
                row.update(reward_stats(fused, "+".join(subset)))

        append_csv_row(csv_path, row)

    def on_predict_epoch_end(self) -> None:
        if not self.hparams.inference:
            # for multi gpu cases, save intermediate file before sending to main csv
            print(f"Saving temporary rows file to : {f'{self.temp_files_path}/temp_pred_{self.trainer.global_rank}.csv'}")
            pd.concat(self.predicted_rows).to_csv(f"{self.temp_files_path}/temp_pred_{self.trainer.global_rank}.csv")
