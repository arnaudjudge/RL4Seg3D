import os
import random
import time
from typing import Dict
import numpy as np
from scipy import ndimage
import torchio as tio
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from rl4seg3d.utils.Metrics import accuracy, dice_score
from rl4seg3d.utils.logging_helper import log_sequence, log_video
from rl4seg3d.utils.test_metrics import full_test_metrics

from patchless_nnunet.models.patchless_nnunet_module import nnUNetPatchlessLitModule

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        intersection = torch.sum(target * output)
        return 1 - ((2. * intersection) / (torch.sum(target) + torch.sum(output)))


class Supervised3DOptimizer(nnUNetPatchlessLitModule):
    def __init__(self, ckpt_path=None, corrector=None, predict_save_dir=None, tta=False,
                 seed=123, load_from_ckpt=None, val_batch_size=4, num_views=0, **kwargs):
        super().__init__(**kwargs)

        self.save_test_results = False
        self.ckpt_path = ckpt_path
        self.predict_save_dir = predict_save_dir
        self.pred_corrector = corrector
        self.tta = tta
        self.load_from_ckpt = load_from_ckpt
        self.seed = seed
        self.val_batch_size = val_batch_size
        print(f"Seed: {seed}")

        if self.load_from_ckpt:
            m_state_dict = torch.load(self.load_from_ckpt)
            self.net.load_state_dict(m_state_dict, strict=False)

    def _get_cond(self, batch) -> Tensor | None:
        """Return a view conditioning tensor for the batch, or None.

        FiLMUNet (view_embed): returns LongTensor of shape (B,) — integer view indices.
        Additive/AdaIN (cond_projectors): returns float one-hot of shape (B, num_views).
        """
        if self.hparams.num_views <= 0 or 'view_as_id' not in batch:
            return None
        view_id = batch['view_as_id'].long().to(self.device)   # shape (B,)
        if hasattr(self.net, 'view_embed'):
            return view_id
        return F.one_hot(view_id, num_classes=self.hparams.num_views).float()

    def _has_cond(self) -> bool:
        return hasattr(self.net, 'view_embed') or hasattr(self.net, 'cond_projectors')

    def forward(self, x, cond=None):
        use_cond = cond is not None and self._has_cond()
        out = self.net.forward(x, cond) if use_cond else self.net.forward(x)
        # if self.net.num_classes > 1:
        #     out = torch.softmax(out, dim=1)
        # else:
        #     out = torch.sigmoid(out).squeeze(1)
        return out

    def sliding_window_inference(self, image):
        cond = getattr(self, '_current_cond', None)
        if cond is not None and self._has_cond():
            def _net(x):
                # LongTensor (FiLM): expand to (B,); float tensor (additive): expand to (B, C)
                c = cond.expand(x.shape[0]) if cond.dim() == 1 else cond.expand(x.shape[0], -1)
                return self.net(x, c)
            return self.inferer(inputs=image, network=_net)
        return self.inferer(inputs=image, network=self.net)

    def configure_optimizers(self):
        # add weight decay so predictions are less certain, more randomness?
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0)

    def training_step(self, batch: dict[str, Tensor], *args, **kwargs) -> Dict:
        x, y = batch['img'].squeeze(0), batch['gt'].squeeze(0)
        cond = self._get_cond(batch)
        if cond is not None:
            cond = cond.expand(x.shape[0]) if cond.dim() == 1 else cond.expand(x.shape[0], -1)
        y_hat = self.forward(x, cond)

        # DC_and_CE_loss expects logits (B, C, ...) and a channel-dim target (B, 1, ...)
        loss = self.loss(y_hat, y.unsqueeze(1).long())

        logs = {
            'loss': loss,
        }

        self.log_dict(logs, prog_bar=True)
        return logs

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        b_imgs, b_gts = batch['img'].squeeze(0), batch['gt'].squeeze(0)
        base_cond = self._get_cond(batch)
        logs = {'val/loss': [],
                "val/acc": [],
                "val/dice": []
                }
        for i in range(0, b_imgs.shape[0], self.hparams.val_batch_size):
            b_img = b_imgs[i:i + self.hparams.val_batch_size]
            b_gt = b_gts[i:i + self.hparams.val_batch_size]
            cond = (base_cond.expand(b_img.shape[0]) if base_cond.dim() == 1
                    else base_cond.expand(b_img.shape[0], -1)) if base_cond is not None else None

            y_pred = self.forward(b_img, cond)

            loss = self.loss(y_pred, b_gt.unsqueeze(1).long())

            if self.net.num_classes > 1:
                y_pred = y_pred.argmax(dim=1)
            else:
                y_pred = torch.round(torch.sigmoid(y_pred))

            acc = accuracy(y_pred, b_img, b_gt)
            dice = dice_score(y_pred, b_gt)

            logs["val/loss"] += [loss]
            logs["val/acc"] += [acc.mean()]
            logs["val/dice"] += [dice.mean()]

            # log images
            if self.trainer.local_rank == 0 and i == 0:
                idx = random.randint(0, len(b_img) - 1)  # which image to log
                log_sequence(self.logger, img=b_img[idx], title='Image', number=batch_idx, epoch=self.current_epoch)
                log_sequence(self.logger, img=b_gt[idx].unsqueeze(0), title='GroundTruth', number=batch_idx,
                             epoch=self.current_epoch)
                log_sequence(self.logger, img=y_pred[idx].unsqueeze(0), title='Prediction', number=batch_idx,
                             img_text=acc[idx].mean(), epoch=self.current_epoch)

        logs = {k: torch.tensor(v, device=self.device).mean() for k, v in logs.items()}
        self.log_dict(logs)

        return logs

    # don't want these functions from parent class
    def on_validation_epoch_end(self) -> None:
        return

    def on_test_epoch_end(self) -> None:
        return

    def test_step(self, batch, batch_idx):
        b_img, b_gt, meta_dict = batch['img'], batch['gt'], batch['image_meta_dict']
        view_str = meta_dict.get('view')[0]

        self._current_cond = self._get_cond(batch)  # stored for sliding_window_inference

        self.patch_size = list([b_img.shape[-3], b_img.shape[-2], self.hparams.sliding_window_len])
        self.inferer.roi_size = self.patch_size

        start_time = time.time()
        if self.tta:
            y_pred = self.tta_predict(b_img)
        else:
            y_pred = self.predict(b_img)
        print(f"\n{'TTA' if self.tta else 'Simple (No TTA)'} Prediction took {round(time.time() - start_time, 4)} (s).")

        if self.num_classes > 1:
            y_pred = y_pred.argmax(dim=1)
        else:
            y_pred = torch.round(y_pred)

        acc = accuracy(y_pred, b_img, b_gt)
        simple_dice = dice_score(y_pred, b_gt)

        y_pred_np_as_batch = y_pred.cpu().numpy().squeeze(0).transpose((2, 0, 1))
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
        voxel_spacing = np.asarray([[abs(meta_dict['resampled_affine'][0,0,0].cpu().numpy()),
                                    abs(meta_dict['resampled_affine'][0,1,1].cpu().numpy())]]).repeat(
            repeats=len(y_pred_np_as_batch), axis=0)


        logs = full_test_metrics(y_pred_np_as_batch, b_gt_np_as_batch, voxel_spacing, self.device, view=view_str)

        # SAVE OUTPUT ], FOR NOW ALWAYS
        prev_actions = y_pred_np_as_batch.transpose((1, 2, 0))
        original_shape = meta_dict.get("original_shape").cpu().detach().numpy()[0]

        fname = meta_dict.get("case_identifier")[0]
        spacing = meta_dict.get("original_spacing").cpu().detach().numpy()[0]
        resampled_affine = meta_dict.get("resampled_affine").cpu().detach().numpy()[0]
        save_dir = os.path.join(self.trainer.default_root_dir, f"testing_raw_supervised/{self.hparams.sliding_window_len}/")

        final_preds = np.expand_dims(prev_actions, 0)
        transform = tio.Resample(spacing)
        croporpad = tio.CropOrPad(original_shape)
        final_preds = croporpad(transform(tio.LabelMap(tensor=final_preds, affine=resampled_affine))).numpy()[0]

        self.save_mask(final_preds, fname, spacing.astype(np.float64), save_dir)

        if self.trainer.local_rank == 0:
            for i in range(len(b_img)):
                log_video(self.logger, img=b_img[i], title='test_Image', number=batch_idx * (i + 1),
                             epoch=self.current_epoch)
                log_video(self.logger, img=b_gt[i].unsqueeze(0), background=b_img[i], title='test_GroundTruth', number=batch_idx * (i + 1),
                             epoch=self.current_epoch)
                log_video(self.logger, img=y_pred[i].unsqueeze(0), background=b_img[i], title='test_Prediction', number=batch_idx * (i + 1),
                          img_text=simple_dice[i].mean(), epoch=self.current_epoch)

        self.log_dict(logs)

        return logs

    def on_test_end(self) -> None:
        self.save()

    def save(self) -> None:
        if self.ckpt_path:
            torch.save(self.net.state_dict(), self.ckpt_path)
