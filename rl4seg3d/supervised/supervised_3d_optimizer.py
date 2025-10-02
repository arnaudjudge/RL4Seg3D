import random
import time
from typing import Dict

import numpy as np
import torch
from rl4seg3d.utils.Metrics import accuracy, dice_score, is_anatomically_valid
from rl4seg3d.utils.logging_helper import log_sequence, log_video
from rl4seg3d.utils.test_metrics import dice, hausdorff
from scipy import ndimage
from torch import nn, Tensor
from torchmetrics.segmentation import DiceScore

from patchless_nnunet.models.patchless_nnunet_module import nnUNetPatchlessLitModule
from vital.data.camus.config import Label


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        intersection = torch.sum(target * output)
        return 1 - ((2. * intersection) / (torch.sum(target) + torch.sum(output)))


class Supervised3DOptimizer(nnUNetPatchlessLitModule):
    def __init__(self, ckpt_path=None, corrector=None, predict_save_dir=None, **kwargs):
        super().__init__(**kwargs)

        self.save_test_results = False
        self.ckpt_path = ckpt_path
        self.predict_save_dir = predict_save_dir
        self.pred_corrector = corrector

        self.dice = DiceScore()

    def forward(self, x):
        out = self.net.forward(x)
        if self.net.num_classes > 1:
            out = torch.softmax(out, dim=1)
        else:
            out = torch.sigmoid(out).squeeze(1)
        return out

    def configure_optimizers(self):
        # add weight decay so predictions are less certain, more randomness?
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0)

    def training_step(self, batch: dict[str, Tensor], *args, **kwargs) -> Dict:
        x, y = batch['img'].squeeze(0), batch['gt'].squeeze(0)

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs, prog_bar=True)
        return logs

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        b_img, b_gt = batch['img'].squeeze(0), batch['gt'].squeeze(0)
        y_pred = self.forward(b_img)

        loss = self.loss(y_pred, b_gt)

        if self.net.num_classes > 1:
            y_pred = y_pred.argmax(dim=1)
        else:
            y_pred = torch.round(y_pred)

        acc = accuracy(y_pred, b_img, b_gt)
        dice = dice_score(y_pred, b_gt)

        logs = {'val_loss': loss,
                'val_acc': acc.mean(),
                'val_dice': dice.mean(),
                }

        # log images
        if self.trainer.local_rank == 0:
            idx = random.randint(0, len(b_img) - 1)  # which image to log
            log_sequence(self.logger, img=b_img[idx], title='Image', number=batch_idx, epoch=self.current_epoch)
            log_sequence(self.logger, img=b_gt[idx].unsqueeze(0), title='GroundTruth', number=batch_idx,
                         epoch=self.current_epoch)
            log_sequence(self.logger, img=y_pred[idx].unsqueeze(0), title='Prediction', number=batch_idx,
                         img_text=acc[idx].mean(), epoch=self.current_epoch)

        self.log_dict(logs)

        return logs

    # don't want these functions from parent class
    def on_validation_epoch_end(self) -> None:
        return

    def on_test_epoch_end(self) -> None:
        return

    def test_step(self, batch, batch_idx):
        b_img, b_gt, meta_dict = batch['img'], batch['gt'], batch['image_meta_dict']

        self.patch_size = list([b_img.shape[-3], b_img.shape[-2], self.hparams.sliding_window_len])
        self.inferer.roi_size = self.patch_size

        start_time = time.time()
        y_pred = self.predict(b_img)
        print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")


        loss = self.loss(y_pred, b_gt.long())

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

        test_dice = dice(y_pred_np_as_batch, b_gt_np_as_batch, labels=(Label.BG, Label.LV, Label.MYO),
                         exclude_bg=True, all_classes=True)
        test_dice_epi = dice((y_pred_np_as_batch != 0).astype(np.uint8), (b_gt_np_as_batch != 0).astype(np.uint8),
                             labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False)

        test_hd = hausdorff(y_pred_np_as_batch, b_gt_np_as_batch, labels=(Label.BG, Label.LV, Label.MYO),
                            exclude_bg=True, all_classes=True, voxel_spacing=voxel_spacing)
        test_hd_epi = hausdorff((y_pred_np_as_batch != 0).astype(np.uint8), (b_gt_np_as_batch != 0).astype(np.uint8),
                                labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False,
                                voxel_spacing=voxel_spacing)['Hausdorff']
        anat_errors = is_anatomically_valid(y_pred_np_as_batch)

        logs = {'test_loss': loss,
                'test_acc': acc.mean(),
                'test_dice': simple_dice.mean(),
                'test_anat_valid': anat_errors.mean(),
                'dice_epi': test_dice_epi,
                'hd_epi': test_hd_epi,
                }
        logs.update(test_dice)
        logs.update(test_hd)

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
