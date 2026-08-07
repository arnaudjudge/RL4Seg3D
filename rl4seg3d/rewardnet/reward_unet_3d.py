import pickle
import random
from typing import Dict

import torch
from lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F
from vital.models.segmentation.unet import UNet

from rl4seg3d.actors.Actors_3d import _forward_with_optional_cond
from rl4seg3d.utils.Metrics import accuracy
from rl4seg3d.utils.logging_helper import log_sequence, log_video

import torch.distributions as distributions


class Reward3DOptimizer(LightningModule):
    def __init__(self, net, loss=nn.MSELoss(), save_model_path=None, var_file=None, num_views=0,
                 do_temp_scale=False, loss_on_logits=None, **kwargs):
        super().__init__(**kwargs)

        self.net = net
        self.num_views = num_views

        self.temperature = nn.Parameter(torch.ones(1).to(self.device))
        self.var_file = var_file

        self.loss = loss
        # Logit-space losses (see rewardnet.reward_losses) must see raw logits; the MSE
        # baselines take post-sigmoid probabilities. Auto-detected from the loss so the two
        # can never fall out of sync, with an explicit override for anything exotic.
        if loss_on_logits is None:
            loss_on_logits = bool(getattr(loss, "expects_logits",
                                          isinstance(loss, nn.BCEWithLogitsLoss)))
        self.loss_on_logits = loss_on_logits
        print(f"[Reward3DOptimizer] loss={type(loss).__name__} on {'logits' if loss_on_logits else 'probabilities'}")
        self.save_model_path = save_model_path
        self.do_temp_scale = do_temp_scale

    def _predict_and_loss(self, x, y, cond):
        """Forward once; return (probabilities for logging/metrics, loss).

        Applying MSE to the post-sigmoid output scales the logit gradient by p(1-p), which
        vanishes precisely where the net confidently endorses a real error -- the case we
        most need to learn from. Logit-space losses avoid that, hence the routing.
        """
        logits = self(x, cond=cond)
        y_pred = torch.sigmoid(logits)
        return y_pred, self.loss(logits if self.loss_on_logits else y_pred, y)

    def _get_cond(self, batch):
        """View conditioning tensor from batch's view_id, or None.

        FiLMUNet (``view_embed``) expects an integer LongTensor of shape (B,); the
        additive/AdaIN ConditionedUNet (``cond_projectors``) expects a one-hot float
        tensor of shape (B, num_views). Plain UNet gets None.
        """
        if self.num_views <= 0 or len(batch) < 3:
            return None
        view_id = batch[2].long().view(-1).to(self.device)
        if hasattr(self.net, 'view_embed'):
            return view_id
        return F.one_hot(view_id, num_classes=self.num_views).float()

    @staticmethod
    def _expand_cond(cond, batch_size):
        """Broadcast a single-view cond to the per-window batch size.

        FiLM cond is a 1-D LongTensor (B,); additive cond is 2-D (B, num_views).
        """
        if cond is None:
            return None
        if cond.dim() == 1:
            return cond.expand(batch_size)
        return cond.expand(batch_size, -1)

    def forward(self, x, cond=None):
        # Positional cond: FiLMUNet.forward takes view_id, additive takes cond.
        return _forward_with_optional_cond(self.net, x, cond)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min")
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val/loss"}

    def training_step(self, batch, *args, **kwargs) -> Dict:
        x, y = batch[0], batch[1]
        cond = self._get_cond(batch)
        if len(x.shape) > 5:
            x = x.squeeze(0)
            y = y.squeeze(0)
            cond = self._expand_cond(cond, x.shape[0])

        y_pred, loss = self._predict_and_loss(x, y, cond)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs)
        return logs

    def validation_step(self, batch, batch_idx: int):
        x, y, view = batch[0], batch[1], batch[2].view(-1)[0].item()
        cond = self._get_cond(batch)
        if len(x.shape) > 5:
            x = x.squeeze(0)
            y = y.squeeze(0)
            cond = self._expand_cond(cond, x.shape[0])

        y_pred, loss = self._predict_and_loss(x, y, cond)

        acc = accuracy(y_pred, x, y)

        self.log_dict({"val/loss": loss,
                       "val/acc": acc.mean()})

        # log images
        if self.trainer.local_rank == 0:
            idx = random.randint(0, len(x) - 1)  # which image to log
            log_sequence(self.logger, img=x[idx], title=f'Image_{view}', number=batch_idx, epoch=self.current_epoch)
            log_sequence(self.logger, img=y[idx], title=f'GroundTruth_{view}', number=batch_idx, epoch=self.current_epoch)
            log_sequence(self.logger, img=y_pred[idx], title=f'Prediction_{view}', number=batch_idx,
                      img_text=acc[idx].mean(), epoch=self.current_epoch)

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y, view = batch[0], batch[1], batch[2].view(-1)[0].item()
        cond = self._get_cond(batch)
        if len(x.shape) > 5:
            x = x.squeeze(0)
            y = y.squeeze(0)
            cond = self._expand_cond(cond, x.shape[0])

        y_pred, loss = self._predict_and_loss(x, y, cond)

        acc = accuracy(y_pred, x, y)

        self.log_dict({"test/loss": loss,
                       "test/acc": acc.mean()})

        if self.trainer.local_rank == 0:
            for i in range(len(x)):
                log_video(self.logger, img=x[i][1].unsqueeze(0), title=f'test_Image_{view}', number=batch_idx * (i + 1), epoch=self.current_epoch)
                log_video(self.logger, img=y[i], title=f'test_GroundTruth_{view}', number=batch_idx * (i + 1), epoch=self.current_epoch)
                log_video(self.logger, img=y_pred[i], title=f'test_Prediction_{view}', number=batch_idx * (i + 1),
                          img_text=acc[i].mean(), epoch=self.current_epoch)

        return {'loss': loss}

    def on_test_end(self) -> None:
        self.save_model()

    def save_model(self):
        if self.save_model_path:
            sd = self.net.state_dict()
            torch.save(sd, self.save_model_path)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        #temperature = self.temperature.unsqueeze(1).expand(logits.size()).to(self.device)
        return torch.div(logits, self.temperature)

    def on_train_end(self) -> None:
        if self.do_temp_scale:
            val_loader = self.trainer.datamodule.val_dataloader()

            logits_list = []
            labels_list = []
            with torch.no_grad():
                for input, label, _ in val_loader:
                    logits = self(input.to(self.device))
                    logits_list.append(torch.stack((1-logits, logits), dim=1).squeeze(2))
                    labels_list.append(label.squeeze(1).to(torch.long))
                # logits = torch.cat(logits_list).to(self.device)
                # labels = torch.cat(labels_list).to(self.device)
            labels_list.reverse()
            logits_list.reverse()
            optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=10000)
            criterion = nn.CrossEntropyLoss().to(self.device)


            def eval():
                optimizer.zero_grad()
                # do this in a loop and add all losses together before backward?
                loss = criterion(self.temperature_scale(logits_list[0].to(self.device)), labels_list[0].to(self.device))
                for i in range(1, len(labels_list)):
                    loss += criterion(self.temperature_scale(logits_list[i].to(self.device)), labels_list[i].to(self.device))
                loss.backward()
                return loss
            optimizer.step(eval)

            print(f"TEMPERATURE: {self.temperature}")
            self.trainer.logger.log_hyperparams({'Temperature factor': self.temperature.detach().cpu().numpy()[0]})

            if self.var_file:
                pickle.dump({"Temperature_factor": self.temperature.detach().cpu().numpy()[0]}, open(self.var_file, "wb"))

