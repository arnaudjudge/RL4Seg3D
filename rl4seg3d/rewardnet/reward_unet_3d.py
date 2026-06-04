import pickle
import random
from typing import Dict

import torch
from lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F
from vital.models.segmentation.unet import UNet

from rl4seg3d.rewardnet.unet_heads import UNet_multihead
from rl4seg3d.utils.Metrics import accuracy
from rl4seg3d.utils.logging_helper import log_sequence

import torch.distributions as distributions


class Reward3DOptimizer(LightningModule):
    def __init__(self, net, loss=nn.MSELoss(), save_model_path=None, var_file=None, num_views=0, **kwargs):
        super().__init__(**kwargs)

        self.net = net
        self.num_views = num_views

        self.temperature = nn.Parameter(torch.ones(1).to(self.device))
        self.var_file = var_file

        self.loss = loss
        self.save_model_path = save_model_path

    def _get_cond(self, batch):
        """One-hot view conditioning tensor from batch's view_id, or None."""
        if self.num_views <= 0 or len(batch) < 3:
            return None
        view_id = batch[2].long().view(-1)
        return F.one_hot(view_id, num_classes=self.num_views).float().to(self.device)

    def forward(self, x, cond=None):
        if cond is not None and hasattr(self.net, 'cond_projectors'):
            out = self.net(x, cond=cond)
        else:
            out = self.net(x)
        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min")
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val_loss"}

    def training_step(self, batch, *args, **kwargs) -> Dict:
        x, y = batch[0], batch[1]
        cond = self._get_cond(batch)
        if len(x.shape) > 5:
            x = x.squeeze(0)
            y = y.squeeze(0)

        y_pred = torch.sigmoid(self(x, cond=cond))
        loss = self.loss(y_pred, y)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs)
        return logs

    def validation_step(self, batch, batch_idx: int):
        x, y = batch[0], batch[1]
        cond = self._get_cond(batch)
        if len(x.shape) > 5:
            x = x.squeeze(0)
            y = y.squeeze(0)

        y_pred = torch.sigmoid(self(x, cond=cond))
        loss = self.loss(y_pred, y)

        acc = accuracy(y_pred, x, y)

        self.log_dict({"val_loss": loss,
                       "val_acc": acc.mean()})

        # log images
        if self.trainer.local_rank == 0:
            idx = random.randint(0, len(x) - 1)  # which image to log
            log_sequence(self.logger, img=x[idx], title='Image', number=batch_idx, epoch=self.current_epoch)
            log_sequence(self.logger, img=y[idx], title='GroundTruth', number=batch_idx, epoch=self.current_epoch)
            log_sequence(self.logger, img=y_pred[idx], title='Prediction', number=batch_idx,
                      img_text=acc[idx].mean(), epoch=self.current_epoch)

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        cond = self._get_cond(batch)
        if len(x.shape) > 5:
            x = x.squeeze(0)
            y = y.squeeze(0)

        y_pred = torch.sigmoid(self(x, cond=cond))
        loss = self.loss(y_pred, y)

        acc = accuracy(y_pred, x, y)

        self.log_dict({"test_loss": loss,
                       "test_acc": acc.mean()})

        if self.trainer.local_rank == 0:
            for i in range(len(x)):
                log_sequence(self.logger, img=x[i], title='test_Image', number=batch_idx * (i + 1), epoch=self.current_epoch)
                log_sequence(self.logger, img=y[i], title='test_GroundTruth', number=batch_idx * (i + 1), epoch=self.current_epoch)
                log_sequence(self.logger, img=y_pred[i], title='test_Prediction', number=batch_idx * (i + 1),
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
        val_loader = self.trainer.datamodule.val_dataloader()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in val_loader:
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

