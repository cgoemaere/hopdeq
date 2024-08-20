from typing import List

import torch
from deq_core.deq import DEQ
from lightning import LightningModule
from madam import Madam

from .onematrixfunction import OneMatrixFunction


class LitOneMatrixDEQ(LightningModule):
    def __init__(
        self,
        batch_size: int,
        lr: float,
        layers: List[int],
        deq_kwargs: dict,
        ham: bool,
    ):
        super().__init__()

        self.save_hyperparameters()

        deqfunction = OneMatrixFunction(layers, ham)

        self.s_init = torch.nn.parameter.Parameter(
            torch.zeros(batch_size, sum(layers[1:])),
            # For now, we don't want to train s_init
            # (in fact, we can't with the functions in backward.py)
            requires_grad=False,
        )

        self.deq = DEQ(
            deqfunction,
            self.s_init,
            self.log,  # Lightning-defined logging callable
            **deq_kwargs,
        )

        self.lr = lr
        self.batch_size = batch_size
        self.y_dim = layers[-1]

    def forward(self, x):
        return self.deq(x)[:, -self.y_dim :]  # only return y-part of s

    def configure_optimizers(self):
        optimizer = Madam(self.parameters(), lr=self.lr, p_scale=1024.0, g_bound=3.0)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=9,  # use nr_of_epochs-1 to make sure last epoch has desired lr
        )
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        _, loss = self._get_pred_and_loss(train_batch)
        self.log("train_error", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        self._log_loss_and_acc(val_batch, "val_")

    def test_step(self, test_batch, batch_idx):
        self._log_loss_and_acc(test_batch, "test/")

    def _get_pred_and_loss(self, batch):
        x, y = batch
        x.requires_grad_(True)  # Autograd is not happy without this line
        y_pred = self.forward(x)
        x.requires_grad_(False)  # But we definitely don't want gradients on our inputs!

        loss = 0.5 * torch.nn.functional.mse_loss(y_pred, y, reduction="sum") / self.batch_size

        return y_pred, loss

    def _log_loss_and_acc(self, batch, mode: str):
        y_pred, loss = self._get_pred_and_loss(batch)

        class_pred = y_pred.argmax(1, keepdim=True)
        y = batch[1]
        class_target = y.argmax(1, keepdim=True)
        accuracy = (class_pred == class_target).float().mean()

        self.log_dict(
            {
                mode + "error": loss,
                mode + "acc": accuracy,
            }
        )
