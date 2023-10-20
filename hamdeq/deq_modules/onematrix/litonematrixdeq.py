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
        onematrix_dims: List[int],
        deq_kwargs: dict,
        ham: bool,
    ):
        super().__init__()

        self.save_hyperparameters()

        deqfunction = OneMatrixFunction(onematrix_dims, ham)

        self.s_init = torch.nn.parameter.Parameter(
            torch.zeros(batch_size, sum(onematrix_dims[1:])),
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
        self.y_dim = onematrix_dims[-1]

    def forward(self, x):
        return self.deq(x)[:, -self.y_dim :]  # only return y-part of s

    def configure_optimizers(self):
        optimizer = Madam(self.parameters(), lr=self.lr, p_scale=1024.0, g_bound=10.0)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        _, loss = self._get_pred_and_loss(train_batch)
        self.log("train_error", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        y_pred, loss = self._get_pred_and_loss(val_batch)

        class_pred = y_pred.argmax(1, keepdim=True)
        y = val_batch[1]
        class_target = y.argmax(1, keepdim=True)
        accuracy = (class_pred == class_target).float().mean()

        self.log_dict({"test_error": loss, "test_acc": accuracy})

    def test_step(self, val_batch, batch_idx):
        self.validation_step(val_batch, batch_idx)

    def _get_pred_and_loss(self, batch):
        x, y = batch
        x.requires_grad_(True)
        y_pred = self.forward(x)
        x.requires_grad_(False)

        loss = 0.5 * torch.nn.functional.mse_loss(y_pred, y, reduction="sum") / self.batch_size

        return y_pred, loss
