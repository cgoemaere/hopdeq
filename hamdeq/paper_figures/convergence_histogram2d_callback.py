import sys

import torch
from lightning.pytorch.callbacks import Callback

sys.path.append("..")  # Adds higher directory to python modules path.
from fixed_point_utils import track_states


class ConvergenceHistogram2DCallback(Callback):
    def on_test_start(self, trainer, pl_module):
        if not (trainer.callbacks is self or trainer.callbacks[0] is self):
            raise NotImplementedError(
                "TimeToConvergenceCallback only works if it is the first callback of the trainer."
            )

        self.states = []
        # Turn on state tracking
        pl_module.deq.f_module.forward = track_states(pl_module.deq.f_module.forward, self.states)

        self.all_rel_res = []

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        # Clear list before next batch starts
        self.states.clear()

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        states = torch.stack(self.states, dim=0)
        rel_residual = (  # shape=(nr_of_time_steps-1, batch_size)
            torch.linalg.vector_norm(states.diff(dim=0), ord=2, dim=-1)
            / torch.linalg.vector_norm(states[1:], ord=2, dim=-1)
        ).float()  # no-op to get good lay-out

        rel_residual = rel_residual.log10().cpu()
        self.all_rel_res.append(rel_residual)

    def on_test_end(self, trainer, pl_module):
        # Turn off state tracking
        pl_module.deq.f_module.forward = track_states(
            pl_module.deq.f_module.forward, self.states, disable=True
        )
