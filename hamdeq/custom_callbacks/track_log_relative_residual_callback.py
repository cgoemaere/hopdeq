import torch
from deq_core.utils.fixed_point_utils import track_states
from lightning.pytorch.callbacks import Callback


class TrackLogRelativeResidualCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        if not (trainer.callbacks is self or trainer.callbacks[0] is self):
            raise NotImplementedError(
                "TrackLogRelativeResidualCallback only works if it is the first callback of the trainer."
            )

        self.states = []
        # Turn on state tracking
        pl_module.deq.f_module.forward = track_states(pl_module.deq.f_module.forward, self.states)

        # Make list of all relative residuals
        self.all_rel_res = []

    def on_fit_end(self, trainer, pl_module):
        # Turn off state tracking
        pl_module.deq.f_module.forward = track_states(
            pl_module.deq.f_module.forward, self.states, disable=True
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Clear list before next batch starts
        self.states.clear()

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        states = torch.stack(self.states, dim=0)
        rel_residual = (  # shape=(nr_of_time_steps-1, batch_size)
            torch.linalg.vector_norm(states.diff(dim=0), ord=2, dim=-1)
            / torch.linalg.vector_norm(states[1:], ord=2, dim=-1)
        ).log10()

        self.all_rel_res.append(rel_residual.cpu())

    ### Test time code ###

    def on_test_start(self, trainer, pl_module):
        self.on_fit_start(trainer, pl_module)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.on_train_batch_end(trainer, pl_module, None, batch, batch_idx)

    def on_test_end(self, trainer, pl_module):
        self.on_fit_end(trainer, pl_module)
