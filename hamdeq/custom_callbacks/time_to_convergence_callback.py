import torch
from deq_core.utils.fixed_point_utils import track_states
from lightning.pytorch.callbacks import Callback


class TimeToConvergenceCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        if not (trainer.callbacks is self or trainer.callbacks[0] is self):
            raise NotImplementedError(
                "TimeToConvergenceCallback only works if it is the first callback of the trainer."
            )

        self.states = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Clear list before next batch starts
        self.states.clear()

        # Turn on state tracking
        pl_module.deq.f_module.forward = track_states(pl_module.deq.f_module.forward, self.states)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Calculate relative residual
        states = torch.stack(self.states, dim=0)
        rel_residual = (  # shape=(nr_of_time_steps-1, batch_size)
            torch.linalg.vector_norm(states.diff(dim=0), ord=2, dim=-1)
            / torch.linalg.vector_norm(states[1:], ord=2, dim=-1)
        ).float()  # no-op to get good lay-out

        # Clear list after it's been processed
        self.states.clear()

        # We define convergence as having a rel_residual below 1e-4
        # and track the first time this happens
        converged_indx = torch.argmax((rel_residual < 1e-4).float(), dim=0)

        # Careful: if a sample did not converge, converged_indx will now be 0
        did_not_converge = converged_indx == 0
        self.log("Did not converge", did_not_converge.int().sum())
        converged_indx[did_not_converge] = rel_residual.size(0)  # =max number of time steps

        # Calculate equivalent forward Euler ODE solver time
        converged_indx = converged_indx.float().mean().cpu()  # (averaged over the batch)
        step_size = 1.0 - pl_module.deq.config["damping_factor"]
        time_to_convergence = converged_indx * step_size
        self.log("Time to convergence", time_to_convergence)

        # Turn off state tracking
        pl_module.deq.f_module.forward = track_states(
            pl_module.deq.f_module.forward, self.states, disable=True
        )

    ### Test time code ###

    def on_test_start(self, trainer, pl_module):
        self.on_fit_start(trainer, pl_module)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        # Now, self.log does on_epoch=True accumulation automatically
        self.on_train_batch_end(trainer, pl_module, None, batch, batch_idx)
