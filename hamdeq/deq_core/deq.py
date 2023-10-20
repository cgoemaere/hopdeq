from typing import Callable

import torch

from . import backward, solvers
from .utils.fixed_point_utils import damped
from .utils.import_utils import get_function_from_package


class DEQ(torch.nn.Module):
    def __init__(
        self,
        f_module: torch.nn.Module,  # nn.Module representing the fixed point function
        z_init: torch.Tensor,  # starting point for fixed point iteration
        logger: Callable,  # logging function to call in solver to track convergence
        forward_kwargs: dict = {
            "solver": "anderson",
            "iter": 40,
        },
        backward_kwargs: dict = {
            "solver": "anderson",
            "iter": 8,
            "method": "backprop",
        },
        damping_factor: float = 0.0,  # how much damping do we want?
    ):
        super().__init__()

        self.config = {
            "forward_kwargs": forward_kwargs,
            "backward_kwargs": backward_kwargs,
            "damping_factor": damping_factor,
        }  # store input arguments for later access

        # Make sure we're not changing any arguments (since we use .pop() later on)
        forward_kwargs = forward_kwargs.copy()
        backward_kwargs = backward_kwargs.copy()

        # Add logger to solver kwargs
        forward_kwargs["logger"] = logger
        backward_kwargs["logger"] = logger

        self.f_module = f_module
        self.z_init = z_init

        if damping_factor:
            self.f_module.forward = damped(self.f_module.forward, beta=1.0 - damping_factor)

        f_solver: Callable = get_function_from_package(
            solvers, forward_kwargs.pop("solver"), forward_kwargs
        )

        b_solver: Callable = get_function_from_package(solvers, backward_kwargs["solver"])
        backward_kwargs["solver"] = b_solver  # Set solver as fixed kwarg for backward_method
        b_method: Callable = get_function_from_package(
            backward, backward_kwargs.pop("method"), backward_kwargs
        )

        class AutogradDEQ(torch.autograd.Function):
            # Note that all these staticmethods always run in no_grad mode
            @staticmethod
            def forward(x):
                Ux = self.f_module.preprocess_inputs(x)

                # Preload input to get a pure function of z to solve for z*
                def f_pure(z, Ux=Ux):
                    return self.f_module(Ux, z)

                z_star = f_solver(f_pure, self.z_init)

                self.last_z_star = z_star  # store for later use in callbacks
                return z_star

            @staticmethod
            def setup_context(ctx, inputs, z_star):
                (x,) = inputs
                ctx.save_for_backward(x, z_star)

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, grad_output):
                # Retrieve stored tensors
                x, z_star = ctx.saved_tensors

                grads = b_method(grad_output, self.f_module, x, z_star)
                return grads

        # Set forward method to be our autograd module
        self.forward: Callable = AutogradDEQ.apply
