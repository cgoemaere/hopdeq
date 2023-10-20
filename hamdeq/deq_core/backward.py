from typing import Any, Callable, Tuple

import torch


def backprop(
    grad_output: torch.Tensor,
    f: torch.nn.Module,
    x: torch.Tensor,
    z_free: torch.Tensor,
    solver: Callable,
    **solver_kwargs: Any
) -> Tuple[torch.Tensor]:
    """
    Calculates the grad using Recurrent Backpropagation.
    (see Related Work in https://proceedings.mlr.press/v80/liao18c/liao18c.pdf)

    This is actually the implicit function theorem, calculated by going through
    consecutive function calls instead of making use of the function's vjp.
    Note that function.backward(grad) is equivalent to vjp(function)(grad)
    So, in essence, this function solves full_adjoint's f_backward in a less
    memory-efficient manner, and using a hardcoded Picard solver. After all, we are
    doing function call after function call, which is vjp after vjp in the backward
    direction (aka Picard solver).

    This backprop implementation is provably equivalent to EqProp, and therefore
    we use the terminology of EqProp (z_free and z_clamped).
    """

    with torch.enable_grad():
        # Activate gradient tracking, but avoid backpropping beyond the input x
        x = x.detach().requires_grad_(x.requires_grad)

        # Calculate Ux
        Ux = f.preprocess_inputs(x)

        # Create pure function of z to solve for z*
        def f_pure(z, Ux=Ux):
            return f(Ux, z)

        # Avoid infinite recursion loop of backpropping again and again through the DEQ
        z_free = z_free.detach()

        # Solve DEQ for a few more steps and perform backprop
        z_clamped = solver(f_pure, z_free, **solver_kwargs)

        #######################
        ### Calculate grads ###
        #######################
        z_clamped.backward(grad_output)

        grads = (x.grad,)

        # Reset grads, so that everything seems fine from the outside
        x.grad = None
        # Note that we do not need to reset the grads from f
        # They are exactly what they are supposed to be right now!

        return grads


def full_adjoint(
    grad_output: torch.Tensor,
    f: torch.nn.Module,
    x: torch.Tensor,
    z_free: torch.Tensor,
    solver: Callable,
    **solver_kwargs: Any
) -> Tuple[torch.Tensor]:
    """
    Calculates the gradients using the full adjoint method.
    To make the programming easier, we make use of the automatic differentiation.
    Crucially, we only backprop through a single layer.
    """
    # Calculate df/d(.), part 1: the input preprocessing calculation
    with torch.enable_grad():
        # Activate gradient tracking, but avoid backpropping beyond the input x
        x = x.detach().requires_grad_(x.requires_grad)

        # Calculate Ux
        Ux = f.preprocess_inputs(x)

    # Create pure function to calculate Jz* (with J the Jacobian of the function)
    def f_pure(z, Ux=Ux.detach()):
        return f(Ux, z)

    # Solve DEQ for a few more steps
    # We do iter-1 steps, because we do a Picard iteration after this line
    # This is not exactly the same as doing one more accelerated step, but since
    # we expect z_free to already be close to z*, the difference should be minimal.
    solver_kwargs["iter"] -= 1
    z_clamped = solver(f_pure, z_free, **solver_kwargs)
    solver_kwargs["iter"] += 1

    # Track a single function call so that we can use automatic differentiation
    with torch.enable_grad():
        z_clamped = f(Ux, z_clamped)

    # Since z is a row vector, we do z*J instead of Jz*
    # This does the exact same operations as the original deq code by locuslab,
    # (https://github.com/locuslab/deq/blob/1fb7059d6d89bb26d16da80ab9489dcc73fc5472/DEQ-Sequence/models/deq_transformer.py#L377)
    # but uses the built-in vjp function to obscure the hard-to-read autograd.grad call.
    _, calculate_vjp = torch.func.vjp(f_pure, z_clamped)

    # Invert the Jacobian using a fixed point iteration
    def f_backward(z):
        # We want to find z* = Jz* + g
        # g  = dL/dz* (w.r.t Appendix A of https://arxiv.org/pdf/1909.01377.pdf)
        return calculate_vjp(z)[0] + grad_output

    transformed_grad_output = solver(f_backward, torch.zeros_like(z_clamped), **solver_kwargs)

    # Calculate df/d(.), part 2: the actual function
    # We also backprop through Ux here.
    with torch.enable_grad():
        z_clamped.backward(transformed_grad_output)

        grads = (x.grad,)

        # Reset grads, so that everything seems fine from the outside
        x.grad = None
        # Note that we do not need to reset the grads from f
        # They are exactly what they are supposed to be right now!

        return grads
