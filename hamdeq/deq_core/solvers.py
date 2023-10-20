from collections import deque
from typing import Callable, Optional

import torch


def picard(
    f: Callable, z_init: torch.Tensor, iter: int, logger: Optional[Callable] = None
) -> torch.Tensor:
    """
    Picard iteration
    Keep in mind that you will likely need to damp f for this to converge.
    """
    z = z_init

    for _ in range(iter - 1):
        z = f(z)

    z_star = f(z)

    # Log final convergence
    if logger is not None:
        mean_rel_residual = (
            torch.linalg.vector_norm(z_star - z, ord=2, dim=-1)
            / torch.linalg.vector_norm(z, ord=2, dim=-1)
        ).mean()
        logger("rel_residual_1step", mean_rel_residual)

    return z_star


def anderson(
    f: Callable, z_init: torch.Tensor, iter: int, m: int = 4, logger: Optional[Callable] = None
) -> torch.Tensor:
    """
    Anderson acceleration, based on Algorithm AA (form 1.1) of Walker & Ni,
    "ANDERSON ACCELERATION FOR FIXED-POINT ITERATIONS", 2011
    Online PDF: https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf
    We use the same notation as the algorithm, even though it is inconsistent with the
    rest of our Module.

    The minimization problem can be solved with a Lagrangian multiplier on the condition
    sum(α)=1. The problem becomes: min_α ||F @ α||^2 + λ * (sum(α)-1)
    We can then take the derivative of the whole system w.r.t α and λ, and set it to zero.
    We find that λ is simply a scaling factor, and can be taken into account post-rem.
    So, all we have to do, is solve for α: (F^T @ F) α = 1 and rescale α afterwards.

    This procedure can also be found in Eq. 15 in https://arxiv.org/pdf/1606.04133.pdf
    The Tikhonov regularization is equivalent to their Algorithm 2.

    As another reference, this procedure is also described by Anderson himself on
    p.218-219 of https://link.springer.com/article/10.1007/s11075-018-0549-4

    This function can be made more efficient by QR-factorization of Fk.
    One could use torch.linalg.qr(mode='r') together with torch.cholesky_solve(upper=True)
    to compute α directly from Fk. However, our timings suggest this is only faster
    when dim_z_even ~ 10.000x m. Alternatively, one could incrementally update R (e.g,
    see https://github.com/mpf/QRupdate.jl). However, this does not yet exist in PyTorch.
    """
    g = f  # use notation from the Algorithm description

    x0 = z_init
    x1 = g(x0)

    # Use length-limited deque for easy memory maintenance
    x_list = deque([x0, x1], maxlen=m)
    g_list = deque([x1, g(x1)], maxlen=m)

    for k in range(2, iter):
        mk = min(m, k)

        x_i = torch.stack(tuple(x_list), dim=-1)  # =batch_size x dim_z_even x m
        g_i = torch.stack(tuple(g_list), dim=-1)  # =batch_size x dim_z_even x m
        Fk = g_i - x_i

        # Solve min ||Fk α||_2 s.t. sum(α) = 1
        A = torch.bmm(Fk.transpose(1, 2), Fk)  # =batch_size x m x m
        # Make A non-singular by adding eps*torch.eye (Tikhonov regularization)
        A.diagonal(dim1=1, dim2=2).add_(1e-10)

        B = torch.ones(mk, device=z_init.device)  # will get broadcasted in torch.solve
        alpha = torch.linalg.solve(A, B)  # =batch_size x m
        alpha = alpha / alpha.sum(dim=1, keepdim=True)

        x_tilde = torch.einsum("bzm,bm->bz", g_i, alpha)

        # Safe-guarding #
        # Check whether AA is better than Picard. If not, we use Picard.
        # We estimate the Picard convergence by looking at the previous time step.
        with torch.no_grad():
            AA_rel_residual = (
                torch.linalg.vector_norm(g_list[-1] - x_tilde, ord=2, dim=-1)
                / torch.linalg.vector_norm(x_tilde, ord=2, dim=-1)
            ).unsqueeze(-1)
            Picard_rel_residual = (
                torch.linalg.vector_norm(g_list[-1] - x_list[-1], ord=2, dim=-1)
                / torch.linalg.vector_norm(x_list[-1], ord=2, dim=-1)
            ).unsqueeze(-1)
        final_x = torch.where(AA_rel_residual < Picard_rel_residual, x_tilde, g_i[:, :, -1])

        # We do not use damping, as this is equivalent to making f damped
        x_list.append(final_x)
        g_list.append(g(final_x))

    # Log final convergence
    if logger is not None:
        mean_rel_residual = (
            torch.linalg.vector_norm(g_list[-1] - x_list[-1], ord=2, dim=-1)
            / torch.linalg.vector_norm(x_list[-1], ord=2, dim=-1)
        ).mean()
        logger("rel_residual_1step", mean_rel_residual)

    return g_list[-1]
