from typing import List

import torch
from deq_core.deq import DEQ

from deq_modules.onematrix.litonematrixdeq import LitOneMatrixDEQ

from .eo_ham import EvenOddFunctionHAM


class LitEvenOddDEQ(LitOneMatrixDEQ):
    def __init__(
        self,
        batch_size: int,
        lr: float,
        onematrix_dims: List[int],
        deq_kwargs: dict,
        ham: bool,
    ):
        super().__init__(batch_size, lr, onematrix_dims, deq_kwargs, ham)

        if ham:
            deqfunction = EvenOddFunctionHAM(onematrix_dims)

            self.s_init = torch.nn.parameter.Parameter(
                torch.zeros(batch_size, sum(onematrix_dims[2::2])),
                # For now, we don't want to train s_init
                # (in fact, we can't with the functions in backward.py)
                requires_grad=False,
            )
        else:
            raise NotImplementedError("Only HAM even-odd splitting is implemented")

        self.deq = DEQ(
            deqfunction,
            self.s_init,
            self.log,  # Lightning-defined logging callable
            **deq_kwargs,
        )
