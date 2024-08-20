from typing import List

import torch
from deq_core.deq import DEQ
from deq_modules.onematrix.litonematrixdeq import LitOneMatrixDEQ

from .evenoddfunction import EvenOddFunction


class LitEvenOddDEQ(LitOneMatrixDEQ):
    def __init__(
        self,
        batch_size: int,
        lr: float,
        layers: List[int],
        deq_kwargs: dict,
        ham: bool,
    ):
        super().__init__(batch_size, lr, layers, deq_kwargs, ham)

        deqfunction = EvenOddFunction(layers, ham)

        if ham:  # make s_init smaller than the one from OneMatrix
            self.s_init = torch.nn.parameter.Parameter(
                torch.zeros(batch_size, sum(layers[2::2])),
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
