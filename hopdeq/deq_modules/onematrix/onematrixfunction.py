import math

import torch


class OneMatrixFunction(torch.nn.Module):
    def __init__(self, layers: list, ham: bool = False, small_world_proba: float = 0.0):
        super().__init__()
        self.dim_x = layers[0]
        self.layers = layers[1:]
        self.small_world_proba = small_world_proba

        self.forward = self.forward_ham if ham else self.forward_hop

        self.b = torch.nn.parameter.Parameter(torch.zeros(1, sum(self.layers)))
        torch.nn.init.normal_(self.b, mean=0.0, std=0.01)

        self.U = torch.nn.parameter.Parameter(torch.zeros(self.dim_x, self.layers[0]))
        # While U is technically unidirectional, we also use Xavier initialization here
        # for consistency with the other weights.
        torch.nn.init.xavier_uniform_(self.U)

        self.make_W_tensor_and_W_mask()

        # Perform fix_W_tensor_grad_format via a backward hook
        # This does the same as the original code, but now integrated into autograd
        # (like in the original code, fix_W_tensor_grad_format is not used in EqProp)
        self.W_tensor.register_hook(self.fix_W_tensor_grad_format)

    @torch.no_grad()
    def make_W_tensor_and_W_mask(self):
        # Based on https://github.com/jgammell/equilibrium_propagation/blob/b8d2ec6f79ca1bbaa867b3217d18ccad5bcd3308/framework/eqp.py
        ##################
        # Regular layers #
        ##################

        # Use 1 large 'transition matrix'
        # Register as parameters and buffer to get Lightning device to work correctly
        sum_dims = sum(self.layers)
        self.W_tensor = torch.nn.parameter.Parameter(torch.zeros(sum_dims, sum_dims))
        self.register_buffer("W_mask", torch.zeros(sum_dims, sum_dims, requires_grad=False))

        # Initialize layers
        for row_slice, col_slice in self.layer_slices():
            self.W_mask[row_slice, col_slice].fill_(1.0)
            self.W_mask[col_slice, row_slice].fill_(1.0)

            ## Initialize transition matrices with proper in_features dimension##
            # Since every W works bidirectionally, we need to use Xavier initialization,
            # which is symmetrical.
            torch.nn.init.xavier_uniform_(self.W_tensor[row_slice, col_slice])

            # Here, you can choose whether to make this symmetrical or not
            # (this line is the only thing that decides this)
            self.W_tensor[col_slice, row_slice] = self.W_tensor[row_slice, col_slice].T

        # Optional: check if we have the correct pattern (before applying small world model)
        # print(torch.unique_consecutive(torch.unique_consecutive(self.W_mask, dim=1), dim=0))

        #####################
        # Small-world-model #
        #####################
        # DIFFERENCE WITH NAIVE MODEL: here, it is impossible to get small-world connections
        # from x to a hidden layer. This will probably have a big impact on the gradients.

        # Paper: https://www.frontiersin.org/articles/10.3389/fncom.2021.627357/full

        # Optional (see footnote 2 of small-world paper): delete existing connections
        # If sum_dims is large enough, the approach below will result in an amount of lost
        # connections that will be very close to the amount of connections made, though there are
        # no guarantees. But this approach is a lot more computationally efficient.
        # self.W_mask *= 1.-self.small_world_proba
        # self.W_mask = torch.bernoulli(self.W_mask) #Prob[1 -> 0] = small_world_proba
        # No need for explicit masking, we can do this at 'Final masking'

        # Avoid self connections for stability
        no_self_connections_mask = torch.block_diag(
            *[torch.ones(i, i, dtype=bool) for i in self.layers]
        )
        # Generate random uniform numbers between 0 and 1, in the shape of W_tensor
        random_numbers = torch.rand(sum_dims, sum_dims).masked_fill_(
            no_self_connections_mask, value=2.0
        )  # mask with 2. to avoid selection

        # Select only 'lucky' few and make these connections in the tensor
        selection = random_numbers < self.small_world_proba
        self.W_mask[selection].fill_(1.0)  # W_mask might already be 1., but that's not a problem
        # value taken from small-world paper
        self.W_tensor[selection].uniform_(-math.sqrt(3 / sum_dims), math.sqrt(3 / sum_dims))

        #################
        # Final masking #
        #################
        self.W_tensor *= self.W_mask

    def layer_slices(self):
        """
        Returns the slices for every layer, assuming a sequentially connected Hopfield network.

        The Hopfield matrix looks like this:
        [ 0   W0  0   0   0 ]
        [W0.T 0   W1  0   0 ]
        [ 0  W1.T 0   W2  0 ]
        [ 0   0  W2.T 0   W3]
        [ 0   0   0  W3.T 0 ]

        Usage:
        for i, layer_slice in enumerate(self.layer_slices()):
            print('Layer', i, 'looks like this:', self.W_tensor[layer_slice])
        """
        layer_indices = torch.cumsum(torch.tensor([0] + self.layers), 0)
        for i, j, k in zip(layer_indices[:-2], layer_indices[1:-1], layer_indices[2:]):
            yield (slice(i, j), slice(j, k))

        # # Extra skip connections (just to try)
        # # Note that these have to be even-odd splittable
        # print("Using extra skip connections")
        # if len(layer_indices) >= 5:
        #     yield (
        #         slice(layer_indices[0], layer_indices[1]),
        #         slice(layer_indices[3], layer_indices[4]),
        #     )  # layer 1 -> 4

        # if len(layer_indices) >= 7:
        #     yield (
        #         slice(layer_indices[0], layer_indices[1]),
        #         slice(layer_indices[5], layer_indices[6]),
        #     )  # layer 1 -> 6

    def fix_W_tensor_grad_format(self, W_tensor_grad):
        # Buffers and JIT don't go well together, so we need to fix the device
        # (see https://github.com/pytorch/pytorch/issues/43815
        # and https://github.com/pytorch/pytorch/issues/28280)
        self.W_mask = self.W_mask.to(W_tensor_grad.device)

        # Make W_grad symmetric and masked
        return self.W_mask * (W_tensor_grad + W_tensor_grad.T) / 2

    def rho(self, x):
        return torch.sigmoid(4 * (x - 0.5))

    def rhod(self, x):
        return 4 * self.rho(x) * (1 - self.rho(x))

    @torch.jit.export
    def preprocess_inputs(self, x):
        """
        Returns U @ ρ(x), where U = W0 in the original model.
        This represents the influence that x has on the first hidden layer
        (x has no influence on any other hidden layer).
        """
        return self.rho(x) @ self.U  # shape=(batch_size, self.layers[0])

    def forward_hop(self, Ux, s):
        """
        Returns the new s = ρ'(s) * (W @ ρ(s) + b + U @ ρ(x))
        """
        new_s = self.forward_ham(Ux, s)
        new_s *= self.rhod(s)
        return new_s

    def forward_ham(self, Ux, s):
        """
        Returns the new s = W @ ρ(s) + b + U @ ρ(x)
        """
        new_s = self.rho(s) @ self.W_tensor + self.b
        new_s[:, : self.layers[0]] += Ux
        return new_s
