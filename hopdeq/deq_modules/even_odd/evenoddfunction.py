import math

import torch


class EvenOddFunction(torch.nn.Module):
    def __init__(self, layers: list, ham: bool = False, small_world_proba: float = 0.0):
        super().__init__()
        self.dim_x = layers[0]
        self.layers = [0] + layers[1:]  # remove input layer, but keep even and odd layer indices
        self.small_world_proba = small_world_proba

        self.forward = self.forward_ham if ham else self.forward_hop

        self.sum_dims_even = sum(self.layers[::2])
        self.sum_dims_odd = sum(self.layers[1::2])

        self.b_even = torch.nn.parameter.Parameter(torch.zeros(1, self.sum_dims_even))
        self.b_odd = torch.nn.parameter.Parameter(torch.zeros(1, self.sum_dims_odd))
        torch.nn.init.normal_(self.b_even, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.b_odd, mean=0.0, std=0.01)

        self.U = torch.nn.parameter.Parameter(torch.zeros(self.dim_x, self.layers[1]))
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
        ##################
        # Regular layers #
        ##################
        # this makes sure that input and output layer are both even
        assert len(self.layers) % 2 == 1, "Number of layers must be odd!"

        # Note that we are using row vectors as state => sum_dims_even is the first dimension
        # Register as parameters and buffer to get Lightning device to work correctly
        self.W_tensor = torch.nn.parameter.Parameter(
            torch.zeros(self.sum_dims_even, self.sum_dims_odd)
        )
        self.register_buffer(
            "W_mask", torch.zeros(self.sum_dims_even, self.sum_dims_odd, requires_grad=False)
        )

        for row_slice, col_slice in self.layer_slices():
            ## Initialize transition matrices with proper in_features dimension##
            # Since we have an auto-encoder, we need to make sure that both the forward and
            # the backward pass are stabilized.
            # We can do this using the symmetric xavier init.
            torch.nn.init.xavier_uniform_(self.W_tensor[row_slice, col_slice])

            self.W_mask[row_slice, col_slice].fill_(1.0)

        # Optional: check if we have the correct staircase pattern
        # Note that this function does not work as expected for 3 layers
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

        # Self-connections are impossible in even-odd, so we don't have to worry about that

        # Generate random uniform numbers between 0 and 1, in the shape of W_tensor
        random_numbers = torch.rand(self.sum_dims_even, self.sum_dims_odd)

        # Select only 'lucky' few and make these connections in the tensor
        selection = random_numbers < self.small_world_proba
        self.W_mask[selection].fill_(1.0)  # W_mask might already be 1., but that's not a problem
        # value taken from small-world paper, adapted to non-square W
        sum_dims = (self.sum_dims_even + self.sum_dims_odd) / 2
        self.W_tensor[selection].uniform_(-math.sqrt(3 / sum_dims), math.sqrt(3 / sum_dims))

        #################
        # Final masking #
        #################
        self.W_tensor *= self.W_mask

    def layer_slices(self):
        """
        Returns the slices for every layer, assuming a sequentially connected Hopfield network.

        If the original Hopfield matrix looks like this:
        [ 0   W0  0   0   0  ]
        [W0.T 0   W1  0   0  ]
        [ 0  W1.T 0   W2  0  ]
        [ 0   0  W2.T 0   W3 ]
        [ 0   0   0  W3.T 0  ]

        Then the transformed one looks like this:
        [ 0   0   0 | W0  0  ]
        [ 0   0   0 |W1.T W2 ]
        [ 0   0   0 | 0  W3.T]
        ----------------------
        [W0.T W1  0 | 0   0  ]
        [ 0  W2.T W3| 0   0  ]

        The submatrix on the bottom left is what we use as W_tensor (=staircase-like)
        However, since s_even is modelled as a row vector, we must transpose W_tensor.

        Usage:
        for i, layer_slice in enumerate(self.layer_slices()):
            print('Layer', i, 'looks like this:', self.W_tensor[layer_slice])
        """
        # No need for additional [0]+ in even layers, as this is already done in __init__
        layer_indices_even = torch.cumsum(torch.tensor(self.layers[::2]), 0)
        layer_indices_odd = torch.cumsum(torch.tensor([0] + self.layers[1::2]), 0)

        # Note that we are using row vectors as state
        # This means that we are working with the transposed matrix
        # Therefore, layer_indices_even is the row dimension
        row_counter = iter(zip(layer_indices_even[:-1], layer_indices_even[1:]))
        col_counter = iter(zip(layer_indices_odd[:-1], layer_indices_odd[1:]))

        # The first layer W0 is always in the top left corner
        i, j = next(row_counter)
        k, l = next(col_counter)

        # However, since we are explicitly modelling the x dependence with U, we are actually
        # starting at W1.T. This means that we must initially move right, instead of down.
        update_row_indices = False

        while True:
            yield (slice(i, j), slice(k, l))

            try:
                if update_row_indices:
                    i, j = next(row_counter)  # Drop one row down
                else:
                    k, l = next(col_counter)  # Move one col to the right
            except StopIteration:
                break  # We reached the end of our staircase

            update_row_indices = not update_row_indices

        # # Extra skip connections (just to try)
        # # Note that these have to be even-odd splittable
        # print("Using extra skip connections")
        # if len(self.layers) >= 4:
        #     yield (
        #         slice(layer_indices_even[1], layer_indices_even[2]),
        #         slice(layer_indices_odd[0], layer_indices_odd[1]),
        #     )  # layer 1 -> 4

        # if len(self.layers) >= 6:
        #     yield (
        #         slice(layer_indices_even[2], layer_indices_even[3]),
        #         slice(layer_indices_odd[0], layer_indices_odd[1]),
        #     )  # layer 1 -> 6

    @torch.no_grad()
    def fix_W_tensor_grad_format(self, W_tensor_grad):
        # Buffers and JIT don't go well together, so we need to fix the device
        # (see https://github.com/pytorch/pytorch/issues/43815
        # and https://github.com/pytorch/pytorch/issues/28280)
        self.W_mask = self.W_mask.to(W_tensor_grad.device)

        # Make W_grad masked (no need for symmetry here)
        return self.W_mask * W_tensor_grad

    @torch.no_grad()
    def from_OneMatrix(self, om):
        """
        Creates an EvenOddFunction that is equivalent to the given OneMatrixFunction

        Usage:
        eo = EvenOddFunction(config['onematrix_dims'])
        hop.deq.f_module = eo.from_OneMatrix(hop.deq.f_module)
        hop.deq.z_init = torch.nn.parameter.Parameter(
            hop.deq.z_init[:, :sum(eo.layers[::2])], requires_grad=False
            )
        """
        transpose = True
        for layer_slice, om_layer_slice in zip(self.layer_slices(), om.layer_slices()):
            W_i = om.W_tensor[om_layer_slice].detach().clone()
            self.W_tensor[layer_slice] = W_i.T if transpose else W_i
            transpose = not transpose

        self.U = torch.nn.parameter.Parameter(om.U.detach().clone())

        b_layers = torch.split(om.b.detach().clone(), self.layers, dim=1)
        self.b_even = torch.nn.parameter.Parameter(torch.cat(b_layers[::2], dim=1))
        self.b_odd = torch.nn.parameter.Parameter(torch.cat(b_layers[1::2], dim=1))

        return self

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
        return self.rho(x) @ self.U  # shape=(batch_size, self.layers[1])

    def forward_hop(self, Ux, s):
        """
        Returns the new s = W.T ρ(W @ ρ(s) + b_odd + U @ ρ(x)) + b_even
        """
        s_even, s_odd = torch.split(s, [self.sum_dims_even, self.sum_dims_odd], dim=-1)

        # First, update s_odd
        C_odd = self.rho(s_even) @ self.W_tensor + self.b_odd
        C_odd[:, : self.layers[1]] += Ux

        # Damped Picard iteration to get to local energy minimum
        for _ in range(10):
            # Problem: we need to add damping, but we can't access the damping parameter from here
            # Hotfix: let's just say that damping=0.5 in Hop's (which is the value we use in the paper)
            s_odd = 0.5 * s_odd + 0.5 * self.rhod(s_odd) * C_odd

        # Then, based on damped_s_odd, update s_even
        C_even = self.rho(s_odd) @ self.W_tensor.T + self.b_even

        # Damped Picard iteration to get to local energy minimum
        for _ in range(10):
            s_even = 0.5 * s_even + 0.5 * self.rhod(s_even) * C_even

        return torch.cat([s_even, s_odd], dim=-1)

    def forward_ham(self, Ux, s):
        """
        Returns the new s = W.T ρ(W @ ρ(s) + b_odd + U @ ρ(x)) + b_even
        """
        s_odd = self.rho(s) @ self.W_tensor + self.b_odd
        s_odd[:, : self.layers[1]] += Ux
        return self.rho(s_odd) @ self.W_tensor.T + self.b_even
