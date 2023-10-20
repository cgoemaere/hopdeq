import torch


class EvenOddFunctionHAM(torch.nn.Module):
    def __init__(self, layers, small_world_proba: float = 0.0):
        super().__init__()
        self.dim_x = layers[0]
        self.layers = [0] + layers[1:]  # remove input layer, but keep even and odd layer indices
        self.small_world_proba = small_world_proba

        self.b_even = torch.nn.parameter.Parameter(torch.zeros(1, sum(self.layers[::2])))
        self.b_odd = torch.nn.parameter.Parameter(torch.zeros(1, sum(self.layers[1::2])))

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

        # Use 1 large 'transition matrix'
        sum_dims_even = sum(self.layers[::2])
        sum_dims_odd = sum(self.layers[1::2])

        # Note that we are using row vectors as state => sum_dims_even is the first dimension
        # Register as parameters and buffer to get Lightning device to work correctly
        self.W_tensor = torch.nn.parameter.Parameter(torch.zeros(sum_dims_even, sum_dims_odd))
        self.register_buffer(
            "W_mask", torch.zeros(sum_dims_even, sum_dims_odd, requires_grad=False)
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
        Disclaimer: this function was not rigorously tested
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

    @torch.jit.export
    def preprocess_inputs(self, x):
        """
        Returns b + U @ ρ(x), where U = W0 in the original model.
        This represents the influence that x has on the first hidden layer
        (x has no influence on any other hidden layer).
        """
        return self.rho(x) @ self.U  # shape=(batch_size, self.layers[1])

    def forward(self, Ux, s):
        """
        Returns the new s = W.T ρ(W @ ρ(s) + b_odd + U @ ρ(x)) + b_even
        """
        s_odd = self.rho(s) @ self.W_tensor + self.b_odd
        s_odd[:, : self.layers[1]] += Ux
        return self.rho(s_odd) @ self.W_tensor.T + self.b_even
