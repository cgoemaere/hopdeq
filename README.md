
# hopdeq
Code for the paper "Accelerating Hopfield Network Dynamics: Beyond Synchronous Updates and Forward Euler", presented at the ML-DE workshop at ECAI 2024.

Link to paper: [arXiv](https://arxiv.org/abs/2311.15673v2)

**TL;DR:**  Hopfield networks can be made much faster by casting them as Deep Equilibrium Models and alternating optimization between the even and the odd layers.

> Code for the paper "Accelerating Hierarchical Associative Memory: A Deep Equilibrium Approach" (AMHN workshop, NeurIPS 2023) can be found in a separate branch. The master branch includes both HAM and CHN models and uses a few tricks for training larger architectures. Other than that, little has changed.

## Why this repo?
Besides reproducibility, this repo is intended to provide the community with an easy-to-use, minimalistic DEQ framework, applied but not limited to Hopfield networks.

**Note**: concurrently to this work, the original DEQ research team released a new [DEQ framework](https://github.com/locuslab/torchdeq) that is way more extensive than this repo, and just as user-friendly. If you're looking for a good DEQ repo to work with, you should probably take a look ;)

## Repo structure
- ***deq_core***: a minimalistic reimplementation of a generic DEQ framework, based on the [original DEQ repo](https://github.com/locuslab/deq/tree/master) and the [DEQ paper](https://arxiv.org/abs/1909.01377)
- ***deq_modules***: [Lightning](https://lightning.ai/) implementations of the Hopfield networks in the paper (OneMatrix = Eqs. 5-6; EvenOdd = Eqs. 7 & 10)
- ***custom_callbacks***: custom Lightning callbacks, to track the convergence of DEQs
- ***paper_figures***: the code used to generate Figs. 3-4 in Appendix G (note: these files assume you already ran `sweep.py`)
- `sweep.py`: launches a [wandb sweep](https://docs.wandb.ai/guides/sweeps) that trains 5 runs of every model kind in Table 1. While a sweep is technically meant for hyperparameter search, we do not use it as such here.
- `madam.py`: copy of the original [*madam.py*](https://github.com/jxbz/madam/blob/master/pytorch/optim/madam.py), with proper referencing.  
  **Important**: Unlike the other files in this repo, it comes with a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license, as detailed in the [original Madam repo](https://github.com/jxbz/madam).

## How to make your own DEQ
First, you must define your implicit function $f$ for which you want to find the fixed point $\mathbf{z}^\* = f(\mathbf{z}^\*, \mathbf{x})$.

As an example, we show the structure for a standard DEQ of the form $f(\mathbf{z}^\*, \mathbf{x}) = \sigma(\mathbf{Wz}^\*+\mathbf{Ux}+\mathbf{b})$:
~~~python
import torch

class MyDEQFunction(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        
        # Initialize parameters and store args
        ...

    def preprocess_inputs(self, x):
        """
        As x is kept constant, it is often 
        possible to process it once beforehand,
        and use this preprocessed input in the
        iterative process to save some computation.
        """
        return x @ self.U + self.b

    def forward(self, Ux, z):
        """
        Returns f(z, x)
        Receives preprocessed input
        """
        return self.sigma(z @ self.W + Ux)
~~~

Next, you can define your DEQ as follows:
~~~python
from hopdeq import DEQ
from mydeqfunction import MyDEQFunction

# Initial state (often chosen to be zero)
z_init = torch.zeros(batch_size, dims, requires_grad=False)
# We can't train z_init in the DEQ framework (at least not without full backprop)

deq = DEQ(
    f_module = MyDEQFunction(...),
    z_init = z_init,
    logger = f_log,  # optional logger function (e.g., self.log in Lightning)
    **deq_kwargs,
)
~~~

As `deq_kwargs`, the following structure is expected:
~~~python
deq_kwargs  =  dict(
	forward_kwargs  =  dict(
		solver  =  [str],
		iter  =  [int],
	),
	backward_kwargs  =  dict(
		solver  =  [str],
		iter  =  [int],
		method  =  [str],
	),
	damping_factor  =  [float],
),
~~~
* `solver`: the solver used for finding $\mathbf{z}^*$ (forward) or the adjoint gradient (backward) (options: `"picard"` & `"anderson"`)
* `iter`: the number of iterations the solver gets (for now, there is no early stopping)
* `method`: how you want to calculate the gradients (options: `"backprop"` (for recurrent backprop)  & `"full_adjoint"`)
* `damping_factor`: number between 0 and 1 to indicate how much damping should be used (0 = no damping).  
  **Note**: in general DEQs, you'll probably need to add some damping for the Picard solver to converge (but not if you're using HAMs...)

**And that's all!**  
You can now treat `deq` as if it was a regular `torch.nn.Module`, providing it with inputs and getting gradients from backpropping over it. In fact, you can even put it in a `torch.nn.Sequential` if you want!

## Installation
To install the repo with the exact dependencies from the paper, run this in the command line:
```
git clone https://github.com/cgoemaere/hopdeq
cd hopdeq/
conda create --name hopdeq_test_env --file requirements.txt -c conda-forge -c pytorch
conda activate hopdeq_test_env
python3 -c "from hopdeq import DEQ; print(DEQ)" #just a little test; should print "<class 'hopdeq.deq_core.deq.DEQ'>"
```

Next, to get the models training, run `sweep.py` as follows:
```
cd hopdeq/
python3 sweep.py #prepend with 'nohup' to run the sweep in the background
```
## Other DEQ frameworks (for reference)
* PyTorch
	* [locuslab/torchdeq: Modern Fixed Point Systems using Pytorch (github.com)](https://github.com/locuslab/torchdeq): very nicely structured DEQ framework that came out concurrently to this repo. You should probably check it out!
	* [deq-flow/code.v.2.0/core/deq at main · locuslab/deq-flow](https://github.com/locuslab/deq-flow/tree/main/code.v.2.0/core/deq): DEQ v2.0
	* [locuslab/deq: [NeurIPS'19] Deep Equilibrium Models](https://github.com/locuslab/deq): original DEQ repo, but not very straightforward to work with
* JAX
	* [akbir/deq-jax: [NeurIPS'19] Deep Equilibrium Models Jax Implementation](https://github.com/akbir/deq-jax)
	* [jackd/deqx: Deep Equilibrium Models in jax](https://github.com/jackd/deqx)
	* [google/jaxopt: Hardware accelerated, batchable and differentiable optimizers in JAX.](https://github.com/google/jaxopt)

## Citation
If you found this repo useful, please consider citing [the paper](https://openreview.net/forum?id=Vmndp6HnfR):
```bibtex
@inproceedings{goemaere2024hopdeq,
  title={Accelerating Hopfield Network Dynamics: Beyond Synchronous Updates and Forward Euler},
  author={C{\'e}dric Goemaere and Johannes Deleu and Thomas Demeester},
  booktitle = {ML-DE Workshop at ECAI 2024},
  volume = {255},
  year={2024},
}
```
