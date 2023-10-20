import gc
import os
from functools import partial

import lightning
import torch
import torchvision
import wandb
from custom_callbacks.time_to_convergence_callback import TimeToConvergenceCallback
from deq_modules.even_odd.litevenodddeq import LitEvenOddDEQ
from deq_modules.onematrix.litonematrixdeq import LitOneMatrixDEQ

os.environ["WANDB_DISABLE_CODE"] = "true"  # do not track code git diffs
# os.environ["WANDB_NOTEBOOK_NAME"] = "EqProp_training.ipynb"
# os.environ["WANDB_SHOW_RUN"] = "false"
# os.environ["WANDB_MODE"] = "disabled"


wandb.login()


# 0: load datasets


batch_size = 64

dataset_split = "mnist"
dataset = partial(torchvision.datasets.EMNIST, split=dataset_split)

data_train = dataset(
    root="../data",
    train=True,
    download=True,
)
# Put data on device (and implement transforms yourself)
dataset_train = torch.utils.data.TensorDataset(
    (data_train.data / 255).unsqueeze(1).flatten(start_dim=1).cuda(),
    torch.nn.functional.one_hot(data_train.targets).to(torch.float32).cuda(),
)

data_test = dataset(
    root="../data",
    train=False,
    download=True,
)
dataset_test = torch.utils.data.TensorDataset(
    (data_test.data / 255).unsqueeze(1).flatten(start_dim=1).cuda(),
    torch.nn.functional.one_hot(data_test.targets).to(torch.float32).cuda(),
)

loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=False, drop_last=True
)
loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, drop_last=True
)
loader_test_shuffled = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=True, drop_last=True
)


# 1: Define training function
def wandb_run_sweep():
    logger = lightning.pytorch.loggers.WandbLogger(
        project="HopDEQ", entity="hopfield", mode="online"
    )

    if logger.experiment.config["HAM"]:  # HAMs are stable enough for simple Picard iteration
        factor = 1

    if logger.experiment.config["AA"]:
        deq_kwargs = dict(
            forward_kwargs=dict(
                solver="anderson",
                iter=40 * factor,
            ),
            backward_kwargs=dict(
                solver="anderson",
                iter=8 * factor,
                method="full_adjoint",
            ),
            damping_factor=1.0 - 1.0 / factor,
        )
    else:
        deq_kwargs = dict(
            forward_kwargs=dict(
                solver="picard",
                iter=40 * factor,
            ),
            backward_kwargs=dict(
                solver="picard",
                iter=8 * factor,
                method="full_adjoint",
            ),
            damping_factor=1.0 - 1.0 / factor,
        )

    config = dict(
        batch_size=batch_size,
        lr=0.01,
        onematrix_dims=[784, 512, 10],
        deq_kwargs=deq_kwargs,
        ham=logger.experiment.config["HAM"],
    )

    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[
            TimeToConvergenceCallback(),
        ],
        max_epochs=10,
    )

    if logger.experiment.config["EvenOdd"]:
        hop = LitEvenOddDEQ(**config)
    else:
        hop = LitOneMatrixDEQ(**config)

    trainer.fit(hop, loader, loader_test)

    # Release all CUDA memory that you can
    hop = None
    trainer = None
    gc.collect()
    torch.cuda.empty_cache()


# 2: Define the search space
sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "test_error"},
    "parameters": {
        "experiment_count": {"values": list(range(5))},
        "EvenOdd": {"values": [False, True]},
        "AA": {"values": [False, True]},
        "HAM": {"values": [True]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="HopDEQ")
wandb.agent(sweep_id, function=wandb_run_sweep)
