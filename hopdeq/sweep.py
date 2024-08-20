import os
from functools import partial

import lightning
import torch
import torchvision
import wandb
from custom_callbacks import TimeToConvergenceCallback
from deq_modules.even_odd.litevenodddeq import LitEvenOddDEQ
from deq_modules.onematrix.litonematrixdeq import LitOneMatrixDEQ

os.environ["WANDB_DISABLE_CODE"] = "true"  # do not track code git diffs
# os.environ["WANDB_MODE"] = "disabled" # do not run Wandb


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
    dataset_train, batch_size=batch_size, shuffle=True, drop_last=True
)
loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, drop_last=True
)
# loader_test_shuffled = torch.utils.data.DataLoader(
#     dataset_test, batch_size=batch_size, shuffle=True, drop_last=True
# )


# 1: Define training function
def wandb_run_sweep():
    logger = lightning.pytorch.loggers.WandbLogger(
        project="HopDEQ", entity="hopfield", mode="online"
    )

    # User settings
    nr_of_layers = logger.experiment.config["nr_of_layers"]

    match nr_of_layers:
        case 3:
            lr = 0.01
            layers = [784, 1990, 10]
        case 5:
            lr = 0.005
            layers = [784, 1280, 510, 200, 10]
        case 7:
            lr = 0.005
            layers = [784, 1024, 512, 256, 128, 70, 10]

    damping_factor = 0.5 * (
        not logger.experiment.config["HAM"] and not logger.experiment.config["EvenOdd"]
    )  # only for Hop!

    # Deeper requires more time steps
    # Damped means slowed down in time, so more time step required too
    iter_scaling_factor = int((nr_of_layers // 2) / (1 - damping_factor))

    if logger.experiment.config["AA"]:
        deq_kwargs = dict(
            forward_kwargs=dict(
                solver="anderson",
                iter=40 * iter_scaling_factor,
            ),
            backward_kwargs=dict(
                solver="picard",
                iter=8 * iter_scaling_factor,
                method="backprop",
            ),
            damping_factor=damping_factor,
        )
    else:
        deq_kwargs = dict(
            forward_kwargs=dict(
                solver="picard",
                iter=40 * iter_scaling_factor,
            ),
            backward_kwargs=dict(
                solver="picard",
                iter=8 * iter_scaling_factor,
                method="backprop",
            ),
            damping_factor=damping_factor,
        )

    config = dict(
        batch_size=batch_size,
        lr=lr,  # with decay to lr/10
        layers=layers,
        deq_kwargs=deq_kwargs,
        ham=logger.experiment.config["HAM"],
    )

    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[
            TimeToConvergenceCallback(),
            lightning.pytorch.callbacks.LearningRateMonitor(),
        ],
        max_epochs=10,
        inference_mode=True,  # only during validation,
    )

    if logger.experiment.config["EvenOdd"]:
        hop = LitEvenOddDEQ(**config)
    else:
        hop = LitOneMatrixDEQ(**config)

    trainer.fit(hop, loader, loader_test)

    # Test results
    trainer.test(hop, loader_test, verbose=True)

    # Release all CUDA memory that you can
    hop = None
    trainer = None
    lightning.pytorch.utilities.memory.garbage_collection_cuda()


# 2: Define the search space
sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "val_error"},
    "parameters": {
        "experiment_count": {"values": list(range(5))},
        "nr_of_layers": {"values": [3, 5, 7]},
        "EvenOdd": {"values": [True, False]},
        "AA": {"values": [True, False]},
        "HAM": {"values": [True]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="HopDEQ")
wandb.agent(sweep_id, function=wandb_run_sweep)
