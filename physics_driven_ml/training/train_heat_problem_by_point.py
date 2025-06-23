import os
import argparse
import functools

import torch
import torch.optim as optim
import torch.autograd as torch_ad

from tqdm.auto import tqdm, trange

from torch.utils.data import DataLoader

from firedrake import *
from firedrake_adjoint import *
from firedrake.ml.pytorch import torch_operator

from physics_driven_ml.dataset_processing import PointDataset
from physics_driven_ml.models import PointNN
from physics_driven_ml.utils import get_logger
from physics_driven_ml.evaluation import evaluate


def train(model, device, train_dl, dev_dl):
    """
    Train the model on a given dataset.
    """
    learning_rate = 5e-5
    epochs = 50
    device = device

    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    max_grad_norm = 1.0
    best_error = 0.

    # Training loop
    for epoch_num in trange(epochs):
        logger.info(f"Epoch num: {epoch_num}")

        model.train()

        total_loss = 0.0
        train_steps = len(train_dl)
        for step_num, batch in tqdm(enumerate(train_dl), total=train_steps):
            model.zero_grad()

            # separate input and target f out of the data tensor
            inputs = batch[:,:4]
            target_f = batch[:,4]

            # forward pass
            network_f = model(inputs)[:,0]

            # Define L2-loss using PyTorch: 0.5 * ||f - f_exact||^{2}_{L2}
            l2_loss = torch.nn.MSELoss()
            loss = l2_loss(network_f, target_f)

            # Backprop and perform Adam optimisation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimiser.step()

        print(f'Finished epoch {epoch_num}, latest loss {loss}')

    logger.info(f"Total loss: {total_loss/train_steps}")
        

    return model

if __name__ == "__main__":
    logger = get_logger("Training")

    data_dir = os.path.join("/Users/Jemma/Nell/code/physics-driven-ml/data/datasets/heat_problem_example_data")
    batch_size = 1
    device = "cpu"

    # Set the model
    model = PointNN()

    # -- Load dataset -- #

    # Load train dataset
    train_dataset = PointDataset(numpy_data=os.path.join(data_dir, "numpy_point_train_data.npy"),
                                 data_dir=data_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

     # Load test dataset
    test_dataset = PointDataset(numpy_data=os.path.join(data_dir, "numpy_point_test_data.npy"),
                                 data_dir=data_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set double precision to match types
    model.double()
    # Move model to device
    model.to(device)

    # -- Training -- #

    train(model, device=device, train_dl=train_dataloader, dev_dl=test_dataloader)