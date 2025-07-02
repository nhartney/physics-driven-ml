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
from physics_driven_ml.evaluation import evaluate_by_point


def train(model, device, train_dl, dev_dl):
    """
    Train the model on a given dataset.
    """
    learning_rate = 5e-5
    epochs = 20
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
            inputs = batch[:, 1:5]
            target_f = batch[:,0]

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

        # Evaluate this version of the model on the test set
        error = evaluate_by_point(model, dev_dl, disable_tqdm=True)
        logger.info(f"L2 error from this model, evaluated on the test set: {error}")

        # Save best-performing model
        if error < best_error or epoch_num == 0:
            best_error = error
            # Create directory for trained models
            name_dir = f"heat_problem_by_point_epoch-{epoch_num}-error_{best_error:.5f}"
            model_dir = "/Users/Jemma/Nell/code/physics-driven-ml/data/saved_models"
            model_dir = os.path.join(model_dir, "heat_problem_by_point", name_dir)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            # Save model
            logger.info(f"Saving model checkpoint to {model_dir}\n")
            # Take care of distributed/parallel training
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(model_dir, "model.pt"))

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