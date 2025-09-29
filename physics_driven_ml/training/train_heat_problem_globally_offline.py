import os
# import traceprint

from os.path import abspath, dirname

import torch
import torch.optim as optim

from tqdm.auto import tqdm, trange

from torch.utils.data import DataLoader

from firedrake import *
from firedrake_adjoint import *
from firedrake.ml.pytorch import fem_operator, to_torch

from physics_driven_ml.dataset_processing import PointDataset, PDEDatasetOffline, BatchedElementOffline
from physics_driven_ml.models import PointNN
from physics_driven_ml.forward_models import HeatEquation, GustoHeatEquationModel
from physics_driven_ml.utils import get_logger

from physics_driven_ml.dataset_processing.heat_problem_data_tools import (sub_sample_point_data,
                                                                          forward_pass)

from physics_driven_ml.evaluation import evaluate_globally


def train(model, pde_model, device, train_point_dl, train_global_dl,
          test_point_dl, test_global_dl, batch_size, H, num_epochs,
          logger):

    """
    Train the model on a given dataset.
    """
    learning_rate = 5e-10
    epochs = num_epochs
    device = device

    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    max_grad_norm = 1.0
    best_error = 0.

    
    # Extract mesh and function space from the global dataloader
    mesh = train_global_dl.dataset.mesh
    fs = train_global_dl.dataset.fs

    # Training loop
    for epoch_num in trange(epochs):
        logger.info(f"Epoch num: {epoch_num}")

        model.train()

        total_loss = 0.0

        train_steps = len(train_global_dl)

        point_train_data_subsets = sub_sample_point_data(train_point_dl)

        if len(point_train_data_subsets) != train_steps:
            print("The number of data subsets does not match the number of global samples")
        
        for step_num, (subset, global_sample) in tqdm(enumerate(list(zip(point_train_data_subsets, train_global_dl))),
                                                    total=train_steps):
            model.zero_grad()

            batch = BatchedElementOffline(*[x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in global_sample])
            global_target_f = batch.f_target

            # Do a forward pass on all points in the data subset
            global_network_pred = forward_pass(subset, model, batch_size)
  
            # Define L2-loss using Firedrake
            loss = H(global_network_pred, global_target_f)
            total_loss += loss.item()
           
            # Backprop and perform Adam optimisation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimiser.step()

        print(f'Finished epoch {epoch_num}, latest loss {loss}')

        logger.info(f"Total loss: {total_loss/train_steps}")

        # Evaluate this version of the model on the test set
        error = evaluate_globally(model, device, test_point_dl, test_global_dl, write_out_results=False)
        logger.info(f"L2 error from this model, evaluated on the test set: {error}")

        # Save best-performing model
        if error < best_error or epoch_num == 0:
            best_error = error
            # Create directory for trained models
            name_dir = f"heat_problem_globally_epoch-{epoch_num}-error_{best_error:.5f}"
            model_dir = "/Users/Jemma/Nell/code/physics-driven-ml/data/saved_models"
            model_dir = os.path.join(model_dir, "heat_problem_globally", name_dir)
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

    # Set up for NN
    data_dir = f'{abspath(dirname(__file__))}/../../data/datasets'
    data_file_name = "heat_problem_global_gusto_data"
    batch_size = 1
    device = "cpu"
    num_epochs = 20

    # Set the model
    model = PointNN()

    # -- Load datasets -- #

    # Point train
    point_train_dataset = PointDataset(
        numpy_data=os.path.join(data_dir,
                                data_file_name,
                                "numpy_point_train_data.npy"),
        data_dir=data_dir)
    train_point_dl = DataLoader(point_train_dataset,
                                batch_size=batch_size, shuffle=False)

    # Point test
    point_test_dataset = PointDataset(
        numpy_data=os.path.join(data_dir,
                                data_file_name,
                                "numpy_point_test_data.npy"),
        data_dir=data_dir)
    test_point_dl = DataLoader(point_test_dataset,
                               batch_size=batch_size, shuffle=False)

    # Global train
    global_train_dataset = PDEDatasetOffline(
        dataset=os.path.join(data_dir, data_file_name, "train_global_data.h5"),
        data_dir=data_dir)
    train_global_dl = DataLoader(global_train_dataset,
                                 batch_size=batch_size,
                                 collate_fn=global_train_dataset.collate,
                                 shuffle=False)

    # Global test
    global_test_dataset = PDEDatasetOffline(
        dataset=os.path.join(data_dir, data_file_name, "test_global_data.h5"),
        data_dir=data_dir)
    test_global_dl = DataLoader(global_test_dataset,
                                batch_size=batch_size,
                                collate_fn=global_train_dataset.collate,
                                shuffle=False)

    # Set double precision to match types
    model.double()
    # Move model to device
    model.to(device)

    # Set up PDE dynamics problem
    mesh = global_train_dataset.mesh
    # Set PDE forward model
    pde_model = GustoHeatEquationModel(mesh, dt=0.001)
    # pde_model = HeatEquation(mesh, dt=0.001)

    # -- Define the Firedrake operations to be composed with PyTorch -- #

    def assemble_L2_error(x, x_exact):
        # Assemble L2 loss
        return assemble(0.5 * (x - x_exact) ** 2 * dx)

    # -- Construct the Firedrake torch operators -- #

    f_pred = Function(pde_model.V)
    f_exact = Function(pde_model.V)

    # Set tape locally to only record the operations relevant to H on the computational graph
    with set_working_tape() as tape:
        # Define PyTorch operator for computing the L2-loss
        F = ReducedFunctional(assemble_L2_error(f_pred, f_exact), [Control(f_pred), Control(f_exact)])
        H = fem_operator(F)

    # -- Training -- #

    train(model, pde_model, device=device,
          train_point_dl=train_point_dl,
          train_global_dl=train_global_dl,
          test_point_dl=test_point_dl,
          test_global_dl=test_global_dl,
          batch_size=batch_size, H=H,
          num_epochs=num_epochs, logger=logger)
