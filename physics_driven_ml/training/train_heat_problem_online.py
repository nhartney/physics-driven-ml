import os

from os.path import abspath, dirname

import torch
import torch.optim as optim

from tqdm.auto import tqdm, trange

from torch.utils.data import DataLoader

from firedrake import *
from firedrake_adjoint import *
from firedrake.ml.pytorch import fem_operator, to_torch

from physics_driven_ml.dataset_processing import PointDataset, PDEDatasetOnline, BatchedElementOnline
from physics_driven_ml.models import PointNN
from physics_driven_ml.forward_models import HeatEquation, GustoHeatEquationModel
from physics_driven_ml.utils import get_logger

from physics_driven_ml.dataset_processing.heat_problem_data_tools import (sub_sample_point_data,
                                                                          interpolate_to_firedrake_function,
                                                                          forward_pass_by_point)


def train(model, pde_model, device, train_point_dl, train_global_dl,
          test_point_dl, test_global_dl, batch_size, H, num_epochs,
          logger,
          max_rollout_steps=2, ndt=1):
    """
    Train the model on a given dataset.
    """
    learning_rate = 5e-5
    epochs = num_epochs
    device = device

    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    max_grad_norm = 1.0

    # set up some functions
    V = pde_model.V
    u = Function(V)
    dyn_out = Function(V)

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

        if len(point_train_data_subsets) == train_steps:
            print("correct! the number of data subsets matches the number of global samples")

        for step_num, (subset, global_sample) in tqdm(
                enumerate(list(zip(point_train_data_subsets, train_global_dl))),
                total=train_steps):

            model.zero_grad()

            # get initial condition from data
            global_batch = BatchedElementOnline(*[x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in global_sample])
            global_u0 = global_batch.u0_fd[0]

            # set initial values for dynamics and network
            u.assign(global_u0)
            nn_in = subset
            for rollout_step in range(max_rollout_steps):

                # run forward PDE model for ndt timesteps
                pde_model.advance(dyn_out, u, ndt)

                # Produce a network prediction for f using the same
                # initial condition
                # Do a forward pass on all points in the
                # data subset
                nn_out, ordered_coords = forward_pass_by_point(nn_in, model, batch_size)
                # Interpolate this to a Firedrake function
                nn_out = interpolate_to_firedrake_function(mesh, fs,
                                                           nn_out,
                                                           ordered_coords)

                # Add the network's prediction to the dynamics solution
                # This is the input for the next dynamics step
                u.interpolate(dyn_out + nn_out)

                # The point data from this solution becomes input for the
                # next call to the network
                nn_in = convert_to_points(u)

            # Extract the target to compare u to
            target = global_batch.u_target

            # Now make the output of the rollout model a pytorch tensor
            u_tensor = to_torch(u, requires_grad=True)

            # Loss is difference between two tensors
            loss = H(u_tensor, target)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_grad_norm)
            optimiser.step()

        logger.info(f"Total loss: {total_loss/train_steps}")


# Convert the output from the dynamics step to point data for the network
def convert_to_points(u):
    point_data_list = []
    mesh = u.function_space().mesh()
    for i, j in mesh.coordinates.dat.data:
        point_u = u.at(i, j)
        point_data_list.append((point_u, i, j))
    point_dataset = PointDataset(numpy_data=point_data_list)
    return point_dataset


if __name__ == "__main__":
    logger = get_logger("Training")

    # Set up for NN
    data_dir = f'{abspath(dirname(__file__))}/../../data/datasets'
    data_file_name = "heat_problem_online_data"
    batch_size = 1
    device = "cpu"

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
    global_train_dataset = PDEDatasetOnline(
        dataset=os.path.join(data_dir, data_file_name, "train_global_data.h5"),
        data_dir=data_dir)
    train_global_dl = DataLoader(global_train_dataset,
                                 batch_size=batch_size,
                                 collate_fn=global_train_dataset.collate,
                                 shuffle=False)

    # Global test
    global_test_dataset = PDEDatasetOnline(
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
    # pde_model = GustoHeatEquationModel(mesh, dt=0.001)
    pde_model = HeatEquation(mesh, dt=0.001)

    # -- Define the Firedrake operations to be composed with PyTorch -- #

    def assemble_L2_error(x, x_exact):
        # Assemble L2 loss
        return assemble(0.5 * (x - x_exact) ** 2 * dx)

    # -- Construct the Firedrake torch operators -- #

    u_pred = Function(pde_model.V)
    u_exact = Function(pde_model.V)

    # Set tape locally to only record the operations relevant to H on the computational graph
    with set_working_tape() as tape:
        # Define PyTorch operator for computing the L2-loss
        F = ReducedFunctional(assemble_L2_error(u_pred, u_exact), [Control(u_pred), Control(u_exact)])
        H = fem_operator(F)

    # -- Training -- #

    train(model, pde_model, device=device,
          train_point_dl=train_point_dl,
          train_global_dl=train_global_dl,
          test_point_dl=test_point_dl,
          test_global_dl=test_global_dl,
          batch_size=batch_size, H=H,
          num_epochs=4, logger=logger)
