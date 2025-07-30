import os
import argparse
import functools

import numpy as np

from os.path import abspath, dirname

import torch
import torch.optim as optim
import torch.autograd as torch_ad

from tqdm.auto import tqdm, trange

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from torch.autograd import Variable

from firedrake import *
from firedrake_adjoint import *
from firedrake.ml.pytorch import fem_operator, to_torch
from firedrake.__future__ import interpolate

from physics_driven_ml.dataset_processing import PointDataset, PDEDataset2, BatchedElement2
from physics_driven_ml.models import PointNN
from physics_driven_ml.utils import get_logger

from physics_driven_ml.dataset_processing.heat_problem_data_tools import sub_sample_point_data


def train(model, device, train_point_dl, train_global_dl, test_point_dl, test_global_dl,
         bcs):
    """
    Train the model on a given dataset.
    """
    learning_rate = 5e-5
    epochs = 4
    device = device

    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    max_grad_norm = 1.0
    best_error = 0.

    # Training loop
    for epoch_num in trange(epochs):
        logger.info(f"Epoch num: {epoch_num}")

        model.train()

        total_loss = 0.0

        train_steps = len(train_global_dl)

        point_train_data_subsets = sub_sample_point_data(train_point_dl)

        if len(point_train_data_subsets) == train_steps:
            print("correct! the number of data subsets matches the number of global samples")
        
        # Extract the mesh from the data
        mesh = train_global_dl.dataset.mesh
        V = FunctionSpace(mesh, "CG", 1)
        
        for step_num, (subset, global_sample) in tqdm(enumerate(list(zip(point_train_data_subsets, train_global_dl))),
                                                    total=train_steps):
            
            model.zero_grad()

            # Compute a dynamics solution after one timestep by solving the PDE with no forcing
            global_batch = BatchedElement2(*[x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in global_sample])
            global_u0 = global_batch.u0_fd[0]

            dynamics1 = solve_pde_without_forcing(mesh, ntimesteps=1, dt=0.001, V=V, IC=global_u0, bcs=bcs)

            # Produce a network prediction for f using the same initial condition
            # Do a forward pass on all points in the data subset
            network_out1 = forward_pass_by_point(subset, model)
            # Interpolate this to a Firedrake function
            point_dl1 = DataLoader(subset, batch_size=batch_size, shuffle=False)
            # with torch.no_grad():
            network_f1 = interpolate_to_firedrake_function(train_global_dl, point_dl1, network_out1)

            # Add the network's prediction for f to the dynamics solution
            u1 = network_f1 + dynamics1
            # Convert u1 to a Firedrake function
            u1_func = Function(V).interpolate(u1)

            # The point data from this solution becomes input for the next call to
            # the network
            u1_point_data = convert_dy_out_to_NN_in(u1_func)

            # Take another timestep (dynamics and then network f)
            dynamics2 = solve_pde_without_forcing(mesh, ntimesteps=1, dt=0.001, V=V, IC=u1_func, bcs=bcs)
            # Produce a network prediction for f using the pointdata version of the same initial condition
            point_dl2 = DataLoader(u1_point_data, batch_size=batch_size, shuffle=False)
            # Do a forward pass on all points in the data
            network_out2 = forward_pass_by_point(u1_point_data, model)
            # Interpolate this to a Firedrake function
            # with torch.no_grad():
            network_f2 = interpolate_to_firedrake_function(train_global_dl, point_dl2, network_out2)
            # Add the network's prediction for f to the dynamics solution
            u2 = network_f2 + dynamics2
            # Convert this to a Firedrake function
            u2_func = Function(V).interpolate(u2)

            # Extract the target to compare u2 to
            target = global_batch.u_target

            # Now make the Firedrake function a PyTorch tensor
            u2_tensor = to_torch(u2_func)

            # Loss is difference between two Firedrake functions
            loss = H(u2_tensor, target)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimiser.step()

        logger.info(f"Total loss: {total_loss/train_steps}")


def forward_pass_by_point(point_train_data_subset, model):
    """
    This takes in a dataset (a subset of the full point dataset where all the labels are the same), sets
    up a dataloader for that dataset and does a forward pass on all the samples in that dataloader. It
    returns a list of network predictions at each point, which can then be interpolated to a Firedrake
    function for a global network estimate of f.
    """
    network_out = []
    point_train_dl = DataLoader(point_train_data_subset, batch_size=batch_size, shuffle=False)
    train_steps = len(point_train_dl)
    for step_num, batch in enumerate(point_train_dl):
        # model.zero_grad()
        # extract inputs from the tensor
        inputs = batch[:, 0:3]
        # forward pass
        network_point_f = model(inputs)[:,0]
        # extract value from the output tensor
        network_point_f_value = network_point_f.item()
        # add value to list
        network_out.append(network_point_f_value)
    return np.asarray(network_out)


def create_VOM(train_global_dl, subset_dl):
    mesh = train_global_dl.dataset.mesh
    coordinates = []
    for data_sample in subset_dl:
        x_locs = data_sample[:,1].item()
        y_locs = data_sample[:,2].item()
        coordinates.append((x_locs,y_locs))
    vom = VertexOnlyMesh(mesh, coordinates, redundant=False)
    return vom


def interpolate_to_firedrake_function(train_global_dl, subset_dl, network_f):
    vom = create_VOM(train_global_dl, subset_dl)

    # start with a VOM that is structured like the data
    P0DG_io = FunctionSpace(vom.input_ordering, "DG", 0)
    field_vomio = Function(P0DG_io)
    field_vomio.dat.data_wo[:] = network_f

    # Next interpolate onto the vertex only mesh that does not have
    # the input ordering
    P0DG = FunctionSpace(vom, "DG", 0)
    field_vom = assemble(interpolate(field_vomio, P0DG))

    # interpolate from this VOM to the parent mesh (global data mesh)
    src_mesh = train_global_dl.dataset.mesh
    # find the function space that the global target function is on
    Vsrc = train_global_dl.dataset.fs

    I = Interpolator(TestFunction(Vsrc), P0DG)
    f_star = field_vom.riesz_representation(riesz_map="l2")
    f_data_star = Cofunction(Vsrc.dual())
    I.interpolate(f_star, adjoint=True, output=f_data_star)

    # Do this with the new interpolate behaviour
    # f_data_star = assemble(interpolate(f_star, P0DG, adjoint=True))

    field = f_data_star.riesz_representation(riesz_map="l2")
    return field


# Solve the heat equation without forcing (dynamics step)
def solve_pde_without_forcing(mesh, ntimesteps, dt, V, IC, bcs):
    x, y = SpatialCoordinate(mesh)
    k = Constant(1)
    u = Function(V)
    u_ = Function(V)
    v = TestFunction(V)
    u_.assign(IC)
    for n in range(ntimesteps):
        F = (inner((u - u_)/dt, v) + inner(k * grad(u), grad(v))) * dx
        # Solve PDE (using LU factorisation)
        solve(F == 0, u, bcs=bcs)
        u_.assign(u)
    return u


# Convert the output from the dynamics step to point data for the network
def convert_dy_out_to_NN_in(u):
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
    point_train_dataset = PointDataset(numpy_data=os.path.join(data_dir, data_file_name, "numpy_point_train_data.npy"),
                                        data_dir=data_dir)
    train_point_dl = DataLoader(point_train_dataset, batch_size=batch_size, shuffle=False)

    # Point test
    point_test_dataset = PointDataset(numpy_data=os.path.join(data_dir, data_file_name, "numpy_point_test_data.npy"),
                                        data_dir=data_dir)
    test_point_dl = DataLoader(point_test_dataset, batch_size=batch_size, shuffle=False)

    # Global train
    global_train_dataset = PDEDataset2(dataset=os.path.join(data_dir, data_file_name, "train_global_data.h5"),
                                       data_dir=data_dir)
    train_global_dl = DataLoader(global_train_dataset, batch_size=batch_size,
                                 collate_fn=global_train_dataset.collate, shuffle=False)
    
    # Global test
    global_test_dataset = PDEDataset2(dataset=os.path.join(data_dir, data_file_name, "test_global_data.h5"),
                                      data_dir=data_dir)
    test_global_dl = DataLoader(global_test_dataset, batch_size=batch_size,
                                collate_fn=global_train_dataset.collate, shuffle=False)

    # Set double precision to match types
    model.double()
    # Move model to device
    model.to(device)

    # Set up PDE dynamics problem
    dt = 0.001
    mesh = global_train_dataset.mesh
    V = FunctionSpace(mesh, "CG", 1)
    bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]

    # -- Define the Firedrake operations to be composed with PyTorch -- #

    def assemble_L2_error(x, x_exact):
        # Assemble L2 loss
        return assemble(0.5 * (x - x_exact) ** 2 * dx)
    
    # -- Construct the Firedrake torch operators -- #

    u_pred = Function(V)
    u_exact = Function(V)

    # Set tape locally to only record the operations relevant to H on the computational graph
    with set_working_tape() as tape:
        # Define PyTorch operator for computing the L2-loss (for computing κ -> 0.5 * ||κ - κ_exact||^{2}_{L2})
        F = ReducedFunctional(assemble_L2_error(u_pred, u_exact), [Control(u_pred), Control(u_exact)])
        H = fem_operator(F)

    # -- Training -- #

    train(model, device=device, train_point_dl=train_point_dl, train_global_dl=train_global_dl,
        test_point_dl=test_point_dl, test_global_dl=test_global_dl, bcs=bcs)