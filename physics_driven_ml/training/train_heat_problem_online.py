import os
import argparse
import functools

import numpy as np

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

def train(model, device, train_point_dl, train_global_dl, test_point_dl, test_global_dl,
          mesh, V, bcs):
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
        
        for step_num, (subset, global_sample) in tqdm(enumerate(list(zip(point_train_data_subsets, train_global_dl))),
                                                    total=train_steps):
            
            model.zero_grad()

            # Compute a dynamics solution  after one timestep by solving the PDE with no forcing
            global_batch = BatchedElement2(*[x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in global_sample])
            global_u0 = global_batch.u0_fd[0] # this is a list of functions
            dynamics1 = solve_pde_without_forcing(mesh, ntimesteps=1, dt=0.001, V=V, IC=global_u0, bcs=bcs)

            # Produce a network prediction for f using the same initial condition
            point_dl1 = DataLoader(subset, batch_size=batch_size, shuffle=False)
            # Do a forward pass on all points in the data subset
            network_out1 = forward_pass_by_point(point_dl1, model)
            # Interpolate this to a Firedrake function
            # with torch.no_grad():
            network_f1 = interpolate_to_firedrake_function(train_global_dl, point_dl1, network_out1)

            # Add the network's prediction for f to the dynamics solution
            u1 = network_f1 + dynamics1

            # The point data from this solution becomes input for the next call to
            # the network
            u1_point_data = dynamics_output_to_NN_input(u1)

            # Take another timestep (dynamics and then network f)
            dynamics2 = solve_pde_without_forcing(mesh, ntimesteps=1, dt=0.001, V=V, IC=u1, bcs=bcs)
            # Produce a network prediction for f using the pointdata version of the same initial condition
            point_dl2 = DataLoader(u1_point_data, batch_size=batch_size, shuffle=False)
            # Do a forward pass on all points in the data
            network_out2 = forward_pass_by_point(point_dl2, model)
            # Interpolate this to a Firedrake function
            # with torch.no_grad():
            network_f2 = interpolate_to_firedrake_function(train_global_dl, point_dl2, network_out2)
            # Add the network's prediction for f to the dynamics solution
            u2 = network_f2 + dynamics2

            # u2 is the thing we compare the target to


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
        inputs = batch[:, 1:4]
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
        x_locs = data_sample[:,2].item()
        y_locs = data_sample[:,3].item()
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
    u0 = Function(V).interpolate(IC)
    u_in = u0
    for n in range(ntimesteps):
        F = (inner((u - u_)/dt, v) + inner(k * grad(u), grad(v))) * dx
        u_.assign(u_in)
        # Solve PDE (using LU factorisation)
        solve(F == 0, u, bcs=bcs)
        u_.assign(u)
    return u

# Convert the output from the dynamics step to point data for the network
def dynamics_output_to_NN_input(u):
    point_data_list = []
    mesh = u.mesh
    for i, j in mesh.coordinates.dat.data:
        point_u = u.at(i, j)
        point_data_list.append((point_u, i, j))
    point_dataset = PointDataset(numpy_data=point_data_list)
    return point_dataset
    

if __name__ == "__main__":
    logger = get_logger("Training")

    # Set up PDE dynamics problem
    Lx = Ly = 1
    nx = ny = 5
    mesh = RectangleMesh(nx, ny, Lx, Ly, name="mesh")
    dt = 0.001
    V = FunctionSpace(mesh, "CG", 1)
    bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]

    # Set up for NN
    data_dir = os.path.join("/Users/Jemma/Nell/code/physics-driven-ml/data/datasets")
    data_file_name = "heat_problem_example_global_data"
    batch_size = 1
    device = "cpu"

    # Set the model
    model = PointNN()

    # -- Load dataset -- #