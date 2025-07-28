""""
A script to produce train/test data by solving the heat equation with a forcing term,
using various initial conditions. The equation is solved in time with the forward Euler
method. The data is saved both by point and globally, with the point data
(u at a point, x, y) to be used as input for the network and the global data to give the
target full PDE solution.
"""

import os
import numpy as np
from firedrake import *
from numpy import random

from physics_driven_ml.utils import get_logger


# Define the list of initial conditions to use to generate solutions from
def generate_initial_conditions(mesh, n):
    ICs_list = []
    x, y = SpatialCoordinate(mesh)
    # Produce n random samples for initial conditions
    for r in range(n):
        x_pos = random.rand()
        y_pos = random.rand()
        a = 1 + random.rand()
        IC = a*exp(-((x-x_pos)**2)/0.01-((y-y_pos)**2)/0.01)
        ICs_list.append(IC)
    return ICs_list


# for splitting data into train-test sets
def train_test_split(point_data_list, global_data_list, train_proportion):
    total_point_samples = len(point_data_list)

    # TODO: implement a check to make sure that the test-train split specified will keep all
    # point samples with the same labels together - this depends on the number of point samples

    n_point_train = int(train_proportion*total_point_samples)
    n_point_test = int(total_point_samples - n_point_train)

    point_train, point_test = point_data_list[:n_point_train], point_data_list[:n_point_test]

    # find the highest label in the point train set; this must become the last label in the global train
    # set too
    point_train_labels = []
    for p in point_train:
        label_p = p[-1]
        point_train_labels.append(label_p)
    max_point_train_label = max(point_train_labels)
    # this must become the last label in the global train set too

    n_global_train = max_point_train_label
    n_global_test = len(global_data_list) - n_global_train

    global_train, global_test = global_data_list[:n_global_train], global_data_list[:n_global_test]

    # check that the labels for point and global data in each set match
    global_train_labels = []
    for g in global_train:
        label_g = g[-1]
        global_train_labels.append(label_g)
    for l in point_train_labels:
        if l not in global_train_labels:
            raise Exception("The point data label has no corresponding global data label")

    return point_train, point_test, global_train, global_test


# Solve the heat equation with forcing
def solve_pde_with_forcing(mesh, ntimesteps, dt, V, IC, bcs):
    x, y = SpatialCoordinate(mesh)
    k = Constant(1)
    u = Function(V)
    u_ = Function(V)
    v = TestFunction(V)
    u0 = Function(V).interpolate(IC)
    u_in = u0
    for n in range(ntimesteps):
        t = n*dt
        f = Function(V).interpolate(u*t*sin(pi*x)*sin(pi*y))
        F = (inner((u - u_)/dt, v) + inner(k * grad(u), grad(v)) - inner(f, v)) * dx
        u_.assign(u_in)
        # Solve PDE (using LU factorisation)
        solve(F == 0, u, bcs=bcs)
        u_.assign(u)
    return u0, u


if __name__ == "__main__":
    logger = get_logger("Generating synthetic train and test data")
    # Set up problem
    # Domain
    Lx = Ly = 1
    nx = ny = 5
    mesh = RectangleMesh(nx, ny, Lx, Ly, name="mesh")
    dt = 0.001
    V = FunctionSpace(mesh, "CG", 1)

    bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]

    # Produce u for 2 timesteps, for a range of initial conditions
    point_data_list = []
    global_data_list = []
    label = 0
    initial_conditions = generate_initial_conditions(mesh, 50)

    for IC in initial_conditions:
        label += 1
        u0, u = solve_pde_with_forcing(mesh=mesh, ntimesteps=2, dt=dt, V=V, IC=IC, bcs=bcs)
        # extract u at (x,y) points from solutions to use as input data
        for i, j in mesh.coordinates.dat.data:
            point_u0 = u0.at(i, j)
            point_data_list.append((point_u0, i, j, label))
        global_data_list.append((u0, u, label))

    # Split the data into train and test sets
    point_train, point_test, global_train, global_test = train_test_split(point_data_list, global_data_list, 0.8)

    # Save point data and global data
    dataset_dir = os.path.join(
        "/Users/Jemma/Nell/code/physics-driven-ml/data/datasets",
        "heat_problem_online_data")

    point_train_dir = os.path.join(dataset_dir, 'numpy_point_train_data')
    point_test_dir = os.path.join(dataset_dir, 'numpy_point_test_data')
    global_train_dir = os.path.join(dataset_dir, "train_global_data.h5")
    global_test_dir = os.path.join(dataset_dir, "test_global_data.h5")

    # global data
    with CheckpointFile(global_train_dir, "w") as afile:
        afile.h5pyfile["n"] = len(global_train)
        afile.save_mesh(mesh)
        for i, (u0, u, label) in enumerate(global_train):
            afile.save_function(u, idx=i, name="target_u")

    with CheckpointFile(os.path.join(dataset_dir, "test_global_data.h5"), "w") as afile:
        afile.h5pyfile["n"] = len(global_test)
        afile.save_mesh(mesh)
        for i, (u0, u, label) in enumerate(global_test):
            afile.save_function(u, idx=i, name="target_u")

    # point data (u,t,x,y,label)
    np.save(point_train_dir, point_train)
    np.save(point_test_dir, point_test)

    print(f'Point training data ({len(point_train)} samples) saved in {point_train_dir}.npy')
    print(f'Point testing data ({len(point_test)} samples) saved in {point_test_dir}.npy')
    print(f'Global training data ({len(global_train)} samples) saved in {global_train_dir}')
    print(f'Global testing data ({len(global_test)} samples) saved in {global_test_dir}')
