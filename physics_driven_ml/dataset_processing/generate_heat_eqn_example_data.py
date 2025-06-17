""""
A script to solve the heat equation with a forcing term.
The equation is solved in time with the forward Euler method.
"""
import os
from firedrake import (PeriodicRectangleMesh, SpatialCoordinate, Function,
                       CheckpointFile, interpolate, RectangleMesh,
                       IcosahedralSphereMesh, Constant, TestFunction, sin, pi,
                       grad, inner, dx)
from firedrake import *
from gusto import *
import numpy as np
from numpy.random import default_rng
from tqdm.auto import tqdm, trange

def advance_one_timestep(u_in, mesh, V, bcs, time):
    k = Constant(1.)
    dt = Constant(0.1)
    u = Function(V)
    u_ = Function(V)
    v = TestFunction(V)
    x,y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(u*time*sin(pi*x)*sin(pi*y))
    F = (inner((u - u_)/dt, v) + inner(k * grad(u), grad(v)) - inner(f, v)) * dx
    u_.assign(u_in)
    # Solve PDE (using LU factorisation)
    solve(F == 0, u)
    u_.assign(u)
    return u

def solve_with_IC(mesh, ntimesteps, dt, IC):

    x,y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)

    # Define bcs and IC
    bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
    u0 = Function(V).interpolate(IC)

    # Solve the PDE for n timesteps
    stepped_sln = []
    u_in = u0
    for n in range(ntimesteps):
        t = n*dt
        u = advance_one_timestep(u_in, mesh=mesh, V=V, bcs=bcs, time=t)
         # Compute f from the u solution (this is the network's target)
        f = Function(V).interpolate(u*t*sin(pi*x)*sin(pi*y))
        stepped_sln.append([f, u, t])
        u_in = u

    return stepped_sln

# Set up test problem
# Domain
Lx = Ly = 1
nx = ny = 5
mesh = RectangleMesh(nx, ny, Lx, Ly, name="mesh")
x,y = SpatialCoordinate(mesh)
dt = 0.1

# Define the list of initial conditions to use to generate solutions from
initial_condtions = [0.1*sin(pi*x)*sin(pi*y)]

# # make a list of x and y coordinates for point evaluation
# mesh_coords_x, mesh_coords_y = mesh.coordinates
# # Make functions with the coordinates
# V0 = FunctionSpace(mesh, "DG", 0)
# x_coords = Function(V0).interpolate(mesh_coords_x).dat.data
# y_coords = Function(V0).interpolate(mesh_coords_y).dat.data
# print(x_coords)
# print(y_coords)

# mesh_coords = mesh.coordinates
# cell = mesh.ufl_cell().cellname()
# DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
# vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)
# coords_dg = Function(vec_DG1).interpolate(mesh_coords)
x_coords = mesh.coordinates.dat.data[:, 0]
y_coords = mesh.coordinates.dat.data[:, 1]
print(x_coords)
print(y_coords)

# Produce u, f and t for 10 timesteps, for each of these initial conditions
data_list = []
for IC in initial_condtions:
    sln = solve_with_IC(mesh=mesh, ntimesteps=10, dt=0.1, IC=IC)
    # extract f and u at (x,y) points from (f,u,t) solutions
    for s in sln:
        f = s[0]
        u = s[1]
        t = s[2]
        print("this is the time:", t)
        for i in x_coords:
            for j in y_coords:
                print("this is (x,y):", i,j)
                f_eval = f.at(i,j)
                u_eval = u.at(i,j)
                # f, u, t, x, y must all be Firedrake functions
                fn_s = mesh.coordinates.function_space()
                f_func = Function(fn_s).assign(Constant(f_eval))
                u_func = Function(fn_s).assign(Constant(u_eval))
                t_func = Function(fn_s).assign(Constant(t))
                x_func = Function(fn_s).assign(Constant(i))
                y_func = Function(fn_s).assign(Constant(j))
                # concatenate list of (f,u,t,x,y) function solutions
                # data_list.append((f_eval, u_eval, t, i, j))
                data_list.append((f_func, u_func, t_func, x_func, y_func))


# Save the data in train and test sets to checkpoint file
dataset_dir = os.path.join(
        "/Users/Jemma/Nell/code/physics-driven-ml/data/datasets",
        "heat_problem_example_data")
ntrain = 0.8 * len(data_list)
ntest= 0.2 * len(data_list)
print("Number of training examples:", ntrain)
print("Number of testing examples:", ntest)

# Save train data
with CheckpointFile(os.path.join(dataset_dir, "train_data.h5"), "w") as afile:
    afile.h5pyfile["n"]  = ntrain
    afile.save_mesh(mesh)
    for i, (f, u, t, x, y) in enumerate(data_list):
        afile.save_function(f, idx=i, name="target_f")
        afile.save_function(u, idx=i, name="u")
        afile.save_function(t, idx=i, name="t")
        afile.save_function(x, idx=i, name="x")
        afile.save_function(y, idx=i, name="y")

# Save test data
with CheckpointFile(os.path.join(dataset_dir, "test_data.h5"), "w") as afile:
    afile.h5pyfile["n"]  = ntest
    afile.save_mesh(mesh)
    for i, (f, u, t, x, y) in enumerate(data_list):
        afile.save_function(f, idx=i, name="target_f")
        afile.save_function(u, idx=i, name="u")
        afile.save_function(t, idx=i, name="t")
        afile.save_function(x, idx=i, name="x")
        afile.save_function(y, idx=i, name="y")