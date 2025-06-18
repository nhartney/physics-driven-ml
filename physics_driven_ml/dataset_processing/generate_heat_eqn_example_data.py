""""
A script to solve the heat equation with a forcing term.
The equation is solved in time with the forward Euler method.
"""
import os
from firedrake import *
from gusto import *
import numpy as np
from numpy import random
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


# Define the list of initial conditions to use to generate solutions from
def generate_initial_conditions(n):
    ICs_list = []
    # Produce n random samples for initial conditions
    for r in range(n):
        x_pos = random.rand()
        y_pos = random.rand()
        a = 1 + random.rand()
        IC = a*exp(-((x-x_pos)**2)/0.01-((y-y_pos)**2)/0.01)
        ICs_list.append(IC)
    return ICs_list


# Set up test problem
# Domain
Lx = Ly = 1
nx = ny = 5
mesh = RectangleMesh(nx, ny, Lx, Ly, name="mesh")
x,y = SpatialCoordinate(mesh)
dt = 0.001

# Produce u, f and t for 10 timesteps, for a range of initial conditions
point_data_list = []
global_data_list = []
initial_conditions = generate_initial_conditions(5)

for IC in initial_conditions:
    print("this is a new initial condition")
    sln = solve_with_IC(mesh=mesh, ntimesteps=10, dt=0.1, IC=IC)
    # extract f and u at (x,y) points from (f,u,t) solutions
    for s in sln:
        f = s[0]
        u = s[1]
        t = s[2]
        print("this is the time:", t)
        for i, j in mesh.coordinates.dat.data:
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
            point_data_list.append((f_func, u_func, t_func, x_func, y_func))
        # concatenate global data as a list of (f,u,t) functions
        global_data_list.append((s[0], s[1], t_func))


# Save the data in train and test sets to checkpoint file
dataset_dir = os.path.join(
        "/Users/Jemma/Nell/code/physics-driven-ml/data/datasets",
        "heat_problem_example_data")
ntrain_pointwise = 0.8 * len(point_data_list)
ntest_pointwise = 0.2 * len(point_data_list)
print("Number of point-data training examples:", ntrain_pointwise)
print("Number of point-data testing examples:", ntest_pointwise)
ntrain_global = 0.8 * len(global_data_list)
ntest_global = 0.2 * len(global_data_list)
print("Number of global training examples:", ntrain_global)
print("Number of global testing examples:", ntest_global)

# Save point-wise train data
with CheckpointFile(os.path.join(dataset_dir, "train_point_data.h5"), "w") as afile:
    afile.h5pyfile["n"]  = ntrain_pointwise
    afile.save_mesh(mesh)
    for i, (f, u, t, x, y) in enumerate(point_data_list):
        afile.save_function(f, idx=i, name="target_f")
        afile.save_function(u, idx=i, name="u")
        afile.save_function(t, idx=i, name="t")
        afile.save_function(x, idx=i, name="x")
        afile.save_function(y, idx=i, name="y")

# Save point-wise test data
with CheckpointFile(os.path.join(dataset_dir, "test_point_data.h5"), "w") as afile:
    afile.h5pyfile["n"]  = ntest_pointwise
    afile.save_mesh(mesh)
    for i, (f, u, t, x, y) in enumerate(point_data_list):
        afile.save_function(f, idx=i, name="target_f")
        afile.save_function(u, idx=i, name="u")
        afile.save_function(t, idx=i, name="t")
        afile.save_function(x, idx=i, name="x")
        afile.save_function(y, idx=i, name="y")

# Save global train data
with CheckpointFile(os.path.join(dataset_dir, "train_global_data.h5"), "w") as afile:
    afile.h5pyfile["n"]  = ntrain_global
    afile.save_mesh(mesh)
    for i, (f, u, t) in enumerate(global_data_list):
        afile.save_function(f, idx=i, name="target_f")
        afile.save_function(u, idx=i, name="u")
        afile.save_function(t, idx=i, name="t")

# Save global test data
with CheckpointFile(os.path.join(dataset_dir, "test_global_data.h5"), "w") as afile:
    afile.h5pyfile["n"]  = ntest_global
    afile.save_mesh(mesh)
    for i, (f, u, t) in enumerate(global_data_list):
        afile.save_function(f, idx=i, name="target_f")
        afile.save_function(u, idx=i, name="u")
        afile.save_function(t, idx=i, name="t")