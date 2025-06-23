""""
A script to produce data by solving the heat equation with a forcing term, using variuos initial
conditions. The equation is solved in time with the forward Euler method.
"""
import os
from firedrake import *
from gusto import *
import numpy as np
from numpy import random
from tqdm.auto import tqdm, trange
from sklearn.model_selection import train_test_split

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
    solve(F == 0, u, bcs=bcs)
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
    sln = solve_with_IC(mesh=mesh, ntimesteps=10, dt=dt, IC=IC)
    # extract f and u at (x,y) points from (f,u,t) solutions
    for s in sln:
        f = s[0]
        u = s[1]
        t = s[2]
        for i, j in mesh.coordinates.dat.data:
            f_eval = f.at(i,j)
            u_eval = u.at(i,j)
            # concatenate list of (f,u,t,x,y) solutions
            point_data_list.append((f_eval, u_eval, t, i, j))
        # concatenate global data as a list of (f,u,t) functions
        global_data_list.append((s[0], s[1], t))


# Split point data into train and test sets using scikitlearn
train_pointwise, test_pointwise = train_test_split(point_data_list, test_size=0.2, random_state=42)
print("Number of point-data training examples:", len(train_pointwise))
print("Number of point-data testing examples:", len(test_pointwise))

# Save the data in train and test sets as numpy arrays
dataset_dir = os.path.join(
        "/Users/Jemma/Nell/code/physics-driven-ml/data/datasets",
        "heat_problem_example_data")
# Save pointdata as numpy arrays
np.save(os.path.join(dataset_dir, 'numpy_point_train_data'), train_pointwise)
np.save(os.path.join(dataset_dir, 'numpy_point_test_data'), test_pointwise)