""""
A script to produce validation data by solving the heat equation with a forcing term, using various
initial conditions. The equation is solved in time with the forward Euler method.
"""
import os
import numpy as np
from firedrake import RectangleMesh, SpatialCoordinate
from generate_heat_eqn_example_data import generate_initial_conditions, solve_with_IC

# Set up test problem
# Domain
Lx = Ly = 1
nx = ny = 6
mesh = RectangleMesh(nx, ny, Lx, Ly, name="mesh")
# x,y = SpatialCoordinate(mesh)
dt = 0.0015

# Produce u, f and t for 10 timesteps, for a range of initial conditions
point_data_list = []
initial_conditions = generate_initial_conditions(mesh, 1)

for IC in initial_conditions:
    sln = solve_with_IC(mesh=mesh, ntimesteps=5, dt=dt, IC=IC)
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

# Save the data as numpy arrays
dataset_dir = os.path.join(
        "/Users/Jemma/Nell/code/physics-driven-ml/data/datasets",
        "heat_problem_validation_data")
# Save pointdata as numpy arrays
print("Number of point-data validation examples:", len(point_data_list))
np.save(os.path.join(dataset_dir, 'numpy_point_validate_data'), point_data_list)