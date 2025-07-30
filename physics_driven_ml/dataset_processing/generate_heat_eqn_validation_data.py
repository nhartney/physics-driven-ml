""""
A script to produce validation data by solving the heat equation with a forcing term, using various
initial conditions. The equation is solved in time with the forward Euler method.
"""
import os
import numpy as np
from firedrake import RectangleMesh, SpatialCoordinate, CheckpointFile
from generate_heat_eqn_example_data import generate_initial_conditions, solve_with_IC

# Set up test problem
# Domain
Lx = Ly = 1
nx = ny = 6
mesh = RectangleMesh(nx, ny, Lx, Ly, name="mesh")
# x,y = SpatialCoordinate(mesh)
dt = 0.0015

nICs = 3
ntimesteps = 5

n_global_validate = nICs*ntimesteps

# Produce u, f and t for 10 timesteps, for a range of initial conditions
point_data_list = []
global_data_list = []
label = 0 

initial_conditions = generate_initial_conditions(mesh, nICs)

for IC in initial_conditions:
    sln = solve_with_IC(mesh=mesh, ntimesteps=ntimesteps, dt=dt, IC=IC)
    # extract f and u at (x,y) points from (f,u,t) solutions
    for s in sln:
        f = s[0]
        u = s[1]
        t = s[2]
        label += 1
        for i, j in mesh.coordinates.dat.data:
            f_eval = f.at(i,j)
            u_eval = u.at(i,j)
            # concatenate list of (f,u,t,x,y) solutions
            point_data_list.append((f_eval, u_eval, t, i, j, label))
        global_data_list.append((f, label))

# Save the point data as numpy array
dataset_dir = os.path.join(
        "/Users/Jemma/Nell/code/physics-driven-ml/data/datasets",
        "heat_problem_global_validation_data")
print("Number of point-data validation examples:", len(point_data_list))
np.save(os.path.join(dataset_dir, 'numpy_point_validate_data'), point_data_list)

# Save the global data to checkpoint file
global_validate_dir = os.path.join(dataset_dir, "global_validate_data.h5")
global_val = global_data_list[:n_global_validate]
print("Number of global data validation examples:", len(global_data_list))
with CheckpointFile(global_validate_dir, "w") as afile:
        afile.h5pyfile["n"] = len(global_val)
        afile.save_mesh(mesh)
        for i, (f, label) in enumerate(global_val):
            afile.save_function(f, idx=i, name="target_f")