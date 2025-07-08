""""
A script to produce train/test data with a global f function by solving the heat equation with a
forcing term, using various initial conditions. The equation is solved in time with the forward Euler
method.
"""

import os
import numpy as np
from firedrake import RectangleMesh, CheckpointFile
from generate_heat_eqn_example_data import generate_initial_conditions, solve_with_IC

# Set up test problem
# Domain
Lx = Ly = 1
nx = ny = 5
mesh = RectangleMesh(nx, ny, Lx, Ly, name="mesh")
dt = 0.001

# Produce numpy array of point data and a checkpoint file with Firedrake function for the
# corresponding global f.
point_data_list = []
global_data_list = []
label = 0

initial_conditions = generate_initial_conditions(mesh, 10)
ntrain = 80
ntest = 20

for IC in initial_conditions:
    sln = solve_with_IC(mesh=mesh, ntimesteps=10, dt=dt, IC=IC)
    # extract f and u at (x,y) points from (f,u,t) solutions
    for s in sln:
        f = s[0]
        u = s[1]
        t = s[2]
        label +=1
        for i, j in mesh.coordinates.dat.data:
            u_eval = u.at(i,j)
            # concatenate list of (u,t,x,y, label) solutions
            point_data_list.append((u_eval, t, i, j, label))
        global_data_list.append((f, label))

# split into train-test
point_train, point_test = point_data_list[:ntrain], point_data_list[:ntest]
global_train, global_test = global_data_list[:ntrain], global_data_list[:ntest]

# Save global f function to checkpoint files
dataset_dir = os.path.join(
            "/Users/Jemma/Nell/code/physics-driven-ml/data/datasets",
            "heat_problem_example_global_data")

with CheckpointFile(os.path.join(dataset_dir, "train_global_data.h5"), "w") as afile:
        afile.h5pyfile["n"] = ntrain
        afile.save_mesh(mesh)
        for i, (f, label) in enumerate(global_train):
            afile.save_function(f, idx=i, name="target_f")

with CheckpointFile(os.path.join(dataset_dir, "test_global_data.h5"), "w") as afile:
        afile.h5pyfile["n"] = ntrain
        afile.save_mesh(mesh)
        for i, (f, label) in enumerate(global_test):
            afile.save_function(f, idx=i, name="target_f")

# Save point data (u,t,x,y,label) to numpy arrays
np.save(os.path.join(dataset_dir, 'numpy_point_train_data'), point_train)
np.save(os.path.join(dataset_dir, 'numpy_point_test_data'), point_test)