""""
A script to produce train/test data with a global f function by solving the heat equation with a
forcing term, using various initial conditions. The equation is solved in time with the forward Euler
method.
"""

import os
import numpy as np
from firedrake import RectangleMesh, CheckpointFile
from generate_heat_eqn_example_data import generate_initial_conditions, solve_with_IC


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
            # print("these are the labels attached to the point data:", label)
        global_data_list.append((f, label))

point_train, point_test, global_train, global_test = train_test_split(point_data_list, global_data_list, 0.8)

# Save global f function to checkpoint files
dataset_dir = os.path.join(
            "/Users/Jemma/Nell/code/physics-driven-ml/data/datasets",
            "heat_problem_example_global_data")

point_train_dir = os.path.join(dataset_dir, 'numpy_point_train_data')
point_test_dir = os.path.join(dataset_dir, 'numpy_point_test_data')
global_train_dir = os.path.join(dataset_dir, "train_global_data.h5")
global_test_dir = os.path.join(dataset_dir, "test_global_data.h5")

with CheckpointFile(global_train_dir, "w") as afile:
        afile.h5pyfile["n"] = len(global_train)
        afile.save_mesh(mesh)
        for i, (f, label) in enumerate(global_train):
            afile.save_function(f, idx=i, name="target_f")

with CheckpointFile(os.path.join(dataset_dir, "test_global_data.h5"), "w") as afile:
        afile.h5pyfile["n"] = len(global_test)
        afile.save_mesh(mesh)
        for i, (f, label) in enumerate(global_test):
            afile.save_function(f, idx=i, name="target_f")

# Save point data (u,t,x,y,label) to numpy arrays
np.save(point_train_dir, point_train)
np.save(point_test_dir, point_test)

print(f'Point training data ({len(point_train)} samples) saved in {point_train_dir}.npy')
print(f'Point testing data ({len(point_test)} samples) saved in {point_test_dir}.npy')
print(f'Global training data ({len(global_train)} samples) saved in {global_train_dir}')
print(f'Global testing data ({len(global_test)} samples) saved in {global_test_dir}')
