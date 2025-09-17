"""
Create training data for the global heat equation problem using Gusto and checkpointing.
"""

import os
import os.path as osp
import re
import numpy as np
from firedrake import (RectangleMesh, FunctionSpace, Function, CheckpointFile,
                       Constant, DirichletBC, SpatialCoordinate, sin, pi,
                       VertexOnlyMesh, assemble)

from firedrake.__future__ import interpolate
from firedrake.output import VTKFile

from physics_driven_ml.forward_models import GustoHeatEquationModel
from physics_driven_ml.dataset_processing import generate_initial_conditions, train_test_split


def normalise(F):
    # A function to normalise data to values between zero and one
    max = F.dat.data.max()
    min = F.dat.data.min()
    F.dat.data[:] = (F.dat.data[:] - min)/(max - min)
    return F

def listed_coordinates(mesh):
    x_list = []
    y_list = []
    coords = mesh.coordinates.dat.data
    for x, y in coords:
        x_list.append(x)
        y_list.append(y)
    return np.asarray(x_list), np.asarray(y_list)


def evaluate_at_points(F, mesh):
    # Function to evalute a Firedrake function at every point in the mesh
    coords = mesh.coordinates.dat.data
    coordinate_list = []
    for x, y in coords:
        coordinate_list.append((x,y))
    vom = VertexOnlyMesh(mesh, coordinate_list)
    P0DG = FunctionSpace(vom, "DG", 0)
    F_at_points = assemble(interpolate(F, P0DG))
    F_values = F_at_points.dat.data_ro
    return F_values

# Set up problem
# Domain
Lx = Ly = 1
nx = ny = 10
mesh = RectangleMesh(nx, ny, Lx, Ly, name="mesh")
dt = 0.001

# Data generation
num_ICs = 10
num_timesteps = 10
num_chkpts = 10
chkptfreq = num_timesteps/num_chkpts

# Set up function space
V = FunctionSpace(mesh, "CG", 1)

# Initial conditions and boundary conditions
initial_conditions = generate_initial_conditions(mesh, num_ICs)
#TODO: How can we include the boundary conditions? This doesn't get passed to the PDE model...
bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]

IC_counter = int(1)
dir_list = []
for IC in initial_conditions:
    u_0 = Function(V).interpolate(IC)
    sln = Function(V)
    dirname=f'gusto_heat_eqn_IC{IC_counter}'
    # Set PDE forward model
    pde_model = GustoHeatEquationModel(mesh, dt, create_training_data=True,
                                       dirname=dirname,
                                       chkptfreq=chkptfreq)
    # Advance the PDE forward model in time
    pde_model.advance(sln, u_0, ndt=10)
    dir_list.append(dirname)
    IC_counter += 1

# Next load the data from the checkpoints and save it
results_path = osp.join("/Users/Jemma/Nell/code/physics-driven-ml/physics_driven_ml/results")

# Produce numpy array of point data and a checkpoint file with Firedrake function for the
# corresponding global f.
point_data_list = []
global_data_list = []
label = 0

chkpt_list = []
for dir in dir_list:
    chkpt_file_name = osp.join(results_path, dir, "chkpt.h5")
    chkpt_list.append(chkpt_file_name)

for chkpt_file in chkpt_list:
    # Is each IC on a different mesh?
    with CheckpointFile(chkpt_file, 'r') as chkfile:
        mesh = chkfile.load_mesh(name='mesh')
        fs = FunctionSpace(mesh, "CG", 1)
        for i in range(num_chkpts):
            idx = chkptfreq*(i+1)
            u = chkfile.load_function(mesh, 'q', idx=idx)
            t = chkfile.get_timestepping_history(mesh, 'q').get('time')[i+1]
            # grad_u = chkfile.load_function(mesh, 'q_gradient', idx=idx)
            # 'q_gradient' is the name of the field in the vtu but this isn't getting written to the chkpt
            x, y = SpatialCoordinate(mesh)
            f = Function(fs, name="target_f").interpolate(u*sin(t+dt)*sin(pi*x)*sin(pi*y))

            # scale f, u and grad_u
            normalised_f = normalise(f)
            normalised_u = normalise(u)
            # grad_u = normalise(grad_u)

            # get values at points
            f_values = evaluate_at_points(normalised_f, mesh)
            u_values = evaluate_at_points(normalised_u, mesh)
            # grad_u_values = evaluate_at_points(grad_u)
            x_values, y_values = listed_coordinates(mesh)

            # append these to point data list
            label += 1
            for f, u, x, y in zip(f_values, u_values, x_values, y_values):
                point_data_list.append((f, u, x, y, t, label))
            # append the global list with the normalised functions
            global_data_list.append((normalised_f, normalised_u, label))

point_train, point_test, global_train, global_test = train_test_split(point_data_list, global_data_list, 0.8)



