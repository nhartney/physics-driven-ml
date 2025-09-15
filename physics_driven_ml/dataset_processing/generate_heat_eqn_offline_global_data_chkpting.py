"""
Create training data for the global heat equation problem using Gusto and checkpointing.
"""

import os
import os.path as osp
import re
import numpy as np
from firedrake import (RectangleMesh, FunctionSpace, Function, CheckpointFile,
                       Constant, DirichletBC, SpatialCoordinate, sin, pi)

from firedrake.output import VTKFile

from physics_driven_ml.forward_models import GustoHeatEquationModel
from physics_driven_ml.dataset_processing import generate_initial_conditions, train_test_split

# Set up problem
# Domain
Lx = Ly = 1
nx = ny = 5
mesh = RectangleMesh(nx, ny, Lx, Ly, name="mesh")
dt = 0.001

# Data generation
num_ICs = 2
num_timesteps = 10
num_chkpts = 2
chkptfreq = num_timesteps/num_chkpts

# Set up function space
V = FunctionSpace(mesh, "CG", 1)

# Produce numpy array of point data and a checkpoint file with Firedrake function for the
# corresponding global f.
point_data_list = []
global_data_list = []
label = 0

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

chkpt_list = []
for dir in dir_list:
    chkpt_path = osp.join(results_path, dir, "chkpts")
    for i in range(1, num_chkpts+1):
       chkpt_n = int(i*chkptfreq)
       chkpt_file_name = osp.join(chkpt_path, f'chkpt{chkpt_n}.h5')
       chkpt_list.append(chkpt_file_name)

for chkpt_file in chkpt_list:
    print("this is the path to another chkpt:", chkpt_file)
    # Should they all be a different meshes? This is where one single file with the function at different
    # times would be better...
    with CheckpointFile(chkpt_file, 'r') as chkfile:
        mesh = chkfile.load_mesh(name='mesh')
        fs = FunctionSpace(mesh, "CG", 1)
        u = chkfile.load_function(mesh, 'q')
        t = chkfile.get_attr("/", 'time')
        x, y = SpatialCoordinate(mesh)
        f = Function(fs, name="target_f").interpolate(u*sin(t)*sin(pi*x)*sin(pi*y))

        # output functions to look at f and u
        # name each output function by intial condition and checkpoint time
        split_point = '/chkpts/'
        IC_s, _, chkpt_s = chkpt_file.partition(split_point)
        IC_no = str(re.findall(r'\d+', IC_s)[0])
        chkpt_no = str(re.findall(r'\d+', chkpt_s)[0])

        outfile = VTKFile(f'results/plots/IC_{IC_no}_chkpt_{chkpt_no}.pvd')
        outfile.write(u, f)

        # scale f and u
        u_max = u.dat.data.max()
        u_min = u.dat.data.min()
        u.dat.data[:] = (u.dat.data[:] - u_min)/(u_max - u_min)
        f_max = f.dat.data.max()
        f_min = f.dat.data.min()
        f.dat.data[:] = (f.dat.data[:] - f_min)/(f_max - f_min)

        # output functions to look at scaled f and u
        outfile = VTKFile(f'results/plots/scaled_values_IC_{IC_no}_chkpt_{chkpt_no}.pvd')
        outfile.write(u, f)
