"""
Create training data for the global heat equation problem using Gusto and checkpointing.
"""

import os
import numpy as np
from firedrake import (RectangleMesh, FunctionSpace, Function, CheckpointFile,
                       Constant, DirichletBC)

from physics_driven_ml.forward_models import GustoHeatEquationModel
from physics_driven_ml.dataset_processing import generate_initial_conditions, train_test_split

# Set up problem
# Domain
Lx = Ly = 1
nx = ny = 5
mesh = RectangleMesh(nx, ny, Lx, Ly, name="mesh")
dt = 0.001
V = FunctionSpace(mesh, "CG", 1)

# Produce numpy array of point data and a checkpoint file with Firedrake function for the
# corresponding global f.
point_data_list = []
global_data_list = []
label = 0

# Initial conditions and boundary conditions
initial_conditions = generate_initial_conditions(mesh, 10)
bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]

# Set PDE forward model
pde_model = GustoHeatEquationModel(mesh, dt, create_training_data=True)

# Advance the PDE forwrd model in time
for IC in initial_conditions:
    u_0 = Function(V).interpolate(IC)
    sln = Function(V)
    pde_model.advance(sln, u_0, ndt=10)
