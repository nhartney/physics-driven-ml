"""
Create training data for the global heat equation problem using Gusto and checkpointing.
"""

import os
import numpy as np
from firedrake import RectangleMesh, CheckpointFile, Constant

from physics_driven_ml.forward_models import GustoHeatEquationModel

# Set up problem
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

# Set PDE forward model
pde_model = GustoHeatEquationModel(mesh, dt, create_training_data=True)
