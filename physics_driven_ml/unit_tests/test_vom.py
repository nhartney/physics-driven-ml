import os

from os.path import abspath, dirname

from firedrake import (Function, SpatialCoordinate, exp, VertexOnlyMesh,
                       assemble, FunctionSpace)

from firedrake.__future__ import interpolate

from firedrake.output import VTKFile

from torch.utils.data import DataLoader
from physics_driven_ml.dataset_processing import PointDataset, PDEDataset2
from physics_driven_ml.training import interpolate_to_firedrake_function
from physics_driven_ml.dataset_processing.heat_problem_data_tools import sub_sample_point_data

# Start with a function and extract the data from it. This is our fake
# network output that we want to interpolate to a Firedrake function.
# The test checks if the Firedrake output matches what we started with.

# Create a mesh, a FunctionSpace and a Function with some data in it
# Create the VOM from the point data dataloader


def load_data():
    data_dir = f'{abspath(dirname(__file__))}/../../data/datasets'
    data_file_name = "heat_problem_online_data"
    batch_size = 1

    global_train_dataset = PDEDataset2(
        dataset=os.path.join(data_dir, data_file_name, "train_global_data.h5"),
        data_dir=data_dir)
    global_dl = DataLoader(global_train_dataset,
                           batch_size=batch_size,
                           collate_fn=global_train_dataset.collate,
                           shuffle=False)

    return global_dl


def create_data(global_dl):
    fs = global_dl.dataset.fs
    mesh = global_dl.dataset.mesh
    # Make a Gaussian function on the mesh -
    # this is the function we want to recover at the end
    x, y = SpatialCoordinate(mesh)
    V = Function(fs, name="orig_func").interpolate(exp(-(x-0.5)**2-(y-0.5)**2))
    # extract the data from V and write it to a list
    # first get the coordinates in the same order as the data list
    # loop over the mesh??
    coordinate_list = []
    coords = mesh.coordinates.dat.data
    for (x,y) in coords:
        coordinate_list.append((x,y))
    vom = VertexOnlyMesh(mesh, coordinate_list)
    P0DG = FunctionSpace(vom, "DG", 0)
    # Interpolation performs point evaluation
    V_at_points = assemble(interpolate(V, P0DG))
    V_data_list = V_at_points.dat.data_ro
    return V, V_data_list, coordinate_list, mesh, fs


def check_interpolate_to_firedrake_function():
    # pass global_dl to create_data because it needs it to extract function space
    # and mesh from
    global_dl= load_data()
    orig_func, data_list, coordinate_list, mesh, fs = create_data(global_dl)
    new_func = interpolate_to_firedrake_function(mesh, fs, data_list, coordinate_list)
    # check how new_func compares to orig_func
    # do this with a Firedrake norm
    # first check what they look like
    return orig_func, new_func


orig_func, new_func = check_interpolate_to_firedrake_function()
outfile = VTKFile("checking_vom_outfile.pvd")
outfile.write(orig_func, new_func)
