import os

from os.path import abspath, dirname

from firedrake import (Function, SpatialCoordinate, exp, VertexOnlyMesh,
                       assemble, FunctionSpace, errornorm)

from firedrake.__future__ import interpolate

from torch.utils.data import DataLoader
from physics_driven_ml.dataset_processing import PDEDataset2
from physics_driven_ml.training import interpolate_to_firedrake_function

# Start with a function and extract the data from it. This is our fake
# network output that we want to interpolate to a Firedrake function.
# The test checks if the Firedrake output matches what we started with.


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

    # We now need to get the point values of V and a list of the
    # coordinates of the points in the same order

    # First get the coordinates of the mesh
    coords = mesh.coordinates.dat.data

    # Create a VOM with these coordinates
    vom = VertexOnlyMesh(mesh, coords)

    # Create a point valued function space on the VOM and interpolate V to it
    P0DG = FunctionSpace(vom, "DG", 0)
    V_at_points_ = assemble(interpolate(V, P0DG))

    # Create point valued function space on the VOM that has the same
    # ordering as the coordinates and interpolate the point values V to it
    P0DG_io = FunctionSpace(vom.input_ordering, "DG", 0)
    V_at_points = assemble(interpolate(V_at_points_, P0DG_io))

    # We now have the values of V in the same order as the coordinates
    # specified above
    V_data_list = V_at_points.dat.data_ro

    return V, V_data_list, coords, mesh, fs


def test_interpolate_to_firedrake_function():
    # pass global_dl to create_data because it needs it to extract the function
    # space and mesh from
    global_dl = load_data()
    orig_func, data_list, coordinate_list, mesh, fs = create_data(global_dl)
    new_func = interpolate_to_firedrake_function(mesh, fs, data_list, coordinate_list)
    # check how new_func compares to orig_func using Firedrake error norm
    diff = errornorm(orig_func, new_func)
    assert diff < 1e-12
