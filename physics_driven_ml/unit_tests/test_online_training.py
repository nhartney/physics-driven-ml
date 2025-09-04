import os

from os.path import abspath, dirname

from firedrake import *
from firedrake_adjoint import *
from firedrake.ml.pytorch import fem_operator

from physics_driven_ml.training.train_heat_problem_online import train
from physics_driven_ml.models import PointNN
from physics_driven_ml.dataset_processing import PointDataset, PDEDatasetOnline
from torch.utils.data import DataLoader
from physics_driven_ml.forward_models import HeatEquation, GustoHeatEquationModel
from physics_driven_ml.utils import get_logger


def model_setup():
    # Set up logger
    logger = get_logger("Training")

    # Set up neural network
    data_dir = f'{abspath(dirname(__file__))}/../../data/datasets'
    data_file_name = "heat_problem_online_data"
    batch_size = 1
    device = "cpu"

    # Set the model
    model = PointNN()

    # -- Load training datasets -- #
    # Point train
    point_train_dataset = PointDataset(
        numpy_data=os.path.join(data_dir,
                                data_file_name,
                                "numpy_point_train_data.npy"),
        data_dir=data_dir)
    point_train_dl = DataLoader(point_train_dataset,
                                batch_size=batch_size, shuffle=False)
    # Point test
    point_test_dataset = PointDataset(
        numpy_data=os.path.join(data_dir,
                                data_file_name,
                                "numpy_point_test_data.npy"),
        data_dir=data_dir)
    point_test_dl = DataLoader(point_test_dataset,
                               batch_size=batch_size, shuffle=False)
    # Global train
    global_train_dataset = PDEDatasetOnline(
        dataset=os.path.join(data_dir, data_file_name, "train_global_data.h5"),
        data_dir=data_dir)
    global_train_dl = DataLoader(global_train_dataset,
                                 batch_size=batch_size,
                                 collate_fn=global_train_dataset.collate,
                                 shuffle=False)
    # Global test
    global_test_dataset = PDEDatasetOnline(
        dataset=os.path.join(data_dir, data_file_name, "test_global_data.h5"),
        data_dir=data_dir)
    global_test_dl = DataLoader(global_test_dataset,
                                batch_size=batch_size,
                                collate_fn=global_train_dataset.collate,
                                shuffle=False)

    # Set double precision to match types
    model.double()
    # Move model to device
    model.to(device)
    # Set up mesh
    mesh = global_train_dataset.mesh

    return (model, device, mesh, point_train_dl, global_train_dl,
            point_test_dl, global_test_dl, logger)


def assemble_L2_error(x, x_exact):
    # Assemble L2 loss
    return assemble(0.5 * (x - x_exact) ** 2 * dx)


def test_online_training_heat_eqn(batch_size=1):
    print("Testing heat equation")
    # Set up NN model
    model, device, mesh, point_train_dl, global_train_dl, point_test_dl, global_test_dl, logger = model_setup()
    # Set PDE forward model
    pde_model = HeatEquation(mesh, dt=0.001)
    # Construct Firedrake torch operators
    u_pred = Function(pde_model.V)
    u_exact = Function(pde_model.V)
    # Set tape locally to only record the operations relevant to H on the computational graph
    with set_working_tape() as tape:
        # Define PyTorch operator for computing the L2-loss
        F = ReducedFunctional(assemble_L2_error(u_pred, u_exact), [Control(u_pred), Control(u_exact)])
        H = fem_operator(F)
    # Training
    train(model, pde_model, device,
          point_train_dl,
          global_train_dl,
          point_test_dl,
          global_test_dl, batch_size, H,
          num_epochs=1, logger=logger)


def test_online_training_Gusto_heat_eqn(batch_size=1):
    print("Testing Gusto heat equation")
    # Set up NN model
    model, device, mesh, point_train_dl, global_train_dl, point_test_dl, global_test_dl, logger = model_setup()
    # Set PDE forward model
    pde_model = GustoHeatEquationModel(mesh, dt=0.001)
    # Construct Firedrake torch operators
    u_pred = Function(pde_model.V)
    u_exact = Function(pde_model.V)
    # Set tape locally to only record the operations relevant to H on the computational graph
    with set_working_tape() as tape:
        # Define PyTorch operator for computing the L2-loss
        F = ReducedFunctional(assemble_L2_error(u_pred, u_exact), [Control(u_pred), Control(u_exact)])
        H = fem_operator(F)
    # Training
    train(model, pde_model, device,
          point_train_dl,
          global_train_dl,
          point_test_dl,
          global_test_dl, batch_size, H,
          num_epochs=1, logger=logger)
