import os

from os.path import abspath, dirname

# TODO: get rid of this!!
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# import traceprint

import numpy as np

import torch
import torch.optim as optim

from tqdm.auto import trange

from torch.utils.data import DataLoader


from firedrake import *
from firedrake_adjoint import *
from firedrake.ml.pytorch import torch_operator

from physics_driven_ml.dataset_processing import PointDataset, PDEDataset2, BatchedElement2
from physics_driven_ml.models import SimplePointNN
from physics_driven_ml.utils import get_logger


from train_heat_problem_globally import sub_sample_point_data


def train(model, device, train_point_dl, train_global_dl, train_global_dataset):
    """
    Train the model on a given dataset.
    """
    epochs = 1
    device = device
    learning_rate = 1e-2

    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    max_grad_norm = 1.0

    model.train()

    train_steps = len(train_global_dl)

    point_train_data_subsets = sub_sample_point_data(train_point_dl)

    if len(point_train_data_subsets) != train_steps:
        print("The number of data subsets does not match the number of global samples")

    for step_num, (subset, global_sample) in enumerate(list(zip(point_train_data_subsets, train_global_dl))):

        model.zero_grad()

        global_network_pred = forward_pass(subset, model)

        batch = BatchedElement2(*[x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in global_sample])
        global_target_f = batch.target

        func_loss = H(global_network_pred, global_target_f)

        total_func_loss = func_loss.item() # only to keep track of for printing
        print("function loss:", total_func_loss)

        # Backwards pass
        func_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimiser.step()

    return model


def forward_pass(point_train_data_subset, model):
    """
    This takes in a dataset (a subset of the full point dataset where all the labels are the same), sets
    up a dataloader for that dataset and does a forward pass on all the samples in that dataloader. If 
    output_global_loss is True then it returns the loss as the sum of all point losses in the subset.
    If output_global_loss is False it returns a list of network predictions at each point, which can
    then be interpolated to a Firedrake function for a global network estimate of f.
    """

    batch_size = 1
    tensor_list = []
    point_train_dl = DataLoader(point_train_data_subset, batch_size=batch_size, shuffle=False)

    for step_num, batch in enumerate(point_train_dl):

        # extract inputs and target from the tensor
        inputs = batch[:, 1:5]

        # forward pass
        network_point_f = model(inputs)[:,0]

        # add to list of tensor solutions
        tensor_list.append(network_point_f)

    # concatenate the tensor list together
    f_tensor = torch.cat(tuple(tensor_list))

    return f_tensor


if __name__ == "__main__":
    logger = get_logger("Training")

    # data_dir = os.path.join("/Users/Jemma/Nell/code/physics-driven-ml/data/datasets")
    data_dir = f'{abspath(dirname(__file__))}/../../data/datasets'
    data_file_name = "heat_problem_example_global_data"
    batch_size = 1
    device = "cpu"

    # Set the model
    model = SimplePointNN()

    # -- Load dataset -- #

    # Load train point dataset
    point_train_dataset = PointDataset(numpy_data=os.path.join(data_dir, data_file_name, "numpy_point_train_data.npy"),
                                        data_dir=data_dir)
    train_point_dl = DataLoader(point_train_dataset, batch_size=batch_size, shuffle=False)
    
   
    # Load train global dataset
    global_train_dataset = PDEDataset2(dataset=os.path.join(data_dir, data_file_name, "train_global_data.h5"),
                                       data_dir=data_dir)
    train_global_dl = DataLoader(global_train_dataset, batch_size=batch_size,
                                 collate_fn=global_train_dataset.collate, shuffle=False)
    

    # Set double precision to match types
    model.double()
    # Move model to device
    model.to(device)

    # -- Define the Firedrake operations to be composed with PyTorch -- #

    def assemble_L2_error(x, x_exact):
        # Assemble L2 loss
        return assemble(0.5 * (x - x_exact) ** 2 * dx)

    # -- Construct the Firedrake torch operators -- #
    mesh = global_train_dataset.mesh
    V = FunctionSpace(mesh, "CG", 1)
    f_pred = Function(V)
    f_exact = Function(V)

    # Set tape locally to only record the operations relevant to H on the computational graph
    with set_working_tape() as tape:
        # Define PyTorch operator for computing the L2-loss
        F = ReducedFunctional(assemble_L2_error(f_pred, f_exact), [Control(f_pred), Control(f_exact)])
        # H = fem_operator(F)
        H = torch_operator(F)

    # -- Training -- #

    train(model, device=device, train_point_dl=train_point_dl, train_global_dl=train_global_dl,
          train_global_dataset=point_train_dataset)