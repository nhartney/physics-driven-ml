import os
import argparse
import functools

import torch
import torch.optim as optim
import torch.autograd as torch_ad

from tqdm.auto import tqdm, trange

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from firedrake import *
from firedrake_adjoint import *
from firedrake.ml.pytorch import torch_operator
from firedrake.__future__ import interpolate

from physics_driven_ml.dataset_processing import PointDataset, PDEDataset2, BatchedElement2
from physics_driven_ml.models import PointNN
from physics_driven_ml.utils import get_logger
from physics_driven_ml.evaluation import evaluate_by_point

from collections import defaultdict


def train(model, device, train_point_dl, train_global_dl, test_point_dl, test_global_dl):
    """
    Train the model on a given dataset.
    """
    learning_rate = 5e-5
    epochs = 20
    device = device

    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    max_grad_norm = 1.0
    best_error = 0.

    # Training loop
    for epoch_num in trange(epochs):
        logger.info(f"Epoch num: {epoch_num}")

        model.train()

        total_loss = 0.0

        train_steps = len(train_global_dl)

        point_train_data_subsets = sub_sample_point_data(train_point_dl)

        if len(point_train_data_subsets) == train_steps:
            print("correct! the number of data subsets matches the number of global samples")
        
        for step_num, (subset, global_sample) in tqdm(enumerate(list(zip(point_train_data_subsets, train_global_dl))),
                                                    total=train_steps):
            model.zero_grad()

            subset_dl = DataLoader(subset, batch_size=batch_size, shuffle=False)
            batch = BatchedElement2(*[x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in global_sample])
            global_target_f = batch.target_fd[0]

            # Do a forward pass on all points in the data subset
            network_out = forward_pass_by_point(subset, model)
            
            # Interpolate this to a Firedrake function
            with torch.no_grad():
                global_network_f = interpolate_to_firedrake_function(train_global_dl, subset_dl, network_out)

            # put target_f on to the same mesh as network_f (the VOM)
            fs = FunctionSpace(global_network_f.function_space().mesh(), "DG", 0)
            target_f_on_data_mesh = Function(fs).interpolate(global_target_f)
            
            # Define L2-loss using Firedrake
            loss = H(global_network_f, target_f_on_data_mesh)

            # Backprop and perform Adam optimisation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimiser.step()

        print(f'Finished epoch {epoch_num}, latest loss {loss}')

        logger.info(f"Total loss: {total_loss/train_steps}")

        # # Evaluate this version of the model on the test set
        # error = evaluate_by_point(model, dev_dl, disable_tqdm=True)
        # logger.info(f"L2 error from this model, evaluated on the test set: {error}")

        # # Save best-performing model
        # if error < best_error or epoch_num == 0:
        #     best_error = error
        #     # Create directory for trained models
        #     name_dir = f"heat_problem_by_point_epoch-{epoch_num}-error_{best_error:.5f}"
        #     model_dir = "/Users/Jemma/Nell/code/physics-driven-ml/data/saved_models"
        #     model_dir = os.path.join(model_dir, "heat_problem_by_point", name_dir)
        #     if not os.path.exists(model_dir):
        #         os.makedirs(model_dir)
        #     # Save model
        #     logger.info(f"Saving model checkpoint to {model_dir}\n")
        #     # Take care of distributed/parallel training
        #     model_to_save = (model.module if hasattr(model, "module") else model)
        #     torch.save(model_to_save.state_dict(), os.path.join(model_dir, "model.pt"))

    return model


def forward_pass_by_point(point_train_data_subset, model):
    """
    This takes in a dataset (a subset of the full point dataset where all the labels are the same), sets
    up a dataloader for that dataset and does a forward pass on all the samples in that dataloader. It
    returns a list of network predictions at each point, which can then be interpolated to a Firedrake
    function for a global network estimate of f.
    """
    network_out = []
    point_train_dl = DataLoader(point_train_data_subset, batch_size=batch_size, shuffle=False)
    train_steps = len(point_train_dl)
    for step_num, batch in tqdm(enumerate(point_train_dl), total=train_steps):
        model.zero_grad()
        # extract inputs from the tensor
        inputs = batch[:, 1:5]
        # forward pass
        network_point_f = model(inputs)[:,0]
        # extract value from the output tensor
        network_point_f_value = network_point_f.item()
        # add value to list
        network_out.append(network_point_f_value)
    return np.asarray(network_out)


def list_duplicates(labels):
    # this method returns a list of tuples of (label, [indices]) where [indices] is a list of where
    # the label occurs
    tally = defaultdict(list)
    for i, item in enumerate(labels):
        tally[item].append(i)
    return (indices for indices in tally.items() if len(indices)>1)


def sub_sample_point_data(point_train_dataloader):
    labels = []
    for step_num, point_sample in enumerate(point_train_dataloader):
        point_label = point_sample[:,-1].item()
        labels.append(point_label)
    # define a list of lists of indices where all the labels match
    index_list = []
    for lables, indices in sorted(list_duplicates(labels)):
        index_list.append(indices)

    # use index_list to sub-sample the point data by labels
    subsets = []
    for l in index_list:
        subset = Subset(point_train_dataloader.dataset, l)
        subsets.append(subset)
    # return a list of datasets, where all examples in each dataset belongs to one global sample
    return subsets


def create_VOM(train_global_dl, subset_dl):
    mesh = train_global_dl.dataset.mesh
    coordinates = []
    for data_sample in subset_dl:
        x_locs = data_sample[:,2].item()
        y_locs = data_sample[:,3].item()
        coordinates.append((x_locs,y_locs))
    print("these are the coords:", coordinates)
    vom = VertexOnlyMesh(mesh, coordinates, redundant=False)
    return vom


def interpolate_to_firedrake_function(train_global_dl, subset_dl, network_f):
    vom = create_VOM(train_global_dl, subset_dl)

    # P0DG = FunctionSpace(vom, "DG", 0)
    # P0DG_input_ordering = FunctionSpace(vom.input_ordering, "DG", 0)
    # point_data_input_ordering = Function(P0DG_input_ordering)
    # point_data_input_ordering.dat.data_wo[:] = network_f
    # point_data = assemble(interpolate(point_data_input_ordering, P0DG))

    P0DG_io = FunctionSpace(vom.input_ordering, "DG", 0)
    field_vomio = Function(P0DG_io)
    field_vomio.dat.data_wo[:] = network_f

    # We now interpolate onto the vertex only mesh that does not have
    # the input ordering
    P0DG = FunctionSpace(vom, "DG", 0)
    field_vom = interpolate(field_vomio, P0DG)

    # interpolate from VOM to the parent mesh (global data mesh)
    src_mesh = train_global_dl.dataset.mesh
    # find the function space that the global target function is on
    Vsrc = train_global_dl.dataset.fs
    I = Interpolator(TestFunction(Vsrc), P0DG)
    f_star = field_vom.riesz_representation(riesz_map="l2")
    f_data_star = Cofunction(Vsrc.dual())
    I.interpolate(f_star, transpose=True, output=f_data_star)
    field = f_data_star.riesz_representation(riesz_map="l2")
    return point_data


def assemble_L2_error(network_f, target_f):
     # for debugging:
    print("this is the type returned by assemble_L2_error:", type(assemble(0.5 * (network_f - target_f) ** 2 * dx)))
    return assemble(0.5 * (network_f - target_f) ** 2 * dx)


if __name__ == "__main__":
    logger = get_logger("Training")

    data_dir = os.path.join("/Users/Jemma/Nell/code/physics-driven-ml/data/datasets")
    data_file_name = "heat_problem_example_global_data"
    batch_size = 1
    device = "cpu"

    # Set the model
    model = PointNN()

    # -- Load dataset -- #

    # Load train point dataset
    point_train_dataset = PointDataset(numpy_data=os.path.join(data_dir, data_file_name, "numpy_point_train_data.npy"),
                                        data_dir=data_dir)
    train_point_dl = DataLoader(point_train_dataset, batch_size=batch_size, shuffle=False)
    
    # Load test point dataset
    point_test_dataset = PointDataset(numpy_data=os.path.join(data_dir, data_file_name, "numpy_point_test_data.npy"),
                                        data_dir=data_dir)
    test_point_dl = DataLoader(point_test_dataset, batch_size=batch_size, shuffle=False)

   
    # Load train global dataset
    global_train_dataset = PDEDataset2(dataset="heat_problem_example_global_data",
                                       dataset_split="train", data_dir=data_dir)
    train_global_dl = DataLoader(global_train_dataset, batch_size=batch_size,
                                 collate_fn=global_train_dataset.collate, shuffle=False)

    # Load test global dataset
    global_test_dataset = PDEDataset2(dataset="heat_problem_example_global_data",
                                     dataset_split="test", data_dir=data_dir)
    test_global_dl = DataLoader(global_test_dataset, batch_size=batch_size,
                                collate_fn=global_train_dataset.collate, shuffle=False)

    # Set double precision to match types
    model.double()
    # Move model to device
    model.to(device)

    # -- Construct the Firedrake torch operators -- #
 
    # extract a single dataloader from the subset to get the mesh coordinates from
    point_train_data_subsets = sub_sample_point_data(train_point_dl)
    one_subset = point_train_data_subsets[0]
    one_subset_dl = DataLoader(one_subset, batch_size=batch_size, shuffle=False)
    # use the coordinates from this dataloader to define the VertexOnlyMesh
    vom = create_VOM(train_global_dl, one_subset_dl)
    # set up functions for the predicted f and the target f
    V = FunctionSpace(vom, "DG", 0)
    global_network_f = Function(V)
    target_f_on_data_mesh = Function(V)

    # Set tape locally to only record the operations relevant to H on the computational graph
    with set_working_tape() as tape:
        # Define PyTorch operator for computing the L2-loss (for computing κ -> 0.5 * ||f - f_exact||^{2}_{L2})
        F = ReducedFunctional(assemble_L2_error(global_network_f, target_f_on_data_mesh),
                                [Control(global_network_f), Control(target_f_on_data_mesh)])
        print("this is the type of F:", type(F))
        H = torch_operator(F)
        print("this is the type of H:", type(H))

    # -- Training -- #

    train(model, device=device, train_point_dl=train_point_dl, train_global_dl=train_global_dl,
        test_point_dl=test_point_dl, test_global_dl=test_global_dl)