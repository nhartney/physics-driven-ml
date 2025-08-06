import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import argparse
import functools

import torch
import torch.optim as optim
import torch.autograd as torch_ad

from tqdm.auto import tqdm, trange

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from torch.autograd import Variable

from firedrake import *
from firedrake_adjoint import *
from firedrake.ml.pytorch import fem_operator, to_torch
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
    epochs = 4
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
            global_target_f = batch.target

            # Do a forward pass on all points in the data subset
            network_out = forward_pass_by_point(subset, model)
            
            # Interpolate this to a Firedrake function
            # with torch.no_grad():
            global_network_f = interpolate_to_firedrake_function(train_global_dl, subset_dl, network_out)
            
            # Now make the Firedrake function a PyTorch tensor
            global_network_prediction = to_torch(global_network_f)
          
            # Define L2-loss using Firedrake
            loss = H(global_network_prediction, global_target_f)
            total_loss += loss.item()
            print("has the custom backward been called already?")
            print("this is the type of our loss that backward will be called on:", type(loss))

            print("this is the dict information about loss:", loss.__dict__)
            print("this is the dir information about loss:", loss.__dir__)

            # Set requires_grad for the loss to be True
            # loss = Variable(loss, requires_grad=True)
            # loss.requires_grad = True
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
    for step_num, batch in enumerate(point_train_dl):
        # model.zero_grad()
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
    vom = VertexOnlyMesh(mesh, coordinates, redundant=False)
    return vom


def interpolate_to_firedrake_function(train_global_dl, subset_dl, network_f):
    vom = create_VOM(train_global_dl, subset_dl)

    # start with a VOM that is structured like the data
    P0DG_io = FunctionSpace(vom.input_ordering, "DG", 0)
    field_vomio = Function(P0DG_io)
    field_vomio.dat.data_wo[:] = network_f

    # Next interpolate onto the vertex only mesh that does not have
    # the input ordering
    P0DG = FunctionSpace(vom, "DG", 0)
    field_vom = assemble(interpolate(field_vomio, P0DG))

    # interpolate from this VOM to the parent mesh (global data mesh)
    src_mesh = train_global_dl.dataset.mesh
    # find the function space that the global target function is on
    Vsrc = train_global_dl.dataset.fs

    I = Interpolator(TestFunction(Vsrc), P0DG)
    f_star = field_vom.riesz_representation(riesz_map="l2")
    f_data_star = Cofunction(Vsrc.dual())
    I.interpolate(f_star, adjoint=True, output=f_data_star)

    # Do this with the new interpolate behaviour
    # f_data_star = assemble(interpolate(f_star, P0DG, adjoint=True))

    field = f_data_star.riesz_representation(riesz_map="l2")
    return field


def assemble_L2_error(network_f, target_f):
     # for debugging:
    return assemble(0.5 * (network_f - target_f) ** 2 * dx)


def calculate_global_loss(point_train_data_subset, train_global_dl, model, output_loss_fn=False):
    """
    This takes in a dataset (a subset of the full point dataset where all the labels are the same), sets
    up a dataloader for that dataset and does a forward pass on all the samples in that dataloader. It
    returns a list of network predictions at each point, which can then be interpolated to a Firedrake
    function for a global network estimate of f.
    """
    # print("in calculate_global_loss, calculating global loss on a new subset")
    batch_size = 1
    l2_loss = torch.nn.MSELoss()
    loss_list = []
    f_list = []
    point_train_dl = DataLoader(point_train_data_subset, batch_size=batch_size, shuffle=False)
    train_steps = len(point_train_dl)
    for step_num, batch in enumerate(point_train_dl):
        # model.zero_grad()
        # extract inputs and target from the tensor
        inputs = batch[:, 1:5]
        target_f = batch[:,0]
        # print("before the forward pass in calculating global loss, these are the inputs:", inputs)
        # print("before the forward pass in calculating global loss, this is the target:", target_f)
        # print("this is the u input and f target:", inputs[:,0].item(), target_f.item())
        # just for debugging purposes
        # label = batch[:, 5]
        # print("before the forward pass in calculating global loss, this is the label:", label)
        # forward pass
        network_point_f = model(inputs)[:,0]

        if output_loss_fn:
            # for plotting purposes
            network_point_f_value = network_point_f.item()
            f_list.append(network_point_f_value)

        # for debugging
        print("prediction, target:", network_point_f.item(), target_f.item())
        
        # compute L2 loss at the point
        loss = l2_loss(network_point_f, target_f)
        # add point loss to total loss list
        loss_list.append(loss)
    # sum together all point losses in the list
    global_loss = sum(loss_list)

    if output_loss_fn:
        # print("in calculate global loss, this is the network's point f predictions for this global sample:", f_list)
        f_func = interpolate_to_firedrake_function(train_global_dl, point_train_dl, f_list)
        return global_loss, f_func
    else:
        return global_loss


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
 
    # set up functions for the predicted f and the target f
    V = train_global_dl.dataset.fs
    f_pred = Function(V)
    f_exact = Function(V)

    # Set tape locally to only record the operations relevant to H on the computational graph
    with set_working_tape() as tape:
        # Define PyTorch operator for computing the L2-loss (for computing κ -> 0.5 * ||f - f_exact||^{2}_{L2})
        F = ReducedFunctional(assemble_L2_error(f_pred, f_exact),
                                [Control(f_pred), Control(f_exact)])
        print("this is the type of F:", type(F))
        print("this is the dir information about F:", F.__dir__)
        H = fem_operator(F)
        print("this is the the type of H:", type(H))
        print("this is the dir information about H:", H.__dir__)

    # -- Training -- #

    train(model, device=device, train_point_dl=train_point_dl, train_global_dl=train_global_dl,
        test_point_dl=test_point_dl, test_global_dl=test_global_dl)