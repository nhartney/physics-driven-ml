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

from train_heat_problem_globally import sub_sample_point_data


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

        if len(point_train_data_subsets) != train_steps:
            print("The number of data subsets does not match the number of global samples")

        for step_num, (subset, global_sample) in tqdm(enumerate(list(zip(point_train_data_subsets, train_global_dl))),
                                                    total=train_steps):
            model.zero_grad()

            subset_dl = DataLoader(subset, batch_size=batch_size, shuffle=False)

            # Do a forward pass on all points in the data subset and accumulate loss
            loss = calculate_global_loss(subset, model)

            # Total loss
            total_loss += loss.item()
            # print("loss for this sample:", loss.item())

            # Backprop and perform Adam optimisation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimiser.step()

        logger.info(f"Total loss: {total_loss/train_steps}")

    return model


def calculate_global_loss(point_train_data_subset, model):
    """
    This takes in a dataset (a subset of the full point dataset where all the labels are the same), sets
    up a dataloader for that dataset and does a forward pass on all the samples in that dataloader. It
    returns a list of network predictions at each point, which can then be interpolated to a Firedrake
    function for a global network estimate of f.
    """
    l2_loss = torch.nn.MSELoss()
    loss_list = []
    point_train_dl = DataLoader(point_train_data_subset, batch_size=batch_size, shuffle=False)
    train_steps = len(point_train_dl)
    for step_num, batch in enumerate(point_train_dl):
        # model.zero_grad()
        # extract inputs and target from the tensor
        inputs = batch[:, 1:5]
        target_f = batch[:,0]
        # forward pass
        network_point_f = model(inputs)[:,0]
        # compute L2 loss at the point
        loss = l2_loss(network_point_f, target_f)  
        # add point loss to total loss list
        loss_list.append(loss)
    # sum together all point losses in the list
    global_loss = sum(loss_list)
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

    # -- Training -- #

    train(model, device=device, train_point_dl=train_point_dl, train_global_dl=train_global_dl,
        test_point_dl=test_point_dl, test_global_dl=test_global_dl)


