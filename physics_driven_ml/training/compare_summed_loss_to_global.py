import os

import torch
import torch.optim as optim

from tqdm.auto import trange

from torch.utils.data import DataLoader


from firedrake import *
from firedrake_adjoint import *
from firedrake.ml.pytorch import fem_operator, to_torch

from physics_driven_ml.dataset_processing import PointDataset, PDEDataset2, BatchedElement2
from physics_driven_ml.models import PointNN, SimplePointNN
from physics_driven_ml.utils import get_logger


from train_heat_problem_globally import sub_sample_point_data, interpolate_to_firedrake_function


def train(model, device, train_point_dl, train_global_dl, train_global_dataset):
    """
    Train the model on a given dataset.
    """
    epochs = 1
    device = device
    learning_rate = 1e-5

    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    max_grad_norm = 1.0

    # Training loop
    for epoch_num in trange(epochs):
        logger.info(f"Epoch num: {epoch_num}")

        model.train()

        # total_summing_loss = 0
        train_steps = len(train_global_dl)

        point_train_data_subsets = sub_sample_point_data(train_point_dl)

        if len(point_train_data_subsets) != train_steps:
            print("The number of data subsets does not match the number of global samples")

        subset = point_train_data_subsets[0]
        global_sample_list = []
        for sample in train_global_dl:
            global_sample_list.append(sample)

        global_sample = global_sample_list[0]

        model.zero_grad()

        subset_dl = DataLoader(subset, batch_size=batch_size, shuffle=False)

        # Compare estimates of the loss in both ways
        # 1. Do a forward pass on all points in the data subset and accumulate loss
        print("this is the forward pass where we will accumulate loss")
        summing_loss = forward_pass(subset, train_global_dl, model, output_global_loss=True)
        total_summing_loss = summing_loss.item()

      
        # 2. Do a forward pass on all points and return point estimates, then turn this into Firedrake function
        #  and compute the loss as the errror norm between two functions
        print("this is the forward pass where the loss comes from comparing functions")
        network_f = forward_pass(subset, train_global_dl, model, output_global_loss=False)
        fd_f = interpolate_to_firedrake_function(train_global_dl, subset_dl, network_f)
        batch = BatchedElement2(*[x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in global_sample])
        global_target_f = batch.target
        global_network_prediction = to_torch(fd_f, requires_grad=True)
        func_loss = H(global_network_prediction, global_target_f)
        total_func_loss = func_loss.item()

        print("this is the summing loss:", total_summing_loss, "does it requires_grad?", summing_loss.requires_grad)
        print("this is the function loss:", total_func_loss, "does it requires_grad?", func_loss.requires_grad)

        # print("this is the model:", print(model))
        print("these are the weights and biases of the layers in the network, before any backwards pass:")
        for name, param in model.state_dict().items():
            print(name)
            print(param)

        # Do one backwards pass on the summing loss
        summing_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimiser.step()
        print("these are the weights and biases after one backwards pass on the summing loss:")
        for name, param in model.state_dict().items():
            print(name)
            print(param)

    return model


def forward_pass(point_train_data_subset, train_global_dl, model, output_loss_fn=False,
                 output_global_loss=True):
    """
    This takes in a dataset (a subset of the full point dataset where all the labels are the same), sets
    up a dataloader for that dataset and does a forward pass on all the samples in that dataloader. If 
    output_global_loss is True then it returns the loss as the sum of all point losses in the subset.
    If output_global_loss is False it returns a list of network predictions at each point, which can
    then be interpolated to a Firedrake function for a global network estimate of f.
    """

    batch_size = 1
    l2_loss = torch.nn.MSELoss()
    loss_list = []
    f_list = []
    point_train_dl = DataLoader(point_train_data_subset, batch_size=batch_size, shuffle=False)

    for step_num, batch in enumerate(point_train_dl):

        # extract inputs and target from the tensor
        inputs = batch[:, 1:5]
        target_f = batch[:,0]

        # forward pass
        network_point_f = model(inputs)[:,0]

        # for debugging
        # print("input1, input2, input3, input4:", inputs)
        # print("prediction, target:", network_point_f.item(), target_f.item())
        
        if output_global_loss:
            # compute L2 loss at the point
            loss = l2_loss(network_point_f, target_f)
            # add point loss to total loss list
            loss_list.append(loss)
            # sum together all point losses in the list
            global_loss = sum(loss_list)
        
        else:
            network_point_f_value = network_point_f.item()
            f_list.append(network_point_f_value)

    if output_global_loss:
        if output_loss_fn:
            # print("in calculate global loss, this is the network's point f predictions for this global sample:", f_list)
            f_func = interpolate_to_firedrake_function(train_global_dl, point_train_dl, f_list)
            return global_loss, f_func
        else:
            return global_loss
    else:
        return np.asarray(f_list)


if __name__ == "__main__":
    logger = get_logger("Training")

    data_dir = os.path.join("/Users/Jemma/Nell/code/physics-driven-ml/data/datasets")
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
    
    # Load test point dataset
    point_test_dataset = PointDataset(numpy_data=os.path.join(data_dir, data_file_name, "numpy_point_test_data.npy"),
                                        data_dir=data_dir)
    test_point_dl = DataLoader(point_test_dataset, batch_size=batch_size, shuffle=False)

   
    # Load train global dataset
    global_train_dataset = PDEDataset2(dataset=os.path.join(data_dir, data_file_name, "train_global_data.h5"),
                                       data_dir=data_dir)
    train_global_dl = DataLoader(global_train_dataset, batch_size=batch_size,
                                 collate_fn=global_train_dataset.collate, shuffle=False)

    # Load test global dataset
    global_test_dataset = PDEDataset2(dataset=os.path.join(data_dir, data_file_name, "test_global_data.h5"),
                                      data_dir=data_dir)
    test_global_dl = DataLoader(global_test_dataset, batch_size=batch_size,
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
        H = fem_operator(F)

    # -- Training -- #

    train(model, device=device, train_point_dl=train_point_dl, train_global_dl=train_global_dl,
          train_global_dataset=point_train_dataset)