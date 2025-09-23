import os

# TODO: get rid of this!!
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

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
    learning_rate = 1e-2

    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    max_grad_norm = 1.0


    model.train()

    # total_summing_loss = 0
    train_steps = len(train_global_dl)

    point_train_data_subsets = sub_sample_point_data(train_point_dl)

    if len(point_train_data_subsets) != train_steps:
        print("The number of data subsets does not match the number of global samples")

    # subset = point_train_data_subsets[0]
    # global_sample_list = []
    # for sample in train_global_dl:
    #     global_sample_list.append(sample)

    # global_sample = global_sample_list[0]

    for step_num, (subset, global_sample) in enumerate(list(zip(point_train_data_subsets, train_global_dl))):

        model.zero_grad()

        subset_dl = DataLoader(subset, batch_size=batch_size, shuffle=False)

        # Compare estimates of the loss in both ways
        # 1. Do a forward pass on all points in the data subset and accumulate loss
        # print("this is the forward pass where we will accumulate loss")
        summing_loss = forward_pass(subset, train_global_dl, model, output_summed_loss=True)
        # print("does summing loss require grad?", summing_loss.requires_grad)
        total_summing_loss = summing_loss.item()

      
        # 2. Do a forward pass on all points and return point estimates, then turn this into Firedrake function
        #  and compute the loss as the error norm between two functions
        # print("this is the forward pass where the loss comes from comparing functions")
        network_f, ordered_coords, f_function = forward_pass(subset, train_global_dl, model, output_summed_loss=False, output_loss_fn=True)
        mesh = train_global_dl.dataset.mesh
        fs = train_global_dl.dataset.fs
        fd_f = interpolate_to_firedrake_function(mesh, fs, network_f, ordered_coords)
        batch = BatchedElement2(*[x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in global_sample])
        global_target_f = batch.target
        # print(f"this is global_target_f (requires_grad? {global_target_f.requires_grad}):", global_target_f)
        # global_network_prediction = to_torch(fd_f, requires_grad=True)
        global_network_prediction = to_torch(fd_f)
        global_network_prediction.requires_grad_()
        # print(f"this is global_network_prediction (requires_grad? {global_network_prediction.requires_grad}):", global_network_prediction)
        func_loss = H(global_network_prediction, global_target_f)
        # print("does func_loss have requires_grad?", func_loss.requires_grad)
        total_func_loss = func_loss.item()

        print("summing loss:", total_summing_loss, "function loss:", total_func_loss)


        # Do a backwards pass on the summing loss
        func_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimiser.step()

        # Do a backwards pass on the function loss
        # func_loss.backward()
        # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # optimiser.step()

         # print("this is the model:", print(model))
        print("these are the weights and biases of the layers in the network:")
        for name, param in model.state_dict().items():
            print(name)
            print(param)
            # print("this is requires_grad of the parameter:", param.requires_grad)
            # print("this is the gradient of the parameter:", param.grad)
            # print("is the parameter a leaf tensor (should be)?", param.is_leaf)
            # print("this is .grad_fn on the parameter:", param.grad_fn)

        # print("this is model.state_dict_hooks:", model._state_dict_hooks)
        # print("gradients of the weights:", model[0].weight.grad)

        # optimiser.step()

    return model


def forward_pass(point_train_data_subset, train_global_dl, model, output_summed_loss=True,
                 output_loss_fn=False):
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
    ordered_coords = []
    point_train_dl = DataLoader(point_train_data_subset, batch_size=batch_size, shuffle=False)

    for step_num, batch in enumerate(point_train_dl):

        # extract inputs and target from the tensor
        inputs = batch[:, 1:5]
        target_f = batch[:,0]
        x = batch[:, 3].item()
        y = batch[:, 4].item()
        ordered_coords.append((x,y))

        # forward pass
        network_point_f = model(inputs)[:,0]

        # for debugging
        # print("input1, input2, input3, input4:", inputs)
        # print("prediction, target:", network_point_f.item(), target_f.item())
        
        if output_summed_loss:
            # compute L2 loss at the point
            loss = l2_loss(network_point_f, target_f)
            # add point loss to total loss list
            loss_list.append(loss)
        
        else:
            network_point_f_value = network_point_f.item()
            f_list.append(network_point_f_value)

    if output_summed_loss:
        # Loss function based on sum of individual losses
        summed_loss = sum(loss_list)
        return summed_loss
    
    else:
        # Return the point estimates, ready to be interpolated to a Firedrake function 
        if output_loss_fn:
            # print(f"this is the network's point f predictions for this global sample (lenth:{len(f_list)}):", f_list)
            mesh = train_global_dl.dataset.mesh
            fs = train_global_dl.dataset.fs
            f_func = interpolate_to_firedrake_function(mesh, fs, f_list, ordered_coords)
            # print(f"this is that data turned into a function: (lenth: {len(f_func.dat.data)})", f_func.dat.data)
            return np.asarray(f_list), ordered_coords, f_func
        else:
            return np.asarray(f_list), ordered_coords


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
        # return assemble(0.5 * (x - x_exact) ** 2 * dx)
        return errornorm(x_exact, x, norm_type='L2')

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