import os
from os.path import abspath, dirname

import torch
import torch.optim as optim

from tqdm.auto import tqdm, trange

from torch.utils.data import DataLoader

from firedrake import *
from firedrake_adjoint import *
from firedrake.ml.pytorch import fem_operator

from physics_driven_ml.dataset_processing import PointDataset, PDEDataset2, BatchedElement2
from physics_driven_ml.models import PointNN
from physics_driven_ml.utils import get_logger

from physics_driven_ml.evaluation import evaluate_globally


from train_heat_problem_globally import sub_sample_point_data


def train(model, device, batch_size, train_point_dl, train_global_dl, test_point_dl, test_global_dl):
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

            batch = BatchedElement2(*[x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in global_sample])
            global_target_f = batch.target

            # Do a forward pass on all points in the data subset
            global_network_pred = forward_pass(subset, model, batch_size)
          
            # Define L2-loss using Firedrake
            loss = H(global_network_pred, global_target_f)

            total_loss += loss.item()

            # Backprop and perform Adam optimisation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimiser.step()

        logger.info(f"Total loss: {total_loss/train_steps}")

        # Evaluate this version of the model on the test set
        error = evaluate_globally(model, device, test_point_dl, test_global_dl, write_out_results=False)
        logger.info(f"L2 error from this model, evaluated on the test set: {error}")

        # Save best-performing model
        if error < best_error or epoch_num == 0:
            best_error = error
            # Create directory for trained models
            name_dir = f"heat_problem_globally_epoch-{epoch_num}-error_{best_error:.5f}"
            model_dir = "/Users/Jemma/Nell/code/physics-driven-ml/data/saved_models"
            model_dir = os.path.join(model_dir, "heat_problem_globally", name_dir)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            # Save model
            logger.info(f"Saving model checkpoint to {model_dir}\n")
            # Take care of distributed/parallel training
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(model_dir, "model.pt"))

    return model


def forward_pass(point_train_data_subset, model, batch_size):
    """
    This takes in a dataset (a subset of the full point dataset where all the labels are the same), sets
    up a dataloader for that dataset and does a forward pass on all the samples in that dataloader. It
    builds a list of network predictions at each point (tensors), which are then concatenated into one
    tensor to give a global network estimate of f, which is what is returned by the method.
    """

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

    data_dir = f'{abspath(dirname(__file__))}/../../data/datasets'
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

    train(model, device=device, batch_size=batch_size, train_point_dl=train_point_dl,
          train_global_dl=train_global_dl, test_point_dl=test_point_dl,
          test_global_dl=test_global_dl)
