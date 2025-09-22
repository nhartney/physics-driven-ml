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
from physics_driven_ml.evaluation import evaluate_globally_by_point

from train_heat_problem_globally import sub_sample_point_data, calculate_global_loss, forward_pass


def train(model, device, train_point_dl, train_global_dl, test_point_dl, test_global_dl):
    """
    Train the model on a given dataset.
    """
    learning_rate = 5e-5
    epochs = 1
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
            loss = calculate_global_loss(subset, train_global_dl, model)
            # loss = forward_pass(subset, train_global_dl, model, output_global_loss=True)

            # print("these are the weights and biases, before the next backward pass:")
            # for name, param in model.state_dict().items():
            #     print(name)
            #     print(param)

            # Total loss
            total_loss += loss.item()
            # print("loss for this sample:", loss.item())
            # Backprop and perform Adam optimisation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimiser.step()


        logger.info(f"Total loss: {total_loss/train_steps}")

        # # Evaluate this version of the model on the test set
        # error = evaluate_globally_by_point(model, device, test_point_dl, test_global_dl, write_out_results=False)
        # logger.info(f"L2 error from this model, evaluated on the test set: {error}")

        # # Save best-performing model
        # if error < best_error or epoch_num == 0:
        #     best_error = error
        #     # Create directory for trained models
        #     name_dir = f"heat_problem_globally_epoch-{epoch_num}-error_{best_error:.5f}"
        #     model_dir = "/Users/Jemma/Nell/code/physics-driven-ml/data/saved_models"
        #     model_dir = os.path.join(model_dir, "heat_problem_globally", name_dir)
        #     if not os.path.exists(model_dir):
        #         os.makedirs(model_dir)
        #     # Save model
        #     logger.info(f"Saving model checkpoint to {model_dir}\n")
        #     # Take care of distributed/parallel training
        #     model_to_save = (model.module if hasattr(model, "module") else model)
        #     torch.save(model_to_save.state_dict(), os.path.join(model_dir, "model.pt"))

    return model


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

    # -- Training -- #

    train(model, device=device, train_point_dl=train_point_dl, train_global_dl=train_global_dl,
        test_point_dl=test_point_dl, test_global_dl=test_global_dl)


