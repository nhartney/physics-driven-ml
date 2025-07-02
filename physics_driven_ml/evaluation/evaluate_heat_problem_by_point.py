import os
import argparse

import torch
import firedrake as fd
import firedrake.ml.pytorch as fd_ml

from torch.utils.data import DataLoader

from functools import partial
from tqdm.auto import tqdm

from physics_driven_ml.models import PointNN
from physics_driven_ml.utils import ModelConfig, get_logger
from physics_driven_ml.dataset_processing import PointDataset, BatchedElement


def evaluate_by_point(model, dataloader, disable_tqdm=False):
    """
    Evaluate the model on a given dataset.
    Compute the L2 error of the NN at every point for the test set, and then add these errors up
    to give an overall error for that model.
    """

    model.eval()

    eval_steps = len(dataloader)
    total_error = 0.0
    for step_num, batch in tqdm(enumerate(dataloader), total=eval_steps, disable=disable_tqdm):

        # Extract input data
        inputs = batch[:, 1:5]
        # Extract target f from the data tensor
        f_exact = batch[:, 0]

        with torch.no_grad():
            network_f = model(inputs)
            # Error at one point
            total_error += eval_error(network_f, f_exact)

        if step_num == eval_steps - 1:
            break

    # L2 error is the square root of the sum of the errors
    L2_error = total_error**0.5
    L2_error /= eval_steps
    return L2_error


def eval_error(x, x_exact):
    """Compute the L2 error between x and x_exact."""
    l2_eval_error = (x - x_exact) ** 2
    return l2_eval_error.item()


if __name__ == "__main__":
    logger = get_logger("Evaluation")

    data_dir = os.path.join("/Users/Jemma/Nell/code/physics-driven-ml/data/")
    dataset = "heat_problem_example_data"
    batch_size = 1
    device = "cpu"
    evaluation_metric = "L2"
    eval_set = "test"
    model_dir = "/Users/Jemma/Nell/code/physics-driven-ml/data/saved_models/heat_problem_by_point/"
    model_version = "heat_problem_by_point_epoch-19-error_0.00000"

    # Load dataset
    dataset_dir = os.path.join(data_dir, "datasets", dataset)
    logger.info(f"Loading dataset from {dataset_dir}\n")
    dataset = PointDataset(numpy_data=os.path.join(dataset_dir, "numpy_point_test_data.npy"),
                           data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model_dir = os.path.join(model_dir, model_version)

    logger.info(f"Loading model checkpoint from {model_dir}\n")
    model = PointNN()
    # Load pretrained model state dict
    pretrained = torch.load(os.path.join(model_dir, "model.pt"))
    model.load_state_dict(pretrained)
    
    # Set double precision (default Firedrake type)
    model.double()
    # Move model to device
    model.to(device)

    # Evaluate model
    error = evaluate_by_point(model, dataloader)
    logger.info(f"\n\t Error (metric: {evaluation_metric}): {error:.4e}")