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
from physics_driven_ml.dataset_processing import PointDataset, PDEDataset2

from physics_driven_ml.training.train_heat_problem_globally import sub_sample_point_data, calculate_global_loss


def evaluate_globally_by_point(model, point_dl, global_dl, disable_tqdm=False):
    """
    Evaluate the model on a given dataset.
    Compute the L2 error of the NN for every sample in the evaluation set, and then
    add these errors up to give an overall error for that model. (The 'sample' in the
    evaluation set is a global example, made up of all points in the domain.)
    """
    batch_size = 1

    model.eval()

    eval_steps = len(global_dl)
    total_error = 0.0

    point_data_subsets = sub_sample_point_data(point_dl)

    if len(point_data_subsets) != eval_steps:
        print("The number of data subsets does not match the number of global samples")
 
    
    for step_num, (subset, global_sample) in tqdm(enumerate(list(zip(point_data_subsets, global_dl))),
                                                    total=eval_steps, disable=disable_tqdm):
        
        subset_dl = DataLoader(subset, batch_size=batch_size, shuffle=False)

        # Do a forward pass on all points in the data subset and accumulate loss
        with torch.no_grad():
            loss = calculate_global_loss(subset, model)
            total_error += loss

        if step_num == eval_steps - 1:
            break

    # L2 error is the square root of the total error
    L2_error = total_error**0.5
    L2_error /= eval_steps
    return L2_error

if __name__ == "__main__":

    logger = get_logger("Evaluation")

    data_dir = os.path.join("/Users/Jemma/Nell/code/physics-driven-ml/data/")
    dataset = "heat_problem_global_validation_data"
    batch_size = 1
    device = "cpu"
    evaluation_metric = "L2"
    model_dir = "/Users/Jemma/Nell/code/physics-driven-ml/data/saved_models/heat_problem_globally/"
    model_version = "heat_problem_globally_epoch-18-error_0.00494"

    # Load dataset
    dataset_dir = os.path.join(data_dir, "datasets", dataset)
    logger.info(f"Loading dataset from {dataset_dir}\n")

    point_dataset = PointDataset(numpy_data=os.path.join(dataset_dir, "numpy_point_validate_data.npy"),
                           data_dir=data_dir)
    point_dataloader = DataLoader(point_dataset, batch_size=batch_size, shuffle=False)

    global_dataset = PDEDataset2(dataset=os.path.join(dataset_dir, "global_validate_data.h5"),
                                       data_dir=dataset_dir)
    global_dataloader = DataLoader(global_dataset, batch_size=batch_size,
                                 collate_fn=global_dataset.collate, shuffle=False)

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
    error = evaluate_globally_by_point(model, point_dataloader, global_dataloader)
    logger.info(f"\n\t Error (metric: {evaluation_metric}): {error:.4e}")