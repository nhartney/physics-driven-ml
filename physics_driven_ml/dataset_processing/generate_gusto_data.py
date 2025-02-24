""""
A script to solve the diffusion equation using Gusto, with a number of different
diffusion coefficients. The equation is solved in time with the forward Euler
method.
"""
import os
from firedrake import (RectangleMesh, SpatialCoordinate, Function,
                       CheckpointFile, interpolate)
from gusto import *
import numpy as np
from numpy.random import default_rng
from tqdm.auto import tqdm, trange


def random_field(V, N, m=5, σ=0.6, tqdm=False, seed=2023):
    """Generate N 2D random fields with m modes."""
    rng = default_rng(seed)
    x, y = SpatialCoordinate(V.ufl_domain())
    fields = []
    for _ in trange(N, disable=not tqdm):
        r = 0
        for _ in range(m):
            a, b = rng.standard_normal(2)
            k1, k2 = rng.normal(0, σ, 2)
            θ = 2 * pi * (k1 * x + k2 * y)
            r += Constant(a) * cos(θ) + Constant(b) * sin(θ)
        fields.append(interpolate(sqrt(1 / m) * r, V))
    return fields


def generate_pde_solutions(domain, V, kappas, f, dt):
    
    solutions = []
    for kappa in kappas:
        diffusion_params = DiffusionParameters(kappa=kappa)
        equation = DiffusionEquation(domain, V, "q",
                                     diffusion_parameters=diffusion_params)

        # I/O
        output = OutputParameters(
            dirname="gusto_diffusion_eqn_kappa="+str(kappa))
        io = IO(domain, output)

        # Time stepper
        stepper = Timestepper(equation, ForwardEuler(domain), io)
        
        stepper.fields('q').interpolate(f)
           
        # ---------------------------------------------------------------- #
        # Run
        # ---------------------------------------------------------------- #
        
        stepper.run(0, tmax=1)
        logger.info(f'Evaluated solution for kappa = {kappa}')
        
        solutions.append(stepper.fields('q'))

    return solutions


def generate_diffusion_data(ntrain, ntest):
    
    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    Lx = Ly = 1
    nx = ny = 50
    dt = 0.01

    mesh = RectangleMesh(nx, ny, Lx, Ly, name="mesh")
    domain = Domain(mesh, dt, "CG", 1)

    V = domain.spaces("DG")
    Vu = VectorFunctionSpace(mesh, "CG", 1)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    x,y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(sin(pi * x) * sin(pi * y))

    # kappas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    # generate random kappas
    kappas = random_field(V, N=ntrain+ntest, tqdm=True)
    
    logger.info('Generate PDE solutions')

    # Generate PDE solutions
    solutions = generate_pde_solutions(domain, V, kappas, f, dt)
    
    # Add noise to PDE solutions
    logger.info('Add noise to PDE solutions')
    noisy_solns = []
    scale_noise = float(1)
    for q in solutions:
        sample = Function(V).assign(q)
        noise = scale_noise * np.random.rand(V.dim())
        # add noise to PDE solutions
        sample.dat.data[:] += noise
        noisy_solns.append(sample)

    # Split data into test and training sets
    train_kappas, test_kappas = kappas[:ntrain], kappas[ntrain:]
    train_solns, test_solns = solutions[:ntrain], solutions[ntrain:]
    train_obs, test_obs = noisy_solns[:ntrain], noisy_solns[ntrain:]

    # save training and testing data
    dataset_dir = os.path.join(
        "/Users/Jemma/Nell/code/physics-driven-ml/data/datasets",
        "gusto_diffusion_data")

    # Save train data
    with CheckpointFile(os.path.join(dataset_dir, "train_data.h5"), "w") as afile:
        afile.h5pyfile["n"] = ntrain
        afile.save_mesh(mesh)
        for i, (k, u, u_obs) in enumerate(zip(train_kappas, train_solns,
                                              train_obs)):
            afile.save_function(k, idx=i, name="kappa")
            afile.save_function(u_obs, idx=i, name="u_obs")

    # Save test data
    with CheckpointFile(os.path.join(dataset_dir, "test_data.h5"), "w") as afile:
        afile.h5pyfile["n"] = ntest
        afile.save_mesh(mesh)
        for i, (k, u, u_obs) in enumerate(zip(test_kappas, test_solns,
                                              test_obs)):
            afile.save_function(k, idx=i, name="kappa")
            afile.save_function(u_obs, idx=i, name="u_obs")


# call function to generate data
generate_diffusion_data(4, 2)
