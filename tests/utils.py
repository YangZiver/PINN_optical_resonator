# utils.py
import numpy as np
from numpy.typing import NDArray
import torch
from torch.autograd import grad
import multiprocessing
import parameters
import os
from typing import Callable

def normalize_input(x: NDArray | torch.Tensor,
                    max_x: float = None,
                    min_x: float = None) -> torch.Tensor | NDArray:
    """normalize x to [-1, 1]"""
    return 2.0 * (x - min_x) / (max_x - min_x) - 1.0

def normalize_input_ndarray(x: NDArray) -> NDArray:
    """normalize x to [0, 1]"""
    return x / np.max(x)


def setting_device() -> torch.device:
    """set computation device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f">>> CUDA available | GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        total_cores = multiprocessing.cpu_count()
        calc_cores = max(1, total_cores - 2)
        torch.set_num_threads(calc_cores)
        print(f"used CPU cores: {calc_cores} / {total_cores}")
    print(f">>> Start computing | used device: {device}")
    return device

def gradients(u: torch.Tensor,
              x: torch.Tensor) -> torch.Tensor:
    return grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]


def latin_hypercube_sampling(n: int,
    dim: int,
    bounds: list[list[float]]=None) -> NDArray:
    samples = np.zeros((n, dim))
    for i in range(dim):
        samples[:, i] = np.random.permutation(n)

    samples = (samples + np.random.rand(n, dim)) / n
    if bounds is not None:
        for i in range(dim):
            low, high = bounds[i]
            samples[:, i] = low + samples[:, i] * (high - low)
    return samples



def adam_optimizer(model: torch.nn.Module,
                  loss_fn: Callable[[], torch.Tensor],
                  model_name: str) -> None:
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=parameters.lr_adam)
    for epoch in range(parameters.num_epochs_adam):
        optimizer_adam.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer_adam.step()
        if epoch % 1000 == 0:
            print(f"[{model_name}] Train Status: {epoch} rounds | Total Loss: {loss.item():.3e}")

def lbfgs_optimizer(model: torch.nn.Module,
                    loss_fn: Callable[[], torch.Tensor],
                    model_name: str) -> None:
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), 
                                        lr=parameters.lr_lbfgs, 
                                        max_iter=parameters.num_epochs_lbfgs,
                                        max_eval= parameters.lbfgs_max_eval,
                                        line_search_fn="strong_wolfe")
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    optimizer_lbfgs.step(closure)
    final_loss = loss_fn()
    print(f"[{model_name}]  | Total Loss: {final_loss.item():.3e}")


def create_directories() -> None:
    """create directories to save images and data"""
    sections = ["EDF", "SMF1", "SA", "SMF2"]
    for section in sections:
        os.makedirs(f"Images/{section}", exist_ok=True)
        os.makedirs(f"Data/{section}", exist_ok=True)

def compute_wavelength_axis() -> tuple[NDArray, NDArray]:
    """calculate corresponding wave length axis and sorted indices"""
    n = len(parameters.T_grid)
    dt = parameters.T_grid[2] - parameters.T_grid[1]  
    # frequency grid
    V = 2 * np.pi * np.linspace(-1/(2*dt), 1/(2*dt), n, endpoint=False)
    freq = V / (2 * np.pi)
    # center frenquency
    f0 = parameters.omega0 / (2 * np.pi)
    # calculate wave length
    wl = parameters.c_nm_ps / (freq + f0)
    # sort by indices
    sorted_indices = np.argsort(wl)
    wl_sorted = wl[sorted_indices]
    return wl_sorted, sorted_indices