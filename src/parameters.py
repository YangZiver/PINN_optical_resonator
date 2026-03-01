# parameters.py
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
# neural networks parameters
input_dim: int = 2
output_dim: int = 2
hidden_dim: int = 128
num_hidden: int = 4
fourier_dim: int = 256
fourier_scale: float = 300.0  

# PINN solver parameters
T0_width: float = 50.0  
T_window: float = 40 * T0_width  
T_min: float = -T_window / 2
T_max: float = T_window / 2
T_grid: NDArray = np.linspace(T_min, T_max, 8192, endpoint=False)
# train PINN
lr_adam: float = 1e-4# Adam optimizer learning rate
lr_lbfgs: float = 0.01# L-bfgs optimizer learning rate  
num_epochs_adam: int = 20000  
num_epochs_lbfgs: int = 2000  
lbfgs_max_eval: int = 4000
initial_weights: dict[str, int] = {"pde": 500, "ic": 500, "data": 100}
rounds: int = 200 # total train rounds
# number of sampling points
numIC: int = len(T_grid)  
numPDE: int = 10000  # 

# physical parameters
wavelength_nm: float = 1560.0
c_nm_ps: float = 299792458 * 1e9 / 1e12
omega0: float = 2 * np.pi * c_nm_ps / wavelength_nm
wll: float = 1545e-9
wlh: float = 1575e-9
cc: float = 3e-4
Omega_g: float = 2.0 * (cc / wll - cc / wlh) * np.pi #disperison bandwidth
# initial pulse at z = 0
power: float = 0.8 # initial pulse power
Chrip: float = 0.0
initial_pulse: NDArray = (
    np.sqrt(power)
    * 1
    / np.cosh(T_grid / T0_width)
    * np.exp(-1j * 0.5 * Chrip * (T_grid / T0_width) ** 2)
)
# device parameters
# EDF(Erbium-Doped Fiber)
EDF_PARAMS: dict[str, float] = {
    "L": 10.0,
    "alpha": 0.0,
    "beta2": 0.028,
    "beta3": 0.0005,
    "gamma": 0.002,
    "g_ss": 3.0,
    "E_s": 10.0,
    "Omega_g": Omega_g,
    "omega0": omega0,
    "T_R": 0.0,
}
# SMF(Single-Mode Fiber)
SMF1_PARAMS: dict[str, float] = {
    "L": 25,
    "alpha": 0.0001,
    "beta2": -0.0216,
    "beta3": 0.0005,
    "gamma": 0.0013,
    "g_ss": 0.0,
    "E_s": 10.0,
    "Omega_g": Omega_g,
    "omega0": omega0,
    "T_R": 0.0,
}
SMF2_PARAMS: dict[str, float] = SMF1_PARAMS
# SA(Saturable Absorber)
SA_PARAMS: dict[str, float] = {"Esat": 220.0, "Tns": 0.4858, "deltaT": 0.2673}

# save precise data got from SSF
REF_DATA_DIR = "Data/SSF_Reference"
