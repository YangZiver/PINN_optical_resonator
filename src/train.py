# train.py
import pinn
import torch
import numpy as np
from numpy.typing import NDArray
import utils
import parameters
from typing import Union
import os
class Train:
    def __init__(self, 
                 section_name: str,
                 rounds: int,
                 params: dict[str, float],
                 input_pulse: NDArray | torch.Tensor):
        self.section_name = section_name
        self.rounds = rounds
        self.params = params
        self.input_pulse = input_pulse
        if self.rounds <= 30:
            use_fourier = False
        else:
            use_fourier = True
        self.model = pinn.Pinn(parameters.input_dim,
                               parameters.output_dim,
                               parameters.hidden_dim,
                               parameters.num_hidden,
                               self.rounds,
                               use_fourier=use_fourier,
                               fourier_dim=parameters.fourier_dim,
                               fourier_scale=parameters.fourier_scale)
        self.device = utils.setting_device()

    def get_input_pulse(self) -> None:
        """get a real pulse and a imag pulse"""
        device = self.device
        if isinstance(self.input_pulse, np.ndarray):
            if self.input_pulse.dtype in [np.float32, np.float64]:
                input_tensor = torch.tensor(
                    self.input_pulse, dtype=torch.float32
                )
                input_pulse_real = input_tensor.unsqueeze(1).to(device)
                input_pulse_imag = torch.zeros_like(input_pulse_real).to(
                    device
                )
            else:
                input_pulse_real = (
                    torch.tensor(self.input_pulse.real, dtype=torch.float32)
                    .unsqueeze(1)
                    .to(device)
                )
                input_pulse_imag = (
                    torch.tensor(self.input_pulse.imag, dtype=torch.float32)
                    .unsqueeze(1)
                    .to(device)
                )
        elif torch.is_tensor(self.input_pulse):
            if self.input_pulse.is_complex():
                input_pulse_real = self.input_pulse.real.unsqueeze(1).to(
                    device
                )
                input_pulse_imag = self.input_pulse.imag.unsqueeze(1).to(
                    device
                )
            else:
                input_pulse_real = self.input_pulse.unsqueeze(1).to(
                    device
                )
                input_pulse_imag = torch.zeros_like(input_pulse_real).to(
                    device
                )
        else:
            raise ValueError(f"input pulse type not supported: {type(self.input_pulse)}")
        self.u = input_pulse_real
        self.v = input_pulse_imag

    def normalize_zT(self,
                     z: torch.Tensor | NDArray,
        t: torch.Tensor | NDArray) -> tuple[torch.Tensor, ...] |tuple[NDArray, ...]:
        """normalize (z, t) to [-1, 1]"""
        z_norm = utils.normalize_input(z,
                                       self.params['L'],
                                       0.0)
        t_norm = utils.normalize_input(t,
                                       parameters.T_max,
                                       parameters.T_min)
        return (z_norm, t_norm)

    def sampling_points(self) -> None:
        """sample some points to train"""
        bounds = [[0.0, self.params["L"]], [parameters.T_min, parameters.T_max]]
        samples = utils.latin_hypercube_sampling(parameters.numPDE, 2, bounds)
        # PDE data
        z_physical_pde = samples[:, 0:1]  # keep 2 dimension
        t_physical_pde = samples[:, 1:2]
        z_norm_pde, t_norm_pde = self.normalize_zT(z_physical_pde, t_physical_pde)
        # ensure row vector
        z_norm_pde = z_norm_pde.reshape(-1, 1)
        t_norm_pde = t_norm_pde.reshape(-1, 1)
        self.z_pde = torch.tensor(z_norm_pde, 
                                  dtype=torch.float32,
                                  device=self.device,
                                  requires_grad=True)
        self.t_pde = torch.tensor(t_norm_pde,
                                  dtype=torch.float32,
                                  device=self.device,
                                  requires_grad=True)
        # ic_data
        t_physical_ic = np.linspace(parameters.T_min, 
                                    parameters.T_max,
                                    parameters.numIC)
        z_physical_ic = np.zeros_like(t_physical_ic)
        z_norm_ic, t_norm_ic = self.normalize_zT(z_physical_ic, t_physical_ic)
        z_norm_ic = z_norm_ic.reshape(-1, 1)
        t_norm_ic = t_norm_ic.reshape(-1, 1)

        self.z_ic = torch.tensor(z_norm_ic,
                                 dtype=torch.float32,
                                 device=self.device,
                                 requires_grad=True)
        self.t_ic = torch.tensor(t_norm_ic,
                                 dtype=torch.float32,
                                 device=self.device,
                                 requires_grad=True)
        self.z_physical_pde = z_physical_pde.flatten()
        self.t_physical_pde = t_physical_pde.flatten()
        self.z_physical_ic = z_physical_ic.flatten()
        self.t_physical_ic = t_physical_ic.flatten()

    def query_pinn_solution(self) -> torch.Tensor:
        """
        get complex pulse solution from PINN
        """
        z_end = self.params['L']
        z_physical_data = np.full_like(parameters.T_grid, z_end)
        t_physical_data = parameters.T_grid
        z_norm, t_norm = self.normalize_zT(z_physical_data, t_physical_data)
        z = torch.tensor(z_norm, dtype=torch.float32, device=self.device).reshape(-1, 1)
        t = torch.tensor(t_norm, dtype=torch.float32, device=self.device).reshape(-1, 1)
        with torch.no_grad():
            u, v = self.model.forward(z, t)
        return (u + 1j * v).squeeze()

    def compute_intensity_and_spectrum(self,
                            u_end_np: NDArray,
                            v_end_np: NDArray) -> tuple[NDArray, NDArray]:
        """compute pulse intensity and specturm intensity"""
        intensity = u_end_np ** 2 + v_end_np ** 2
        pulse_complex_np = u_end_np + 1j * v_end_np
        spec_complex = np.fft.fftshift(np.fft.fft(pulse_complex_np))
        spec_intensity = np.abs(spec_complex)**2
        spec_db = 10 * np.log10(spec_intensity + 1e-16)
        wl_axis, sorted_indices = self.compute_wavelength_axis()
        spec_db_sorted = spec_db[sorted_indices]
        s_min, s_max = np.min(spec_db_sorted), np.max(spec_db_sorted)
        if s_max > s_min:
            spec_norm = (spec_db_sorted - s_min) / (s_max - s_min)
        else:
            spec_norm = spec_db_sorted
        return intensity, spec_norm

    def compute_wavelength_axis(self) -> tuple[NDArray, NDArray]:
        """
        compute wave length axis
        """
        n = len(parameters.T_grid)
        dt = parameters.T_grid[2] - parameters.T_grid[1]  
        V = 2 * np.pi * np.linspace(-1/(2*dt), 1/(2*dt), n, endpoint=False)
        freq = V / (2 * np.pi)
        f0 = parameters.omega0 / (2 * np.pi)
        wl = parameters.c_nm_ps / (freq + f0)
        sorted_indices = np.argsort(wl)
        wl_sorted = wl[sorted_indices]
        return wl_sorted, sorted_indices


    def analyze_model(self) -> dict[str, Union[torch.Tensor, NDArray]]:
        output_pulse = self.query_pinn_solution()
        u_end_np = (output_pulse.real).detach().cpu().numpy()
        v_end_np = (output_pulse.imag).detach().cpu().numpy()
        intensity, spec_norm = self.compute_intensity_and_spectrum(u_end_np,
                                                                   v_end_np)
        return {
            'output_pulse': output_pulse,
            'intensity': intensity,
            'spec_norm': spec_norm,
        }

    def calculate_derivatives(self,
                            z: torch.Tensor, 
                            t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """calculate derivatives in pde"""
        u_pred, v_pred = self.model(z, t)
        d_z = 2.0 / self.params["L"]
        d_t = 2.0 / (parameters.T_max - parameters.T_min)
        intensity = u_pred ** 2 + v_pred ** 2
        messWithu = intensity * u_pred
        messWithv = intensity * v_pred
        u_z = utils.gradients(u_pred, z) * d_z
        v_z = utils.gradients(v_pred, z) * d_z
        u_t = utils.gradients(u_pred, t) * d_t
        v_t = utils.gradients(v_pred, t) * d_t
        u_tt = utils.gradients(u_t, t) * d_t
        #u_ttt = utils.gradients(u_tt, t) * d_t
        v_tt = utils.gradients(v_t, t) * d_t
        #v_ttt = utils.gradients(v_tt, t) * d_t
        intensity_t = utils.gradients(intensity, t) * d_t
        messWithu_t = utils.gradients(messWithu, t) * d_t
        messWithv_t = utils.gradients(messWithv, t) * d_t
        return (
            u_pred,
            v_pred,
            messWithu,
            messWithv,
            u_z,
            v_z,
            u_tt,
            v_tt,
            # u_ttt,
            # v_ttt,
            intensity_t,
            messWithu_t,
            messWithv_t,
        )


    def pde_loss(self) -> torch.Tensor:
        """calculate pde loss"""
        beta2 = self.params["beta2"]
        beta3 = self.params["beta3"]
        alpha = self.params["alpha"]
        gamma = self.params["gamma"]
        g = self.params["g_ss"]
        Omega_g = self.params["Omega_g"]
        omega0 = self.params["omega0"]
        T_R = self.params["T_R"]
        (
            u,
            v,
            messWithu,
            messWithv,
            u_z,
            v_z,
            u_tt,
            v_tt,
            # u_ttt,
            # v_ttt,
            intensity_t,
            messWithu_t,
            messWithv_t,
        ) = self.calculate_derivatives(self.z_pde, self.t_pde)
        residual_real = (
            u_z
            + (alpha - g) / 2 * u
            - beta2 / 2 * v_tt
            #- beta3 / 6 * u_ttt
            - g / (2 * Omega_g**2) * u_tt
            + gamma * messWithv
            + gamma / omega0 * messWithu_t
            - gamma * T_R * v * intensity_t
        )
        residual_imag = (
            v_z
            + (alpha - g) / 2 * v
            + beta2 / 2 * u_tt
            #- beta3 / 6 * v_ttt
            - g / (2 * Omega_g**2) * v_tt
            - gamma * messWithu
            + gamma / omega0 * messWithv_t
            + gamma * T_R * u * intensity_t
        )
        pde_loss = torch.mean(residual_real**2 + residual_imag**2)
        return pde_loss

    def ic_loss(self) -> torch.Tensor:
        """calculate initial condition loss"""
        u_ic, v_ic = self.model(self.z_ic, self.t_ic)
        ic_loss = torch.mean(
            (u_ic - self.u) ** 2 + (v_ic - self.v) ** 2
        )
        return ic_loss

    def data_loss(self) -> torch.Tensor:
        """calculate data loss"""
        self.load_reference_output(self.rounds, self.section_name)
        if hasattr(self, 'z_data') and self.z_data is not None:
            u_data, v_data = self.model(self.z_data, self.t_data)
            data_loss = torch.mean((u_data - self.ref_real) ** 2 + (v_data - self.ref_imag) ** 2)
        else:
            data_loss = torch.tensor(0.0, device=self.device)
        # total_loss =  data_loss
        # if not hasattr(self, "history"):
        #     self.history = {
        #         "total_loss": [],
        #     }
        # self.history["total_loss"].append(total_loss.item())
        return data_loss

    def losses(self) -> torch.Tensor:
        """calcualte total loss"""
        pde_w = parameters.initial_weights['pde']
        ic_w = parameters.initial_weights['ic']
        data_w = parameters.initial_weights['data']
        self.pde_loss_val = self.pde_loss()
        self.ic_loss_val = self.ic_loss()
        self.data_loss_val = self.data_loss()
        self.total_loss = pde_w * self.pde_loss_val + ic_w * self.ic_loss_val \
        + data_w * self.data_loss_val
        if not hasattr(self, "history"):
            self.history = {
                "total_loss": [],
                "pde_loss": [],
                "ic_loss": [],
                "data_loss": [],
            }
        self.history["total_loss"].append(self.total_loss.item())
        self.history["pde_loss"].append(self.pde_loss_val.item())
        self.history["ic_loss"].append(self.ic_loss_val.item())
        self.history["data_loss"].append(self.data_loss_val.item())
        return self.total_loss
    
    def load_reference_output(self,
                              round_num: int,
                              section_name: str):
        """
        load reference output from Data/SSF_refernce
        """
        file_path = os.path.join(parameters.REF_DATA_DIR, f"{section_name}_round{round_num}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reference file not exist!: {file_path}")
        ref_np = np.load(file_path)                     
        ref_tensor = torch.tensor(ref_np, dtype=torch.complex64, device=self.device)
        self.set_reference_output(ref_tensor)           
    
    def set_reference_output(self, ref_output_tensor):
        """save reference output and generate data sampling points"""
        self.ref_real = ref_output_tensor.real.unsqueeze(1).to(self.device)
        self.ref_imag = ref_output_tensor.imag.unsqueeze(1).to(self.device)
        z_physical_data = np.full_like(parameters.T_grid, self.params['L'])
        t_physical_data = parameters.T_grid
        z_norm_data, t_norm_data = self.normalize_zT(z_physical_data, t_physical_data)
        self.z_data = torch.tensor(z_norm_data, dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.t_data = torch.tensor(t_norm_data, dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.z_data.requires_grad_(False)
        self.t_data.requires_grad_(False)

    def train_model(self):
        """use PINN to solve pde"""
        self.sampling_points()
        self.get_input_pulse()
        utils.adam_optimizer(model=self.model, 
                            loss_fn=lambda: self.losses(),
                            model_name=self.section_name)
        utils.lbfgs_optimizer(model=self.model,
                              loss_fn=lambda: self.losses(),
                               model_name = self.section_name)
        # Optional, visualize loss history
        # if hasattr(model, 'history') and model.history:
        #plotting.plot_loss_history(model.history, model_name, round_num)
        return self.model

    