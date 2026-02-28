# ssf_solver.py
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from numpy.typing import NDArray
import torch

class SSFSolver:
    """
    For EDF, g in NLSE vary with energy. PINN difficult to solve.
    The best solution I can think is SSF.
    """
    
    def __init__(self):
        pass
    
    def GNLSE_EDF(self,
                  input_pulse: NDArray | torch.Tensor,
                  params: dict[str, float],
                  T_grid: NDArray) -> NDArray:
        """
        use SSF to solve GNLSE in EDf
        """
        # ensure input pulse is NDArray
        if torch.is_tensor(input_pulse):
            A0 = input_pulse.detach().cpu().numpy()
        else:
            A0 = input_pulse
        # get coefficents in GNLSE
        L = params['L']
        beta2 = params['beta2']
        gamma = params['gamma']
        gss = params['g_ss']
        Es = params['E_s']
        alpha = params['alpha']
        T_R = params['T_R']
        Omega_g = params['Omega_g']
        omega0 = params['omega0']
        # time window parameters
        N = len(T_grid)
        twidth = T_grid[-1] - T_grid[0]
        dT = T_grid[1] - T_grid[0]
        # frenquency grid
        V = 2 * np.pi * np.linspace(-1/(2*dT), 1/(2*dT), N, endpoint=False)
        # calculate disperison bandwidth
        if Omega_g is None:
            wll = 1545e-9
            wlh = 1575e-9
            cc = 3e-4
            Omega_g = 2.0 * (cc / wll - cc / wlh) * np.pi
        
        # SSF parameters
        h = 0.1  # step(m)
        Save_step = 1
        Seg_num = int(L / h)
        Save_num = 1 + int(Seg_num // Save_step)
        
        # initialize
        As = A0.copy()
        # SSF(Split Step Fourier)
        for ind_a in range(1, Seg_num + 1):
            # Nonliner step(formmer part)
            abs_As_sq = np.abs(As)**2
            N_pre = 1j * gamma * (
                abs_As_sq + 
                (1j/omega0) * (-1j*V) * abs_As_sq +
                T_R * abs_As_sq
            )
            As = As * np.exp(0.5 * h * N_pre)
            # Liner step(frequency domain)
            E_z = np.trapezoid(np.abs(As)**2, dx=dT)
            g = gss * np.exp(-E_z/Es)
            beta3 = params['beta3']  
            D = (1j/2) * beta2 * V**2 + g/2 - g/(2*Omega_g**2) * V**2 - alpha/2 + (beta3/6) * 1j * V**3
            # fourier transoform to frequency domain
            A_f = fftshift(ifft(fftshift(As))) * N
            A_f = A_f * np.exp(h * D)
            # back to time domain
            As = fftshift(fft(fftshift(A_f))) / N
            # Nonliner step(latter part)
            abs_As_sq = np.abs(As)**2
            N_post = 1j * gamma * (
                abs_As_sq + 
                (1j/omega0) * (-1j*V) * abs_As_sq +
                T_R * abs_As_sq
            )
            As = As * np.exp(h * N_post)
        
        return As
    
    def apply_to_edf(self,
                     input_pulse: NDArray | torch.Tensor,
                     params: dict[str, float],
                     T_grid: NDArray,
                     device: str = 'cuda') -> tuple[torch.Tensor, NDArray]:
        """
        get output pulse
        """
        output_pulse_np = self.GNLSE_EDF(input_pulse, params, T_grid)
        output_pulse_tensor = torch.tensor(
            output_pulse_np, 
            dtype=torch.complex64,
            device=device
        )
        return output_pulse_tensor, output_pulse_np
