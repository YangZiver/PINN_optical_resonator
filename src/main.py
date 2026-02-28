# main.py
import numpy as np
import torch
from numpy.fft import fftshift, ifft
from numpy.typing import NDArray
import parameters
import plotting
import ssf_solver
import train
import utils 
from SA import SaturableAbsorber

def edf_spec_norm(output_pulse: NDArray) ->NDArray:
    fft_edf = fftshift(ifft(fftshift(output_pulse))) * len(parameters.T_grid)
    spec_edf = np.abs(fft_edf) ** 2
    spec_db = 10 * np.log10(spec_edf + 1e-16)
    wl_axis, wl_sorted_indices = utils.compute_wavelength_axis()
    spec_db_sorted = spec_db[wl_sorted_indices]
    s_min, s_max = np.min(spec_db_sorted), np.max(spec_db_sorted)
    spec_norm_edf = (
        (spec_db_sorted - s_min) / (s_max - s_min)
        if s_max > s_min
        else spec_db_sorted
    )
    return spec_norm_edf

def main():
    utils.create_directories()
    print(f"\n{'=' * 60}")
    print(f"Start Pulse Simulation")
    print(f"Total Rounds: {parameters.rounds}")
    print(f"{'=' * 60}")
    ssf = ssf_solver.SSFSolver()
    device = utils.setting_device()
    current_pulse = torch.tensor(
        parameters.initial_pulse, dtype=torch.complex64, device=device
    )
    wl_axis, wl_sorted_indices = utils.compute_wavelength_axis()
    for round_num in range(1, parameters.rounds + 1):
        print(f"\n{'=' * 60}")
        print(f">> {round_num}/{parameters.rounds} start")
        print(f"{'=' * 60}")
        # 1. EDF use SSF
        print(f"\n>> {round_num}rounds 1. SSF simulate EDF...")
        current_pulse, edf_output_np = ssf.apply_to_edf(
            current_pulse, 
            parameters.EDF_PARAMS, 
            parameters.T_grid, device=device
        )
        edf_intensity = np.abs(edf_output_np) ** 2
        spec_norm_edf = edf_spec_norm(edf_output_np)
        # plot EDF results
        

        plotting.plot_time(parameters.T_grid,
                                edf_intensity,
                                "EDF",
                                round_num)
        plotting.plot_spec(wl_axis,
                                spec_norm_edf,
                                "EDF",
                                round_num)
        # 2. SMF1 use PINN
        print(f"\n>> {round_num} 2. PINN simulate SMF1 ...")
        smf1_train = train.Train('SMF1',
                                      round_num,
                                      parameters.SMF1_PARAMS,
                                      current_pulse)
        smf1_train_model = smf1_train.train_model() 
        smf1_result_dict = smf1_train.analyze_model()
        current_pulse = smf1_result_dict["output_pulse"]
        plotting.plot_time(
            parameters.T_grid,
            smf1_result_dict["intensity"],
            "SMF1",
            round_num
        )
        plotting.plot_spec(wl_axis,
                           smf1_result_dict["spec_norm"],
                           "SMF1",
                           round_num)
        plotting.plot_loss_history(smf1_train.history, "SMF1", round_num)
        # 3. Saturable Absorber
        print(f"\n>> {round_num} rounds 3. Apply saturable absorber...")
        SA_train = SaturableAbsorber(parameters.SA_PARAMS)
        sa_output = SA_train.apply(current_pulse)
        current_pulse = sa_output
        plotting.plot_time(parameters.T_grid,
                           SA_train.intensity,
                           "SA",
                           round_num)
        # 4. SMF2 use PINN
        print(f"\n>> {round_num} 4. PINN simulate SMF2 ...")
        smf2_train = train.Train('SMF2',
                                      round_num,
                                      parameters.SMF2_PARAMS,
                                      current_pulse)
        smf2_train_model = smf2_train.train_model() 
        smf2_result_dict = smf2_train.analyze_model()
        current_pulse = smf2_result_dict["output_pulse"]
        plotting.plot_time(
            parameters.T_grid,
            smf2_result_dict["intensity"],
            "SMF2",
            round_num
        )
        plotting.plot_spec(wl_axis,
                           smf2_result_dict["spec_norm"],
                           "SMF2",
                           round_num)
        plotting.plot_loss_history(smf2_train.history, "SMF2", round_num)
        print(f"\n>> {round_num} rounds finish！")
    print(f"\n{'=' * 60}")
    print(f"{'=' * 60}")
if __name__ == "__main__":
    main()
