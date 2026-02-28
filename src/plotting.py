# plotting.py
import os
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Union

# -------------------- helper functions --------------------
def ensure_dir(filepath: str) -> None:
    """ensure directionary exist"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def save_figure(fig: plt.Figure, base_path: str, dpi: int = 300) -> None:
    """
    save figures in .png
    if also want .pdf, just cancel the comment
    """
    pdf_path = base_path + ".pdf"
    png_path = base_path + ".png"
    ensure_dir(pdf_path)
    #fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight") 

def save_data_npy(data: NDArray, filepath: str) -> None:
    """save data in .npy"""
    ensure_dir(filepath)
    np.save(filepath, data)

def save_data_txt(data: NDArray, filepath: str, header: str) -> None:
    """optional, save data in .txt"""
    ensure_dir(filepath)
    np.savetxt(filepath, data, fmt="%10.6f", header=header, encoding="utf-8")

# -------------------- general line plot --------------------
def plot_line(
    x: NDArray,
    y: NDArray,
    section_name: str,
    plot_type: str,
    rounds: int,
    xlabel: str,
    ylabel: str,
    title: str,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    save_data: bool = True,
    #data_header: Optional[str] = None,
) -> None:
    """
    general function to plot line
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (Round {rounds})")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # save figures
    img_base = os.path.join("Images", section_name, f"{plot_type}_round{rounds}")
    save_figure(fig, img_base, dpi=300)
    plt.close(fig)
    # save plot data
    if save_data:
        data = np.column_stack((x, y))
        data_path = os.path.join("Data", section_name, f"{plot_type}_data_round{rounds}.npy")
        save_data_npy(data, data_path)
        # if data_header:
        #     txt_path = os.path.join("Data", section_name, f"{plot_type}_data_round{rounds}.txt")
        #     save_data_txt(data, txt_path, data_header)

# -------------------- plot application interface --------------------
def plot_time(
    T_grid: NDArray,
    intensity: NDArray,
    section_name: str,
    rounds: int,
) -> None:
    """time-domain plot"""
    plot_line(
        x=T_grid,
        y=intensity,
        section_name=section_name,
        plot_type="time",
        rounds=rounds,
        xlabel="Time (ps)",
        ylabel="Intensity (W)",
        title="Time Domain",
        xlim=(-150, 150),
        #data_header="Time(ps) Intensity(W)",
    )

def plot_spec(
    wave_len: NDArray,
    spec_intensity: NDArray,
    section_name: str,
    rounds: int,
) -> None:
    """specturm plot"""
    plot_line(
        x=wave_len,
        y=spec_intensity,
        section_name=section_name,
        plot_type="spec",
        rounds=rounds,
        xlabel="Wavelength (nm)",
        ylabel="Normalized Intensity",
        title="Spectrum",
        xlim=(1545, 1575),
        ylim=(0.0, 1.1),
        #data_header="Wavelength(nm) NormIntensity(dB)",
    )

def plot_sampling_points(
    z_pde: NDArray,
    t_pde: NDArray,
    z_ic: NDArray,
    t_ic: NDArray,
    params: Dict[str, float],
    section_name: str,
    round_num: int,
) -> None:
    """sampling points distribution plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(z_pde, t_pde, c="blue", alpha=0.6, s=5, label=f"PDE ({len(z_pde)} points)")
    ax.scatter(z_ic, t_ic, c="red", alpha=0.8, s=10, label=f"IC ({len(z_ic)} points)")
    ax.set_xlabel("Length z (m)", fontsize=12)
    ax.set_ylabel("Time T (ps)", fontsize=12)
    ax.set_title(f"{section_name} - Sampling Points (Round {round_num})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    # fiber end
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.axvline(x=params["L"], color="k", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.text(0, ax.get_ylim()[1] * 0.95, "z=0", fontsize=10, ha="right")
    ax.text(params["L"], ax.get_ylim()[1] * 0.95, f"z={params['L']}", fontsize=10, ha="left")
    ax.set_xlim(-0.1, params["L"] * 1.1)
    fig.tight_layout()
    # save figures
    img_base = os.path.join("Images", section_name, f"sampling_round{round_num}")
    save_figure(fig, img_base, dpi=300)
    plt.close(fig)
    # save data(pde and ic)
    data_pde = np.column_stack((z_pde, t_pde))
    data_ic = np.column_stack((z_ic, t_ic))
    save_data_npy(data_pde, os.path.join("Data", section_name, f"sampling_pde_round{round_num}.npy"))
    save_data_npy(data_ic, os.path.join("Data", section_name, f"sampling_ic_round{round_num}.npy"))
    # Optional, save data in .txt
    # save_data_txt(data_pde, os.path.join("Data", section_name, f"sampling_pde_round{round_num}.txt"), "z(m) T(ps)")
    # save_data_txt(data_ic, os.path.join("Data", section_name, f"sampling_ic_round{round_num}.txt"), "z(m) T(ps)")

def plot_loss_history(
    loss_history: Dict[str, List[float]],
    section_name: str,
    round_num: int,
) -> None:
    """
    plot loss curve
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes = axes.flatten()
    epochs = np.arange(1, len(loss_history["total_loss"]) + 1)
    # total loss
    ax = axes[0]
    ax.semilogy(epochs, loss_history["total_loss"], "b-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title(f"{section_name} - Total Loss (Round {round_num})")
    ax.grid(True, alpha=0.3)
    # pde loss, ic loss, data loss
    ax = axes[1]
    ax.semilogy(epochs, loss_history["pde_loss"], "r-", linewidth=2, label="PDE Loss")
    ax.semilogy(epochs, loss_history["ic_loss"], "g-", linewidth=2, label="IC Loss")
    if "data_loss" in loss_history:
        ax.semilogy(epochs, loss_history["data_loss"], "m-", linewidth=2, label="Data Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{section_name} - PDE & IC Loss (Round {round_num})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # all losses comprison
    ax = axes[2]
    ax.semilogy(epochs, loss_history["total_loss"], "b-", linewidth=2, alpha=0.7, label="Total")
    ax.semilogy(epochs, loss_history["pde_loss"], "r-", linewidth=1, alpha=0.7, label="PDE")
    ax.semilogy(epochs, loss_history["ic_loss"], "g-", linewidth=1, alpha=0.7, label="IC")
    if "data_loss" in loss_history:
        ax.semilogy(epochs, loss_history["data_loss"], "m-", linewidth=1, alpha=0.7, label="Data")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{section_name} - Loss Comparison (Round {round_num})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    # save figures
    img_base = os.path.join("Images", section_name, f"loss_round{round_num}")
    save_figure(fig, img_base, dpi=300)
    plt.close(fig)
    # save data, .npy or .txt
    if "data_loss" in loss_history:
        loss_data = np.column_stack((
            epochs,
            loss_history["total_loss"],
            loss_history["pde_loss"],
            loss_history["ic_loss"],
            loss_history["data_loss"],
        ))
        #header = "Epoch Total_Loss PDE_Loss IC_Loss Data_Loss"
    else:
        loss_data = np.column_stack((
            epochs,
            loss_history["total_loss"],
            loss_history["pde_loss"],
            loss_history["ic_loss"],
        ))
        #header = "Epoch Total_Loss PDE_Loss IC_Loss"
    data_path = os.path.join("Data", section_name, f"loss_data_round{round_num}.npy")
    save_data_npy(loss_data, data_path)
    # txt_path = os.path.join("Data", section_name, f"loss_data_round{round_num}.txt")
    # save_data_txt(loss_data, txt_path, header)