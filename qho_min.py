# -*- coding: utf-8 -*-
"""
QHO — Minimal Finite-Difference Demo (clean & runnable)
- Build 1D harmonic oscillator Hamiltonian (tridiagonal, 2nd-order FD)
- Solve lowest k eigenpairs
- Plot eigenfunctions over potential
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

def build_grid(n_points=400, x_max=6.0):
    x = np.linspace(-x_max, x_max, n_points, dtype=np.float64)
    dx = x[1] - x[0]
    return x, dx

def build_hamiltonian(x, dx, m=1.0, omega=1.0, hbar=1.0):
    n = x.size
    # Kinetic: - (hbar^2 / 2m) d^2/dx^2  → tri-diagonal
    diag_T = np.full(n, (hbar**2) / (m * dx**2), dtype=np.float64)
    off_T  = np.full(n-1, -(hbar**2) / (2*m*dx**2), dtype=np.float64)
    # Potential
    V = 0.5 * m * (omega**2) * x**2
    diag = diag_T + V
    off = off_T
    return diag, off, V

def normalize(psi, dx):
    return psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)

def solve_lowest_k(diag, off, k=6, dx=1.0):
    e, v = eigh_tridiagonal(diag, off, select="i", select_range=(0, k-1))
    v = np.column_stack([normalize(v[:, i], dx) for i in range(v.shape[1])])
    return e, v

def plot_eigen(x, V, energies, states, save_path="qho_min_levels.png", show=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, V, label="V(x)=1/2 m ω² x²", alpha=0.9)
    for n in range(states.shape[1]):
        ax.axhline(energies[n], linestyle="--", linewidth=0.8)
        ax.plot(x, 0.6*np.real(states[:, n]) + energies[n], label=f"n={n}, E≈{energies[n]:.4f}")
    ax.set_xlabel("x"); ax.set_ylabel("Energy / Offsets")
    ax.set_title("QHO — Minimal Eigenstates over Potential")
    ax.grid(True); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(save_path, dpi=150)
    if show: plt.show()
    plt.close(fig)

def main():
    # Params (edit freely)
    n_points, x_max = 400, 6.0
    m, omega, hbar = 1.0, 1.0, 1.0
    k = 6

    x, dx = build_grid(n_points, x_max)
    diag, off, V = build_hamiltonian(x, dx, m, omega, hbar)
    energies, states = solve_lowest_k(diag, off, k=k, dx=dx)

    # quick log vs theory En = (n+1/2) ħω
    for i, E in enumerate(energies):
        Eth = (i + 0.5) * hbar * omega
        print(f"n={i}: E_num={E:.8f}, E_theory={Eth:.8f}, rel_err={(abs(E-Eth)/Eth):.2e}")

    plot_eigen(x, V, energies, states, save_path="qho_min_levels.png", show=False)
    print("[done] Saved figure: qho_min_levels.png")

if __name__ == "__main__":
    main()
