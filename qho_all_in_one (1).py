# -*- coding: utf-8 -*-
"""
Quantum Harmonic Oscillator — Finite-Difference Solver & Analyzer (All-in-One)
Features:
  • Build 1D QHO tridiagonal Hamiltonian (2nd-order FD)
  • Solve lowest k eigenpairs (dense tridiagonal) + optional sparse eigensolver
  • Continuous normalization; phase convention for consistent plots
  • Virial theorem check: <T> ≈ <V> ≈ E/2
  • Orthonormality (Gram) matrix with conjugate transpose (complex-safe)
  • Analytic Hermite–Gaussian comparison (L2 error)
  • Heisenberg check via Δx·Δp ≥ ħ/2 (using <p²>=2m<T>)
  • Parity (even/odd) L2 error check
  • Convergence scan over (n_points, x_max)
  • Plot eigenfunctions (energy-overlay / stacked)
  • Crank–Nicolson time propagation (Dirichlet), optional clamp_edges & norm trace
  • Coherent/Gaussian initial state; record ⟨x⟩(t), Energy(t), Norm(t)
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal, solve_banded
from numpy.polynomial.hermite import hermval

class QuantumHarmonicOscillator:
    def __init__(self, n_points=400, x_max=6.0, m=1.0, omega=1.0, hbar=1.0):
        self.n_points = int(n_points)
        self.x_max = float(x_max)
        self.m = float(m)
        self.omega = float(omega)
        self.hbar = float(hbar)
        self.x = np.linspace(-self.x_max, self.x_max, self.n_points, dtype=np.float64)
        self.dx = self.x[1] - self.x[0]
        # matrices & results
        self.diag_T = None
        self.off_T = None
        self.V = None
        self.diag = None
        self.off = None
        self.energies = None
        self.wavefuncs = None  # columns = eigenstates (real for stationary problem)

    # ---------- builders ----------
    def build_hamiltonian(self):
        """H = T + V using 2nd-order central differences."""
        dx, m, omega, hbar, n = self.dx, self.m, self.omega, self.hbar, self.n_points
        # Kinetic:  - (hbar^2 / 2m) d^2/dx^2  → tri-diagonal (stored as diag_T/off_T)
        self.diag_T = np.full(n, (hbar**2) / (m * dx**2), dtype=np.float64)
        self.off_T = np.full(n - 1, -(hbar**2) / (2 * m * dx**2), dtype=np.float64)
        # Potential: V = 1/2 m ω^2 x^2
        self.V = 0.5 * m * (omega**2) * self.x**2
        # H = T + V
        self.diag = self.diag_T + self.V
        self.off = self.off_T
        return self

    # ---------- eigen solvers ----------
    def solve_eigen(self, k=6):
        """Dense tridiagonal solver for lowest k eigenpairs."""
        if self.diag is None or self.off is None:
            raise ValueError("Hamiltonian not built yet. Call build_hamiltonian() first.")
        e, v = eigh_tridiagonal(self.diag, self.off, select="i", select_range=(0, k - 1))
        v_norm = [self.normalize(v[:, i]) for i in range(v.shape[1])]
        self.energies = e
        self.wavefuncs = np.column_stack(v_norm)
        return self

    def solve_eigen_sparse(self, k=6):
        """Optional: sparse solver for large grids (N ≳ 1e4)."""
        from scipy.sparse import diags
        from scipy.sparse.linalg import eigsh
        H = diags([self.off, self.diag, self.off], offsets=[-1, 0, 1], format="csr")
        e, v = eigsh(H, k=k, which="SA")
        idx = np.argsort(e)
        e, v = e[idx], v[:, idx]
        v = np.column_stack([self.normalize(v[:, i]) for i in range(v.shape[1])])
        self.energies, self.wavefuncs = e, v
        return self

    # ---------- numerics ----------
    def normalize(self, psi):
        """Continuous normalization: sum |psi|^2 * dx = 1."""
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        return psi / norm

    def tri_matvec(self, diag, off, psi):
        """y = H ψ for tri-diagonal H."""
        y = diag * psi
        y[:-1] += off * psi[1:]
        y[1:] += off * psi[:-1]
        return y

    # ---------- expectations (complex-safe) ----------
    def expectation_H(self, psi):
        e_diag = np.sum(self.diag * (np.abs(psi)**2))
        e_off = 2.0 * np.sum(self.off * np.real(np.conj(psi[:-1]) * psi[1:]))
        return (e_diag + e_off) * self.dx

    def expectation_T(self, psi):
        t_diag = np.sum(self.diag_T * (np.abs(psi)**2))
        t_off = 2.0 * np.sum(self.off_T * np.real(np.conj(psi[:-1]) * psi[1:]))
        return (t_diag + t_off) * self.dx

    def expectation_V(self, psi):
        return np.sum(self.V * (np.abs(psi)**2)) * self.dx

    def expectation_x(self, psi):
        return np.sum(self.x * (np.abs(psi)**2)) * self.dx

    def expectation_p(self, psi):
        """<p> ≈ ∫ ψ* (-i ħ d/dx) ψ dx via central difference."""
        dpsi = np.zeros_like(psi, dtype=np.complex128)
        dpsi[1:-1] = (psi[2:] - psi[:-2]) / (2 * self.dx)
        ppsi = -1j * self.hbar * dpsi
        return np.real(np.sum(np.conj(psi) * ppsi) * self.dx)

    def virial_check(self, n=0, verbose=True):
        """For QHO eigenstate n, expect <T>≈<V>≈E/2."""
        psi = self.wavefuncs[:, n]
        E = self.energies[n]
        T = self.expectation_T(psi)
        V = self.expectation_V(psi)
        if verbose:
            print(f"[virial] n={n}: E={E:.8f}, <T>={T:.8f}, <V>={V:.8f}, 2(<T>-<V>)={2*(T-V):.2e}")
        return E, T, V

    # ---------- orthonormality & phase ----------
    def orthonormality_matrix(self):
        if self.wavefuncs is None:
            raise ValueError("Solve eigenstates first.")
        return self.wavefuncs.conj().T @ (self.wavefuncs * self.dx)

    def phase_convention_all(self):
        if self.wavefuncs is None:
            raise ValueError("Solve eigenstates first.")
        npts = self.n_points
        left = npts // 2 - 1
        right = left + 1
        for n in range(self.wavefuncs.shape[1]):
            psi = self.wavefuncs[:, n]
            if n % 2 == 0:
                center_avg = 0.5 * (psi[left] + psi[right])
                if np.real(center_avg) < 0:
                    self.wavefuncs[:, n] = -psi
            else:
                center_slope = psi[right] - psi[left]
                if np.real(center_slope) < 0:
                    self.wavefuncs[:, n] = -psi

    # ---------- analytic comparison ----------
    def analytic_psi(self, n):
        alpha = self.m * self.omega / self.hbar
        xi = np.sqrt(alpha) * self.x
        coeffs = [0.0] * n + [1.0]
        Hn = hermval(xi, coeffs)
        pref = (alpha / np.pi) ** 0.25 / np.sqrt((2.0 ** n) * math.factorial(n))
        psi = pref * Hn * np.exp(-0.5 * alpha * self.x**2)
        return self.normalize(psi)

    def compare_with_analytic(self, k=5):
        if self.wavefuncs is None:
            raise ValueError("Solve eigenstates first.")
        k = min(k, self.wavefuncs.shape[1])
        print("[check] Numeric vs Analytic (L2, dx-weighted)")
        for n in range(k):
            psi_num = self.wavefuncs[:, n]
            psi_ana = self.analytic_psi(n)
            l2 = np.sqrt(np.sum(np.abs(psi_num - psi_ana)**2) * self.dx)
            print(f"  n={n}: ||ψ_num-ψ_ana||₂ ≈ {l2:.3e}")

    # ---------- uncertainty & parity ----------
    def heisenberg_check(self, psi=None, n=0, verbose=True):
        if psi is None:
            if self.wavefuncs is None:
                raise ValueError("Solve eigenstates first or pass psi.")
            psi = self.wavefuncs[:, n]
        x_mean = self.expectation_x(psi)
        x2_mean = np.sum((self.x**2) * (np.abs(psi)**2)) * self.dx
        dx_std = np.sqrt(max(x2_mean - x_mean**2, 0.0))
        p_mean = self.expectation_p(psi)
        T_mean = self.expectation_T(psi)
        p2_mean = 2.0 * self.m * T_mean
        dp_std = np.sqrt(max(p2_mean - p_mean**2, 0.0))
        product = dx_std * dp_std
        if verbose:
            print(f"[heisenberg] Δx≈{dx_std:.6f}, Δp≈{dp_std:.6f}, ΔxΔp≈{product:.6f}, ħ/2={0.5*self.hbar:.6f}")
        return dx_std, dp_std, product

    def parity_error(self, n=0):
        psi = self.wavefuncs[:, n]
        psi_flip = psi[::-1]
        if n % 2 == 0:
            err = np.sqrt(np.sum(np.abs(psi - psi_flip)**2) * self.dx)
        else:
            err = np.sqrt(np.sum(np.abs(psi + psi_flip)**2) * self.dx)
        print(f"[parity] n={n} parity L2 error ≈ {err:.3e}")
        return err

    # ---------- plotting ----------
    def plot_wavefunctions(self, max_level=6, mode="energy", scale=0.6,
                           save_path="qho_levels.png", show=True, overlay_potential=True):
        if self.wavefuncs is None or self.energies is None:
            raise ValueError("Eigenstates not computed yet. Call solve_eigen() first.")
        fig, ax = plt.subplots(figsize=(8, 6))
        if overlay_potential:
            ax.plot(self.x, self.V, label="V(x) = 1/2 m ω² x²", alpha=0.9)
        L = min(max_level, self.wavefuncs.shape[1])
        for n in range(L):
            psi = self.wavefuncs[:, n]
            if mode == "energy":
                y = scale * np.real(psi) + self.energies[n]
                ax.axhline(self.energies[n], linestyle="--", linewidth=0.8)
                label = f"n={n}, E≈{self.energies[n]:.3f}"
            else:
                y = scale * np.real(psi) + n * 0.5
                label = f"n={n}, E≈{self.energies[n]:.3f}"
            ax.plot(self.x, y, label=label)
        ax.set_title("Quantum Harmonic Oscillator — Eigenstates")
        ax.set_xlabel("x")
        ax.set_ylabel("Energy / Offsets")
        ax.grid(True)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    # ---------- convergence scan ----------
    def scan_convergence(self, n_points_list=(200, 400, 800), x_max_list=(5.0, 6.0, 8.0), k=1):
        rows = []
        E0_th = 0.5 * self.hbar * self.omega
        for N in n_points_list:
            for X in x_max_list:
                x = np.linspace(-X, X, N)
                dx = x[1] - x[0]
                hbar, m, omega = self.hbar, self.m, self.omega
                diag_T = np.full(N, (hbar**2) / (m * dx**2))
                off_T = np.full(N - 1, -(hbar**2) / (2 * m * dx**2))
                V = 0.5 * m * (omega**2) * x**2
                diag = diag_T + V
                off = off_T
                e, _ = eigh_tridiagonal(diag, off, select="i", select_range=(0, k - 1))
                E0 = e[0]
                rows.append({"n_points": N, "x_max": X, "E0": E0, "rel_err": abs(E0 - E0_th) / E0_th})
        return rows

    # ---------- initial states ----------
    def coherent_gaussian(self, x0=1.0, p0=0.0):
        alpha = self.m * self.omega / self.hbar
        psi = np.exp(-0.5 * alpha * (self.x - x0)**2 + 1j * p0 * self.x / self.hbar)
        return self.normalize(psi)

    # ---------- Crank–Nicolson propagation ----------
    def propagate_cn(self, psi0, dt=0.01, steps=400, renormalize=True,
                     record=("x", "E", "norm"), clamp_edges=False):
        if self.diag is None or self.off is None:
            raise ValueError("Hamiltonian not built yet.")
        n = self.n_points
        a = dt / (2.0 * self.hbar)
        ab = np.zeros((3, n), dtype=np.complex128)
        ab[1, :] = 1.0 + 1j * a * self.diag
        ab[0, 1:] = 1j * a * self.off
        ab[2, :-1] = 1j * a * self.off
        psi = psi0.astype(np.complex128)
        t = np.arange(steps + 1) * dt
        recs = {}
        if "x" in record:    recs["x"] = np.zeros(steps + 1)
        if "E" in record:    recs["E"] = np.zeros(steps + 1)
        if "norm" in record: recs["norm"] = np.zeros(steps + 1)
        def _norm(z): return np.sqrt(np.sum(np.abs(z)**2) * self.dx)
        if "x" in record:    recs["x"][0] = self.expectation_x(psi)
        if "E" in record:    recs["E"][0] = self.expectation_H(psi)
        if "norm" in record: recs["norm"][0] = _norm(psi)
        for k in range(steps):
            Hpsi = self.tri_matvec(self.diag, self.off, psi)
            rhs = psi - 1j * a * Hpsi
            psi = solve_banded((1, 1), ab, rhs)
            if clamp_edges:
                psi[0] = 0.0
                psi[-1] = 0.0
            if renormalize:
                psi = self.normalize(psi)
            if "x" in record:    recs["x"][k + 1] = self.expectation_x(psi)
            if "E" in record:    recs["E"][k + 1] = self.expectation_H(psi)
            if "norm" in record: recs["norm"][k + 1] = _norm(psi)
        return t, recs, psi

def demo_all():
    qho = QuantumHarmonicOscillator(n_points=400, x_max=6.0, m=1.0, omega=1.0, hbar=1.0)
    print("[kernel-log] 建立哈密頓量...")
    qho.build_hamiltonian()
    print("[kernel-log] 求解最低能階...")
    qho.solve_eigen(k=6)  # 或改用：qho.solve_eigen_sparse(k=6)
    qho.phase_convention_all()
    print(f"[kernel-log] 取得 {len(qho.energies)} 個能階")
    for i, E in enumerate(qho.energies):
        E_th = (i + 0.5) * qho.hbar * qho.omega
        rel = abs(E - E_th) / E_th
        print(f"  n={i}: E_num={E:.8f}, E_theory={E_th:.8f}, 相對誤差={rel:.2e}")
    print("[kernel-log] 維利定理檢查（n=0..2）")
    for i in range(3):
        qho.virial_check(n=i)
    print("[kernel-log] 解析解對照（前 4 態）...")
    qho.compare_with_analytic(k=4)
    G = qho.orthonormality_matrix()
    max_dev = np.max(np.abs(G - np.eye(G.shape[0])))
    print(f"[kernel-log] 正交性檢查：max |G - I| ≈ {max_dev:.2e}")
    print("[kernel-log] Heisenberg（基態）")
    qho.heisenberg_check(n=0)
    print("[kernel-log] Parity 檢查（n=0..3）")
    for i in range(4):
        qho.parity_error(n=i)
    print("[kernel-log] 繪圖（疊在能量線與勢能上）...")
    qho.plot_wavefunctions(max_level=6, mode='energy', scale=0.6,
                           save_path="qho_levels_energy.png", show=True)
    print("[kernel-log] 繪圖（垂直堆疊視圖）...")
    qho.plot_wavefunctions(max_level=6, mode='stack', scale=0.8,
                           save_path="qho_levels_stack.png", show=False)
    print("[kernel-log] 收斂掃描（E0 相對誤差）...")
    rows = qho.scan_convergence(n_points_list=(200, 400, 800),
                               x_max_list=(5.0, 6.0, 8.0))
    for r in rows:
        print(f"  N={r['n_points']:>4}, X={r['x_max']:.1f}: E0={r['E0']:.8f}, rel_err={r['rel_err']:.2e}")
    print("[kernel-log] 時間演化示範（Coherent Gaussian）...")
    psi0 = qho.coherent_gaussian(x0=1.0, p0=0.0)
    dt, steps = 0.01, 800
    t, recs, psiT = qho.propagate_cn(
        psi0, dt=dt, steps=steps, renormalize=True,
        record=("x", "E", "norm"),
        clamp_edges=False
    )
    # 理論：<x>(t) = x0 cos(ω t) + p0/(m ω) sin(ω t)；此處 p0=0
    x_theory = 1.0 * np.cos(qho.omega * t)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, recs["x"], label="<x>(t)")
    ax.plot(t, x_theory, linestyle="--", label="theory x0 cos(ω t)")
    ax.set_xlabel("t")
    ax.set_ylabel("<x>(t)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("qho_time_x.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, recs["E"])
    ax.set_xlabel("t")
    ax.set_ylabel("Energy(t)")
    ax.set_title("Energy conservation (Crank–Nicolson)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("qho_time_energy.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, recs["norm"])
    ax.set_xlabel("t")
    ax.set_ylabel("‖ψ‖")
    ax.set_title("Norm trace")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("qho_time_norm.png", dpi=150)
    plt.close(fig)
    print("[kernel-log] 輸出完成：")
    print("  - qho_levels_energy.png")
    print("  - qho_levels_stack.png")
    print("  - qho_time_x.png")
    print("  - qho_time_energy.png")
    print("  - qho_time_norm.png")
    print("[done] All tasks completed.")

if __name__ == "__main__":
    demo_all()
