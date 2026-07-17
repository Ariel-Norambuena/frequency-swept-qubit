"""Exact thermal phase damping for exponential-cutoff power-law spectra.

The commuting interaction gives a closed decoherence functional.  Generalized
Gauss-Laguerre quadrature is used only to evaluate the exact integrals and to
verify convergence for sub-, Ohmic, and super-Ohmic exponents.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import roots_genlaguerre

from powerlaw_spectral_analysis import calibrated_coupling


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
FIGURE_DIR = ROOT / "output" / "figures"


def phase_damping_powerlaw(
    t: np.ndarray,
    alpha_b: float,
    omega_c: float,
    s: float,
    beta: float,
    quadrature_order: int = 180,
) -> dict[str, np.ndarray]:
    """Evaluate Gamma_beta, gamma_phi, Delta E_B, and Sigma for initial |+>."""
    nodes, weights = roots_genlaguerre(quadrature_order, s - 1.0)
    x = omega_c * np.asarray(t)
    y = beta * omega_c
    phases = np.outer(x, nodes)
    thermal = 1.0 / np.tanh(0.5 * y * nodes)

    # Weight is u^(s-1) exp(-u); the first residual has a finite u -> 0 limit.
    one_minus_cos_over_u = 2.0 * np.sin(0.5 * phases) ** 2 / nodes
    decoherence = 8.0 * alpha_b * np.dot(one_minus_cos_over_u * thermal, weights)
    dephasing_rate = 8.0 * alpha_b * omega_c * np.dot(
        np.sin(phases) * thermal, weights
    )
    bath_energy = 4.0 * alpha_b * omega_c * np.dot(
        2.0 * np.sin(0.5 * phases) ** 2, weights
    )

    coherence = np.exp(-decoherence)
    p_plus = np.clip(0.5 * (1.0 + coherence), 1e-15, 1.0 - 1e-15)
    system_entropy = -p_plus * np.log(p_plus) - (1.0 - p_plus) * np.log(1.0 - p_plus)
    entropy_production = system_entropy + beta * bath_energy
    return {
        "decoherence": decoherence,
        "coherence": coherence,
        "dephasing_rate": dephasing_rate,
        "bath_energy": bath_energy,
        "system_entropy": system_entropy,
        "entropy_production": entropy_production,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    gamma0 = 0.8
    omega_b = 1.0
    omega_c = 0.5
    beta = 10.0
    exponents = (0.5, 1.0, 3.0)
    t = np.linspace(0.0, 10.0, 1401)

    results = {}
    summary = []
    for s in exponents:
        alpha_b = calibrated_coupling(gamma0, omega_b, omega_c, s)
        fine = phase_damping_powerlaw(t, alpha_b, omega_c, s, beta, 320)
        coarse = phase_damping_powerlaw(t, alpha_b, omega_c, s, beta, 160)
        error = float(np.max(np.abs(fine["decoherence"] - coarse["decoherence"])))
        results[s] = fine
        summary.append(
            [
                s,
                alpha_b,
                fine["decoherence"][-1],
                fine["coherence"][-1],
                np.min(fine["dephasing_rate"]),
                fine["entropy_production"][-1],
                error,
            ]
        )
        np.savetxt(
            DATA_DIR / f"phase_damping_thermal_s_{s:g}.csv",
            np.column_stack(
                [
                    t,
                    fine["decoherence"],
                    fine["coherence"],
                    fine["dephasing_rate"],
                    fine["bath_energy"],
                    fine["system_entropy"],
                    fine["entropy_production"],
                ]
            ),
            delimiter=",",
            header="t,Gamma_beta,coherence,gamma_phi,bath_energy,system_entropy,Sigma_beta",
            comments="",
        )

    np.savetxt(
        DATA_DIR / "phase_damping_thermal_summary.csv",
        np.asarray(summary),
        delimiter=",",
        header=(
            "s,alpha_b,Gamma_final,coherence_final,min_gamma_phi,Sigma_final,"
            "max_Gamma_GL160_GL320_error"
        ),
        comments="",
    )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9,
            "savefig.dpi": 300,
        }
    )
    colors = {0.5: "#1b9e77", 1.0: "#d95f02", 3.0: "#7570b3"}
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    tau = gamma0 * t
    for s, data in results.items():
        label = rf"$s={s:g}$"
        axes[0].plot(tau, data["coherence"], color=colors[s], lw=1.8, label=label)
        axes[1].plot(tau, data["dephasing_rate"] / gamma0, color=colors[s], lw=1.55, label=label)
        axes[2].plot(tau, data["entropy_production"], color=colors[s], lw=1.7, label=label)
    axes[0].set(xlabel=r"$\tau=\gamma_0t$", ylabel=r"$|\rho_{eg}(t)|/|\rho_{eg}(0)|$", title="Thermal coherence")
    axes[1].axhline(0.0, color="0.25", lw=0.8)
    axes[1].set(xlabel=r"$\tau$", ylabel=r"$\gamma_\phi/\gamma_0$", title="Exact dephasing rate")
    axes[1].set_yscale("symlog", linthresh=0.02, linscale=0.8)
    axes[2].set(xlabel=r"$\tau$", ylabel=r"$\Sigma_\beta(t)$", title="Microscopic entropy production")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(FIGURE_DIR / "phase_damping_thermal_powerlaw.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "phase_damping_thermal_powerlaw.pdf", bbox_inches="tight")
    plt.close(fig)

    print("s, alpha_b, Gamma(tf), coherence(tf), min gamma_phi, Sigma(tf), quadrature error")
    for row in summary:
        print(", ".join(f"{value:.12g}" for value in row))


if __name__ == "__main__":
    main()
