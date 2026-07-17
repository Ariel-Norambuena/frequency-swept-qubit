"""Exact Volterra comparison for a bounded tanh detuning.

The spectral densities are rate matched at the initial transition frequency.
The Lorentzian trajectory is analytic (Gauss hypergeometric) and integrated as
an ODE for dense plotting; the power-law trajectories solve their exact
microscopic Volterra equations by converged product integration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

from powerlaw_spectral_analysis import _memory_derivative, calibrated_coupling
from tanh_detuning_exact import solve_tanh_ode


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
FIGURE_DIR = ROOT / "output" / "figures"


def solve_tanh_powerlaw(
    gamma0: float,
    omega_b: float,
    delta_max: float,
    time_scale: float,
    omega_c: float,
    s: float,
    t_final: float,
    n_steps: int = 5000,
) -> dict[str, np.ndarray | float]:
    alpha_b = calibrated_coupling(gamma0, omega_b, omega_c, s)
    t = np.linspace(0.0, t_final, n_steps + 1)
    h = t[1] - t[0]
    detuning = delta_max * np.tanh(t / time_scale)
    omega = omega_b - detuning
    phase = omega_b * t - delta_max * time_scale * np.log(np.cosh(t / time_scale))
    c = np.empty(n_steps + 1, dtype=complex)
    dc = np.empty_like(c)
    c[0] = 1.0
    dc[0] = 0.0

    for n in range(n_steps):
        c[n + 1] = c[n] + h * dc[n]
        for _ in range(3):
            derivative_trial = _memory_derivative(
                n + 1, t, phase, c, h, alpha_b, omega_c, s
            )
            c[n + 1] = c[n] + 0.5 * h * (dc[n] + derivative_trial)
        dc[n + 1] = _memory_derivative(
            n + 1, t, phase, c, h, alpha_b, omega_c, s
        )

    population = np.abs(c) ** 2
    population_rate_t = 2.0 * np.real(np.conjugate(c) * dc)
    population_rate_tau = population_rate_t / gamma0
    heat_power = omega * population_rate_t
    probability_backflow = np.concatenate(
        ([0.0], cumulative_trapezoid(np.maximum(population_rate_t, 0.0), t))
    )
    energy_backflow = np.concatenate(
        ([0.0], cumulative_trapezoid(np.maximum(heat_power, 0.0), t))
    )
    safe_p = np.clip(population, 1e-14, 1.0 - 1e-14)
    mutual_information = -2.0 * (
        safe_p * np.log(safe_p) + (1.0 - safe_p) * np.log(1.0 - safe_p)
    )
    return {
        "t": t,
        "tau": gamma0 * t,
        "detuning": detuning,
        "omega": omega,
        "c": c,
        "dc": dc,
        "population": population,
        "population_rate_tau": population_rate_tau,
        "heat_power": heat_power,
        "mutual_information": mutual_information,
        "probability_backflow": float(probability_backflow[-1]),
        "energy_backflow": float(energy_backflow[-1]),
        "final_population": float(population[-1]),
        "alpha_b": float(alpha_b),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    gamma0 = 0.8
    omega_b = 1.0
    lam = 0.5
    omega_c = lam
    delta_max = 0.5
    time_scale = 5.0
    t_final = 10.0
    ell = lam / gamma0
    d = delta_max / gamma0
    theta = gamma0 * time_scale
    exponents = (0.5, 1.0, 3.0)

    lorentz_raw = solve_tanh_ode(ell, d, theta, gamma0 * t_final, n_times=5001)
    lorentz_population = np.asarray(lorentz_raw["population"])
    lorentz_rate = np.asarray(lorentz_raw["population_rate"])
    lorentz_tau = np.asarray(lorentz_raw["tau"])
    lorentz_omega = omega_b - delta_max * np.tanh(lorentz_tau / theta)
    lorentz_heat_dimensionless = (lorentz_omega / gamma0) * lorentz_rate
    lorentz_probability_backflow = float(
        np.trapezoid(np.maximum(lorentz_rate, 0.0), lorentz_tau)
    )
    lorentz_energy_backflow_over_gamma0 = float(
        np.trapezoid(np.maximum(lorentz_heat_dimensionless, 0.0), lorentz_tau)
    )
    safe_lorentz = np.clip(lorentz_population, 1e-14, 1.0 - 1e-14)
    lorentz_information = -2.0 * (
        safe_lorentz * np.log(safe_lorentz)
        + (1.0 - safe_lorentz) * np.log(1.0 - safe_lorentz)
    )
    trajectories: dict[str, dict[str, np.ndarray | float]] = {
        "Lorentzian": {
            "tau": lorentz_tau,
            "population": lorentz_population,
            "population_rate_tau": lorentz_rate,
            "mutual_information": lorentz_information,
            "final_population": float(lorentz_population[-1]),
            "probability_backflow": lorentz_probability_backflow,
            "energy_backflow_over_gamma0": lorentz_energy_backflow_over_gamma0,
        }
    }

    rows = []
    for s in exponents:
        fine = solve_tanh_powerlaw(
            gamma0,
            omega_b,
            delta_max,
            time_scale,
            omega_c,
            s,
            t_final,
            5000,
        )
        coarse = solve_tanh_powerlaw(
            gamma0,
            omega_b,
            delta_max,
            time_scale,
            omega_c,
            s,
            t_final,
            2500,
        )
        maximum_population_error = float(
            np.max(np.abs(np.asarray(fine["population"])[::2] - np.asarray(coarse["population"])))
        )
        backflow_error = abs(
            float(fine["probability_backflow"]) - float(coarse["probability_backflow"])
        )
        fine["energy_backflow_over_gamma0"] = float(fine["energy_backflow"]) / gamma0
        trajectories[f"s={s:g}"] = fine
        rows.append(
            [
                s,
                fine["alpha_b"],
                fine["final_population"],
                fine["probability_backflow"],
                fine["energy_backflow_over_gamma0"],
                maximum_population_error,
                backflow_error,
            ]
        )
        np.savetxt(
            DATA_DIR / f"tanh_powerlaw_s_{s:g}_trajectory.csv",
            np.column_stack(
                [
                    fine["t"],
                    fine["tau"],
                    fine["detuning"],
                    fine["population"],
                    fine["population_rate_tau"],
                    fine["heat_power"],
                    fine["mutual_information"],
                ]
            ),
            delimiter=",",
            header="t,tau,detuning,population,dP_dtau,heat_power,mutual_information",
            comments="",
        )

    lorentz_row = [
        np.nan,
        np.nan,
        trajectories["Lorentzian"]["final_population"],
        trajectories["Lorentzian"]["probability_backflow"],
        trajectories["Lorentzian"]["energy_backflow_over_gamma0"],
        np.nan,
        np.nan,
    ]
    np.savetxt(
        DATA_DIR / "tanh_spectral_comparison_summary.csv",
        np.asarray([lorentz_row] + rows, dtype=float),
        delimiter=",",
        header=(
            "s,alpha_b,final_population,probability_backflow,energy_backflow_over_gamma0,"
            "coarse_fine_max_P_error,coarse_fine_backflow_error"
        ),
        comments="",
    )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "legend.fontsize": 8.5,
            "savefig.dpi": 300,
        }
    )
    styles = {
        "Lorentzian": ("black", "-"),
        "s=0.5": ("#1b9e77", "-"),
        "s=1": ("#d95f02", "--"),
        "s=3": ("#7570b3", "-."),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    for label, data in trajectories.items():
        color, line_style = styles[label]
        axes[0].plot(data["tau"], data["population"], color=color, ls=line_style, lw=1.7, label=label)
        axes[1].plot(data["tau"], data["population_rate_tau"], color=color, ls=line_style, lw=1.45, label=label)
        axes[2].plot(data["tau"], data["mutual_information"], color=color, ls=line_style, lw=1.55, label=label)
    axes[0].set(xlabel=r"$\tau=\gamma_0t$", ylabel=r"$P(\tau)$", title="Bounded-chirp survival")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=False)
    axes[1].axhline(0.0, color="0.25", lw=0.8)
    axes[1].set(xlabel=r"$\tau$", ylabel=r"$\mathrm{d}P/\mathrm{d}\tau$", title="Backflow if positive")
    axes[2].set(xlabel=r"$\tau$", ylabel=r"$\mathcal{I}_{S:B}$", title="Vacuum correlations")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)
    fig.savefig(FIGURE_DIR / "tanh_spectral_comparison.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "tanh_spectral_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    print("reservoir, P_final, N_P, E_back/gamma0, max P error, N_P error")
    print(
        "Lorentzian, "
        f"{lorentz_row[2]:.12g}, {lorentz_row[3]:.12g}, {lorentz_row[4]:.12g}, -, -"
    )
    for row in rows:
        print(
            f"s={row[0]:g}, {row[2]:.12g}, {row[3]:.12g}, {row[4]:.12g}, "
            f"{row[5]:.3e}, {row[6]:.3e}"
        )


if __name__ == "__main__":
    main()
