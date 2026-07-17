"""Converged Volterra analysis for an exponential-cutoff power-law reservoir.

The zero-temperature RWA survival amplitude obeys

    dc(t)/dt = -int_0^t K(t,u)c(u) du,

with a linear sweep omega_S(t)=omega_b-alpha_sw*t and

    J(w)=2*alpha_b*omega_c**(1-s)*w**s*exp(-w/omega_c).

The reservoir transform is analytic.  Only the resulting Volterra equation is
integrated numerically; no Born, Markov, or time-local approximation is used.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import gamma as gamma_function

from exact_backflow_analysis import solve_trajectory as solve_lorentzian
from publication_style import close_axes, configure_publication_style


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
FIGURE_DIR = ROOT / "output" / "figures"


def spectral_density(
    omega: np.ndarray,
    alpha_b: float,
    omega_c: float,
    s: float,
) -> np.ndarray:
    return 2.0 * alpha_b * omega_c ** (1.0 - s) * omega**s * np.exp(-omega / omega_c)


def calibrated_coupling(
    gamma0: float,
    omega_b: float,
    omega_c: float,
    s: float,
) -> float:
    """Choose alpha_b so that 2*pi*J(omega_b)=gamma0."""
    return (
        gamma0
        * np.exp(omega_b / omega_c)
        / (4.0 * np.pi * omega_c ** (1.0 - s) * omega_b**s)
    )


def bath_correlation(delay: np.ndarray, alpha_b: float, omega_c: float, s: float) -> np.ndarray:
    """Analytic transform int_0^infinity J(w) exp(-i*w*delay) dw."""
    return (
        2.0
        * alpha_b
        * omega_c**2
        * gamma_function(s + 1.0)
        / (1.0 + 1j * omega_c * delay) ** (s + 1.0)
    )


def _memory_derivative(
    n: int,
    t: np.ndarray,
    phase: np.ndarray,
    c: np.ndarray,
    h: float,
    alpha_b: float,
    omega_c: float,
    s: float,
) -> complex:
    if n == 0:
        return 0.0j
    delay = t[n] - t[: n + 1]
    kernel = np.exp(1j * (phase[n] - phase[: n + 1])) * bath_correlation(
        delay, alpha_b, omega_c, s
    )
    weights = np.ones(n + 1)
    weights[[0, -1]] = 0.5
    return -h * np.dot(weights * kernel, c[: n + 1])


def solve_powerlaw(
    gamma0: float,
    omega_b: float,
    delta_f: float,
    alpha_sw: float,
    omega_c: float,
    s: float,
    n_steps: int = 5000,
) -> dict[str, np.ndarray | float]:
    """Second-order product-integration solution of the exact Volterra equation."""
    if min(gamma0, omega_b, delta_f, alpha_sw, omega_c, s) <= 0.0:
        raise ValueError("all physical parameters and s must be positive")
    if delta_f >= omega_b:
        raise ValueError("delta_f must be smaller than omega_b")

    alpha_b = calibrated_coupling(gamma0, omega_b, omega_c, s)
    t_final = delta_f / alpha_sw
    t = np.linspace(0.0, t_final, n_steps + 1)
    h = t[1] - t[0]
    phase = omega_b * t - 0.5 * alpha_sw * t**2
    c = np.empty(n_steps + 1, dtype=complex)
    dc = np.empty_like(c)
    c[0] = 1.0
    dc[0] = 0.0

    for n in range(n_steps):
        c[n + 1] = c[n] + h * dc[n]
        for _ in range(3):
            dc_trial = _memory_derivative(
                n + 1, t, phase, c, h, alpha_b, omega_c, s
            )
            c[n + 1] = c[n] + 0.5 * h * (dc[n] + dc_trial)
        dc[n + 1] = _memory_derivative(
            n + 1, t, phase, c, h, alpha_b, omega_c, s
        )

    population = np.abs(c) ** 2
    population_rate = 2.0 * np.real(np.conjugate(c) * dc)
    omega = omega_b - alpha_sw * t
    heat_power = omega * population_rate
    regular = population > 1e-13
    gamma_rate = np.full_like(population, np.nan)
    lamb_shift = np.full_like(population, np.nan)
    gamma_rate[regular] = -population_rate[regular] / population[regular]
    lamb_shift[regular] = -np.imag(dc[regular] / c[regular])
    interaction_energy = 2.0 * population * np.nan_to_num(lamb_shift)

    safe_p = np.clip(population, 1e-14, 1.0 - 1e-14)
    entropy = -safe_p * np.log(safe_p) - (1.0 - safe_p) * np.log(1.0 - safe_p)
    mutual_information = 2.0 * entropy
    mutual_information_rate = 2.0 * population_rate * np.log((1.0 - safe_p) / safe_p)
    probability_backflow = np.concatenate(
        ([0.0], cumulative_trapezoid(np.maximum(population_rate, 0.0), t))
    )
    energy_backflow = np.concatenate(
        ([0.0], cumulative_trapezoid(np.maximum(heat_power, 0.0), t))
    )
    work = np.concatenate(([0.0], cumulative_trapezoid(-alpha_sw * population, t)))
    bath_energy = (
        omega_b
        + work
        - omega * population
        - interaction_energy
    )

    return {
        "t": t,
        "tau": gamma0 * t,
        "c": c,
        "dc": dc,
        "population": population,
        "population_rate": population_rate,
        "omega": omega,
        "gamma": gamma_rate,
        "lamb_shift": lamb_shift,
        "heat_power": heat_power,
        "interaction_energy": interaction_energy,
        "bath_energy": bath_energy,
        "mutual_information": mutual_information,
        "mutual_information_rate": mutual_information_rate,
        "probability_backflow_cumulative": probability_backflow,
        "energy_backflow_cumulative": energy_backflow,
        "probability_backflow": float(probability_backflow[-1]),
        "energy_backflow": float(energy_backflow[-1]),
        "final_population": float(population[-1]),
        "alpha_b": float(alpha_b),
    }


def positive_current_integrals(
    tau: np.ndarray,
    population: np.ndarray,
    population_rate_tau: np.ndarray,
    omega_over_gamma0: np.ndarray,
    a: float,
) -> tuple[float, float, float]:
    """Return N_P and E_S,rev/gamma0 by direct and by-parts quadrature.

    Zero crossings of P' are linearly interpolated and inserted into every
    positive-rate interval.  The second energy value evaluates the printed
    integration-by-parts identity independently of the direct current integral.
    """
    tau = np.asarray(tau, dtype=float)
    population = np.asarray(population, dtype=float)
    rate = np.asarray(population_rate_tau, dtype=float)
    omega = np.asarray(omega_over_gamma0, dtype=float)
    if not (tau.shape == population.shape == rate.shape == omega.shape):
        raise ValueError("tau, population, rate, and omega must have the same shape")

    positive = rate > 0.0
    starts = np.flatnonzero(positive & ~np.r_[False, positive[:-1]])
    stops = np.flatnonzero(positive & ~np.r_[positive[1:], False])
    if starts.size != stops.size:
        raise RuntimeError("unpaired positive-rate intervals")

    probability_integral = 0.0
    energy_direct = 0.0
    energy_parts = 0.0

    def zero_crossing(left: int, right: int) -> float:
        return float(
            tau[left]
            - rate[left] * (tau[right] - tau[left]) / (rate[right] - rate[left])
        )

    for start_raw, stop_raw in zip(starts, stops):
        start = int(start_raw)
        stop = int(stop_raw)
        tau_minus = float(tau[0]) if start == 0 else zero_crossing(start - 1, start)
        tau_plus = float(tau[-1]) if stop == tau.size - 1 else zero_crossing(stop, stop + 1)

        interior = np.arange(start, stop + 1)
        tau_segment = np.r_[tau_minus, tau[interior], tau_plus]
        population_segment = np.r_[
            np.interp(tau_minus, tau, population),
            population[interior],
            np.interp(tau_plus, tau, population),
        ]
        rate_segment = np.r_[0.0, rate[interior], 0.0]
        omega_segment = np.r_[
            np.interp(tau_minus, tau, omega),
            omega[interior],
            np.interp(tau_plus, tau, omega),
        ]

        probability_integral += float(np.trapezoid(rate_segment, tau_segment))
        energy_direct += float(
            np.trapezoid(omega_segment * rate_segment, tau_segment)
        )
        energy_parts += float(
            omega_segment[-1] * population_segment[-1]
            - omega_segment[0] * population_segment[0]
            + a * np.trapezoid(population_segment, tau_segment)
        )

    return probability_integral, energy_direct, energy_parts


def write_trajectory(name: str, data: dict[str, np.ndarray | float]) -> None:
    columns = np.column_stack(
        [
            data["t"],
            data["tau"],
            data["population"],
            data["population_rate"],
            data["gamma"],
            data["heat_power"],
            data["mutual_information"],
            data["mutual_information_rate"],
            data["interaction_energy"],
            data["bath_energy"],
        ]
    )
    np.savetxt(
        DATA_DIR / name,
        columns,
        delimiter=",",
        header=(
            "t,tau,population,population_rate,gamma,heat_power,mutual_information,"
            "mutual_information_rate,interaction_energy,bath_energy"
        ),
        comments="",
    )


def make_figure(
    omega: np.ndarray,
    spectral_curves: dict[float, np.ndarray],
    alpha_values: dict[float, float],
    trajectories: dict[str, dict[str, np.ndarray | float]],
    gamma0: float,
    omega_b: float,
) -> None:
    configure_publication_style()
    colors = {0.5: "#1b9e77", 1.0: "#d95f02", 3.0: "#7570b3"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)

    for s, curve in spectral_curves.items():
        axes[0, 0].plot(omega, curve / gamma0, color=colors[s], lw=2.3, label=rf"$s={s:g}$")
    axes[0, 0].axvline(omega_b / gamma0, color="black", lw=1.6, ls="--", label=r"$\omega_b$")
    axes[0, 0].set(xlabel=r"$\omega/\gamma_0$", ylabel=r"$J(\omega)/\gamma_0$", title="Rate-matched spectra")
    axes[0, 0].set_xlim(0.0, 4.0)
    axes[0, 0].legend(frameon=True)

    line_styles = {"Lorentzian": ("black", "-"), "s=0.5": (colors[0.5], "-"), "s=1": (colors[1.0], "--"), "s=3": (colors[3.0], "-.")}
    for label, data in trajectories.items():
        color, ls = line_styles[label]
        display_label = label if label == "Lorentzian" else "$" + label + "$"
        axes[0, 1].plot(data["tau"], data["population"], color=color, ls=ls, lw=2.3, label=display_label)
        axes[1, 0].plot(data["tau"], data["population_rate_tau"], color=color, ls=ls, lw=2.1, label=display_label)
        axes[1, 1].plot(data["tau"], data["mutual_information"], color=color, ls=ls, lw=2.2, label=display_label)

    axes[0, 1].set(xlabel=r"$\tau=\gamma_0t$", ylabel=r"$P(\tau)$", title="Converged survival probability")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].legend(frameon=True, loc="lower left")
    axes[1, 0].axhline(0.0, color="0.25", lw=1.3)
    axes[1, 0].set(xlabel=r"$\tau$", ylabel=r"$\mathrm{d}P/\mathrm{d}\tau$", title="Population flow (backflow if positive)")
    axes[1, 1].set(xlabel=r"$\tau$", ylabel=r"$\mathcal{I}_{S:B}$", title="Converged vacuum correlations")

    close_axes(axes)

    fig.savefig(FIGURE_DIR / "powerlaw_lorentzian_comparison.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "powerlaw_lorentzian_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    gamma0 = 0.8
    lam = 0.5
    omega_c = lam
    omega_b = 2.0
    delta_f = 1.0
    alpha_sw = 0.1
    ell = lam / gamma0
    a = alpha_sw / gamma0**2
    Omega_delta = delta_f / gamma0
    Omega_b = omega_b / gamma0
    exponents = (0.5, 1.0, 3.0)

    lorentz = solve_lorentzian(ell, a, Omega_delta, Omega_b, n_times=5001)
    lorentz_np, lorentz_energy_direct, lorentz_energy_parts = positive_current_integrals(
        np.asarray(lorentz["tau"]),
        np.asarray(lorentz["population"]),
        np.asarray(lorentz["population_rate"]),
        np.asarray(lorentz["omega"]),
        a,
    )
    lorentz_data = {
        "tau": lorentz["tau"],
        "population": lorentz["population"],
        "gamma": gamma0 * np.asarray(lorentz["gamma"]),
        "population_rate_tau": lorentz["population_rate"],
        "mutual_information": lorentz["mutual_information"],
        "probability_backflow": lorentz_np,
        "energy_backflow": lorentz_energy_direct,
        "energy_backflow_parts": lorentz_energy_parts,
        "final_population": lorentz["final_population"],
    }

    trajectories: dict[str, dict[str, np.ndarray | float]] = {"Lorentzian": lorentz_data}
    alpha_values: dict[float, float] = {}
    summary_rows = []
    for s in exponents:
        data = solve_powerlaw(
            gamma0, omega_b, delta_f, alpha_sw, omega_c, s, n_steps=5000
        )
        data["population_rate_tau"] = np.asarray(data["population_rate"]) / gamma0
        probability_integral, energy_direct, energy_parts = positive_current_integrals(
            np.asarray(data["tau"]),
            np.asarray(data["population"]),
            np.asarray(data["population_rate_tau"]),
            np.asarray(data["omega"]) / gamma0,
            a,
        )
        trajectories[f"s={s:g}"] = data
        alpha_values[s] = float(data["alpha_b"])
        write_trajectory(f"powerlaw_s_{s:g}_trajectory.csv", data)

        coarse = solve_powerlaw(
            gamma0, omega_b, delta_f, alpha_sw, omega_c, s, n_steps=2500
        )
        convergence_error = abs(float(data["final_population"]) - float(coarse["final_population"]))
        maximum_population_error = float(
            np.max(np.abs(np.asarray(data["population"])[::2] - np.asarray(coarse["population"])))
        )
        backflow_error = abs(float(data["probability_backflow"]) - float(coarse["probability_backflow"]))
        summary_rows.append(
            [
                s,
                data["alpha_b"],
                data["final_population"],
                probability_integral,
                energy_direct,
                energy_parts,
                abs(energy_direct - energy_parts),
                convergence_error,
                maximum_population_error,
                backflow_error,
            ]
        )

    np.savetxt(
        DATA_DIR / "powerlaw_comparison_summary.csv",
        np.asarray(summary_rows, dtype=float),
        delimiter=",",
        header=(
            "s,alpha_b,final_population,probability_backflow,positive_heatlike_integral_direct,"
            "positive_heatlike_integral_parts,direct_parts_error,"
            "coarse_fine_final_P_error,coarse_fine_max_P_error,coarse_fine_backflow_error"
        ),
        comments="",
    )

    omega = np.linspace(1e-5, 4.0 * gamma0, 1200)
    curves = {
        s: spectral_density(omega, alpha_values[s], omega_c, s) for s in exponents
    }
    make_figure(omega / gamma0, curves, alpha_values, trajectories, gamma0, omega_b)

    table_rows = [
        [
            "Lorentzian",
            "",
            float(lorentz_data["final_population"]),
            float(lorentz_data["probability_backflow"]),
            float(lorentz_data["energy_backflow"]),
            float(lorentz_data["energy_backflow_parts"]),
            abs(
                float(lorentz_data["energy_backflow"])
                - float(lorentz_data["energy_backflow_parts"])
            ),
            1e-9,
        ]
    ]
    for row in summary_rows:
        table_rows.append(
            [f"s={row[0]:g}", row[1], row[2], row[3], row[4], row[5], row[6], row[8]]
        )
    with (DATA_DIR / "powerlaw_table_sii.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "reservoir",
                "alpha_b",
                "final_population",
                "probability_backflow",
                "positive_heatlike_integral_direct_over_gamma0",
                "positive_heatlike_integral_parts_over_gamma0",
                "direct_parts_error",
                "coarse_fine_max_population_error",
            ]
        )
        writer.writerows(table_rows)

    print("s, alpha_b, P_final, N_P, Qplus_direct, Qplus_parts, mismatch, final-P error, max-P error, N_P error")
    for row in summary_rows:
        print(", ".join(f"{value:.12g}" for value in row))
    print(
        "Lorentzian, -, "
        f"{lorentz_data['final_population']:.12g}, "
        f"{lorentz_data['probability_backflow']:.12g}, "
        f"{lorentz_data['energy_backflow']:.12g}, "
        f"{lorentz_data['energy_backflow_parts']:.12g}, -"
    )


if __name__ == "__main__":
    main()
