"""Exact Lorentzian amplitude damping with a bounded hyperbolic chirp.

Dimensionless model

    c'' + [ell + i*d*tanh(tau/theta)] c' + ell*c/2 = 0,
    c(0)=1, c'(0)=0.

After removing the first derivative and setting
    x = (1+tanh(tau/theta))/2,
the equation is Gauss hypergeometric.  The special-function solution is
validated below against direct high-accuracy integration of the original ODE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

from exact_backflow_analysis import solve_trajectory as solve_linear_chirp


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
FIGURE_DIR = ROOT / "output" / "figures"


@dataclass(frozen=True)
class HypergeometricParameters:
    p: complex
    q: complex
    a: complex
    b: complex
    c: complex
    coefficient_1: complex
    coefficient_2: complex


def tanh_rhs(tau: float, state: np.ndarray, ell: float, d: float, theta: float) -> np.ndarray:
    c = state[0] + 1j * state[1]
    v = state[2] + 1j * state[3]
    delta = d * np.tanh(tau / theta)
    acceleration = -(ell + 1j * delta) * v - 0.5 * ell * c
    return np.array([v.real, v.imag, acceleration.real, acceleration.imag])


def solve_tanh_ode(
    ell: float,
    d: float,
    theta: float,
    tau_final: float,
    n_times: int = 4001,
) -> dict[str, np.ndarray | float]:
    tau = np.linspace(0.0, tau_final, n_times)
    solution = solve_ivp(
        tanh_rhs,
        (0.0, tau_final),
        np.array([1.0, 0.0, 0.0, 0.0]),
        t_eval=tau,
        args=(ell, d, theta),
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    c = solution.y[0] + 1j * solution.y[1]
    dc = solution.y[2] + 1j * solution.y[3]
    population = np.abs(c) ** 2
    population_rate = 2.0 * np.real(np.conjugate(c) * dc)
    gamma_rate = np.full_like(population, np.nan)
    regular = population > 1e-13
    gamma_rate[regular] = -population_rate[regular] / population[regular]
    delta = d * np.tanh(tau / theta)
    return {
        "tau": tau,
        "c": c,
        "dc": dc,
        "population": population,
        "population_rate": population_rate,
        "gamma": gamma_rate,
        "delta": delta,
    }


def _basis_value_and_derivative(
    x: mp.mpf,
    exponent_x: mp.mpc,
    exponent_one_minus_x: mp.mpc,
    hyper_a: mp.mpc,
    hyper_b: mp.mpc,
    hyper_c: mp.mpc,
) -> tuple[mp.mpc, mp.mpc]:
    hyper = mp.hyp2f1(hyper_a, hyper_b, hyper_c, x)
    hyper_derivative = (hyper_a * hyper_b / hyper_c) * mp.hyp2f1(
        hyper_a + 1, hyper_b + 1, hyper_c + 1, x
    )
    prefactor = x**exponent_x * (1 - x) ** exponent_one_minus_x
    value = prefactor * hyper
    derivative = prefactor * (
        (exponent_x / x - exponent_one_minus_x / (1 - x)) * hyper
        + hyper_derivative
    )
    return value, derivative


def hypergeometric_parameters(
    ell: float,
    d: float,
    theta: float,
    decimal_digits: int = 60,
) -> HypergeometricParameters:
    mp.mp.dps = decimal_digits
    ell_mp = mp.mpf(ell)
    d_mp = mp.mpf(d)
    theta_mp = mp.mpf(theta)

    p = theta_mp * mp.sqrt((ell_mp - 1j * d_mp) ** 2 - 2 * ell_mp) / 4
    q = theta_mp * mp.sqrt((ell_mp + 1j * d_mp) ** 2 - 2 * ell_mp) / 4
    total = p + q
    # sqrt(1+4R)=1-i*theta*d for R=-theta^2*d^2/4-i*theta*d/2.
    hyper_a = total + 1 - 0.5j * theta_mp * d_mp
    hyper_b = total + 0.5j * theta_mp * d_mp
    hyper_c = 1 + 2 * p

    x0 = mp.mpf("0.5")
    f1, df1 = _basis_value_and_derivative(x0, p, q, hyper_a, hyper_b, hyper_c)
    f2, df2 = _basis_value_and_derivative(
        x0,
        -p,
        q,
        hyper_a - 2 * p,
        hyper_b - 2 * p,
        1 - 2 * p,
    )
    determinant = f1 * df2 - f2 * df1
    target_derivative = ell_mp * theta_mp
    coefficient_1 = (df2 - f2 * target_derivative) / determinant
    coefficient_2 = (f1 * target_derivative - df1) / determinant
    return HypergeometricParameters(
        complex(p),
        complex(q),
        complex(hyper_a),
        complex(hyper_b),
        complex(hyper_c),
        complex(coefficient_1),
        complex(coefficient_2),
    )


def exact_tanh_point(
    tau: float,
    ell: float,
    d: float,
    theta: float,
    parameters: HypergeometricParameters,
    decimal_digits: int = 60,
) -> tuple[complex, complex, complex]:
    mp.mp.dps = decimal_digits
    tau_mp = mp.mpf(tau)
    ell_mp = mp.mpf(ell)
    d_mp = mp.mpf(d)
    theta_mp = mp.mpf(theta)
    p = mp.mpc(parameters.p)
    q = mp.mpc(parameters.q)
    hyper_a = mp.mpc(parameters.a)
    hyper_b = mp.mpc(parameters.b)
    hyper_c = mp.mpc(parameters.c)
    coefficient_1 = mp.mpc(parameters.coefficient_1)
    coefficient_2 = mp.mpc(parameters.coefficient_2)

    u = tau_mp / theta_mp
    x = (1 + mp.tanh(u)) / 2
    f1, df1 = _basis_value_and_derivative(x, p, q, hyper_a, hyper_b, hyper_c)
    f2, df2 = _basis_value_and_derivative(
        x,
        -p,
        q,
        hyper_a - 2 * p,
        hyper_b - 2 * p,
        1 - 2 * p,
    )
    y = coefficient_1 * f1 + coefficient_2 * f2
    dy_dx = coefficient_1 * df1 + coefficient_2 * df2
    delta = d_mp * mp.tanh(u)
    dx_dtau = 2 * x * (1 - x) / theta_mp
    gauge = mp.e ** (
        -ell_mp * tau_mp / 2
        -0.5j * d_mp * theta_mp * mp.log(mp.cosh(u))
    )
    c = gauge * y
    logarithmic_derivative = -(ell_mp + 1j * delta) / 2 + dx_dtau * dy_dx / y
    dc = c * logarithmic_derivative
    return complex(c), complex(dc), complex(logarithmic_derivative)


def validate_hypergeometric_solution(
    ode: dict[str, np.ndarray | float],
    ell: float,
    d: float,
    theta: float,
    n_samples: int = 81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, HypergeometricParameters]:
    parameters = hypergeometric_parameters(ell, d, theta)
    tau_ode = np.asarray(ode["tau"])
    c_ode = np.asarray(ode["c"])
    dc_ode = np.asarray(ode["dc"])
    sample_indices = np.unique(np.linspace(0, tau_ode.size - 1, n_samples, dtype=int))
    rows = []
    exact_c = []
    exact_dc = []
    for index in sample_indices:
        c_value, dc_value, q_value = exact_tanh_point(
            float(tau_ode[index]), ell, d, theta, parameters
        )
        exact_c.append(c_value)
        exact_dc.append(dc_value)
        rows.append(
            [
                tau_ode[index],
                abs(c_value - c_ode[index]),
                abs(dc_value - dc_ode[index]),
                abs(q_value - dc_ode[index] / c_ode[index]),
            ]
        )
    return (
        np.asarray(rows, dtype=float),
        sample_indices,
        np.asarray(exact_c),
        parameters,
    )


def make_comparison_figure(
    tanh_data: dict[str, np.ndarray | float],
    linear_data: dict[str, np.ndarray | float],
    sample_indices: np.ndarray,
    exact_c: np.ndarray,
    ell: float,
    d: float,
    theta: float,
    Omega: float,
) -> None:
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
    tau = np.asarray(tanh_data["tau"])
    population = np.asarray(tanh_data["population"])
    population_rate = np.asarray(tanh_data["population_rate"])
    delta = np.asarray(tanh_data["delta"])
    omega = Omega - delta
    heat_power = omega * population_rate
    safe_p = np.clip(population, 1e-14, 1 - 1e-14)
    mutual_information = -2 * (
        safe_p * np.log(safe_p) + (1 - safe_p) * np.log(1 - safe_p)
    )
    energy_backflow = np.concatenate(
        ([0.0], cumulative_trapezoid(np.maximum(heat_power, 0.0), tau))
    )

    linear_tau = np.asarray(linear_data["tau"])
    linear_delta = (d / theta) * linear_tau
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    axes[0, 0].plot(tau, delta, color="#1b9e77", lw=2.0, label="bounded tanh")
    axes[0, 0].plot(linear_tau, linear_delta, color="#d95f02", lw=1.5, ls="--", label="local linear chirp")
    axes[0, 0].axhline(Omega, color="black", lw=0.8, ls=":", label=r"$\omega_S=0$")
    axes[0, 0].set(xlabel=r"$\tau$", ylabel=r"$\bar\delta(\tau)$", title="Detuning protocols")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(tau, population, color="#1b9e77", lw=2.0, label="tanh ODE")
    axes[0, 1].plot(
        tau[sample_indices],
        np.abs(exact_c) ** 2,
        marker="o",
        ls="none",
        ms=3.0,
        mfc="none",
        mec="black",
        label=r"${}_2F_1$",
    )
    axes[0, 1].plot(linear_tau, linear_data["population"], color="#d95f02", lw=1.35, ls="--", label="linear")
    axes[0, 1].set(xlabel=r"$\tau$", ylabel=r"$P(\tau)$", title="Survival probability")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(tau, population_rate, color="#1b9e77", lw=1.7, label="tanh")
    axes[1, 0].plot(linear_tau, linear_data["population_rate"], color="#d95f02", lw=1.3, ls="--", label="linear")
    axes[1, 0].axhline(0.0, color="0.25", lw=0.8)
    axes[1, 0].fill_between(tau, 0.0, population_rate, where=population_rate > 0, color="#1b9e77", alpha=0.18)
    axes[1, 0].set(xlabel=r"$\tau$", ylabel=r"$\mathrm{d}P/\mathrm{d}\tau$", title="Population backflow")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(tau, mutual_information, color="#7570b3", lw=1.8, label=r"$\mathcal{I}_{S:B}$")
    right = axes[1, 1].twinx()
    right.plot(tau, energy_backflow, color="#e7298a", lw=1.5, ls="--", label=r"$\mathcal{E}_{\rm back}/\gamma_0$")
    axes[1, 1].set(xlabel=r"$\tau$", ylabel="mutual information", title="Correlations and energy backflow")
    right.set_ylabel("cumulative energy backflow")
    handles = axes[1, 1].lines + right.lines
    axes[1, 1].legend(handles, [line.get_label() for line in handles], frameon=False)

    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)
    fig.suptitle(rf"$\ell={ell:g}$, $d={d:g}$, $\theta={theta:g}$")
    fig.savefig(FIGURE_DIR / "tanh_detuning_exact_comparison.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "tanh_detuning_exact_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ell = 0.625
    Omega = 1.25
    theta = 4.0
    d = 0.625
    local_slope = d / theta
    tau_final = 8.0

    tanh_data = solve_tanh_ode(ell, d, theta, tau_final)
    validation, sample_indices, exact_c, parameters = validate_hypergeometric_solution(
        tanh_data, ell, d, theta
    )
    linear_data = solve_linear_chirp(ell, local_slope, Omega, n_times=4001)
    make_comparison_figure(
        tanh_data,
        linear_data,
        sample_indices,
        exact_c,
        ell,
        d,
        theta,
        Omega,
    )
    np.savetxt(
        DATA_DIR / "tanh_hypergeometric_validation.csv",
        validation,
        delimiter=",",
        header="tau,abs_c_hypergeom_minus_ode,abs_dc_hypergeom_minus_ode,abs_logderivative_error",
        comments="",
    )
    np.savetxt(
        DATA_DIR / "tanh_hypergeometric_parameters.csv",
        np.asarray(
            [
                [
                    parameters.p.real,
                    parameters.p.imag,
                    parameters.q.real,
                    parameters.q.imag,
                    parameters.a.real,
                    parameters.a.imag,
                    parameters.b.real,
                    parameters.b.imag,
                    parameters.c.real,
                    parameters.c.imag,
                ]
            ]
        ),
        delimiter=",",
        header="Re_p,Im_p,Re_q,Im_q,Re_a,Im_a,Re_b,Im_b,Re_c,Im_c",
        comments="",
    )
    max_rate = float(np.max(np.asarray(tanh_data["population_rate"])))
    print(f"max |c_hyp-c_ode| = {validation[:, 1].max():.12e}")
    print(f"max |dc_hyp-dc_ode| = {validation[:, 2].max():.12e}")
    print(f"max |q_hyp-q_ode| = {validation[:, 3].max():.12e}")
    print(f"max dP/dtau = {max_rate:.12e}")
    print(f"final P_tanh = {np.asarray(tanh_data['population'])[-1]:.12e}")
    print(f"final P_linear = {np.asarray(linear_data['population'])[-1]:.12e}")


if __name__ == "__main__":
    main()
