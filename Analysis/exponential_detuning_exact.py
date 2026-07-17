"""Exact Lorentzian dynamics for delta(tau)=d*(1-exp(-tau/theta)).

After eliminating the first derivative, x=exp(-tau/theta) maps the equation
to the Morse/Whittaker equation.  This script validates the Whittaker solution
against direct integration of the original Lorentzian ODE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

from tanh_detuning_exact import solve_tanh_ode


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
FIGURE_DIR = ROOT / "output" / "figures"


@dataclass(frozen=True)
class WhittakerParameters:
    kappa: complex
    mu: complex
    coefficient_m: complex
    coefficient_w: complex


def exponential_rhs(
    tau: float,
    state: np.ndarray,
    ell: float,
    d: float,
    theta: float,
) -> np.ndarray:
    c = state[0] + 1j * state[1]
    dc = state[2] + 1j * state[3]
    delta = d * (1.0 - np.exp(-tau / theta))
    ddc = -(ell + 1j * delta) * dc - 0.5 * ell * c
    return np.array([dc.real, dc.imag, ddc.real, ddc.imag])


def solve_exponential_ode(
    ell: float,
    d: float,
    theta: float,
    tau_final: float,
    n_times: int = 4001,
) -> dict[str, np.ndarray]:
    tau = np.linspace(0.0, tau_final, n_times)
    solution = solve_ivp(
        exponential_rhs,
        (0.0, tau_final),
        np.array([1.0, 0.0, 0.0, 0.0]),
        args=(ell, d, theta),
        t_eval=tau,
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
    return {
        "tau": tau,
        "c": c,
        "dc": dc,
        "population": population,
        "population_rate": population_rate,
        "delta": d * (1.0 - np.exp(-tau / theta)),
    }


def _whittaker_values(
    z: mp.mpc,
    kappa: mp.mpc,
    mu: mp.mpc,
) -> tuple[mp.mpc, mp.mpc, mp.mpc, mp.mpc]:
    value_m = mp.whitm(kappa, mu, z)
    value_w = mp.whitw(kappa, mu, z)
    derivative_m = mp.diff(lambda argument: mp.whitm(kappa, mu, argument), z)
    derivative_w = mp.diff(lambda argument: mp.whitw(kappa, mu, argument), z)
    return value_m, derivative_m, value_w, derivative_w


def whittaker_parameters(
    ell: float,
    d: float,
    theta: float,
    decimal_digits: int = 60,
) -> WhittakerParameters:
    mp.mp.dps = decimal_digits
    ell_mp = mp.mpf(ell)
    d_mp = mp.mpf(d)
    theta_mp = mp.mpf(theta)
    kappa = (theta_mp * ell_mp - 1 + 1j * theta_mp * d_mp) / 2
    mu = theta_mp * mp.sqrt((ell_mp + 1j * d_mp) ** 2 - 2 * ell_mp) / 2
    z0 = 1j * theta_mp * d_mp
    value_m, derivative_m, value_w, derivative_w = _whittaker_values(
        z0, kappa, mu
    )
    target_w = mp.mpf(1)
    target_derivative = (1 - ell_mp * theta_mp) / (2 * z0)
    determinant = value_m * derivative_w - value_w * derivative_m
    coefficient_m = (
        target_w * derivative_w - value_w * target_derivative
    ) / determinant
    coefficient_w = (
        value_m * target_derivative - target_w * derivative_m
    ) / determinant
    return WhittakerParameters(
        complex(kappa), complex(mu), complex(coefficient_m), complex(coefficient_w)
    )


def exact_exponential_point(
    tau: float,
    ell: float,
    d: float,
    theta: float,
    parameters: WhittakerParameters,
    decimal_digits: int = 60,
) -> tuple[complex, complex, complex]:
    mp.mp.dps = decimal_digits
    tau_mp = mp.mpf(tau)
    ell_mp = mp.mpf(ell)
    d_mp = mp.mpf(d)
    theta_mp = mp.mpf(theta)
    kappa = mp.mpc(parameters.kappa)
    mu = mp.mpc(parameters.mu)
    coefficient_m = mp.mpc(parameters.coefficient_m)
    coefficient_w = mp.mpc(parameters.coefficient_w)
    x = mp.e ** (-tau_mp / theta_mp)
    z = 1j * theta_mp * d_mp * x
    value_m, derivative_m, value_w, derivative_w = _whittaker_values(z, kappa, mu)
    w = coefficient_m * value_m + coefficient_w * value_w
    derivative_w_z = coefficient_m * derivative_m + coefficient_w * derivative_w
    y = x ** (-mp.mpf("0.5")) * w
    delta = d_mp * (1 - x)
    integral_delta = d_mp * (tau_mp - theta_mp * (1 - x))
    gauge = mp.e ** (-ell_mp * tau_mp / 2 - 0.5j * integral_delta)
    c = gauge * y
    y_log_derivative = (mp.mpf("0.5") - z * derivative_w_z / w) / theta_mp
    q = -0.5 * (ell_mp + 1j * delta) + y_log_derivative
    dc = c * q
    return complex(c), complex(dc), complex(q)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ell = 0.625
    d = 0.625
    theta = 4.0
    tau_final = 8.0
    exponential = solve_exponential_ode(ell, d, theta, tau_final)
    tanh = solve_tanh_ode(ell, d, theta, tau_final)
    parameters = whittaker_parameters(ell, d, theta)
    tau = np.asarray(exponential["tau"])
    indices = np.unique(np.linspace(0, tau.size - 1, 81, dtype=int))
    rows = []
    exact_population = []
    for index in indices:
        c_exact, dc_exact, q_exact = exact_exponential_point(
            float(tau[index]), ell, d, theta, parameters
        )
        c_ode = exponential["c"][index]
        dc_ode = exponential["dc"][index]
        rows.append(
            [
                tau[index],
                abs(c_exact - c_ode),
                abs(dc_exact - dc_ode),
                abs(q_exact - dc_ode / c_ode),
            ]
        )
        exact_population.append(abs(c_exact) ** 2)
    validation = np.asarray(rows)
    np.savetxt(
        DATA_DIR / "exponential_whittaker_validation.csv",
        validation,
        delimiter=",",
        header="tau,abs_c_Whittaker_minus_ode,abs_dc_Whittaker_minus_ode,abs_logderivative_error",
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
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    axes[0].plot(tau, exponential["delta"], color="#d95f02", lw=1.8, label="exponential")
    axes[0].plot(tau, tanh["delta"], color="#1b9e77", lw=1.7, ls="--", label="tanh")
    axes[0].plot(tau, (d / theta) * tau, color="black", lw=1.0, ls=":", label="common tangent")
    axes[0].set(xlabel=r"$\tau$", ylabel=r"$\bar\delta(\tau)$", title="Equal initial slope")
    axes[0].legend(frameon=False)

    axes[1].plot(tau, exponential["population"], color="#d95f02", lw=1.8, label="exponential ODE")
    axes[1].plot(tau[indices], exact_population, "o", ms=3.0, mfc="none", mec="black", label="Whittaker")
    axes[1].plot(tau, tanh["population"], color="#1b9e77", lw=1.35, ls="--", label="tanh")
    axes[1].set(xlabel=r"$\tau$", ylabel=r"$P(\tau)$", title="Survival probability")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(frameon=False)

    axes[2].plot(tau, exponential["population_rate"], color="#d95f02", lw=1.7, label="exponential")
    axes[2].plot(tau, tanh["population_rate"], color="#1b9e77", lw=1.35, ls="--", label="tanh")
    axes[2].axhline(0.0, color="0.25", lw=0.8)
    axes[2].set(xlabel=r"$\tau$", ylabel=r"$\mathrm{d}P/\mathrm{d}\tau$", title="Backflow if positive")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)
    fig.savefig(FIGURE_DIR / "exponential_detuning_whittaker.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "exponential_detuning_whittaker.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"max |c_W-c_ode| = {validation[:, 1].max():.12e}")
    print(f"max |dc_W-dc_ode| = {validation[:, 2].max():.12e}")
    print(f"max |q_W-q_ode| = {validation[:, 3].max():.12e}")
    print(f"P_final_exponential = {exponential['population'][-1]:.12e}")
    print(f"P_final_tanh = {tanh['population'][-1]:.12e}")
    print(f"max dP/dtau exponential = {np.max(exponential['population_rate']):.12e}")


if __name__ == "__main__":
    main()
