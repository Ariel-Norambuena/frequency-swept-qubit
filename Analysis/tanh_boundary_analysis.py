"""Upper non-Markovian boundary for a bounded tanh detuning.

The saturation amplitude d is fixed and theta=d/a_eff, so the slope at the
resonance is a_eff.  The tangency is obtained from Re q=Re q'=0 using the
desingularized y equation, not exponentially small population derivatives.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from asymptotic_boundary_analysis import solve_boundary as solve_linear_boundary


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
FIGURE_DIR = ROOT / "output" / "figures"


def y_rhs(tau: float, state: np.ndarray, ell: float, d: float, theta: float) -> np.ndarray:
    y = state[0] + 1j * state[1]
    yp = state[2] + 1j * state[3]
    u = tau / theta
    delta = d * np.tanh(u)
    delta_prime = (d / theta) / np.cosh(u) ** 2
    potential = (
        0.5 * ell
        - 0.25 * ell**2
        + 0.25 * delta**2
        - 0.5j * (ell * delta + delta_prime)
    )
    ypp = -potential * y
    return np.array([yp.real, yp.imag, ypp.real, ypp.imag])


def logarithmic_derivative(tau: float, ell: float, d: float, theta: float) -> complex:
    solution = solve_ivp(
        y_rhs,
        (0.0, tau),
        np.array([1.0, 0.0, 0.5 * ell, 0.0]),
        args=(ell, d, theta),
        method="DOP853",
        rtol=3e-11,
        atol=3e-13,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    y = solution.y[0, -1] + 1j * solution.y[1, -1]
    yp = solution.y[2, -1] + 1j * solution.y[3, -1]
    delta = d * np.tanh(tau / theta)
    return -0.5 * (ell + 1j * delta) + yp / y


def residual(log_variables: np.ndarray, a_eff: float, d: float) -> np.ndarray:
    mu, scaled_time = np.exp(log_variables)
    ell = 2.0 - mu * np.sqrt(a_eff)
    tau = scaled_time * a_eff ** (-0.25)
    if not 0.0 < ell < 2.0:
        return np.array([10.0 + abs(ell), 10.0 + abs(ell)])
    theta = d / a_eff
    q = logarithmic_derivative(tau, ell, d, theta)
    delta = d * np.tanh(tau / theta)
    qp = -(q * q) - (ell + 1j * delta) * q - 0.5 * ell
    return np.array([q.real, qp.real])


def solve_boundary(a_eff: float, d: float, guess: tuple[float, float]) -> tuple[float, float, complex]:
    solution = root(
        residual,
        np.log(np.asarray(guess)),
        args=(a_eff, d),
        method="hybr",
        options={"xtol": 3e-10, "maxfev": 220},
    )
    if not solution.success or np.linalg.norm(solution.fun) > 2e-7:
        raise RuntimeError(
            f"a={a_eff:g}: {solution.message}; residual={solution.fun}"
        )
    mu, scaled_time = np.exp(solution.x)
    ell = 2.0 - mu * np.sqrt(a_eff)
    tau = scaled_time * a_eff ** (-0.25)
    q = logarithmic_derivative(tau, ell, d, d / a_eff)
    return ell, tau, q


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    d = 0.625
    a_values = np.geomspace(1e-6, 5e-2, 28)
    coefficient = np.sqrt(np.pi) / 2.0**0.75
    guess = (np.sqrt(2.0) * np.pi, np.sqrt(np.sqrt(2.0) * np.pi))
    rows = []
    for a_eff in a_values:
        ell, tau, q = solve_boundary(float(a_eff), d, guess)
        epsilon_linear = a_eff ** (1.0 / 3.0)
        ell_linear, tau_linear, _ = solve_linear_boundary(
            float(a_eff),
            (
                (2.0 - ell) / epsilon_linear**2,
                tau * epsilon_linear,
            ),
        )
        profile_difference = ell - ell_linear
        leading = 2.0 - np.sqrt(2.0) * np.pi * np.sqrt(a_eff)
        two_term = leading + coefficient * a_eff**0.75
        theta = d / a_eff
        rows.append(
            [
                a_eff,
                d,
                theta,
                ell,
                tau,
                ell - leading,
                ell - two_term,
                tau / theta,
                q.imag,
                ell_linear,
                tau_linear,
                profile_difference,
                profile_difference * d**2 / a_eff**2,
            ]
        )
        guess = ((2.0 - ell) / np.sqrt(a_eff), tau * a_eff**0.25)
        print(
            f"a={a_eff:.3e} ell={ell:.12f} tau={tau:.8f} "
            f"two-term error={ell-two_term:+.3e} "
            f"tanh-linear={profile_difference:+.3e} tau/theta={tau/theta:.3e}"
        )

    data = np.asarray(rows)
    np.savetxt(
        DATA_DIR / "tanh_small_a_upper_boundary.csv",
        data,
        delimiter=",",
        header=(
            "a_eff,d,theta,ell_critical,tau_tangent,ell_minus_leading,"
            "ell_minus_two_term,tau_over_theta,imag_q,ell_linear_exact,"
            "tau_linear_exact,ell_tanh_minus_linear,scaled_profile_difference"
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
    a = data[:, 0]
    ell = data[:, 3]
    leading = 2.0 - np.sqrt(2.0) * np.pi * np.sqrt(a)
    two_term = leading + coefficient * a**0.75
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.65), constrained_layout=True)
    axes[0].plot(a, ell, "o", ms=3.2, color="black", label="tanh tangency")
    axes[0].plot(a, leading, color="#d95f02", ls="--", lw=1.35, label=r"linear $O(a^{1/2})$")
    axes[0].plot(a, two_term, color="#1b9e77", lw=1.4, label=r"linear $O(a^{3/4})$")
    axes[0].set_xscale("log")
    axes[0].set(xlabel=r"local slope $a_{\rm eff}=d/\theta$", ylabel=r"$\ell_c$", title="Bounded-chirp boundary")
    axes[0].legend(frameon=False)

    collapse = (ell - leading) / a**0.75
    axes[1].plot(a**0.25, collapse, "o-", ms=3, lw=1.0, color="#7570b3")
    axes[1].axhline(coefficient, color="black", ls="--", lw=1.0, label=rf"$\sqrt{{\pi}}/2^{{3/4}}={coefficient:.6f}$")
    axes[1].set(xlabel=r"$a_{\rm eff}^{1/4}$", ylabel="scaled next coefficient", title="Linear-law recovery")
    axes[1].legend(frameon=False)

    profile_difference = np.abs(data[:, 11])
    axes[2].loglog(a, profile_difference, "o-", ms=3.0, lw=1.0, color="#e7298a", label="tanh minus linear")
    axes[2].loglog(a, 1.07 * a**2 / d**2, color="black", ls="--", lw=1.0, label=r"guide $1.07a^2/d^2$")
    axes[2].set(xlabel=r"$a_{\rm eff}$", ylabel=r"$|\ell_c^{\tanh}-\ell_c^{\rm lin}|$", title="First profile correction")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2, which="both")
    fig.savefig(FIGURE_DIR / "tanh_boundary_asymptotics.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "tanh_boundary_asymptotics.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
