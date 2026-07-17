"""Exact bi-Lorentzian pseudomode boundary near a second-order EP.

The spectral density is a positive sum of two co-centered Lorentzians,

    J(x)/gamma0 = sum_j w_j ell_j**2 / [2*pi*(x**2+ell_j**2)],
    ell_j = r*rho_j,

with sum_j w_j = 1.  Its vacuum dynamics is exactly represented by two damped
pseudomodes.  The script locates the static exceptional point and continues
the first non-Markovian double tangency for a common linear detuning a*tau.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, root

from publication_style import close_axes, configure_publication_style


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
FIGURE_DIR = ROOT / "output" / "figures"

WEIGHTS = np.array([0.65, 0.35], dtype=float)
WIDTH_RATIOS = np.array([1.0, 3.0], dtype=float)


def pole_parameters(r: float) -> tuple[np.ndarray, np.ndarray]:
    ell = r * WIDTH_RATIOS
    coupling = np.sqrt(0.5 * WEIGHTS * ell)
    return ell, coupling


def static_ep_residual(variables: np.ndarray) -> np.ndarray:
    sigma, r = variables
    ell, coupling = pole_parameters(r)
    denominators = sigma + ell
    secular = sigma + np.sum(coupling**2 / denominators)
    derivative = 1.0 - np.sum(coupling**2 / denominators**2)
    return np.array([secular, derivative], dtype=float)


def static_exceptional_point() -> tuple[float, float, np.ndarray]:
    solution = root(static_ep_residual, np.array([-1.0, 1.4]), tol=1e-12)
    if not solution.success or np.linalg.norm(solution.fun) > 1e-9:
        raise RuntimeError(f"static EP solve failed: {solution.message}, {solution.fun}")
    sigma, r = solution.x
    ell, coupling = pole_parameters(float(r))
    generator = np.array(
        [
            [0.0, -1j * coupling[0], -1j * coupling[1]],
            [-1j * coupling[0], -ell[0], 0.0],
            [-1j * coupling[1], 0.0, -ell[1]],
        ],
        dtype=complex,
    )
    eigenvalues = np.linalg.eigvals(generator)
    return float(sigma), float(r), eigenvalues


def rhs(tau: float, state: np.ndarray, r: float, a: float) -> np.ndarray:
    ell, coupling = pole_parameters(r)
    c = state[0]
    modes = state[1:]
    derivative = np.empty_like(state)
    derivative[0] = -1j * np.dot(coupling, modes)
    derivative[1:] = -(ell + 1j * a * tau) * modes - 1j * coupling * c
    return derivative


def endpoint_q(r: float, a: float, tau: float) -> tuple[complex, complex]:
    # Remove the dominant EP decay.  The common exponential cancels from q,
    # but prevents loss of relative accuracy when tau grows as a^(-1/4).
    sigma_shift = -0.94
    solution = solve_ivp(
        lambda time, state: rhs(time, state, r, a) - sigma_shift * state,
        (0.0, tau),
        np.array([1.0 + 0.0j, 0.0j, 0.0j]),
        method="DOP853",
        rtol=2e-10,
        atol=2e-13,
        t_eval=[tau],
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    state = solution.y[:, -1]
    # `state` is exponentially rescaled; applying the original generator gives
    # the physical logarithmic derivatives because the scale is common.
    first = rhs(tau, state, r, a)
    c = state[0]
    c_prime = first[0]
    ell, coupling = pole_parameters(r)
    mode_prime = first[1:]
    c_second = -1j * np.dot(coupling, mode_prime)
    q = c_prime / c
    q_prime = c_second / c - q * q
    return q, q_prime


def tangency_residual(log_variables: np.ndarray, a: float) -> np.ndarray:
    r, tau = np.exp(log_variables)
    q, q_prime = endpoint_q(float(r), a, float(tau))
    return np.array([q.real, q_prime.real], dtype=float)


def continue_boundary(r_ep: float) -> np.ndarray:
    a_descending = np.logspace(-2.0, -6.0, 17)
    rows: list[list[float]] = []
    previous_a: float | None = None
    previous_r: float | None = None
    previous_tau: float | None = None

    for a in a_descending:
        if previous_a is None:
            r_guess = max(0.2, r_ep - 0.35)
            tau_guess = 7.4
        else:
            scale = a / previous_a
            r_guess = r_ep - (r_ep - previous_r) * np.sqrt(scale)
            tau_guess = previous_tau * scale ** (-0.25)
        lower = np.log([0.25 * r_ep, 0.3 * a ** (-0.25)])
        upper = np.log([1.02 * r_ep, 5.0 * a ** (-0.25)])
        solution = least_squares(
            lambda variables: tangency_residual(variables, float(a)),
            np.log([r_guess, tau_guess]),
            bounds=(lower, upper),
            xtol=2e-10,
            ftol=2e-10,
            gtol=2e-10,
            max_nfev=120,
        )
        residual = tangency_residual(solution.x, float(a))
        if not solution.success or np.linalg.norm(residual) > 5e-7:
            raise RuntimeError(
                f"tangency solve failed at a={a}: {solution.message}; residual={residual}"
            )
        r, tau = np.exp(solution.x)
        q, q_prime = endpoint_q(float(r), float(a), float(tau))
        print(
            f"a={a:.6g}, r_c={r:.12g}, tau_*={tau:.12g}, "
            f"residual={np.linalg.norm(residual):.3e}"
        )
        rows.append([a, r, tau, q.real, q.imag, q_prime.real, q_prime.imag])
        previous_a = float(a)
        previous_r = float(r)
        previous_tau = float(tau)

    return np.asarray(rows[::-1], dtype=float)


def fit_fractional_boundary(data: np.ndarray, r_ep: float) -> tuple[np.ndarray, np.ndarray]:
    a = data[:, 0]
    deficit = r_ep - data[:, 1]
    fit_mask = a <= 1e-3
    design = np.column_stack(
        [a[fit_mask] ** 0.5, a[fit_mask] ** 0.75, a[fit_mask]]
    )
    coefficients, *_ = np.linalg.lstsq(design, deficit[fit_mask], rcond=None)
    prediction = (
        coefficients[0] * a**0.5
        + coefficients[1] * a**0.75
        + coefficients[2] * a
    )
    return coefficients, prediction


def make_figure(
    data: np.ndarray,
    r_ep: float,
    sigma_ep: float,
    eigenvalues: np.ndarray,
    coefficients: np.ndarray,
    prediction: np.ndarray,
) -> None:
    configure_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.15), constrained_layout=True)

    ell, _ = pole_parameters(r_ep)
    x = np.linspace(-8.0, 8.0, 700)
    components = [
        weight * width**2 / (2.0 * np.pi * (x**2 + width**2))
        for weight, width in zip(WEIGHTS, ell, strict=True)
    ]
    total = np.sum(components, axis=0)
    for index, component in enumerate(components, start=1):
        axes[0].plot(x, component, lw=1.9, ls="--", label=rf"$J_{index}(\omega)$")
    axes[0].plot(x, total, color="black", lw=2.6, label=r"$J_1(\omega)+J_2(\omega)$")
    axes[0].set(
        xlabel=r"$(\omega-\omega_b)/\gamma_0$",
        ylabel=r"$J(\omega)/\gamma_0$",
        title="Bi-Lorentzian spectrum",
    )
    axes[0].legend(frameon=True)

    a = data[:, 0]
    r = data[:, 1]
    axes[1].semilogx(a, r, "o", ms=5.5, color="#1b9e77", label="exact tangency")
    axes[1].semilogx(a, r_ep - prediction, color="#d95f02", lw=2.3,
                    label=r"$r_*-C_{1/2}a^{1/2}-C_{3/4}a^{3/4}-C_1a$")
    axes[1].axhline(r_ep, color="black", lw=1.5, ls="--", label=rf"$r_*={r_ep:.6f}$")
    axes[1].set(xlabel=r"$a$", ylabel=r"$r_c(a)$", title="Exact swept boundary")
    axes[1].legend(frameon=True)

    scaled = (r_ep - r) / np.sqrt(a)
    axes[2].plot(a**0.25, scaled, "o", ms=5.5, color="#7570b3", label="exact tangency")
    line = coefficients[0] + coefficients[1] * a**0.25 + coefficients[2] * a**0.5
    axes[2].plot(a**0.25, line, color="#e7298a", lw=2.3,
                 label=r"$C_{1/2}+C_{3/4}a^{1/4}+C_1a^{1/2}$")
    axes[2].axhline(coefficients[0], color="black", lw=1.5, ls="--",
                    label=rf"$C_{{1/2}}={coefficients[0]:.6f}$")
    axes[2].set(
        xlabel=r"$a^{1/4}$",
        ylabel=r"$[r_*-r_c(a)]/a^{1/2}$",
        title="Second-order-EP scaling",
    )
    axes[2].legend(frameon=True)

    close_axes(axes)
    fig.savefig(FIGURE_DIR / "bilorentzian_boundary.pdf")
    fig.savefig(FIGURE_DIR / "bilorentzian_boundary.png")
    plt.close(fig)

    separated = eigenvalues[np.argmax(np.abs(eigenvalues - sigma_ep))]
    print(
        f"static EP: sigma={sigma_ep:.12g}, r={r_ep:.12g}, "
        f"third eigenvalue={separated:.12g}"
    )
    print(
        "fractional coefficients: "
        f"C_1/2={coefficients[0]:.12g}, C_3/4={coefficients[1]:.12g}, "
        f"C_1={coefficients[2]:.12g}"
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sigma_ep, r_ep, eigenvalues = static_exceptional_point()
    data = continue_boundary(r_ep)
    coefficients, prediction = fit_fractional_boundary(data, r_ep)
    np.savetxt(
        DATA_DIR / "bilorentzian_boundary.csv",
        data,
        delimiter=",",
        header="a,r_critical,tau_tangency,Re_q,Im_q,Re_q_prime,Im_q_prime",
        comments="",
    )
    summary = np.array(
        [[sigma_ep, r_ep, *coefficients, *WEIGHTS, *WIDTH_RATIOS]], dtype=float
    )
    np.savetxt(
        DATA_DIR / "bilorentzian_summary.csv",
        summary,
        delimiter=",",
        header=(
            "sigma_ep,r_ep,C_half,C_three_quarters,C_one,"
            "weight_1,weight_2,width_ratio_1,width_ratio_2"
        ),
        comments="",
    )
    make_figure(data, r_ep, sigma_ep, eigenvalues, coefficients, prediction)


if __name__ == "__main__":
    main()
