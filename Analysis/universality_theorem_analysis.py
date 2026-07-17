"""Asymptotic universality of the slow-sweep non-Markovian boundary.

For a detuning family

    delta_a(tau) = d f(a tau / d),
    f(x) = x + c_m x^m + O(x^(m+1)),

the first profile-dependent displacement relative to the exactly linear
boundary is predicted to be

    ell_c[f] - ell_c[linear]
      = -K_m c_m a^((3m-1)/4) / d^(m-1) + o(...).

This script validates the cubic constant without subtracting two nearly equal
boundaries by integrating the exact Riccati variational equations.  It also
checks several nonlinear profiles directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import root

from asymptotic_boundary_analysis import solve_boundary as solve_linear_boundary
from publication_style import close_axes, configure_publication_style


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
FIGURE_DIR = ROOT / "output" / "figures"

MU0 = np.sqrt(2.0) * np.pi
K0 = np.sqrt(np.pi / np.sqrt(2.0))


def universal_constant(m: int) -> float:
    """Return K_m = mu_0 k_0^(-m-3) int_0^pi x^m sin^2(x) dx."""
    integral = quad(lambda x: x**m * np.sin(x) ** 2, 0.0, np.pi, epsabs=2e-14)[0]
    return MU0 * integral / K0 ** (m + 3)


K2 = universal_constant(2)
K3 = universal_constant(3)
K2_CLOSED = 2.0 ** (7.0 / 4.0) * (2.0 * np.pi**2 - 3.0) / (12.0 * np.sqrt(np.pi))
K3_CLOSED = (np.pi**2 - 3.0) / 2.0


def linear_variational_rhs(
    tau: float, state: np.ndarray, ell: float, a: float
) -> np.ndarray:
    """Riccati equation and derivatives with respect to ell and zeta.

    The auxiliary parameter zeta is defined by
        delta = a tau + zeta a tau^3.
    Since zeta=beta a^2 for delta=a tau+beta a^3 tau^3,
    d ell_c/d zeta tends to -K_3.
    """
    q = state[0] + 1j * state[1]
    q_ell = state[2] + 1j * state[3]
    q_zeta = state[4] + 1j * state[5]
    delta = a * tau
    common = 2.0 * q + ell + 1j * delta
    qp = -q * q - (ell + 1j * delta) * q - 0.5 * ell
    q_ell_p = -common * q_ell - q - 0.5
    q_zeta_p = -common * q_zeta - 1j * (a * tau**3) * q
    return np.array(
        [qp.real, qp.imag, q_ell_p.real, q_ell_p.imag, q_zeta_p.real, q_zeta_p.imag]
    )


def cubic_boundary_sensitivity(a: float, ell: float, tau: float) -> tuple[float, float, float]:
    """Return d ell_c/d zeta, d tau_*/d zeta, and the tangency residual."""
    solution = solve_ivp(
        linear_variational_rhs,
        (0.0, tau),
        np.zeros(6),
        args=(ell, a),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    q = solution.y[0, -1] + 1j * solution.y[1, -1]
    q_ell = solution.y[2, -1] + 1j * solution.y[3, -1]
    q_zeta = solution.y[4, -1] + 1j * solution.y[5, -1]
    delta = a * tau
    common = 2.0 * q + ell + 1j * delta
    qp = -q * q - (ell + 1j * delta) * q - 0.5 * ell
    qp_ell = -common * q_ell - q - 0.5
    qp_zeta = -common * q_zeta - 1j * (a * tau**3) * q
    qpp = -common * qp - 1j * a * q
    jacobian = np.array(
        [[q_ell.real, qp.real], [qp_ell.real, qpp.real]], dtype=float
    )
    parameter_column = np.array([q_zeta.real, qp_zeta.real], dtype=float)
    ell_zeta, tau_zeta = np.linalg.solve(jacobian, -parameter_column)
    residual = float(np.hypot(q.real, qp.real))
    return float(ell_zeta), float(tau_zeta), residual


def detuning(tau: float, a: float, d: float, profile: str) -> tuple[float, float]:
    """Return delta and its derivative for a selected slow profile."""
    x = a * tau / d
    if profile == "tanh":
        value = d * np.tanh(x)
        derivative = a / np.cosh(x) ** 2
    elif profile == "arctan":
        value = d * np.arctan(x)
        derivative = a / (1.0 + x * x)
    elif profile == "sqrt":
        value = d * x / np.sqrt(1.0 + x * x)
        derivative = a / (1.0 + x * x) ** 1.5
    elif profile == "cubic_plus":
        c3 = 0.25
        value = d * (x + c3 * x**3)
        derivative = a * (1.0 + 3.0 * c3 * x**2)
    elif profile == "exponential":
        value = d * (1.0 - np.exp(-x))
        derivative = a * np.exp(-x)
    else:
        raise ValueError(profile)
    return float(value), float(derivative)


def profile_q(tau: float, ell: float, a: float, d: float, profile: str) -> complex:
    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        q = state[0] + 1j * state[1]
        delta, _ = detuning(time, a, d, profile)
        qp = -q * q - (ell + 1j * delta) * q - 0.5 * ell
        return np.array([qp.real, qp.imag])

    solution = solve_ivp(
        rhs,
        (0.0, tau),
        np.zeros(2),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[0, -1] + 1j * solution.y[1, -1]


def profile_residual(
    log_variables: np.ndarray, a: float, d: float, profile: str
) -> np.ndarray:
    mu, scaled_time = np.exp(log_variables)
    ell = 2.0 - mu * np.sqrt(a)
    tau = scaled_time * a ** (-0.25)
    if not 0.0 < ell < 2.0:
        return np.array([10.0 + abs(ell), 10.0 + abs(ell)])
    q = profile_q(tau, ell, a, d, profile)
    delta, _ = detuning(tau, a, d, profile)
    qp = -q * q - (ell + 1j * delta) * q - 0.5 * ell
    return np.array([q.real, qp.real])


def solve_profile_boundary(
    a: float, d: float, profile: str, guess: tuple[float, float]
) -> tuple[float, float]:
    solution = root(
        profile_residual,
        np.log(np.asarray(guess)),
        args=(a, d, profile),
        method="hybr",
        options={"xtol": 2e-10, "maxfev": 180},
    )
    # MINPACK may report stagnation after the residual has already reached
    # machine precision; the residual, rather than that status flag, is the
    # relevant acceptance criterion here.
    if np.linalg.norm(solution.fun) > 3e-8:
        raise RuntimeError(
            f"profile={profile}, a={a:g}: {solution.message}; residual={solution.fun}"
        )
    mu, scaled_time = np.exp(solution.x)
    return 2.0 - mu * np.sqrt(a), scaled_time * a ** (-0.25)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"K2 quadrature={K2:.12f}, closed={K2_CLOSED:.12f}")
    print(f"K3 quadrature={K3:.12f}, closed={K3_CLOSED:.12f}")
    print(f"C_tanh=K3/3={K3/3.0:.12f}")

    sensitivity_rows = []
    guess = (10.4, 1.35)
    for a in np.geomspace(1e-2, 1e-8, 25):
        epsilon_linear = a ** (1.0 / 3.0)
        ell, tau, _ = solve_linear_boundary(float(a), guess)
        ell_zeta, tau_zeta, residual = cubic_boundary_sensitivity(a, ell, tau)
        sensitivity_rows.append([a, ell, tau, ell_zeta, tau_zeta, residual, ell_zeta + K3])
        guess = ((2.0 - ell) / epsilon_linear**2, tau * epsilon_linear)
        print(
            f"a={a:.2e} d ell/dzeta={ell_zeta:+.10f} "
            f"target={-K3:+.10f} error={ell_zeta+K3:+.3e}"
        )
    sensitivity_data = np.asarray(sensitivity_rows)
    np.savetxt(
        DATA_DIR / "cubic_universality_variational.csv",
        sensitivity_data,
        delimiter=",",
        header="a,ell_linear,tau_tangent,dell_dzeta,dtau_dzeta,tangency_residual,dell_dzeta_plus_K3",
        comments="",
    )

    d = 0.625
    profile_specs = {
        "tanh": (3, -1.0 / 3.0),
        "arctan": (3, -1.0 / 3.0),
        "sqrt": (3, -1.0 / 2.0),
        "cubic_plus": (3, 1.0 / 4.0),
        "exponential": (2, -1.0 / 2.0),
    }
    direct_rows = []
    a_values = np.geomspace(2e-2, 2e-5, 16)
    linear_solver_guess = (10.4, 1.35)
    profile_guesses: dict[str, tuple[float, float]] = {}
    for a in a_values:
        epsilon_linear = a ** (1.0 / 3.0)
        ell_linear, tau_linear, _ = solve_linear_boundary(
            float(a), linear_solver_guess
        )
        linear_solver_guess = (
            (2.0 - ell_linear) / epsilon_linear**2,
            tau_linear * epsilon_linear,
        )
        linear_guess = (
            (2.0 - ell_linear) / np.sqrt(a),
            tau_linear * a**0.25,
        )
        for profile, (m, coefficient) in profile_specs.items():
            ell_profile, tau_profile = solve_profile_boundary(
                float(a), d, profile, profile_guesses.get(profile, linear_guess)
            )
            profile_guesses[profile] = (
                (2.0 - ell_profile) / np.sqrt(a),
                tau_profile * a**0.25,
            )
            exponent = (3.0 * m - 1.0) / 4.0
            scale = -coefficient * a**exponent / d ** (m - 1)
            normalized = (ell_profile - ell_linear) / scale
            target = K2 if m == 2 else K3
            direct_rows.append(
                [
                    a,
                    m,
                    coefficient,
                    ell_linear,
                    ell_profile,
                    tau_linear,
                    tau_profile,
                    ell_profile - ell_linear,
                    normalized,
                    target,
                    normalized - target,
                ]
            )
    direct_data = np.asarray(direct_rows)
    np.savetxt(
        DATA_DIR / "profile_universality_direct.csv",
        direct_data,
        delimiter=",",
        header=(
            "a,m,c_m,ell_linear,ell_profile,tau_linear,tau_profile,"
            "ell_difference,normalized_difference,K_m,normalized_minus_K_m"
        ),
        comments="",
    )

    configure_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), constrained_layout=True)
    a_s = sensitivity_data[:, 0]
    axes[0].plot(a_s**0.25, -sensitivity_data[:, 3], "o-", ms=5.2, lw=2.1, color="#1b9e77")
    axes[0].axhline(K3, color="black", ls="--", lw=1.7, label=rf"$K_3=(\pi^2-3)/2$")
    axes[0].set(
        xlabel=r"$a^{1/4}$",
        ylabel=r"$-\partial\ell_c/\partial\zeta$",
        title="Exact cubic sensitivity",
    )
    axes[0].legend(frameon=True)

    colors = {
        "tanh": "#d95f02",
        "arctan": "#7570b3",
        "sqrt": "#e7298a",
        "cubic_plus": "#66a61e",
    }
    odd_names = list(colors)
    for index, profile in enumerate(profile_specs):
        if profile == "exponential":
            continue
        mask = np.arange(direct_data.shape[0]) % len(profile_specs) == index
        axes[1].plot(
            direct_data[mask, 0] ** 0.25,
            direct_data[mask, 8],
            "o-",
            ms=5.0,
            lw=2.0,
            color=colors[profile],
            label=profile.replace("cubic_plus", "cubic +"),
        )
    axes[1].axhline(K3, color="black", ls="--", lw=1.7)
    axes[1].set(
        xlabel=r"$a^{1/4}$",
        ylabel=r"$(\ell_c^f-\ell_c^{\rm lin})/[-c_3a^2/d^2]$",
        title="Odd-profile collapse",
    )
    axes[1].legend(frameon=True, ncol=2)

    exp_index = list(profile_specs).index("exponential")
    exp_mask = np.arange(direct_data.shape[0]) % len(profile_specs) == exp_index
    axes[2].plot(
        direct_data[exp_mask, 0] ** 0.25,
        direct_data[exp_mask, 8],
        "o-",
        ms=5.2,
        lw=2.1,
        color="#1f78b4",
        label="exponential",
    )
    axes[2].axhline(K2, color="black", ls="--", lw=1.7, label=r"$K_2$")
    axes[2].set(
        xlabel=r"$a^{1/4}$",
        ylabel=r"$(\ell_c^f-\ell_c^{\rm lin})/[-c_2a^{5/4}/d]$",
        title="Quadratic-profile law",
    )
    axes[2].legend(frameon=True)

    close_axes(axes)
    fig.savefig(FIGURE_DIR / "detuning_universality_theorem.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "detuning_universality_theorem.pdf", bbox_inches="tight")
    plt.close(fig)

    # Small-a polynomial extrapolation in epsilon=a^(1/4), reported per profile.
    print("\nDirect-profile extrapolations (quadratic in a^(1/4)):")
    for index, profile in enumerate(profile_specs):
        mask = np.arange(direct_data.shape[0]) % len(profile_specs) == index
        subset = direct_data[mask]
        order = np.argsort(subset[:, 0])[:8]
        x = subset[order, 0] ** 0.25
        y = subset[order, 8]
        intercept = np.polyfit(x, y, 2)[-1]
        target = K2 if profile == "exponential" else K3
        print(f"  {profile:12s}: intercept={intercept:.9f}, target={target:.9f}")


if __name__ == "__main__":
    main()
