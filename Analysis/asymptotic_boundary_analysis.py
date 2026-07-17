"""High-precision tangency calculation for the small-sweep boundary.

The upper non-Markovian boundary is located by solving simultaneously
Re q(tau)=0 and d_tau Re q(tau)=0, where q=c'/c obeys an exact Riccati
equation.  This avoids testing exponentially small absolute population rates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from scipy.optimize import root


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"


def scaled_rhs(x: float, state: np.ndarray, ell: float, epsilon: float) -> np.ndarray:
    """Exponentially desingularized amplitude equation on x=a^(1/3) tau."""
    u = state[0] + 1j * state[1]
    ux = state[2] + 1j * state[3]
    delta = 2.0 - ell
    potential = (
        ell * delta / (4.0 * epsilon**2)
        + epsilon**2 * x**2 / 4.0
        - 0.5j * epsilon
        - 0.5j * ell * x
    )
    uxx = -potential * u
    return np.array([ux.real, ux.imag, uxx.real, uxx.imag])


def q_at_scaled(x: float, ell: float, epsilon: float) -> complex:
    solution = solve_ivp(
        scaled_rhs,
        (0.0, x),
        np.array([epsilon, 0.0, 0.5 * ell, 0.0]),
        args=(ell, epsilon),
        method="DOP853",
        rtol=3e-11,
        atol=3e-13,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    u = solution.y[0, -1] + 1j * solution.y[1, -1]
    ux = solution.y[2, -1] + 1j * solution.y[3, -1]
    return -0.5 * ell - 0.5j * epsilon**2 * x + epsilon * ux / u


def q_at_riccati(tau: float, ell: float, a: float) -> complex:
    def riccati_rhs(time: float, state: np.ndarray) -> np.ndarray:
        q = state[0] + 1j * state[1]
        qp = -(q * q) - (ell + 1j * a * time) * q - 0.5 * ell
        return np.array([qp.real, qp.imag])

    solution = solve_ivp(
        riccati_rhs,
        (0.0, tau),
        np.zeros(2),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[0, -1] + 1j * solution.y[1, -1]


def q_at_parabolic_cylinder(tau: float, ell: float, a: float, dps: int = 140) -> complex:
    """Independent evaluation of q=c'/c from the closed special-function form."""
    with mp.workdps(dps):
        ell_mp = mp.mpf(ell)
        a_mp = mp.mpf(a)
        tau_mp = mp.mpf(tau)
        nu = 1j * ell_mp / (2 * a_mp)
        eta = mp.e ** (-1j * mp.pi / 4) * mp.sqrt(a_mp)

        def z(time: mp.mpf) -> mp.mpc:
            return eta * (time - 1j * ell_mp / a_mp)

        def cylinder(order: mp.mpc, argument: mp.mpc) -> mp.mpc:
            return mp.pcfd(order, argument)

        def cylinder_prime(order: mp.mpc, argument: mp.mpc) -> mp.mpc:
            return 0.5 * argument * cylinder(order, argument) - cylinder(order + 1, argument)

        z0 = z(mp.mpf("0"))
        # D_nu(z) and D_{-nu-1}(i z) form a better-conditioned fundamental pair
        # than D_nu(+-z) for the large imaginary orders of the slow-sweep limit.
        conjugate_order = -nu - 1
        matrix = mp.matrix(
            [
                [cylinder(nu, z0), cylinder(conjugate_order, 1j * z0)],
                [
                    eta * cylinder_prime(nu, z0),
                    1j * eta * cylinder_prime(conjugate_order, 1j * z0),
                ],
            ]
        )
        constants = mp.lu_solve(matrix, mp.matrix([1, ell_mp / 2]))
        zt = z(tau_mp)
        denominator = constants[0] * cylinder(nu, zt) + constants[1] * cylinder(
            conjugate_order, 1j * zt
        )
        numerator = eta * (
            constants[0] * cylinder_prime(nu, zt)
            + 1j * constants[1] * cylinder_prime(conjugate_order, 1j * zt)
        )
        q = -ell_mp / 2 - 0.5j * a_mp * tau_mp + numerator / denominator
        return complex(q)


def tangency_residual(log_variables: np.ndarray, a: float) -> np.ndarray:
    epsilon = a ** (1.0 / 3.0)
    mu = np.exp(log_variables[0])
    x = np.exp(log_variables[1])
    delta = mu * epsilon**2
    ell = 2.0 - delta
    if not 0.0 < ell < 2.0:
        return np.array([10.0 + abs(ell), 10.0 + abs(ell)])
    tau = x / epsilon
    q = q_at_scaled(x, ell, epsilon)
    qp = -(q * q) - (ell + 1j * a * tau) * q - 0.5 * ell
    return np.array([q.real, qp.real])


def solve_boundary(a: float, guess: tuple[float, float]) -> tuple[float, float, complex]:
    mu_guess, x_guess = guess
    epsilon = a ** (1.0 / 3.0)
    solution = root(
        tangency_residual,
        np.log([mu_guess, x_guess]),
        args=(a,),
        method="hybr",
        options={"xtol": 2e-10, "maxfev": 250},
    )
    if not solution.success:
        raise RuntimeError(f"a={a:g}: {solution.message}; residual={solution.fun}")
    mu, x = np.exp(solution.x)
    delta = mu * epsilon**2
    tau = x / epsilon
    ell = 2.0 - delta
    q = q_at_scaled(x, ell, epsilon)
    return ell, tau, q


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    a_values = np.geomspace(1e-2, 1e-10, 33)
    rows: list[tuple[float, ...]] = []
    guess = (10.4, 1.35)
    for a in a_values:
        ell, tau, q = solve_boundary(float(a), guess)
        delta = 2.0 - ell
        epsilon = a ** (1.0 / 3.0)
        coefficient_3_4 = np.sqrt(np.pi) / 2.0 ** 0.75
        ell_asymptotic = 2.0 - np.sqrt(2.0) * np.pi * np.sqrt(a)
        ell_asymptotic_next = ell_asymptotic + coefficient_3_4 * a**0.75
        tau_asymptotic = np.sqrt(np.sqrt(2.0) * np.pi) * a ** (-0.25)
        rows.append(
            (
                a,
                ell,
                delta,
                tau,
                delta / np.sqrt(a),
                tau * a**0.25,
                ell_asymptotic,
                ell - ell_asymptotic,
                ell_asymptotic_next,
                ell - ell_asymptotic_next,
                tau_asymptotic,
                q.imag,
            )
        )
        guess = (delta / epsilon**2, tau * epsilon)
        print(
            f"a={a:.3e} ell_c={ell:.12f} delta={delta:.6e} "
            f"delta/sqrt(a)={delta/np.sqrt(a):.7f} "
            f"two-term error={ell-ell_asymptotic_next:+.3e}"
        )

    np.savetxt(
        DATA_DIR / "small_a_upper_boundary.csv",
        np.asarray(rows),
        delimiter=",",
        header=(
            "a,ell_critical,delta,tau_tangent,delta_over_sqrt_a,"
            "tau_times_a_1_4,ell_leading_asymptotic,ell_error,"
            "ell_two_term_asymptotic,ell_two_term_error,"
            "tau_leading_asymptotic,imag_q"
        ),
        comments="",
    )

    validation_rows = []
    for row_index in (0, 2, 4):
        a, ell, _, tau, *_ = rows[row_index]
        # q_at_scaled uses the exact x=a^(1/3) tau desingularization employed
        # by scaled_rhs; this is independent of the a^(1/4) asymptotic layer.
        epsilon = a ** (1.0 / 3.0)
        q_scaled = q_at_scaled(tau * epsilon, ell, epsilon)
        q_riccati = q_at_riccati(tau, ell, a)
        q_cylinder = q_at_parabolic_cylinder(tau, ell, a) if row_index == 0 else complex(np.nan, np.nan)
        validation_rows.append(
            (
                a,
                ell,
                tau,
                q_scaled.real,
                q_scaled.imag,
                q_riccati.real,
                q_riccati.imag,
                q_cylinder.real,
                q_cylinder.imag,
                abs(q_scaled - q_riccati),
                abs(q_scaled - q_cylinder) if row_index == 0 else np.nan,
            )
        )
        print(
            f"triple validation a={a:.3e}: "
            f"|scaled-Riccati|={abs(q_scaled-q_riccati):.3e}, "
            + (
                f"|scaled-cylinder|={abs(q_scaled-q_cylinder):.3e}"
                if row_index == 0
                else "special-function check reserved for a=10^-2"
            )
        )
    np.savetxt(
        DATA_DIR / "small_a_triple_validation.csv",
        np.asarray(validation_rows),
        delimiter=",",
        header=(
            "a,ell,tau,scaled_real_q,scaled_imag_q,riccati_real_q,riccati_imag_q,"
            "cylinder_real_q,cylinder_imag_q,scaled_riccati_abs_error,"
            "scaled_cylinder_abs_error"
        ),
        comments="",
    )


if __name__ == "__main__":
    main()
