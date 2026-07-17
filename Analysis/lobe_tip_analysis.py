"""Exact finite-sweep boundary and upper-tip analysis for fixed Omega."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar, root


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"


def rhs(tau: float, state: np.ndarray, ell: float, a: float) -> np.ndarray:
    c = state[0] + 1j * state[1]
    v = state[2] + 1j * state[3]
    vp = -(ell + 1j * a * tau) * v - 0.5 * ell * c
    return np.array([v.real, v.imag, vp.real, vp.imag])


def trajectory(ell: float, a: float, Omega: float, n: int = 1601) -> tuple[np.ndarray, np.ndarray]:
    tau_final = Omega / a
    tau = np.linspace(0.0, tau_final, n)
    sol = solve_ivp(
        rhs,
        (0.0, tau_final),
        np.array([1.0, 0.0, 0.0, 0.0]),
        args=(ell, a),
        t_eval=tau,
        dense_output=True,
        method="DOP853",
        rtol=3e-11,
        atol=3e-13,
    )
    c = sol.y[0] + 1j * sol.y[1]
    v = sol.y[2] + 1j * sol.y[3]
    return tau, np.real(v / c)


def maximum_log_rate(ell: float, a: float, Omega: float) -> tuple[float, float]:
    tau, rate = trajectory(ell, a, Omega)
    candidates = []
    for index in range(1, len(tau) - 1):
        if rate[index] >= rate[index - 1] and rate[index] >= rate[index + 1]:
            candidates.append(index)
    candidates.append(len(tau) - 1)
    best_value = -np.inf
    best_tau = tau[-1]
    for index in candidates:
        if index == len(tau) - 1:
            value = rate[-1]
            critical_tau = tau[-1]
        else:
            left = tau[index - 1]
            right = tau[index + 1]

            def objective(time: float) -> float:
                # A short independent integration is robust at the moderate-a tip.
                local = solve_ivp(
                    rhs,
                    (0.0, time),
                    np.array([1.0, 0.0, 0.0, 0.0]),
                    args=(ell, a),
                    method="DOP853",
                    rtol=3e-11,
                    atol=3e-13,
                )
                c = local.y[0, -1] + 1j * local.y[1, -1]
                v = local.y[2, -1] + 1j * local.y[3, -1]
                return -float(np.real(v / c))

            optimum = minimize_scalar(objective, bounds=(left, right), method="bounded")
            value = -float(optimum.fun)
            critical_tau = float(optimum.x)
        if value > best_value:
            best_value = value
            best_tau = critical_tau
    return best_value, best_tau


def endpoint_q(ell: float, a: float, Omega: float) -> complex:
    tau_final = Omega / a
    sol = solve_ivp(
        rhs,
        (0.0, tau_final),
        np.array([1.0, 0.0, 0.0, 0.0]),
        args=(ell, a),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    c = sol.y[0, -1] + 1j * sol.y[1, -1]
    v = sol.y[2, -1] + 1j * sol.y[3, -1]
    return v / c


def endpoint_tip_residual(log_variables: np.ndarray, Omega: float) -> np.ndarray:
    ell, a = np.exp(log_variables)
    q = endpoint_q(ell, a, Omega)
    h = 2e-4 * ell
    derivative = (endpoint_q(ell + h, a, Omega).real - endpoint_q(ell - h, a, Omega).real) / (2.0 * h)
    return np.array([q.real, derivative])


def endpoint_boundary_roots(a: float, Omega: float) -> list[float]:
    ell_grid = np.linspace(0.01, 1.4, 220)
    values = np.array([endpoint_q(float(ell), a, Omega).real for ell in ell_grid])
    roots = []
    for left, right, f_left, f_right in zip(ell_grid[:-1], ell_grid[1:], values[:-1], values[1:]):
        if f_left * f_right < 0.0:
            roots.append(brentq(lambda ell: endpoint_q(ell, a, Omega).real, left, right, xtol=2e-12))
    return roots


def main() -> None:
    Omega = 1.25
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    scan_rows = []
    for a in np.linspace(0.32, 0.43, 12):
        best = (-np.inf, np.nan, np.nan)
        for ell in np.linspace(0.08, 1.15, 72):
            value, tau = maximum_log_rate(float(ell), float(a), Omega)
            if value > best[0]:
                best = (value, ell, tau)
        scan_rows.append((a, best[1], best[2], best[0], Omega / a, best[2] / (Omega / a)))
        print(
            f"a={a:.5f} ell_argmax={best[1]:.6f} tau={best[2]:.6f} "
            f"max Re(q)={best[0]:+.6e} tau/tauf={best[2]/(Omega/a):.6f}"
        )

    tip = root(endpoint_tip_residual, np.log([0.45, 0.39]), args=(Omega,), tol=2e-10)
    if not tip.success:
        raise RuntimeError(tip.message)
    ell_tip, a_tip = np.exp(tip.x)
    tau_tip = Omega / a_tip
    q_tip = endpoint_q(ell_tip, a_tip, Omega)
    qprime_tip = -(q_tip * q_tip) - (ell_tip + 1j * a_tip * tau_tip) * q_tip - 0.5 * ell_tip

    h_a = 8e-5
    h_ell = 4e-4
    g0 = q_tip.real
    g_a = (
        endpoint_q(ell_tip, a_tip - 2 * h_a, Omega).real
        - 8 * endpoint_q(ell_tip, a_tip - h_a, Omega).real
        + 8 * endpoint_q(ell_tip, a_tip + h_a, Omega).real
        - endpoint_q(ell_tip, a_tip + 2 * h_a, Omega).real
    ) / (12 * h_a)
    g_ell_ell = (
        -endpoint_q(ell_tip + 2 * h_ell, a_tip, Omega).real
        + 16 * endpoint_q(ell_tip + h_ell, a_tip, Omega).real
        - 30 * g0
        + 16 * endpoint_q(ell_tip - h_ell, a_tip, Omega).real
        - endpoint_q(ell_tip - 2 * h_ell, a_tip, Omega).real
    ) / (12 * h_ell**2)
    cusp_coefficient = np.sqrt(2.0 * g_a / g_ell_ell)
    print(
        f"endpoint tip: a={a_tip:.12f}, ell={ell_tip:.12f}, "
        f"tau={tau_tip:.12f}, Re(q)={q_tip.real:+.3e}, Im(q)={q_tip.imag:+.12f}, "
        f"Re(q')={qprime_tip.real:+.12f}"
    )
    print(
        f"tip derivatives: g_a={g_a:+.12f}, g_ell_ell={g_ell_ell:+.12f}, "
        f"C_tip={cusp_coefficient:.12f}"
    )

    cusp_rows = []
    for distance in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        a = a_tip - distance
        roots = endpoint_boundary_roots(a, Omega)
        if len(roots) != 2:
            raise RuntimeError(f"Expected two endpoint roots at a={a}, found {roots}")
        lower_asymptotic = ell_tip - cusp_coefficient * np.sqrt(distance)
        upper_asymptotic = ell_tip + cusp_coefficient * np.sqrt(distance)
        cusp_rows.append(
            (
                distance,
                a,
                roots[0],
                roots[1],
                lower_asymptotic,
                upper_asymptotic,
                roots[0] - lower_asymptotic,
                roots[1] - upper_asymptotic,
            )
        )
        print(
            f"tip distance={distance:.1e}: exact=({roots[0]:.9f},{roots[1]:.9f}) "
            f"sqrt-law=({lower_asymptotic:.9f},{upper_asymptotic:.9f})"
        )

    np.savetxt(
        DATA_DIR / "upper_tip_scan.csv",
        np.asarray(scan_rows),
        delimiter=",",
        header="a,ell_argmax,tau_argmax,max_real_q,tau_final,tau_over_tau_final",
        comments="",
    )
    np.savetxt(
        DATA_DIR / "upper_tip_point.csv",
        np.asarray(
            [[
                Omega,
                a_tip,
                ell_tip,
                tau_tip,
                q_tip.real,
                q_tip.imag,
                qprime_tip.real,
                g_a,
                g_ell_ell,
                cusp_coefficient,
            ]]
        ),
        delimiter=",",
        header=(
            "Omega,a_tip,ell_tip,tau_tip,real_q,imag_q,real_qprime,"
            "g_a,g_ell_ell,cusp_coefficient"
        ),
        comments="",
    )
    np.savetxt(
        DATA_DIR / "upper_tip_cusp_validation.csv",
        np.asarray(cusp_rows),
        delimiter=",",
        header=(
            "a_tip_minus_a,a,ell_lower_exact,ell_upper_exact,"
            "ell_lower_sqrt,ell_upper_sqrt,lower_error,upper_error"
        ),
        comments="",
    )


if __name__ == "__main__":
    main()
