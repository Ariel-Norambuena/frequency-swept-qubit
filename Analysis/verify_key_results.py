"""Minimal regression checks for the PRR manuscript's critical claims."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, root

from exact_backflow_analysis import solve_trajectory
from finite_sweep_phase_boundary import switch_residual
from lobe_tip_analysis import endpoint_tip_residual


OMEGA_DELTA = 1.25


def weak_coupling_kernel(x: float) -> float:
    real = quad(lambda y: np.cos(0.5 * y * y), 0.0, x, epsabs=2e-13)[0]
    imag = quad(lambda y: np.sin(0.5 * y * y), 0.0, x, epsabs=2e-13)[0]
    return float(np.real(np.exp(-0.5j * x * x) * (real + 1j * imag)))


def main() -> None:
    switch = root(switch_residual, np.log([0.94, 0.32]), tol=2e-11)
    ell_switch, a_switch = np.exp(switch.x)
    assert switch.success
    assert abs(ell_switch - 0.93618777) < 5e-8
    assert abs(a_switch - 0.31910184) < 5e-8

    x0 = brentq(weak_coupling_kernel, 2.0, 2.3, xtol=2e-14)
    a_birth = (OMEGA_DELTA / x0) ** 2
    assert abs(a_birth - 0.34005247) < 5e-8

    tip = root(
        endpoint_tip_residual,
        np.log([0.49, 0.397]),
        args=(OMEGA_DELTA,),
        tol=2e-11,
    )
    ell_tip, a_tip = np.exp(tip.x)
    assert tip.success
    assert abs(ell_tip - 0.49225744) < 5e-8
    assert abs(a_tip - 0.39663267) < 5e-8

    data = solve_trajectory(0.1, 0.1, 1.25, 2.0, n_times=25001)
    tau = np.asarray(data["tau"])
    target = 9.318
    q_s = float(np.interp(target, tau, data["heat_power"]))
    p_s = float(np.interp(target, tau, data["population"]))
    w_s = -0.1 * p_s
    u_s = q_s + w_s
    gamma_s = float(np.interp(target, tau, data["gamma"]))
    minus_e_b = float(np.interp(target, tau, data["reservoir_to_system_power"]))
    minus_e_i = float(np.interp(target, tau, data["interaction_to_system_power"]))
    assert q_s > 0.0 and minus_e_b < 0.0 and minus_e_i > 0.0
    assert w_s < 0.0 and u_s < 0.0
    assert -gamma_s < 0.1 / (2.0 - 0.1 * target)
    assert abs(q_s - minus_e_b - minus_e_i) < 1e-12
    assert abs(u_s - q_s - w_s) < 1e-15

    print("All critical-coordinate and energy-balance checks passed.")


if __name__ == "__main__":
    main()
