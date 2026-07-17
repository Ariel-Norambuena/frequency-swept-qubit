"""Continue the finite-sweep switch, endpoint birth, and cusp versus range."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares

from lobe_tip_analysis import endpoint_q, endpoint_tip_residual


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
RANGES = (1.0, 1.25, 1.5, 2.0)


def weak_coupling_kernel(x: float) -> float:
    real = quad(lambda y: np.cos(0.5 * y * y), 0.0, x, epsabs=2e-13)[0]
    imag = quad(lambda y: np.sin(0.5 * y * y), 0.0, x, epsabs=2e-13)[0]
    return float(np.real(np.exp(-0.5j * x * x) * (real + 1j * imag)))


def switch_residual(log_variables: np.ndarray, omega_delta: float) -> np.ndarray:
    ell, a = np.exp(log_variables)
    tau_f = omega_delta / a
    q = endpoint_q(ell, a, omega_delta)
    q_prime = -q * q - (ell + 1j * a * tau_f) * q - 0.5 * ell
    return np.array([q.real, q_prime.real])


def solve_range(omega_delta: float, x0: float) -> np.ndarray:
    scale = (omega_delta / 1.25) ** 2
    switch = least_squares(
        switch_residual,
        np.log([0.93618777, 0.31910184 * scale]),
        args=(omega_delta,),
        bounds=(np.log([1e-4, 1e-4]), np.log([4.0, 4.0])),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=3000,
    )
    tip = least_squares(
        lambda z: endpoint_tip_residual(z, omega_delta),
        np.log([0.49225744, 0.39663267 * scale]),
        bounds=(np.log([1e-4, 1e-4]), np.log([4.0, 4.0])),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=3000,
    )
    if max(np.max(np.abs(switch.fun)), np.max(np.abs(tip.fun))) > 1e-9:
        raise RuntimeError(f"Unresolved branch at Omega_delta={omega_delta}")
    ell_switch, a_switch = np.exp(switch.x)
    ell_tip, a_tip = np.exp(tip.x)
    return np.array(
        [omega_delta, ell_switch, a_switch, (omega_delta / x0) ** 2, ell_tip, a_tip]
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    x0 = brentq(weak_coupling_kernel, 2.0, 2.3, xtol=2e-14)
    rows = np.vstack([solve_range(value, x0) for value in RANGES])
    np.savetxt(
        DATA_DIR / "sweep_range_dependence.csv",
        rows,
        delimiter=",",
        header="Omega_delta,ell_switch,a_switch,a_birth,ell_tip,a_tip",
        comments="",
    )
    for row in rows:
        print(", ".join(f"{value:.10f}" for value in row))


if __name__ == "__main__":
    main()
