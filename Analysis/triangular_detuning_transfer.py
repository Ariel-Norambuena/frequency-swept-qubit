"""Exact triangular double passage from parabolic-cylinder transfer matrices.

Each affine-detuning segment is a Weber problem.  Matching the state vector
(c,c') across the turning point gives an exact composite protocol without a
new differential-equation solve.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
FIGURE_DIR = ROOT / "output" / "figures"


def cylinder_prime(order: mp.mpc, argument: mp.mpc) -> mp.mpc:
    return 0.5 * argument * mp.pcfd(order, argument) - mp.pcfd(order + 1, argument)


def fundamental_matrix(
    shifted_time: mp.mpf,
    ell: mp.mpf,
    slope: mp.mpf,
) -> mp.matrix:
    nu = 1j * ell / (2 * slope)
    eta = mp.e ** (-1j * mp.pi / 4) * mp.sqrt(slope)
    z = eta * (shifted_time - 1j * ell / slope)
    gauge = mp.e ** (-ell * shifted_time / 2 - 0.25j * slope * shifted_time**2)
    gauge_log_derivative = -ell / 2 - 0.5j * slope * shifted_time
    first = mp.pcfd(nu, z)
    first_prime = cylinder_prime(nu, z)
    second_order = -nu - 1
    second = mp.pcfd(second_order, 1j * z)
    second_prime = cylinder_prime(second_order, 1j * z)
    return mp.matrix(
        [
            [gauge * first, gauge * second],
            [
                gauge * (gauge_log_derivative * first + eta * first_prime),
                gauge
                * (
                    gauge_log_derivative * second
                    + 1j * eta * second_prime
                ),
            ],
        ]
    )


def segment_propagator(
    delta_initial: float,
    delta_final: float,
    ell: float,
    slope: float,
    decimal_digits: int = 90,
) -> mp.matrix:
    if slope == 0.0:
        raise ValueError("the affine segment slope must be nonzero")
    with mp.workdps(decimal_digits):
        ell_mp = mp.mpf(ell)
        slope_mp = mp.mpf(slope)
        initial_time = mp.mpf(delta_initial) / slope_mp
        final_time = mp.mpf(delta_final) / slope_mp
        initial_matrix = fundamental_matrix(initial_time, ell_mp, slope_mp)
        final_matrix = fundamental_matrix(final_time, ell_mp, slope_mp)
        return final_matrix * initial_matrix**-1


def triangular_delta(tau: float | np.ndarray, d: float, slope: float) -> np.ndarray:
    tau_array = np.asarray(tau)
    turning_time = 2.0 * d / slope
    return np.where(
        tau_array <= turning_time,
        -d + slope * tau_array,
        d - slope * (tau_array - turning_time),
    )


def analytic_state(
    tau: float,
    ell: float,
    d: float,
    slope: float,
    decimal_digits: int = 90,
) -> tuple[complex, complex]:
    turning_time = 2.0 * d / slope
    initial_state = mp.matrix([1, 0])
    if tau <= turning_time:
        delta = -d + slope * tau
        propagator = segment_propagator(-d, float(delta), ell, slope, decimal_digits)
        state = propagator * initial_state
    else:
        first = segment_propagator(-d, d, ell, slope, decimal_digits)
        delta = d - slope * (tau - turning_time)
        second = segment_propagator(d, float(delta), ell, -slope, decimal_digits)
        state = second * first * initial_state
    return complex(state[0]), complex(state[1])


def solve_direct(
    ell: float,
    d: float,
    slope: float,
    n_times: int = 6001,
) -> dict[str, np.ndarray]:
    turning_time = 2.0 * d / slope
    final_time = 2.0 * turning_time
    tau = np.linspace(0.0, final_time, n_times)

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        c = state[0] + 1j * state[1]
        dc = state[2] + 1j * state[3]
        delta = float(triangular_delta(time, d, slope))
        ddc = -(ell + 1j * delta) * dc - 0.5 * ell * c
        return np.array([dc.real, dc.imag, ddc.real, ddc.imag])

    solution = solve_ivp(
        rhs,
        (0.0, final_time),
        np.array([1.0, 0.0, 0.0, 0.0]),
        t_eval=tau,
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
        max_step=0.03,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    c = solution.y[0] + 1j * solution.y[1]
    dc = solution.y[2] + 1j * solution.y[3]
    population = np.abs(c) ** 2
    population_rate = 2.0 * np.real(np.conjugate(c) * dc)
    return {
        "tau": tau,
        "delta": triangular_delta(tau, d, slope),
        "c": c,
        "dc": dc,
        "population": population,
        "population_rate": population_rate,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ell = 0.625
    d = 0.625
    slope = 0.3
    Omega = 1.25
    direct = solve_direct(ell, d, slope)
    tau = direct["tau"]
    indices = np.unique(np.linspace(0, tau.size - 1, 121, dtype=int))
    exact_c = []
    exact_dc = []
    rows = []
    for index in indices:
        c_value, dc_value = analytic_state(float(tau[index]), ell, d, slope)
        exact_c.append(c_value)
        exact_dc.append(dc_value)
        rows.append(
            [
                tau[index],
                abs(c_value - direct["c"][index]),
                abs(dc_value - direct["dc"][index]),
            ]
        )
    validation = np.asarray(rows)
    np.savetxt(
        DATA_DIR / "triangular_transfer_validation.csv",
        validation,
        delimiter=",",
        header="tau,abs_c_transfer_minus_ode,abs_dc_transfer_minus_ode",
        comments="",
    )

    population = direct["population"]
    population_rate = direct["population_rate"]
    omega = Omega - direct["delta"]
    energy_backflow = np.concatenate(
        ([0.0], cumulative_trapezoid(np.maximum(omega * population_rate, 0.0), tau))
    )
    safe_p = np.clip(population, 1e-14, 1.0 - 1e-14)
    mutual_information = -2.0 * (
        safe_p * np.log(safe_p) + (1.0 - safe_p) * np.log(1.0 - safe_p)
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
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    axes[0, 0].plot(tau, direct["delta"], color="#7570b3", lw=2.0)
    axes[0, 0].axhline(0.0, color="0.25", lw=0.8)
    axes[0, 0].set(xlabel=r"$\tau$", ylabel=r"$\bar\delta(\tau)$", title="Triangular double passage")

    axes[0, 1].plot(tau, population, color="#1b9e77", lw=1.8, label="direct ODE")
    axes[0, 1].plot(tau[indices], np.abs(exact_c) ** 2, "o", ms=3.0, mfc="none", mec="black", label="transfer matrix")
    axes[0, 1].set(xlabel=r"$\tau$", ylabel=r"$P(\tau)$", title="Exact population")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(tau, population_rate, color="#d95f02", lw=1.6)
    axes[1, 0].axhline(0.0, color="0.25", lw=0.8)
    axes[1, 0].fill_between(tau, 0.0, population_rate, where=population_rate > 0, color="#d95f02", alpha=0.2)
    axes[1, 0].set(xlabel=r"$\tau$", ylabel=r"$\mathrm{d}P/\mathrm{d}\tau$", title="Memory after two crossings")

    axes[1, 1].plot(tau, mutual_information, color="#7570b3", lw=1.7, label=r"$\mathcal{I}_{S:B}$")
    right = axes[1, 1].twinx()
    right.plot(tau, energy_backflow, color="#e7298a", lw=1.4, ls="--", label=r"$\mathcal{E}_{\rm back}/\gamma_0$")
    axes[1, 1].set(xlabel=r"$\tau$", ylabel="mutual information", title="Correlations and accumulated backflow")
    right.set_ylabel("energy backflow")
    handles = axes[1, 1].lines + right.lines
    axes[1, 1].legend(handles, [line.get_label() for line in handles], frameon=False)

    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)
    fig.savefig(FIGURE_DIR / "triangular_double_passage.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "triangular_double_passage.pdf", bbox_inches="tight")
    plt.close(fig)

    positive_intervals = np.flatnonzero(population_rate > 0)
    print(f"max |c_transfer-c_ode| = {validation[:, 1].max():.12e}")
    print(f"max |dc_transfer-dc_ode| = {validation[:, 2].max():.12e}")
    print(f"P_final = {population[-1]:.12e}")
    print(f"N_P = {np.trapezoid(np.maximum(population_rate, 0.0), tau):.12e}")
    print(f"E_back/gamma0 = {energy_backflow[-1]:.12e}")
    print(f"positive-rate samples = {positive_intervals.size}")


if __name__ == "__main__":
    main()
