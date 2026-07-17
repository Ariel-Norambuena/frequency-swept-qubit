"""Global audit for the slow-sweep non-Markovian boundary.

The local first-tangency theorem is supplemented by a late-time Riccati tube.
At the matching detuning delta_m=a^(1/3), the exact logarithmic derivative is
compared with the attracting frozen root q_s.  A scalar differential
inequality then certifies Re q<0 for all later times when its tube conditions
are satisfied.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from universality_theorem_analysis import (
    detuning,
    solve_profile_boundary,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
FIGURE_DIR = ROOT / "output" / "figures"

PROFILES = ("tanh", "arctan", "sqrt", "exponential")
SATURATION = {
    "tanh": 1.0,
    "arctan": np.pi / 2.0,
    "sqrt": 1.0,
    "exponential": 1.0,
}


def slow_root(ell: float, delta: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Attracting frozen Riccati root and spectral gap."""
    delta_array = np.asarray(delta, dtype=float)
    gap = np.sqrt((ell + 1j * delta_array) ** 2 - 2.0 * ell)
    gap = np.where(np.real(gap) < 0.0, -gap, gap)
    q_s = -0.5 * (ell + 1j * delta_array) + 0.5 * gap
    return q_s, gap


def slow_root_derivative(ell: float, delta: np.ndarray) -> np.ndarray:
    q_s, gap = slow_root(ell, delta)
    del q_s
    return -0.5j + 0.5j * (ell + 1j * delta) / gap


def q_rhs(tau: float, state: np.ndarray, ell: float, a: float, d: float, profile: str) -> np.ndarray:
    q = state[0] + 1j * state[1]
    delta, _ = detuning(tau, a, d, profile)
    qp = -q * q - (ell + 1j * delta) * q - 0.5 * ell
    return np.array([qp.real, qp.imag])


def profile_value(x: float, profile: str) -> float:
    if profile == "tanh":
        return float(np.tanh(x))
    if profile == "arctan":
        return float(np.arctan(x))
    if profile == "sqrt":
        return float(x / np.sqrt(1.0 + x * x))
    if profile == "exponential":
        return float(1.0 - np.exp(-x))
    raise ValueError(profile)


def profile_derivative(x: float, profile: str) -> float:
    if profile == "tanh":
        return float(1.0 / np.cosh(x) ** 2)
    if profile == "arctan":
        return float(1.0 / (1.0 + x * x))
    if profile == "sqrt":
        return float((1.0 + x * x) ** -1.5)
    if profile == "exponential":
        return float(np.exp(-x))
    raise ValueError(profile)


def matching_time(a: float, d: float, profile: str) -> tuple[float, float, float]:
    delta_match = a ** (1.0 / 3.0)
    target = delta_match / d
    saturation = SATURATION[profile]
    if target >= saturation:
        raise ValueError("matching detuning exceeds profile saturation")
    upper = 1.0
    while profile_value(upper, profile) <= target:
        upper *= 2.0
    x_match = brentq(lambda x: profile_value(x, profile) - target, 0.0, upper)
    return d * x_match / a, x_match, delta_match


def trajectory_to_match(
    ell: float,
    a: float,
    d: float,
    profile: str,
    tau_match: float,
    tau_tangent: float,
) -> tuple[np.ndarray, np.ndarray]:
    early_end = min(tau_match, tau_tangent + 12.0)
    early = np.linspace(0.0, early_end, 2400)
    if tau_match > early_end:
        late = np.geomspace(max(early_end, 1e-8), tau_match, 1800)
        times = np.unique(np.concatenate([early, late]))
    else:
        times = early
    solution = solve_ivp(
        q_rhs,
        (0.0, tau_match),
        np.zeros(2),
        t_eval=times,
        args=(ell, a, d, profile),
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y[0] + 1j * solution.y[1]


def late_tube(
    ell: float,
    a: float,
    d: float,
    profile: str,
    x_match: float,
    delta_match: float,
    q_match: complex,
) -> dict[str, float | bool]:
    delta_final = d * SATURATION[profile]
    # The frozen-root extrema are evaluated over the complete remaining
    # detuning interval, including the asymptotic saturation value.
    delta_grid = np.linspace(delta_match, delta_final, 8000)
    q_s, gap = slow_root(ell, delta_grid)
    q_s_delta = slow_root_derivative(ell, delta_grid)
    u_min = float(np.min(np.real(gap)))
    margin_min = float(np.min(-np.real(q_s)))
    delta_prime_max = a * profile_derivative(x_match, profile)
    forcing_max = float(np.max(np.abs(q_s_delta)) * delta_prime_max)
    q_s_match = complex(slow_root(ell, delta_match)[0])
    entry_error = abs(q_match - q_s_match)
    discriminant = u_min * u_min - 4.0 * forcing_max
    if discriminant > 0.0:
        root_disc = np.sqrt(discriminant)
        radius_small = 0.5 * (u_min - root_disc)
        radius_large = 0.5 * (u_min + root_disc)
        certified_radius = max(entry_error, radius_small)
        certified = certified_radius < min(margin_min, radius_large)
    else:
        radius_small = np.nan
        radius_large = np.nan
        certified_radius = np.inf
        certified = False
    return {
        "delta_final": delta_final,
        "u_min": u_min,
        "margin_min": margin_min,
        "delta_prime_max": delta_prime_max,
        "forcing_max": forcing_max,
        "entry_error": entry_error,
        "entry_error_scaled": entry_error / a ** (2.0 / 3.0),
        "discriminant": discriminant,
        "radius_small": radius_small,
        "radius_large": radius_large,
        "certified_radius": certified_radius,
        "certified": certified,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    d = 0.625
    a_values = np.geomspace(1e-2, 1e-5, 10)
    profile_guesses: dict[str, tuple[float, float]] = {}
    rows: list[list[float]] = []

    for a in a_values:
        for profile_index, profile in enumerate(PROFILES):
            default_guess = (np.sqrt(2.0) * np.pi, np.sqrt(np.sqrt(2.0) * np.pi))
            ell, tau_tangent = solve_profile_boundary(
                float(a), d, profile, profile_guesses.get(profile, default_guess)
            )
            profile_guesses[profile] = (
                (2.0 - ell) / np.sqrt(a),
                tau_tangent * a**0.25,
            )
            tau_match, x_match, delta_match = matching_time(a, d, profile)
            times, q = trajectory_to_match(
                ell, a, d, profile, tau_match, tau_tangent
            )
            tube = late_tube(
                ell, a, d, profile, x_match, delta_match, complex(q[-1])
            )
            remote_mask = times >= tau_tangent + 5.0
            remote_max = float(np.max(np.real(q[remote_mask]))) if np.any(remote_mask) else np.nan
            remote_time = (
                float(times[remote_mask][np.argmax(np.real(q[remote_mask]))])
                if np.any(remote_mask)
                else np.nan
            )
            rows.append(
                [
                    a,
                    profile_index,
                    ell,
                    tau_tangent,
                    tau_match,
                    x_match,
                    delta_match,
                    remote_max,
                    remote_time,
                    float(tube["entry_error"]),
                    float(tube["entry_error_scaled"]),
                    float(tube["u_min"]),
                    float(tube["margin_min"]),
                    float(tube["forcing_max"]),
                    float(tube["discriminant"]),
                    float(tube["certified_radius"]),
                    float(bool(tube["certified"])),
                ]
            )
            print(
                f"a={a:.2e} {profile:11s} remote max={remote_max:+.3e} "
                f"entry/a^(2/3)={tube['entry_error_scaled']:.3e} "
                f"tube/margin={tube['certified_radius']/tube['margin_min']:.3e} "
                f"certified={tube['certified']}"
            )

    data = np.asarray(rows)
    np.savetxt(
        DATA_DIR / "global_boundary_audit.csv",
        data,
        delimiter=",",
        header=(
            "a,profile_index,ell_critical,tau_tangent,tau_match,x_match,delta_match,"
            "max_real_q_after_local_layer,time_of_remote_max,entry_error,"
            "entry_error_over_a_2_3,u_min,slow_branch_margin_min,forcing_max,"
            "tube_discriminant,certified_radius,tube_certified"
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
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    colors = ["#d95f02", "#7570b3", "#e7298a", "#1f78b4"]
    for index, profile in enumerate(PROFILES):
        mask = data[:, 1] == index
        subset = data[mask]
        axes[0].loglog(
            subset[:, 0], subset[:, 9], "o-", ms=3.2, lw=1.0,
            color=colors[index], label=profile,
        )
        axes[1].semilogx(
            subset[:, 0], -subset[:, 7], "o-", ms=3.2, lw=1.0,
            color=colors[index], label=profile,
        )
        axes[2].loglog(
            subset[:, 0], subset[:, 15] / subset[:, 12], "o-", ms=3.2, lw=1.0,
            color=colors[index], label=profile,
        )
    guide_a = np.geomspace(a_values.min(), a_values.max(), 100)
    axes[0].loglog(
        guide_a, 2.0 * guide_a ** (2.0 / 3.0), color="black", ls="--", lw=1.0,
        label=r"guide $a^{2/3}$",
    )
    axes[0].set(
        xlabel=r"$a$", ylabel=r"$|q(\tau_m)-q_s(\delta_m)|$",
        title="Matching to the slow branch",
    )
    axes[1].set(
        xlabel=r"$a$", ylabel=r"$-\max_{\tau\geq\tau_*+5}\Re q$",
        title="Intermediate negative margin",
    )
    axes[2].axhline(1.0, color="black", ls="--", lw=1.0)
    axes[2].set(
        xlabel=r"$a$", ylabel="certified tube / slow margin",
        title="All-time late tube",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2, which="both")
    axes[0].legend(frameon=False)
    fig.savefig(FIGURE_DIR / "global_boundary_audit.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "global_boundary_audit.pdf", bbox_inches="tight")
    plt.close(fig)

    print(
        f"Certified rows: {int(np.sum(data[:, 16]))}/{data.shape[0]}; "
        f"largest remote Re q={np.max(data[:, 7]):+.3e}"
    )


if __name__ == "__main__":
    main()
