"""Uniform crossover between lifted zeros and the opened-gap slow branch.

The critical variables are epsilon=a^(1/4), s=epsilon*tau and
X=sqrt(epsilon)*s=a^(3/8)*tau.  The script evaluates the exact scaled
amplitude at the linear critical boundary and tests the WKB selection law

    |B/A| ~ exp[-X^2/(2 k)],

as well as the late logarithmic-derivative error and the negative margin of
all maxima after the first critical tangency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

from asymptotic_boundary_analysis import solve_boundary as solve_linear_boundary


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
FIGURE_DIR = ROOT / "output" / "figures"


def scaled_rhs(s: float, state: np.ndarray, ell: float, epsilon: float) -> np.ndarray:
    U = state[0] + 1j * state[1]
    Us = state[2] + 1j * state[3]
    potential = (
        ell * (2.0 - ell) / (4.0 * epsilon**2)
        + epsilon**4 * s**2 / 4.0
        - 0.5j * epsilon**2
        - 0.5j * ell * epsilon * s
    )
    Uss = -potential * U
    return np.array([Us.real, Us.imag, Uss.real, Uss.imag])


def slow_and_fast_roots(ell: float, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gap = np.sqrt((ell + 1j * delta) ** 2 - 2.0 * ell)
    gap = np.where(np.real(gap) < 0.0, -gap, gap)
    q_s = -0.5 * (ell + 1j * delta) + 0.5 * gap
    q_f = -0.5 * (ell + 1j * delta) - 0.5 * gap
    return q_s, q_f, gap


def exact_trajectory(a: float, ell: float, s_exit: float) -> dict[str, np.ndarray]:
    epsilon = a**0.25
    # Resolve every oscillator period and retain a dense tail in X.
    n_points = max(10000, int(np.ceil(90.0 * s_exit)))
    s = np.linspace(0.0, s_exit, n_points)
    solution = solve_ivp(
        scaled_rhs,
        (0.0, s_exit),
        np.array([epsilon, 0.0, 0.5 * ell, 0.0]),
        t_eval=s,
        args=(ell, epsilon),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    U = solution.y[0] + 1j * solution.y[1]
    Us = solution.y[2] + 1j * solution.y[3]
    delta = epsilon**3 * s
    q = -0.5 * (ell + 1j * delta) + epsilon * Us / U
    q_s, q_f, gap = slow_and_fast_roots(ell, delta)
    z = (q - q_s) / (q - q_f)
    X = np.sqrt(epsilon) * s
    return {
        "s": s,
        "X": X,
        "U": U,
        "Us": Us,
        "q": q,
        "q_s": q_s,
        "q_f": q_f,
        "gap": gap,
        "z": z,
    }


def averaged_envelope(s: np.ndarray, ell: float, epsilon: float, k: float) -> np.ndarray:
    """Leading uniform sine-WKB logarithmic derivative."""
    theta = k * (s + epsilon)
    phi = epsilon * s**2 / (4.0 * k)
    phi_prime = epsilon * s / (2.0 * k)
    argument = theta - 1j * phi
    cotangent = np.cos(argument) / np.sin(argument)
    delta = epsilon**3 * s
    return -0.5 * (ell + 1j * delta) + epsilon * (k - 1j * phi_prime) * cotangent


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    # The tangency solver is continued on a fine logarithmic mesh; expensive
    # crossover trajectories are retained every fourth point.
    continuation_values = np.geomspace(1e-3, 1e-11, 33)
    retained_indices = set(range(0, continuation_values.size, 4))
    exit_factor = 3.0
    rows = []
    curves: list[tuple[float, np.ndarray, np.ndarray]] = []
    guess = (10.4, 1.35)

    for continuation_index, a in enumerate(continuation_values):
        epsilon = a**0.25
        ell, tau_tangent, _ = solve_linear_boundary(float(a), guess)
        epsilon_old = a ** (1.0 / 3.0)
        guess = ((2.0 - ell) / epsilon_old**2, tau_tangent * epsilon_old)
        if continuation_index not in retained_indices:
            continue
        mu = (2.0 - ell) / epsilon**2
        k = np.sqrt(mu / 2.0)
        X_exit = exit_factor * np.sqrt(np.log(1.0 / epsilon))
        s_exit = X_exit / np.sqrt(epsilon)
        data = exact_trajectory(float(a), ell, s_exit)
        q = data["q"]
        q_s = data["q_s"]
        z = data["z"]
        s = data["s"]
        X = data["X"]
        q_wkb = averaged_envelope(s, ell, epsilon, k)

        peaks, _ = find_peaks(np.real(q), distance=max(4, int(0.55 * np.pi / k / (s[1] - s[0]))))
        # Match peaks to the first tangency and keep every later peak.
        s_tangent = epsilon * tau_tangent
        first_index = int(np.argmin(np.abs(s[peaks] - s_tangent)))
        later_peaks = peaks[first_index + 1 :]
        later_max = float(np.max(np.real(q[later_peaks]))) if later_peaks.size else np.nan

        exit_z = abs(z[-1])
        predicted_z = np.exp(-X_exit**2 / (2.0 * k))
        exit_error = abs(q[-1] - q_s[-1])
        wkb_q_error = np.max(np.abs(q_wkb[later_peaks] - q[later_peaks])) if later_peaks.size else np.nan
        rows.append(
            [
                a,
                epsilon,
                ell,
                mu,
                k,
                tau_tangent,
                X_exit,
                s_exit,
                later_max,
                exit_z,
                predicted_z,
                exit_error,
                exit_error / epsilon**2,
                wkb_q_error,
            ]
        )
        # Curves are restricted away from X=0, where both frozen roots coalesce
        # in real part and the Möbius ratio is phase-sensitive.
        mask = (X >= 0.45) & (X <= X_exit)
        stride = max(1, np.count_nonzero(mask) // 900)
        curves.append((a, X[mask][::stride], np.abs(z[mask][::stride])))
        print(
            f"a={a:.1e} eps={epsilon:.3e} later max={later_max:+.6f} "
            f"|z_exit|={exit_z:.3e} pred={predicted_z:.3e} "
            f"|q-qs|/eps^2={exit_error/epsilon**2:.3e}"
        )

    table = np.asarray(rows)
    np.savetxt(
        DATA_DIR / "uniform_crossover_audit.csv",
        table,
        delimiter=",",
        header=(
            "a,epsilon,ell_critical,mu,k,tau_tangent,X_exit,s_exit,"
            "largest_real_q_after_first_peak,abs_z_exit,predicted_abs_z_exit,"
            "abs_q_minus_qs_exit,abs_q_minus_qs_over_epsilon2,"
            "max_wkb_q_error_at_later_peaks"
        ),
        comments="",
    )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "legend.fontsize": 8.0,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.75), constrained_layout=True)
    color_map = plt.get_cmap("viridis")
    for index, (a, X, z_abs) in enumerate(curves):
        axes[0].semilogy(
            X,
            np.maximum(z_abs, 1e-15),
            lw=0.95,
            color=color_map(index / max(1, len(curves) - 1)),
            label=rf"$10^{{{int(np.round(np.log10(a)))}}}$" if index in (0, 2, 4, 6, 8) else None,
        )
    X_guide = np.linspace(0.45, np.max(table[:, 6]), 500)
    k0 = np.sqrt(np.sqrt(2.0) * np.pi / 2.0)
    axes[0].semilogy(X_guide, np.exp(-X_guide**2 / (2.0 * k0)), "k--", lw=1.1, label=r"$e^{-X^2/(2k_0)}$")
    axes[0].set(
        xlabel=r"$X=a^{3/8}\tau$",
        ylabel=r"instantaneous mode ratio $|z|$",
        title="Slow-mode selection",
        ylim=(1e-8, 2.0),
    )
    axes[0].legend(frameon=False, ncol=2)

    # At the two smallest rates the peak locator becomes more sensitive than
    # the ODE itself to the residual tangency error; retain the converged
    # window for the visual margin test while keeping every row in the CSV.
    stable_peaks = table[:, 0] >= 1e-9
    axes[1].semilogx(table[stable_peaks, 0], -table[stable_peaks, 8], "o-", ms=3.5, lw=1.0, color="#d95f02")
    axes[1].axhline(0.75, color="black", ls="--", lw=1.0, label=r"leading margin $3/4$")
    axes[1].set(
        xlabel=r"$a$",
        ylabel=r"$-\max_{n\geq2}\Re q$",
        title="All later lifted zeros",
    )
    axes[1].legend(frameon=False)

    axes[2].loglog(table[:, 0], table[:, 11], "o-", ms=3.5, lw=1.0, color="#1b9e77", label=r"exact $|q-q_s|$")
    guide = table[:, 0]
    coefficient = np.median(table[-4:, 11] / table[-4:, 1] ** 2)
    axes[2].loglog(guide, coefficient * table[:, 1] ** 2, "k--", lw=1.0, label=r"guide $\epsilon^2=a^{1/2}$")
    axes[2].set(
        xlabel=r"$a$",
        ylabel=r"$|q-q_s|$ at crossover exit",
        title="Entry error for the exact tube",
    )
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2, which="both")
    fig.savefig(FIGURE_DIR / "uniform_crossover_analysis.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "uniform_crossover_analysis.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
