"""Discrete-star reconstruction of the exact Lorentzian reservoir.

The script compares the exact pseudomode amplitude with finite star baths
obtained from midpoint quadrature of the continuum spectral density.  It also
illustrates the Lorentzian deconvolution needed when each discrete delta peak
is displayed with a finite Lorentzian broadening.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from publication_style import close_axes, configure_publication_style


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "output" / "figures"
DATA_DIR = ROOT / "output" / "data"


def dimensionless_density(x: np.ndarray, ell: float) -> np.ndarray:
    return ell**2 / (2.0 * np.pi * (x**2 + ell**2))


def normalized_lorentzian(x: np.ndarray, width: float) -> np.ndarray:
    return width / (np.pi * (x**2 + width**2))


def exact_pseudomode(
    ell: float, a: float, tau_final: float, n_times: int = 801
) -> tuple[np.ndarray, np.ndarray]:
    tau = np.linspace(0.0, tau_final, n_times)
    coupling = np.sqrt(ell / 2.0)

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        c, mode = state
        return np.array(
            [
                -1j * coupling * mode,
                -(ell + 1j * a * time) * mode - 1j * coupling * c,
            ],
            dtype=complex,
        )

    solution = solve_ivp(
        rhs,
        (0.0, tau_final),
        np.array([1.0 + 0.0j, 0.0j]),
        t_eval=tau,
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return tau, solution.y[0]


def discrete_star(
    ell: float,
    a: float,
    tau_final: float,
    half_window: float,
    n_modes: int,
    n_times: int = 801,
) -> dict[str, np.ndarray | float]:
    delta_x = 2.0 * half_window / n_modes
    x = -half_window + (np.arange(n_modes) + 0.5) * delta_x
    cell_left = x - 0.5 * delta_x
    cell_right = x + 0.5 * delta_x
    # Exact Lorentzian mass in each frequency cell.  This removes midpoint
    # quadrature error and leaves only the controlled finite-window and
    # recurrence errors discussed in the Supplemental Material.
    weights = ell / (2.0 * np.pi) * (
        np.arctan(cell_right / ell) - np.arctan(cell_left / ell)
    )
    couplings = np.sqrt(weights)
    tau = np.linspace(0.0, tau_final, n_times)

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        c = state[0]
        beta = state[1:]
        phase = np.exp(-1j * (x * time + 0.5 * a * time**2))
        dc = -1j * np.dot(couplings, beta * phase)
        dbeta = -1j * couplings * c * np.conjugate(phase)
        return np.concatenate(([dc], dbeta))

    initial = np.zeros(n_modes + 1, dtype=complex)
    initial[0] = 1.0
    solution = solve_ivp(
        rhs,
        (0.0, tau_final),
        initial,
        t_eval=tau,
        method="DOP853",
        rtol=3e-10,
        atol=3e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    c = solution.y[0]
    beta = solution.y[1:]
    norm = np.abs(c) ** 2 + np.sum(np.abs(beta) ** 2, axis=0)
    return {
        "tau": tau,
        "x": x,
        "half_window": float(half_window),
        "delta_x": float(delta_x),
        "couplings": couplings,
        "c": c,
        "beta": beta,
        "norm": norm,
    }


def broadened_reconstruction(
    ell: float, half_window: float, n_modes: int, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_x = 2.0 * half_window / n_modes
    x = -half_window + (np.arange(n_modes) + 0.5) * delta_x
    envelope = 0.5 * ell * normalized_lorentzian(x, ell - sigma)
    weights = envelope * delta_x
    plot_x = np.linspace(-4.0 * ell, 4.0 * ell, 1001)
    reconstructed = np.sum(
        weights[:, None] * normalized_lorentzian(plot_x[None, :] - x[:, None], sigma),
        axis=0,
    )
    return plot_x, reconstructed, weights


def main() -> None:
    configure_publication_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ell = 5.0 / 8.0
    a = 5.0 / 32.0
    tau_final = 8.0
    configurations = ((200, 8.0), (400, 14.0), (800, 20.0))
    mode_counts = tuple(count for count, _ in configurations)

    tau, c_exact = exact_pseudomode(ell, a, tau_final)
    star_data = {
        count: discrete_star(ell, a, tau_final, half_window, count)
        for count, half_window in configurations
    }

    finest = star_data[mode_counts[-1]]
    delta_x = float(finest["delta_x"])
    sigma = 2.0 * delta_x
    plot_x, reconstructed, broadened_weights = broadened_reconstruction(
        ell, float(finest["half_window"]), mode_counts[-1], sigma
    )
    target = dimensionless_density(plot_x, ell)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.15), constrained_layout=True)

    axes[0].plot(plot_x, target, color="black", lw=2.5, label=r"target $J_L/\gamma_0$")
    axes[0].plot(
        plot_x,
        reconstructed,
        color="#d95f02",
        lw=2.2,
        ls="--",
        label=rf"broadened star, $\sigma={sigma:.3f}$",
    )
    axes[0].set(
        xlabel=r"$x=(\omega-\omega_b)/\gamma_0$",
        ylabel=r"$J(x)/\gamma_0$",
        title="Spectral reconstruction",
    )
    axes[0].legend(frameon=True)

    axes[1].plot(tau, np.abs(c_exact) ** 2, color="black", lw=2.6, label="exact pole")
    colors = ("#1b9e77", "#7570b3", "#e7298a")
    for count, color in zip(mode_counts, colors, strict=True):
        data = star_data[count]
        axes[1].plot(
            data["tau"],
            np.abs(data["c"]) ** 2,
            color=color,
            lw=1.9,
            ls="--" if count < mode_counts[-1] else "-.",
            label=rf"star $N={count}$, $X={data['half_window']:g}$",
        )
    axes[1].set(
        xlabel=r"$\tau=\gamma_0t$",
        ylabel=r"$P(\tau)=|c(\tau)|^2$",
        title="Continuum convergence",
    )
    axes[1].legend(frameon=True)

    beta_final = np.asarray(finest["beta"])[:, -1]
    occupation_density = np.abs(beta_final) ** 2 / delta_x
    axes[2].plot(
        finest["x"],
        occupation_density,
        color="#0072b2",
        lw=2.2,
        label=r"$|\beta_k(\tau_{\max})|^2/\Delta x$",
    )
    axes[2].set_xlim(-4.0 * ell, 4.0 * ell)
    axes[2].set(
        xlabel=r"$x_k=(\omega_k-\omega_b)/\gamma_0$",
        ylabel="mode-occupation density",
        title="Resolved reservoir amplitudes",
    )
    axes[2].legend(frameon=True)

    close_axes(axes)
    fig.savefig(FIGURE_DIR / "discrete_bath_reconstruction.pdf")
    fig.savefig(FIGURE_DIR / "discrete_bath_reconstruction.png")
    plt.close(fig)

    rows = []
    for count in mode_counts:
        data = star_data[count]
        population_error = np.max(np.abs(np.abs(data["c"]) ** 2 - np.abs(c_exact) ** 2))
        norm_error = np.max(np.abs(np.asarray(data["norm"]) - 1.0))
        recurrence_time = 2.0 * np.pi / float(data["delta_x"])
        rows.append(
            [
                count,
                data["half_window"],
                data["delta_x"],
                recurrence_time,
                population_error,
                norm_error,
            ]
        )
        print(
            f"N={count}: X={data['half_window']:.6g}, dx={data['delta_x']:.6g}, "
            f"T_rec={recurrence_time:.6g}, "
            f"max population error={population_error:.3e}, norm error={norm_error:.3e}"
        )

    np.savetxt(
        DATA_DIR / "discrete_bath_convergence.csv",
        np.asarray(rows, dtype=float),
        delimiter=",",
        header=(
            "N_modes,half_window,delta_x,recurrence_time,"
            "max_population_error,max_norm_error"
        ),
        comments="",
    )
    np.savetxt(
        DATA_DIR / "discrete_bath_final_modes.csv",
        np.column_stack(
            [
                finest["x"],
                finest["couplings"],
                np.abs(beta_final) ** 2,
                occupation_density,
                broadened_weights,
            ]
        ),
        delimiter=",",
        header=(
            "x_k,direct_quadrature_coupling,final_mode_population,"
            "final_mode_population_density,broadened_reconstruction_weight"
        ),
        comments="",
    )


if __name__ == "__main__":
    main()
