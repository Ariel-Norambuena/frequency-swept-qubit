"""Bare-energy return and correlation generation for the exact swept TLS model.

Dimensionless variables:
    tau          = gamma0 * t
    ell          = lambda / gamma0
    a            = alpha / gamma0**2
    Omega_delta  = Delta_f / gamma0 (sweep range)
    Omega_b      = omega_b / gamma0 (independent carrier)

The exact amplitude equation is
    c''(tau) + (ell + i*a*tau)c'(tau) + ell*c(tau)/2 = 0.

The script generates the figures and phase-grid data used in the analytical
note.  No Markov or weak-coupling approximation is made in the evolution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

from publication_style import close_axes, configure_publication_style


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "output" / "figures"
DATA_DIR = ROOT / "output" / "data"


def rhs(tau: float, y: np.ndarray, ell: float, a: float) -> np.ndarray:
    """Real-valued representation of the exact complex amplitude equation."""
    c = y[0] + 1j * y[1]
    v = y[2] + 1j * y[3]
    acceleration = -(ell + 1j * a * tau) * v - 0.5 * ell * c
    return np.array([v.real, v.imag, acceleration.real, acceleration.imag])


def solve_trajectory(
    ell: float,
    a: float,
    Omega_delta: float,
    Omega_b: float = 20.0,
    n_times: int = 3001,
) -> dict[str, np.ndarray | float]:
    """Solve a finite detuning sweep while keeping the laboratory gap positive."""
    if ell <= 0 or a <= 0 or Omega_delta <= 0 or Omega_b <= Omega_delta:
        raise ValueError("Require ell,a,Omega_delta>0 and Omega_b>Omega_delta")

    tau_final = Omega_delta / a
    tau = np.linspace(0.0, tau_final, n_times)
    solution = solve_ivp(
        rhs,
        (0.0, tau_final),
        np.array([1.0, 0.0, 0.0, 0.0]),
        args=(ell, a),
        t_eval=tau,
        method="DOP853",
        rtol=2e-10,
        atol=2e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    c = solution.y[0] + 1j * solution.y[1]
    v = solution.y[2] + 1j * solution.y[3]
    population = np.abs(c) ** 2
    population_rate = 2.0 * np.real(np.conjugate(c) * v)
    omega = Omega_b - a * tau
    heat_power = omega * population_rate
    work_power = -a * population
    system_energy = omega * population
    system_energy_rate = work_power + heat_power

    q = np.full_like(c, np.nan + 1j * np.nan)
    regular_c = np.abs(c) > 1e-12
    q[regular_c] = v[regular_c] / c[regular_c]
    lamb_shift = -np.imag(q)
    q_rate = np.full_like(q, np.nan + 1j * np.nan)
    q_rate[regular_c] = (
        -q[regular_c] ** 2
        - (ell + 1j * a * tau[regular_c]) * q[regular_c]
        - 0.5 * ell
    )
    lamb_shift_rate = -np.imag(q_rate)
    interaction_energy = 2.0 * population * lamb_shift
    interaction_energy_rate = 2.0 * (
        population_rate * lamb_shift + population * lamb_shift_rate
    )
    reservoir_to_system_power = heat_power + interaction_energy_rate
    interaction_to_system_power = -interaction_energy_rate

    gamma = np.full_like(population, np.nan)
    regular = population > 1e-12
    gamma[regular] = -population_rate[regular] / population[regular]

    safe_population = np.clip(population, 1e-14, 1.0 - 1e-14)
    binary_entropy = -safe_population * np.log(safe_population)
    binary_entropy -= (1.0 - safe_population) * np.log(1.0 - safe_population)
    mutual_information = 2.0 * binary_entropy
    mutual_information_rate = 2.0 * population_rate * np.log(
        (1.0 - safe_population) / safe_population
    )

    backflow_power = np.maximum(heat_power, 0.0)
    probability_backflow_rate = np.maximum(population_rate, 0.0)
    cumulative_energy_backflow = np.concatenate(
        ([0.0], cumulative_trapezoid(backflow_power, tau))
    )
    cumulative_probability_backflow = np.concatenate(
        ([0.0], cumulative_trapezoid(probability_backflow_rate, tau))
    )

    return {
        "tau": tau,
        "c": c,
        "v": v,
        "population": population,
        "population_rate": population_rate,
        "omega": omega,
        "heat_power": heat_power,
        "work_power": work_power,
        "system_energy": system_energy,
        "system_energy_rate": system_energy_rate,
        "interaction_energy": interaction_energy,
        "interaction_energy_rate": interaction_energy_rate,
        "reservoir_to_system_power": reservoir_to_system_power,
        "interaction_to_system_power": interaction_to_system_power,
        "gamma": gamma,
        "mutual_information": mutual_information,
        "mutual_information_rate": mutual_information_rate,
        "energy_backflow_cumulative": cumulative_energy_backflow,
        "probability_backflow_cumulative": cumulative_probability_backflow,
        "energy_backflow": float(cumulative_energy_backflow[-1]),
        "probability_backflow": float(cumulative_probability_backflow[-1]),
        "final_population": float(population[-1]),
    }


def _rk4_step(
    tau: float,
    c: np.ndarray,
    v: np.ndarray,
    dt: float,
    ell: np.ndarray,
    a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized fixed-step RK4 update for a full row of ell values."""

    def acceleration(time: float, amplitude: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        return -(ell + 1j * a * time) * velocity - 0.5 * ell * amplitude

    k1c = v
    k1v = acceleration(tau, c, v)
    k2c = v + 0.5 * dt * k1v
    k2v = acceleration(tau + 0.5 * dt, c + 0.5 * dt * k1c, v + 0.5 * dt * k1v)
    k3c = v + 0.5 * dt * k2v
    k3v = acceleration(tau + 0.5 * dt, c + 0.5 * dt * k2c, v + 0.5 * dt * k2v)
    k4c = v + dt * k3v
    k4v = acceleration(tau + dt, c + dt * k3c, v + dt * k3v)

    c_next = c + dt * (k1c + 2.0 * k2c + 2.0 * k3c + k4c) / 6.0
    v_next = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
    return c_next, v_next


def phase_grid(
    ell_values: np.ndarray,
    a_values: np.ndarray,
    Omega_delta: float,
    Omega_b: float = 20.0,
    target_dt: float = 0.012,
) -> dict[str, np.ndarray]:
    """Compute exact backflow metrics over the (a, ell) plane."""
    shape = (a_values.size, ell_values.size)
    energy_backflow = np.zeros(shape)
    probability_backflow = np.zeros(shape)
    final_population = np.zeros(shape)
    minimum_population_rate = np.zeros(shape)
    maximum_population_rate = np.zeros(shape)

    ell = ell_values.astype(float)
    for row, a in enumerate(a_values):
        tau_final = Omega_delta / a
        n_steps = max(700, int(np.ceil(tau_final / target_dt)))
        dt = tau_final / n_steps
        c = np.ones_like(ell, dtype=complex)
        v = np.zeros_like(ell, dtype=complex)
        tau = 0.0

        population = np.abs(c) ** 2
        population_rate = 2.0 * np.real(np.conjugate(c) * v)
        omega = Omega_b
        heat_power = omega * population_rate
        min_rate = population_rate.copy()
        max_rate = population_rate.copy()
        eback = np.zeros_like(ell)
        pback = np.zeros_like(ell)

        for _ in range(n_steps):
            c_next, v_next = _rk4_step(tau, c, v, dt, ell, float(a))
            tau_next = tau + dt
            population_next = np.abs(c_next) ** 2
            rate_next = 2.0 * np.real(np.conjugate(c_next) * v_next)
            omega_next = Omega_b - a * tau_next
            heat_next = omega_next * rate_next

            eback += 0.5 * dt * (
                np.maximum(heat_power, 0.0) + np.maximum(heat_next, 0.0)
            )
            pback += 0.5 * dt * (
                np.maximum(population_rate, 0.0) + np.maximum(rate_next, 0.0)
            )
            min_rate = np.minimum(min_rate, rate_next)
            max_rate = np.maximum(max_rate, rate_next)

            tau = tau_next
            c, v = c_next, v_next
            population = population_next
            population_rate = rate_next
            heat_power = heat_next

        energy_backflow[row] = eback
        probability_backflow[row] = pback
        final_population[row] = population
        minimum_population_rate[row] = min_rate
        maximum_population_rate[row] = max_rate

    return {
        "ell": ell_values,
        "a": a_values,
        "Omega_delta": np.array([Omega_delta]),
        "Omega_b": np.array([Omega_b]),
        "energy_backflow": energy_backflow,
        "probability_backflow": probability_backflow,
        "final_population": final_population,
        "minimum_population_rate": minimum_population_rate,
        "maximum_population_rate": maximum_population_rate,
    }


def configure_plotting() -> None:
    configure_publication_style()


def plot_thesis_trajectory(
    data: dict[str, np.ndarray | float],
    ell: float,
    a: float,
    Omega_delta: float,
    Omega_b: float,
) -> None:
    tau = np.asarray(data["tau"])
    population = np.asarray(data["population"])
    gamma = np.asarray(data["gamma"])
    heat_power = np.asarray(data["heat_power"])
    work_power = np.asarray(data["work_power"])
    system_energy_rate = np.asarray(data["system_energy_rate"])
    reservoir_power = np.asarray(data["reservoir_to_system_power"])
    interaction_power = np.asarray(data["interaction_to_system_power"])

    positive = np.asarray(data["population_rate"]) > 0.0
    starts = np.flatnonzero(positive & ~np.r_[False, positive[:-1]])
    stops = np.flatnonzero(positive & ~np.r_[positive[1:], False])
    if starts.size != 1 or stops.size != 1:
        raise RuntimeError("The representative trajectory must contain one revival interval")

    def crossing(left: int, right: int) -> float:
        rate = np.asarray(data["population_rate"])
        return float(
            tau[left]
            - rate[left] * (tau[right] - tau[left]) / (rate[right] - rate[left])
        )

    start = int(starts[0])
    stop = int(stops[0])
    tau_minus = crossing(start - 1, start)
    tau_plus = crossing(stop, stop + 1)
    zoom_left = max(float(tau[0]), tau_minus - 0.28)
    zoom_right = min(float(tau[-1]), tau_plus + 0.28)

    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)

    axes[0, 0].plot(tau, population, color="#1f77b4", lw=2.5)
    axes[0, 0].axvspan(tau_minus, tau_plus, color="#d62728", alpha=0.12)
    axes[0, 0].axvline(tau_minus, color="0.25", lw=1.2, ls="--")
    axes[0, 0].axvline(tau_plus, color="0.25", lw=1.2, ls="--")
    axes[0, 0].set(xlabel=r"$\tau=\gamma_0 t$", ylabel=r"$P(\tau)=|c(\tau)|^2$")
    axes[0, 0].set_ylim(-0.02, 1.02)
    inset = axes[0, 0].inset_axes([0.43, 0.49, 0.53, 0.43])
    inset.plot(tau, population, color="#1f77b4", lw=2.0)
    inset.axvspan(tau_minus, tau_plus, color="#d62728", alpha=0.14)
    inset.axvline(tau_minus, color="0.25", lw=1.0, ls="--")
    inset.axvline(tau_plus, color="0.25", lw=1.0, ls="--")
    inset.set_xlim(zoom_left, zoom_right)
    local = (tau >= zoom_left) & (tau <= zoom_right)
    pmin = float(np.min(population[local]))
    pmax = float(np.max(population[local]))
    inset.set_ylim(pmin - 0.08 * (pmax - pmin), pmax + 0.12 * (pmax - pmin))
    inset.set_xticks([tau_minus, tau_plus], labels=[r"$\tau_-$", r"$\tau_+$"])
    inset.tick_params(labelsize=9)
    inset.grid(alpha=0.18)

    finite_gamma = np.isfinite(gamma)
    axes[0, 1].plot(tau[finite_gamma], gamma[finite_gamma], color="#7f3c8d", lw=2.3)
    axes[0, 1].axhline(0.0, color="0.25", lw=1.3)
    axes[0, 1].fill_between(
        tau,
        0.0,
        np.where(np.isfinite(gamma), gamma, 0.0),
        where=np.isfinite(gamma) & (gamma < 0.0),
        color="#d62728",
        alpha=0.22,
        label=r"$\bar\gamma<0$",
    )
    axes[0, 1].set(xlabel=r"$\tau$", ylabel=r"$\bar\gamma(\tau)=\gamma/\gamma_0$")
    axes[0, 1].axvline(tau_minus, color="0.25", lw=1.2, ls="--")
    axes[0, 1].axvline(tau_plus, color="0.25", lw=1.2, ls="--")
    axes[0, 1].legend(frameon=True)

    axes[1, 0].plot(tau, heat_power, color="#d62728", lw=2.4, label=r"$\dot Q_S$")
    axes[1, 0].plot(tau, reservoir_power, color="#1f77b4", lw=2.2, ls="--", label=r"$-\dot E_B$")
    axes[1, 0].plot(tau, interaction_power, color="#2ca02c", lw=2.2, ls="-.", label=r"$-\dot E_I$")
    axes[1, 0].axhline(0.0, color="0.25", lw=1.3)
    axes[1, 0].fill_between(tau, 0.0, heat_power, where=heat_power > 0.0, color="#d62728", alpha=0.18)
    axes[1, 0].set(xlabel=r"$\tau$", ylabel=r"energy currents $/\gamma_0^2$")
    axes[1, 0].set_xlim(zoom_left, zoom_right)
    microscopic_local = np.concatenate(
        [heat_power[local], reservoir_power[local], interaction_power[local], np.array([0.0])]
    )
    micro_span = float(np.max(microscopic_local) - np.min(microscopic_local))
    axes[1, 0].set_ylim(
        float(np.min(microscopic_local) - 0.10 * micro_span),
        float(np.max(microscopic_local) + 0.12 * micro_span),
    )
    axes[1, 0].axvline(tau_minus, color="0.25", lw=1.2, ls="--")
    axes[1, 0].axvline(tau_plus, color="0.25", lw=1.2, ls="--")
    axes[1, 0].legend(frameon=True, loc="best", ncol=3)

    axes[1, 1].plot(tau, heat_power, color="#d62728", lw=2.4, label=r"$\dot Q_S$")
    axes[1, 1].plot(tau, work_power, color="#9467bd", lw=2.2, ls="--", label=r"$\dot W_S$")
    axes[1, 1].plot(tau, system_energy_rate, color="black", lw=2.2, ls="-.", label=r"$\dot U_S$")
    axes[1, 1].axhline(0.0, color="0.25", lw=1.3)
    axes[1, 1].fill_between(tau, 0.0, system_energy_rate, where=system_energy_rate > 0.0, color="0.25", alpha=0.12)
    axes[1, 1].set(xlabel=r"$\tau$", ylabel=r"system powers $/\gamma_0^2$")
    axes[1, 1].set_xlim(zoom_left, zoom_right)
    system_local = np.concatenate(
        [heat_power[local], work_power[local], system_energy_rate[local], np.array([0.0])]
    )
    system_span = float(np.max(system_local) - np.min(system_local))
    axes[1, 1].set_ylim(
        float(np.min(system_local) - 0.10 * system_span),
        float(np.max(system_local) + 0.12 * system_span),
    )
    axes[1, 1].axvline(tau_minus, color="0.25", lw=1.2, ls="--")
    axes[1, 1].axvline(tau_plus, color="0.25", lw=1.2, ls="--")
    axes[1, 1].legend(frameon=True, loc="best", ncol=3)

    figure.suptitle(
        rf"Finite sweep: $\ell={ell:.4g}$, $a={a:.4g}$, $\Omega_\Delta={Omega_delta:.4g}$, $\Omega_b={Omega_b:.4g}$",
        fontsize=15,
    )
    close_axes([*axes.flat])
    figure.savefig(FIGURE_DIR / "thesis_exact_energy_correlation.png", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / "thesis_exact_energy_correlation.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_phase_diagram(grid: dict[str, np.ndarray], thesis_point: tuple[float, float]) -> None:
    ell = grid["ell"]
    a = grid["a"]
    eback = grid["energy_backflow"]
    pfinal = grid["final_population"]

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    a_edges = np.geomspace(a[0] / np.sqrt(a[1] / a[0]), a[-1] * np.sqrt(a[-1] / a[-2]), a.size + 1)
    ell_edges = np.geomspace(
        ell[0] / np.sqrt(ell[1] / ell[0]),
        ell[-1] * np.sqrt(ell[-1] / ell[-2]),
        ell.size + 1,
    )

    positive = eback[eback > 1e-8]
    floor = max(float(positive.min()) if positive.size else 1e-8, 1e-7)
    norm = mpl.colors.LogNorm(vmin=floor, vmax=max(float(eback.max()), 10.0 * floor))
    image0 = axes[0].pcolormesh(
        ell_edges,
        a_edges,
        np.maximum(eback, floor),
        shading="auto",
        cmap="magma",
        norm=norm,
        rasterized=True,
    )
    axes[0].contour(ell, a, eback, levels=[1e-4], colors="cyan", linewidths=1.2)
    axes[0].axvline(2.0, color="white", lw=1.0, ls="--", alpha=0.9)
    axes[0].plot(thesis_point[0], thesis_point[1], marker="*", ms=12, color="cyan", mec="black", mew=0.6)
    axes[0].set(xlabel=r"$\ell=\lambda/\gamma_0$", ylabel=r"$a=\alpha/\gamma_0^2$", title=r"Integrated positive heat-like current")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    figure.colorbar(image0, ax=axes[0], pad=0.02)

    image1 = axes[1].pcolormesh(
        ell_edges,
        a_edges,
        pfinal,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    axes[1].contour(ell, a, eback, levels=[1e-4], colors="white", linewidths=1.2)
    axes[1].axvline(2.0, color="white", lw=1.0, ls="--", alpha=0.9)
    axes[1].plot(thesis_point[0], thesis_point[1], marker="*", ms=12, color="cyan", mec="black", mew=0.6)
    axes[1].set(xlabel=r"$\ell=\lambda/\gamma_0$", ylabel=r"$a=\alpha/\gamma_0^2$", title=r"Final population $P(\Omega_\Delta/a)$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    figure.colorbar(image1, ax=axes[1], pad=0.02)

    figure.savefig(FIGURE_DIR / "exact_backflow_phase_diagram.png", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / "exact_backflow_phase_diagram.pdf", bbox_inches="tight")
    plt.close(figure)


def write_grid_csv(grid: dict[str, np.ndarray]) -> None:
    ell_mesh, a_mesh = np.meshgrid(grid["ell"], grid["a"])
    table = np.column_stack(
        [
            a_mesh.ravel(),
            ell_mesh.ravel(),
            grid["energy_backflow"].ravel(),
            grid["probability_backflow"].ravel(),
            grid["final_population"].ravel(),
            grid["minimum_population_rate"].ravel(),
            grid["maximum_population_rate"].ravel(),
        ]
    )
    header = "a,ell,positive_heatlike_integral_over_gamma0,probability_backflow,final_population,min_Pdot,max_Pdot"
    np.savetxt(DATA_DIR / "exact_backflow_phase_grid.csv", table, delimiter=",", header=header, comments="")


def write_boundary_csv(grid: dict[str, np.ndarray], rate_threshold: float = 1e-6) -> None:
    """Write the numerical non-Markovian lobe boundaries for each sweep rate."""
    rows = []
    ell = grid["ell"]
    for row, a in enumerate(grid["a"]):
        indices = np.flatnonzero(grid["maximum_population_rate"][row] > rate_threshold)
        if indices.size:
            rows.append((a, ell[indices[0]], ell[indices[-1]]))
        else:
            rows.append((a, np.nan, np.nan))
    np.savetxt(
        DATA_DIR / "exact_nonmarkovian_boundary.csv",
        np.asarray(rows),
        delimiter=",",
        header="a,ell_lower,ell_upper",
        comments="",
    )


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    ell_thesis = 5.0 / 8.0
    a_thesis = 5.0 / 32.0
    Omega_delta = 5.0 / 4.0
    Omega_b = 20.0

    trajectory = solve_trajectory(ell_thesis, a_thesis, Omega_delta, Omega_b)
    plot_thesis_trajectory(
        trajectory, ell_thesis, a_thesis, Omega_delta, Omega_b
    )

    ell_values = np.geomspace(0.01, 4.0, 100)
    a_values = np.geomspace(0.01, 1.0, 72)
    grid = phase_grid(ell_values, a_values, Omega_delta, Omega_b)
    write_grid_csv(grid)
    write_boundary_csv(grid)
    np.savez_compressed(DATA_DIR / "exact_backflow_phase_grid.npz", **grid)
    plot_phase_diagram(grid, (ell_thesis, a_thesis))

    summary = np.array(
        [
            ell_thesis,
            a_thesis,
            Omega_delta,
            Omega_b,
            trajectory["energy_backflow"],
            trajectory["probability_backflow"],
            trajectory["final_population"],
        ],
        dtype=float,
    )
    np.savetxt(
        DATA_DIR / "thesis_parameter_summary.csv",
        summary.reshape(1, -1),
        delimiter=",",
        header="ell,a,Omega_delta,Omega_b,positive_heatlike_integral_over_gamma0,probability_backflow,final_population",
        comments="",
    )
    print(
        "Thesis point: "
        f"ell={ell_thesis:.8f}, a={a_thesis:.8f}, "
        f"Omega_delta={Omega_delta:.8f}, Omega_b={Omega_b:.8f}, "
        f"Q_S_plus/gamma0={trajectory['energy_backflow']:.10f}, "
        f"N_P={trajectory['probability_backflow']:.10f}, "
        f"P_final={trajectory['final_population']:.10f}"
    )
    print(
        "Maximum integrated positive heat-like current "
        f"Q_S_plus/gamma0={grid['energy_backflow'].max():.10f}"
    )


if __name__ == "__main__":
    main()
