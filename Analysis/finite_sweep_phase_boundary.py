"""Resolved finite-sweep non-Markovian lobe and asymptotic validity.

For fixed Omega the upper boundary changes from an interior tangency to an
endpoint condition.  At a second rate the lower boundary detaches from
ell=0, and the two endpoint roots finally merge in a square-root cusp.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, root


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
FIGURE_DIR = ROOT / "output" / "figures"
sys.path.insert(0, str(ROOT / "Analysis"))

from asymptotic_boundary_analysis import solve_boundary
from lobe_tip_analysis import endpoint_q, endpoint_tip_residual
from publication_style import close_axes, configure_publication_style


OMEGA = 1.25


def switch_residual(log_variables: np.ndarray) -> np.ndarray:
    """Endpoint rate and its time derivative vanish simultaneously."""
    ell, a = np.exp(log_variables)
    tau = OMEGA / a
    q = endpoint_q(ell, a, OMEGA)
    q_prime = -(q * q) - (ell + 1j * a * tau) * q - 0.5 * ell
    return np.array([q.real, q_prime.real])


def weak_coupling_kernel(x: float) -> float:
    """Real Fresnel kernel controlling q/ell as ell -> 0."""
    real = quad(lambda y: np.cos(0.5 * y * y), 0.0, x, epsabs=2e-13)[0]
    imag = quad(lambda y: np.sin(0.5 * y * y), 0.0, x, epsabs=2e-13)[0]
    return float(np.real(np.exp(-0.5j * x * x) * (real + 1j * imag)))


def resolved_endpoint_roots(a: float) -> list[float]:
    """Find endpoint roots, including a lower branch born at ell=0."""
    ell_grid = np.unique(
        np.concatenate([np.geomspace(1e-8, 0.1, 72), np.linspace(0.1, 1.4, 180)])
    )
    values = np.asarray([endpoint_q(float(ell), a, OMEGA).real for ell in ell_grid])
    roots: list[float] = []
    for left, right, f_left, f_right in zip(
        ell_grid[:-1], ell_grid[1:], values[:-1], values[1:]
    ):
        if f_left * f_right < 0.0:
            roots.append(
                brentq(
                    lambda ell: endpoint_q(ell, a, OMEGA).real,
                    float(left),
                    float(right),
                    xtol=2e-12,
                )
            )
    return roots


def continued_interior_boundary(a_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ell_values = []
    tau_values = []
    a0 = float(a_values[0])
    ell0 = 2.0 - np.sqrt(2.0) * np.pi * np.sqrt(a0) + np.sqrt(np.pi) / 2.0**0.75 * a0**0.75
    tau0 = np.sqrt(np.sqrt(2.0) * np.pi) * a0**(-0.25)
    epsilon0 = a0 ** (1.0 / 3.0)
    guess = ((2.0 - ell0) / epsilon0**2, tau0 * epsilon0)
    for a in a_values:
        ell, tau, _ = solve_boundary(float(a), guess)
        epsilon = a ** (1.0 / 3.0)
        guess = ((2.0 - ell) / epsilon**2, tau * epsilon)
        ell_values.append(ell)
        tau_values.append(tau)
    return np.asarray(ell_values), np.asarray(tau_values)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    switch = root(switch_residual, np.log([0.94, 0.32]), tol=2e-11)
    if not switch.success:
        raise RuntimeError(switch.message)
    ell_switch, a_switch = np.exp(switch.x)
    q_switch = endpoint_q(ell_switch, a_switch, OMEGA)

    x_zero = brentq(weak_coupling_kernel, 2.0, 2.3, xtol=2e-14)
    a_lower_birth = (OMEGA / x_zero) ** 2

    tip = root(endpoint_tip_residual, np.log([0.49, 0.397]), args=(OMEGA,), tol=2e-11)
    if not tip.success:
        raise RuntimeError(tip.message)
    ell_tip, a_tip = np.exp(tip.x)

    # Interior tangency branch, with a logarithmic slow-sweep segment and a
    # linear segment resolving the mechanism switch.
    a_interior = np.unique(
        np.concatenate(
            [np.geomspace(1e-4, 1e-2, 15), np.linspace(1e-2, a_switch, 33)]
        )
    )
    ell_interior, tau_interior = continued_interior_boundary(a_interior)

    # Endpoint roots.  A single positive root is the upper boundary before
    # the weak-coupling lower branch is born; afterwards there are two roots.
    a_endpoint = np.linspace(a_switch + 2e-4, a_tip - 2e-4, 30)
    lower_endpoint = np.zeros_like(a_endpoint)
    upper_endpoint = np.zeros_like(a_endpoint)
    for index, a in enumerate(a_endpoint):
        roots = resolved_endpoint_roots(float(a))
        if a < a_lower_birth:
            if len(roots) != 1:
                raise RuntimeError(f"Expected one endpoint root at a={a}, got {roots}")
            lower_endpoint[index] = 0.0
            upper_endpoint[index] = roots[0]
        else:
            if len(roots) != 2:
                raise RuntimeError(f"Expected two endpoint roots at a={a}, got {roots}")
            lower_endpoint[index], upper_endpoint[index] = roots

    # Include all analytically distinguished endpoints in the exported curve.
    a_curve = np.concatenate(
        [a_interior, a_endpoint, np.array([a_tip])]
    )
    lower_curve = np.concatenate(
        [np.zeros_like(a_interior), lower_endpoint, np.array([ell_tip])]
    )
    upper_curve = np.concatenate(
        [ell_interior, upper_endpoint, np.array([ell_tip])]
    )
    mechanism = np.concatenate(
        [np.zeros_like(a_interior), np.ones_like(a_endpoint), np.array([2.0])]
    )
    order = np.argsort(a_curve)
    np.savetxt(
        DATA_DIR / "finite_sweep_boundary_structure.csv",
        np.column_stack(
            [a_curve[order], lower_curve[order], upper_curve[order], mechanism[order]]
        ),
        delimiter=",",
        header="a,ell_lower,ell_upper,mechanism_0_interior_1_endpoint_2_tip",
        comments="",
    )

    summary = np.array(
        [[
            OMEGA,
            a_switch,
            ell_switch,
            OMEGA / a_switch,
            q_switch.imag,
            x_zero,
            a_lower_birth,
            a_tip,
            ell_tip,
        ]]
    )
    np.savetxt(
        DATA_DIR / "finite_sweep_transition_points.csv",
        summary,
        delimiter=",",
        header=(
            "Omega,a_interior_to_endpoint,ell_interior_to_endpoint,tau_switch,"
            "imag_q_switch,first_fresnel_zero,a_lower_birth,a_tip,ell_tip"
        ),
        comments="",
    )

    slow = np.genfromtxt(DATA_DIR / "small_a_upper_boundary.csv", delimiter=",", names=True)
    deficit = slow["delta"]
    leading_relative = np.abs(slow["ell_error"]) / deficit
    two_term_relative = np.abs(slow["ell_two_term_error"]) / deficit

    x_grid = np.linspace(0.0, 3.0, 350)
    fresnel_values = np.asarray([weak_coupling_kernel(float(x)) for x in x_grid])

    configure_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.15), constrained_layout=True)

    # Use the resolved branches for the lobe fill.
    curve_a = np.concatenate([a_interior, a_endpoint, [a_tip]])
    curve_low = np.concatenate([np.zeros_like(a_interior), lower_endpoint, [ell_tip]])
    curve_up = np.concatenate([ell_interior, upper_endpoint, [ell_tip]])
    axes[0].fill_between(curve_a, curve_low, curve_up, color="#80cdc1", alpha=0.35)
    axes[0].plot(a_interior, ell_interior, color="#01665e", lw=2.35, label="interior tangency")
    axes[0].plot(a_endpoint, upper_endpoint, color="#d95f02", lw=2.35, label="upper endpoint")
    mask_lower = a_endpoint >= a_lower_birth
    axes[0].plot(a_endpoint[mask_lower], lower_endpoint[mask_lower], color="#7570b3", lw=2.35,
                 label="lower endpoint")
    axes[0].plot([a_lower_birth, a_switch, a_tip], [0.0, ell_switch, ell_tip],
                 "o", ms=6.0, color="black")
    axes[0].annotate("mechanism switch", (a_switch, ell_switch), xytext=(0.22, 1.22),
                     arrowprops={"arrowstyle": "->", "lw": 1.2}, fontsize=11)
    axes[0].annotate("weak-coupling birth", (a_lower_birth, 0.0), xytext=(0.27, 0.32),
                     arrowprops={"arrowstyle": "->", "lw": 1.2}, fontsize=11)
    omega_label = r"5/4" if np.isclose(OMEGA, 1.25) else f"{OMEGA:g}"
    axes[0].set(xlabel=r"$a$", ylabel=r"$\ell$", title=rf"Resolved lobe ($\Omega_\Delta={omega_label}$)",
                xlim=(0.0, 0.415), ylim=(-0.02, 2.03))
    axes[0].legend(frameon=True, loc="upper right")

    axes[1].loglog(slow["a"], leading_relative, "o-", ms=5.2, lw=2.0, label=r"$a^{1/2}$ term")
    axes[1].loglog(slow["a"], two_term_relative, "o-", ms=5.2, lw=2.0, label=r"through $a^{3/4}$")
    axes[1].axhline(0.01, color="black", ls="--", lw=1.5, label=r"$1\%$")
    axes[1].axhline(0.05, color="0.4", ls=":", lw=1.5, label=r"$5\%$")
    axes[1].set(xlabel=r"$a$", ylabel=r"error$/\,(2-\ell_c)$", title="Asymptotic validity")
    axes[1].legend(frameon=True)

    axes[2].plot(x_grid, fresnel_values, color="#5e3c99", lw=2.35)
    axes[2].axhline(0.0, color="black", lw=1.3)
    axes[2].axvline(x_zero, color="#e66101", ls="--", lw=1.7,
                    label=rf"$x_0={x_zero:.6f}$")
    axes[2].set(xlabel=r"$x=\sqrt{a}\,\tau$", ylabel=r"$\Re[e^{-ix^2/2}\int_0^x e^{iy^2/2}dy]$",
                title="Universal weak-coupling edge")
    axes[2].legend(frameon=True)

    close_axes(axes)
    fig.savefig(FIGURE_DIR / "finite_sweep_boundary_structure.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "finite_sweep_boundary_structure.pdf", bbox_inches="tight")
    plt.close(fig)

    print(
        f"switch: a={a_switch:.12f}, ell={ell_switch:.12f}; "
        f"weak birth: x0={x_zero:.12f}, a={a_lower_birth:.12f}; "
        f"tip: a={a_tip:.12f}, ell={ell_tip:.12f}"
    )


if __name__ == "__main__":
    main()
