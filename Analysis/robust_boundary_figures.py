"""Publication figures for the exact slow-sweep boundary and endpoint cusp."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from publication_style import close_axes, configure_publication_style


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"
FIGURE_DIR = ROOT / "output" / "figures"


def load_csv(name: str) -> np.ndarray:
    return np.genfromtxt(DATA_DIR / name, delimiter=",", names=True)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    slow = load_csv("small_a_upper_boundary.csv")
    tip = load_csv("upper_tip_point.csv")
    cusp = load_csv("upper_tip_cusp_validation.csv")

    configure_publication_style()

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.05), constrained_layout=True)

    a = slow["a"]
    order = np.argsort(a)
    a = a[order]
    ell = slow["ell_critical"][order]
    leading = slow["ell_leading_asymptotic"][order]
    two_term = slow["ell_two_term_asymptotic"][order]

    axes[0].plot(a, ell, "o", ms=5.2, color="black", label="exact tangency")
    axes[0].plot(a, leading, lw=2.1, color="#d95f02", ls="--", label=r"$O(a^{1/2})$")
    axes[0].plot(a, two_term, lw=2.2, color="#1b9e77", label=r"$O(a^{3/4})$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"sweep rate $a$")
    axes[0].set_ylabel(r"upper boundary $\ell_{\rm c}$")
    axes[0].set_title("Slow-sweep boundary")
    axes[0].legend(frameon=True, loc="lower left")
    axes[0].grid(alpha=0.22, which="both")

    coefficient = (ell - leading) / a**0.75
    exact_coefficient = np.sqrt(np.pi) / 2.0**0.75
    axes[1].plot(a**0.25, coefficient, "o-", ms=5.0, lw=2.0, color="#7570b3")
    axes[1].axhline(exact_coefficient, color="black", ls="--", lw=1.7, label=rf"$\sqrt{{\pi}}/2^{{3/4}}={exact_coefficient:.6f}$")
    axes[1].set_xlabel(r"$a^{1/4}$")
    axes[1].set_ylabel(r"$[\ell_{\rm c}-2+\sqrt{2}\pi\sqrt{a}]/a^{3/4}$")
    axes[1].set_title("Coefficient collapse")
    axes[1].legend(frameon=True, loc="best")
    axes[1].grid(alpha=0.22)

    a_tip = float(tip["a_tip"])
    ell_tip = float(tip["ell_tip"])
    c_tip = float(tip["cusp_coefficient"])
    a_cusp = cusp["a"]
    sort_tip = np.argsort(a_cusp)
    a_cusp = a_cusp[sort_tip]
    lower = cusp["ell_lower_exact"][sort_tip]
    upper = cusp["ell_upper_exact"][sort_tip]
    distance_line = np.linspace(0.0, max(a_tip - a_cusp) * 1.08, 200)
    a_line = a_tip - distance_line
    axes[2].plot(a_cusp, lower, "o", ms=5.3, color="#1f78b4", label="exact endpoint roots")
    axes[2].plot(a_cusp, upper, "o", ms=5.3, color="#1f78b4")
    axes[2].plot(a_line, ell_tip - c_tip * np.sqrt(distance_line), color="#e31a1c", lw=2.0, ls="--", label="square-root cusp")
    axes[2].plot(a_line, ell_tip + c_tip * np.sqrt(distance_line), color="#e31a1c", lw=2.0, ls="--")
    axes[2].plot(a_tip, ell_tip, marker="*", ms=13, color="black", label="tip")
    axes[2].set_xlabel(r"sweep rate $a$")
    axes[2].set_ylabel(r"endpoint boundary $\ell_\pm$")
    omega_label = r"5/4" if np.isclose(float(tip["Omega"]), 1.25) else f"{float(tip['Omega']):g}"
    axes[2].set_title(rf"Finite-sweep tip ($\Omega={omega_label}$)")
    axes[2].legend(frameon=True, loc="best")
    axes[2].grid(alpha=0.22)

    close_axes(axes)

    figure.savefig(FIGURE_DIR / "exact_boundary_asymptotics.png", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / "exact_boundary_asymptotics.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
