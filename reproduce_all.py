"""Regenerate every figure and numerical regression used in the manuscript.

Run from any working directory with the environment in ``requirements.txt``.
The generated PDF figures are copied next to ``main.tex`` and
``supplement.tex`` so that the Overleaf upload has no directory dependency.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "Analysis"
OUTPUT_FIGURES = ROOT / "output" / "figures"
OVERLEAF = ROOT / "Overleaf_PRR_submission"

SCRIPTS = (
    "asymptotic_boundary_analysis.py",
    "lobe_tip_analysis.py",
    "robust_boundary_figures.py",
    "universality_theorem_analysis.py",
    "finite_sweep_phase_boundary.py",
    "exact_backflow_analysis.py",
    "powerlaw_spectral_analysis.py",
    "multilorentzian_boundary.py",
    "discrete_bath_reconstruction.py",
    "sweep_range_dependence.py",
)

MANUSCRIPT_FIGURES = (
    "exact_boundary_asymptotics.pdf",
    "detuning_universality_theorem.pdf",
    "finite_sweep_boundary_structure.pdf",
    "thesis_exact_energy_correlation.pdf",
    "powerlaw_lorentzian_comparison.pdf",
    "bilorentzian_boundary.pdf",
    "discrete_bath_reconstruction.pdf",
)

REGRESSIONS = (
    "verify_key_results.py",
    "verify_table_ii.py",
)


def run(script: str) -> None:
    print(f"[reproduce] {script}", flush=True)
    subprocess.run(
        [sys.executable, str(ANALYSIS / script)], cwd=ROOT, check=True
    )


def main() -> None:
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OVERLEAF.mkdir(parents=True, exist_ok=True)

    for script in SCRIPTS:
        run(script)

    missing = []
    for name in MANUSCRIPT_FIGURES:
        source = OUTPUT_FIGURES / name
        if not source.exists():
            missing.append(name)
        else:
            shutil.copy2(source, OVERLEAF / name)
    if missing:
        raise FileNotFoundError(
            "Missing regenerated manuscript figures: " + ", ".join(missing)
        )

    for script in REGRESSIONS:
        run(script)

    print(
        "[reproduce] Seven figures and all numerical regressions passed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
