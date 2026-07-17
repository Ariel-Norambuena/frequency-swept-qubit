"""Regression tests for Supplemental Table SII."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from exact_backflow_analysis import solve_trajectory
from powerlaw_spectral_analysis import positive_current_integrals


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "output" / "data" / "powerlaw_table_sii.csv"


def main() -> None:
    ell = 5.0 / 8.0
    a = 5.0 / 32.0
    Omega_delta = 5.0 / 4.0
    Omega_b = 5.0 / 2.0

    trajectory = solve_trajectory(
        ell, a, Omega_delta, Omega_b, n_times=20001
    )
    probability, direct, parts = positive_current_integrals(
        np.asarray(trajectory["tau"]),
        np.asarray(trajectory["population"]),
        np.asarray(trajectory["population_rate"]),
        np.asarray(trajectory["omega"]),
        a,
    )

    assert abs(probability - 0.0159382984) < 5e-8
    assert abs(direct - 0.0277612129) < 5e-8
    assert abs(direct - parts) < 2e-8
    assert abs(direct - float(trajectory["energy_backflow"])) < 2e-8

    with TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    lorentz = rows[0]
    assert lorentz["reservoir"] == "Lorentzian"
    assert abs(
        float(lorentz["positive_heatlike_integral_direct_over_gamma0"])
        - 0.0277612129
    ) < 5e-8

    for row in rows:
        direct_table = float(row["positive_heatlike_integral_direct_over_gamma0"])
        parts_table = float(row["positive_heatlike_integral_parts_over_gamma0"])
        recorded_error = float(row["direct_parts_error"])
        assert abs(abs(direct_table - parts_table) - recorded_error) < 5e-13
        assert recorded_error < 2e-6

    print(
        "Table-SII regression passed: "
        f"N_P={probability:.10f}, E_direct={direct:.10f}, E_parts={parts:.10f}."
    )


if __name__ == "__main__":
    main()
