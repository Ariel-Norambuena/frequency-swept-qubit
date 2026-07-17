# Frequency-swept qubit in a structured reservoir

This repository contains the reproducible numerical code, generated data, figures, and LaTeX sources associated with the manuscript

> *Exceptional-point unfolding and critical bare-energy return in a frequency-swept qubit*

by Francisco Ahumada, Felipe Barra, and Ariel Norambuena.

The project studies the exact single-excitation dynamics of a frequency-controlled two-level system coupled to a structured bosonic reservoir. The repository reproduces the non-Markovian revival boundary, its exceptional-point asymptotics and control-protocol dependence, the microscopic energy balance, and the extensions reported in the Supplemental Material.

## Repository structure

- `Analysis/`: numerical analysis, figure generators, and regression tests.
- `output/data/`: numerical data used in the analyses and figures.
- `output/figures/`: generated figures in PDF and PNG formats.
- `Overleaf_PRR_submission/`: self-contained LaTeX sources and publication figures for the article and Supplemental Material.
- `reproduce_all.py`: regenerates the seven manuscript figures and runs the numerical regressions.
- `verify_key_results.py`: compact entry point for the principal checks.

## Reproducing the results

The tested environment is Python 3.12 with the package versions recorded in `requirements.txt`.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python reproduce_all.py
```

The last command regenerates the publication figures, copies them next to the LaTeX sources, and executes the key-result and Table SII regression tests. The checks can also be run separately:

```bash
python verify_key_results.py
python Analysis/verify_table_ii.py
```

To compile the manuscript and Supplemental Material with a standard TeX installation:

```bash
cd Overleaf_PRR_submission
latexmk -pdf main.tex
latexmk -pdf supplement.tex
```

## Citation

Please cite the associated article when using this code or data. The final journal citation and persistent archive DOI will be added after publication and archival deposition.

## Contact

Corresponding author: Ariel Norambuena (`ariel.norambuena@usm.cl`).
