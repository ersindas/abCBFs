# Robust Adaptive Backup Control Barrier Functions

This repository contains the implementation of **Robust Adaptive Backup Control Barrier Functions**, submitted to IEEE Control Systems Letters (L-CSS), 2026.

Ersin Daş, David E. J. van Wijk, Tamas G. Molnar, Aaron D. Ames, and Joel W. Burdick

Paper: https://arxiv.org/pdf/2607.20842

## Prerequisites

- Python 3.9 or later.
- Required Python libraries:
  - `numpy`
  - `scipy`
  - `jax`
  - `cvxpy`
  - `quadprog`
  - `matplotlib`

## Contents

The `quadrotor` folder contains the planar quadrotor example from the paper: the true and estimated dynamics, the DREM parameter estimator, the vanilla and robust adaptive bCBF-QP safety filters, and the scripts used to run the simulation and generate the figure.

- `main_sim.py` runs a single simulation (vanilla, adaptive, or nominal, see `--mode`).
- `collect_data.py` runs all three cases and saves the data used for the paper figure.
- `make_fig.py` generates the figure from the saved data.

## Supplementary material

The journal version of the paper has a page limit, so the proof that the backup controller keeps the backup set forward invariant (Proposition 1) was moved out of the paper and put here instead: see `SUPPLEMENTARY.md`. The same conditions are checked numerically in `quadrotor/control.py`, in `checkValidBackup`.
