# Robust Adaptive Backup Control Barrier Functions

This repository contains the implementation of **Robust Adaptive Backup Control Barrier Functions**, accepted for publication in IEEE Control Systems Letters (L-CSS), 2026.

Ersin Daş, David E. J. van Wijk, Tamas G. Molnar, Aaron D. Ames, and Joel W. Burdick

[Paper](https://arxiv.org/pdf/2607.20842) · [Figure](media/figure.pdf) · [Video](media/simulation.mp4)

[![Simulation of the planar quadrotor](media/simulation.gif)](media/simulation.mp4)

## Reproducing the results

The `quadrotor` folder contains the planar quadrotor example from the paper: the true and estimated dynamics, the DREM parameter estimator, the nominal and robust adaptive bCBF-QP safety filters, and the scripts used to generate the figure and animation.

Use Python 3.12. Install the packages, run the simulations, and generate the figure and animation:

```bash
python -m pip install -r requirements.txt
python quadrotor/collect_data.py
python quadrotor/make_fig.py
python quadrotor/animate.py
```

Saved data are included in `quadrotor/data`. To use them, skip `collect_data.py` and run the two plotting commands. Figures are saved in `quadrotor/figures`. 

To run one controller, use `main_sim.py` with `--mode adaptive`, `--mode vanilla` for the nominal bCBF, or `--mode nominal` for the primary controller alone:

```bash
python quadrotor/main_sim.py --mode adaptive
```

The simulation uses a 1 s backup horizon and a 10 ms sampling period. 

## Supplementary material

[Supplementary Material.pdf](Supplementary%20Material.pdf) gives the proof that the backup controller keeps the backup set forward invariant (Proposition 1). The same conditions are checked numerically in `quadrotor/control.py`, in `checkValidBackup`.
