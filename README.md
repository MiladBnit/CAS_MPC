# Control-aware Stein filtering for stochastic MPC

Code for the numerical section of the paper *Control-aware Stein filtering for stochastic MPC* (L4DC).

A Rao-Blackwellized Stein filter (RBSGD) estimates the states and the weights of a learned model component. Its kernel
compares particles by their predicted outputs and constraint margins under the planned input (the control-aware
kernel), and the particles serve as the scenarios of a stochastic MPC with chance constraints.

The case study is a continuous bioreactor, controlled by the dilution rate:
- the mixing efficiency is unknown and drifts during the run;
- the growth rate is learned online by a neural network;
- the biomass must stay inside a corridor.

## Layout

| path | content |
|---|---|
| `notebooks/run_bioreactor.ipynb` | the bioreactor model and its parameters, one closed-loop run of every method, closed-loop figure |
| `notebooks/montecarlo_bioreactor.ipynb` | Monte Carlo over seeds 1–100: metrics, constraint violations, box-plot figure |
| `config/bioreactor.yaml` | the benchmark: plant, model, filters, controller, reference (loaded on top of `config/default.yaml`) |
| `src/carbsgd/` | the implementation |
| `results/bioreactor_mc_corridor_100/` | the Monte Carlo results of the paper (metrics per run and the configuration used) |

Modules of `src/carbsgd/` (equations are referred to by their labels in the paper, e.g. `eq:score`):

| module | content |
|---|---|
| `bioreactor.py` | plant, gray-box model with the learned growth rate, inhibition laws |
| `ekf.py`, `score.py` | per-particle EKF, one-step marginal likelihood and its score |
| `kernels.py`, `features.py` | parameter and control-aware kernels, feature map |
| `filters.py`, `constraints.py` | RBSGD (Stein variational Newton step, constraints on the parameters), RBPF |
| `scenarios.py`, `mpc.py` | scenario propagation, mixture chance constraints, scenario SMPC and certainty-equivalent MPC (IPOPT) |
| `closed_loop.py`, `bio_setup.py` | closed loop, setup of the case study, metrics of a run |
| `metrics.py`, `bio_mc.py`, `plots.py` | CRPS of Gaussian mixtures, Monte Carlo with saved results, figures |

## Setup

Python 3.11 or later.

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[notebook]"      # Linux/macOS: .venv/bin/python
```

Then open the notebooks with the `.venv` environment as the kernel.

## Reproducing the results

- **`notebooks/montecarlo_bioreactor.ipynb`.** It loads the saved Monte Carlo results from
  `results/bioreactor_mc_corridor_100/` and reproduces the tables and `paper/figures/bio_mc.pdf`.
  - To recompute the runs, set `rerun = True`. That is 500 closed-loop runs (5 methods × 100 seeds), which take about
    1.5 hours with 10 parallel processes.
- **`notebooks/run_bioreactor.ipynb`.** It runs seed 11 of every method (about 2 minutes) and writes
  `paper/figures/bio_closed_loop.pdf`.

All parameters are in `config/`. Seeds are fixed, and every result folder holds the configuration that produced it.
The notebooks reuse saved results when they run again with the same configuration.

## License

MIT, see `LICENSE`.
