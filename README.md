# Quantum Teleportation and Dense Coding in Open Quantum Systems

Python research code for simulating two-qubit quantum correlations in an open-system setting. The project builds Cooper-pair density matrices from Hubbard/BdG Green's functions, evolves them under non-Markovian noise channels, and evaluates quantities relevant to quantum teleportation, dense coding, and entanglement preservation.

## What this project does

The main pipeline performs the following steps:

1. Solves the BdG self-consistency equations for the Hubbard Cooper-pair model.
2. Computes spatial Green's functions `G(r, θ)` and anomalous correlations `F(r, θ)`.
3. Builds two-qubit density matrices `ρ(θ, r)`.
4. Evolves each density matrix under non-Markovian noise.
5. Computes physical observables such as concurrence, singlet fraction, purity, and von Neumann entropy.
6. Saves numerical results as compressed NumPy files and optionally generates plots.

Optional analysis modules also support fidelity and Holevo-quantity calculations, which are useful for studying teleportation quality and dense-coding capacity.

## Repository structure

```text
.
├── main.py                 # Main end-to-end simulation runner
├── config.py               # Physical parameters, grids, noise settings, output names
├── kspace_setup.py         # 2D k-space grid and spin-resolved dispersion
├── self_consistency.py     # BdG self-consistency solver
├── greens_functions.py     # Green's-function calculation over r and θ
├── density_matrix.py       # Construction and validation of two-qubit density matrices
├── noise_channels.py       # Kraus operators for supported open-system noise models
├── evolution.py            # Time evolution of density matrices
├── observables.py          # Concurrence, singlet fraction, purity, entropy
├── fidelity.py             # Uhlmann fidelity and fidelity-grid utilities
├── holevo.py               # Holevo quantity and dense-coding analysis tools
├── visualization.py        # Main plots for concurrence and observables
├── visualize_*.py          # Additional plotting scripts
├── data/                   # Suggested location for generated .npz data files
└── figures/                # Suggested location for generated figures
```

## Supported noise models

The simulation currently supports:

- `kind="amp", bath="independent"` — independent non-Markovian amplitude damping.
- `kind="dephasing", bath="independent"` — independent Ornstein-Uhlenbeck dephasing.
- `kind="dephasing", bath="common"` — collective/common-bath dephasing.

Common-bath evolution is intended for dephasing only.

## Requirements

Use Python 3.10+ if possible.

Install the required scientific Python packages:

```bash
pip install numpy scipy matplotlib
```

There is currently no `requirements.txt` in the repository, so the command above installs the packages used by the source files.

## Installation

Clone the repository:

```bash
git clone https://github.com/AmirhoseynpowAsghari/quantum-teleportation-and-dense-coding-in-open-quantum-systems.git
cd quantum-teleportation-and-dense-coding-in-open-quantum-systems
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
```

Install dependencies:

```bash
pip install --upgrade pip
pip install numpy scipy matplotlib
```

## Quick start

Run the full pipeline without showing plots:

```bash
python main.py --no-plot
```

Run the default amplitude-damping simulation and save plots:

```bash
python main.py --kind amp --bath independent --plot
```

Run independent dephasing:

```bash
python main.py --kind dephasing --bath independent --plot
```

Run common-bath dephasing:

```bash
python main.py --kind dephasing --bath common --plot
```

Set custom noise parameters:

```bash
python main.py \
  --kind amp \
  --bath independent \
  --gammaA 4.5 \
  --GammaA 0.01 \
  --gammaB 4.5 \
  --GammaB 0.01 \
  --theta 0 \
  --plot
```

Use a custom filename prefix for figures:

```bash
python main.py --kind amp --bath independent --prefix test_run --plot
```

## Important configuration options

Edit `config.py` to change the physical model, grid sizes, noise parameters, and output settings.

Common parameters:

```python
# Physical parameters
t_hop = 1.0
U = 75 * t_hop
Delta_t = 4.5
V_SO = 0.0 * t_hop
n_elec = 1.875

# k-space grid
Nk = 100

# Real-space grid
R_MIN, R_MAX, N_R = 0.0, 10.0, 200
THETA_DEG = [0, 15, 30, 45, 60, 75, 90]

# Time grid
T_MIN, T_MAX, N_T = 0.0, 80.0, 161

# Default noise
NOISE_KIND = "amp"
NOISE_BATH = "independent"
GAMMA_A = 4.5
BIG_GAMMA_A = 0.01
GAMMA_B = 4.5
BIG_GAMMA_B = 0.01
```

For a faster test run, reduce the grid sizes before running:

```python
Nk = 30
N_R = 40
N_T = 41
```

After confirming the code runs, restore larger values for production-quality results.

## Outputs

The main pipeline saves compressed NumPy data files:

```text
rho_rt_results.npz    # main simulation results
C_rt_heatmap.npz      # concurrence heatmap data
```

The main results file contains arrays such as:

```text
t_grid        # time values
theta_vals    # angular grid in radians
r_vals        # radial grid
C_rt          # concurrence C(r,t)
Fs_rt         # singlet fraction Fs(r,t)
purity_rt     # purity Tr(rho^2)
entropy_rt    # von Neumann entropy S(rho)
```

When plotting is enabled, figures are generated for concurrence and other observables, including heatmaps and line plots versus `r` and `t`.

## Loading results in Python

```python
import numpy as np

results = np.load("rho_rt_results.npz")

r_vals = results["r_vals"]
t_grid = results["t_grid"]
C_rt = results["C_rt"]
Fs_rt = results["Fs_rt"]
purity_rt = results["purity_rt"]
entropy_rt = results["entropy_rt"]

print(C_rt.shape)  # expected shape: (N_R, N_T)
print("max concurrence:", C_rt.max())
```

## Optional analyses

### Fidelity

`fidelity.py` includes utilities for Uhlmann fidelity and fidelity grids. Use it when you want to compare evolved states against either the initial density matrix or a chosen reference state.

### Holevo quantity / dense coding

`holevo.py` computes the Holevo quantity:

```text
χ = S(ρ_tilde_AB) - S(ρ_AB)
```

This is useful for dense-coding analysis because it measures the classical information gain associated with the encoded two-qubit state.

## First-run troubleshooting

### 1. `ImportError` from `config.py`

Some helper modules expect additional configuration constants. If you see an error such as:

```text
ImportError: cannot import name 'KBT' from 'config'
```

add the missing constants to `config.py`. Example defaults:

```python
# Output folders used by utils.py, visualization.py, fidelity.py, and holevo.py
DATA_DIR = "data"
FIG_DIR = "figures"

# BdG / fsolve defaults used by self_consistency.py
KBT = 0.0
DELTA0_INIT = 1.0
DELTAS_INIT = 1.0
FSOLVE_XTOL = 1e-10
FSOLVE_MAXFEV_FACTOR = 1000
FSOLVE_N_RESTARTS = 5
```

Adjust these values according to the physical regime you want to study.

### 2. The run is slow

The default grid can be expensive because the calculation loops over k-space, radial distance, angle, and time. For debugging, reduce:

```python
Nk = 30
N_R = 40
N_T = 41
```

Then run:

```bash
python main.py --no-plot
```

### 3. Plot windows block the script

Use:

```bash
python main.py --no-plot
```

or set:

```python
MAKE_PLOTS = False
```

inside `config.py`.

### 4. Common-bath amplitude damping fails

Use common bath only with dephasing:

```bash
python main.py --kind dephasing --bath common --plot
```

For amplitude damping, use:

```bash
python main.py --kind amp --bath independent --plot
```

## Reproducibility

You can pass a seed to the self-consistency stage:

```bash
python main.py --seed 42 --no-plot
```

For reproducible studies, record:

- Git commit hash.
- Full `config.py` values.
- Command-line arguments used for the run.
- Python and package versions.

Example:

```bash
python --version
pip freeze > environment.txt
git rev-parse HEAD > commit.txt
```

## Suggested citation

If this code is used in a paper, thesis, or report, cite the repository and include the commit hash used for the simulations.

```bibtex
@software{quantum_open_systems_repo,
  title  = {Quantum Teleportation and Dense Coding in Open Quantum Systems},
  author = {AmirhoseynpowAsghari},
  url    = {https://github.com/AmirhoseynpowAsghari/quantum-teleportation-and-dense-coding-in-open-quantum-systems},
  year   = {2026}
}
```


