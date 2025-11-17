# SKEWD

Code for [Skewness-Robust Causal Discovery in Location-Scale Noise Models]()

## Project Structure

In the following, we provide the general project structure.

```bash
skewd/
├── data # benchmark datasets
├── environment.yaml # .yaml for environment generation
├── example_plots # exemplary plots of skew-noise pairs
├── experiments # skewd experimental code: scripts and results for all benchmarks
├── plot_examples.ipynb # generates figures in example_plots
├── requirements.txt # requirements file
├── skewDGP 
│   └── generate_skewed_LSNMs.R # generates skew-LSNM benchmarks
└── src # source code for skewd
```

## Installation Instructions

### Create a Conda Environment

To ensure compatibility, use the ``environment.yaml'' file to create an environment:

```
conda env create -f environment.yaml
```

To run our code, activate the Conda environment:

```
conda activate skewd-env
```

Alternatively, we provide the ``requirements.txt'' file for environment creation.

## Running SKEWD

### Reproduce Results
For replicating the experiments, see the README.md inside the experiments folder. Another example for a single pair is provided in "src/example.py", showing how to execute SKEWD on a skew-noise pair.

### Random Dataset

Run SKEWD with a randomly generated dataset:
```python
import numpy as np
from src.skewd import skewd
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

x = np.random.randn(300).reshape(-1, 1)
eps = np.random.randn(300).reshape(-1, 1)
y = x**3 + (x**2) * eps

indep_score, indep_dir, lik_score, lik_dir, _, _, _, _  = skewd(x, y, cma_iter=200, ecm_max_iter=200)
print(f"SKEWD-IT infers direction {indep_dir}")
print(f"SKEWD-LL infers direction {lik_dir}")
```
The inferred direction is given by indep_dir and lik_dir for independence testing and likelihood scoring, respectively. 
If the direction is 1 x -> y is inferred, while if it is 0 y-> x is inferred.