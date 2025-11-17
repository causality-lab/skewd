# SKEWD Experimental Runs

## Folder Structure

```bash
skewd/experiments/
├── ANLSMN_experiments.py # used to run corresponding experiments
├── Benchmark_simulated_experiments.py # used to run corresponding experiments
├── Dataverse_pairs_experiments.py # used to run corresponding experiments
├── Tuebingen_experiments.py # used to run corresponding experiments
├── baseline_results # contains baseline results in .csv format 
├── evaluation.ipynb # run to compute and save result metrics
├── evaluation_cmaes.ipynb # run to compute and save result metrics for cmaes
├── results # contains results in .csv format and .pkl for every single experiment
└── skew_experiments.py # used to run corresponding experiments
```

## Run the Experiments
In total there are 5 classes of pair benchmark datasets. For each, there is a single .py file used to run the associated experiments. The scripts
require an environmental variable "PBS_ARRAYID" as specified in the respective scripts. 

### Skew-noise datasets
To run the skew-noise experiments for the different datasets, one has to specify the PBS_ARRAYID. 
To run SKEWD on the first pair, run the following code on the terminal.
```terminal
set PBS_ARRAYID=1
python -m experiments.skew_experiments
```
This is the same for the different datasets types. 
Files can be sequentially run through a loop over all IDs.
The IDs that result in the data used for our experiments are 1,2...,1000.

### ANLSMN pairs
To run SKEWD on the first pair, run the following code on the terminal.
```terminal
set PBS_ARRAYID=1
python -m experiments.ANLSMN_experiments
```
For ANLSMN, consider the IDs 1,2,...,500.

### Benchmark simulated 
To run SKEWD on the first pair, run the following code on the terminal.
```terminal
set PBS_ARRAYID=1
python -m experiments.Benchmark_simulated_experiments
```
For Benchmark simulated, consider the IDs 1,2,...,400.

### Dataverse pairs
To run SKEWD on the first pair, run the following code on the terminal.
```terminal
set PBS_ARRAYID=1
python -m experiments.Dataverse_pairs_experiments
```
For Dataverse pairs, consider the IDs 1,2,...,300.

### Tuebingen
To run SKEWD on the first pair, run the following code on the terminal.
```terminal
set PBS_ARRAYID=1
python -m experiments.Tuebingen_experiments
```
For Tuebingen, consider the IDs 1,2,...,108, except 47, 52, 53, 54, 55, 70, 71, 105, 107.

## Evaluation
Evaluation was carried out using the evaluation.ipynb and the evaluation_cmaes.ipynb.