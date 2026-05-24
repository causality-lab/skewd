"""Create multivariate skew datasets, skewness = 0.985."""

from src.graphgen import set_random_seed, simulate_dag, simulate_nonlinear_sem
import pickle
import numpy as np

# Specs:
n = 1000
d = 10
s0 = 10
reps = 100
# LSNM specs
sem_type = "skew-LS-s"
n = 1000
snn = "uniform"
hnf = 1

for rep in range(reps):
    set_random_seed(rep)
    B = simulate_dag(d=d, s0=s0, graph_type="ER")
    X, _, _, _ = simulate_nonlinear_sem(B, n, sem_type, source_node_noise=snn, het_noise_fraction=hnf)
    np.savetxt(f"data/MVskew10/MVskew{rep+1}.csv", X, delimiter=",")
    XB = {"X": X, "B": B}
    with open(f"data/MVskew10/MVskew{rep+1}.pkl", "wb") as f:
        pickle.dump(XB, f)