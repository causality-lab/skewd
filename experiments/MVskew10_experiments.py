"""Multivariate skew LSNM experiments with 10 nodes."""

import pickle 
from src.skewdmv import skewdmv
from src.graphutils import build_oracle_MEC
from pathlib import Path
import os
import time

rep_id = int(os.getenv("PBS_ARRAYID", "unknown")) # global: 1-200 on SLURM
#rep_id = 139

if __name__ == "__main__":
    start = time.time()

    if rep_id > 100:
        pid = rep_id - 100
        with open(f"data/MVskew10/MVskew{pid}.pkl", "rb") as file:
            XB = pickle.load(file)
        X, B = XB["X"], XB["B"]
        # Determine oracle MEC:
        oracle_MEC = build_oracle_MEC(B) # 101-200 use oracle MEC
    else:
        pid = rep_id
        with open(f"data/MVskew10/MVskew{pid}.pkl", "rb") as file:
            XB = pickle.load(file)
        X = XB["X"]
        # No oracle MEC
        oracle_MEC = None # 1-100 use pc algorithm

    best_graph_KDE, score_KDE, best_graph_N, score_N, MEC = skewdmv(
        X,
        rng_seed=912,
        N_knots=8,
        Z_knots=5,
        N_deg=3,
        Z_deg=3,
        alpha_pc=0.05,
        indep_test_pc="kci",
        oracle_MEC = oracle_MEC, 
        cma_sigma=1,
        cma_lr_adapt=True,
        cma_population_size=100,
        cma_iter=5000,
        cma_crit=10**-6,
        cma_n_crit=200,
        bounds_rho=None,
        bounds_lambda=None,
        n_calls_boptim=20,
        n_inits_boptim=10,
        log_bounds_alpha=None,
        log_bounds_kappa=None,
        n_cv_splits_boptim=5,
        ecm_tol=10**-6,
        ecm_n_crit=25,
        ecm_max_iter=3000,
        ecm_max_iter_rho=100,
        ecm_track_history=False,
        n_cores_max = 6
    )

    full_results = {"best_graph_KDE": best_graph_KDE, "score_KDE": score_KDE, 
                    "best_graph_N": best_graph_N, "score_N": score_N, "MEC": MEC }

    save_file = (
        Path("experiments")
        / "results"
        / "MVskew10"
        / f"result_{rep_id}.pkl"
    )
    save_file.parent.mkdir(parents=True, exist_ok=True)
    with open(save_file, "wb") as file:
        pickle.dump(full_results, file)

    print("Done!")
    end = time.time()
    print(end-start)
