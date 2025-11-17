"""Run Tuebingen pairs skewd experiments."""

import os
from src.datasets import Tuebingen
from sklearn.preprocessing import StandardScaler
from src.skewd import skewd
from pathlib import Path
import pickle


pid = int(os.getenv("PBS_ARRAYID", "unknown"))  
# variable on SLURM 1-108 except [47, 52, 53, 54, 55, 70, 71, 105, 107];

# Pair 69 was run using a max of 500 ECM iterations, while all other pairs had 5000.

# Pairs [47, 52, 53, 54, 55, 70, 71, 105, 107] should not be run.

if pid == 69:
    mi = 500
else:
    mi = 5000

dataset = Tuebingen(pid, preprocessor=StandardScaler(), double=True)

x = dataset.cause.flatten().numpy().reshape((-1, 1))
y = dataset.effect.flatten().numpy().reshape((-1, 1))
(
    indep_score,
    indep_dir,
    lik_score,
    lik_dir,
    res_FW,
    res_RV,
    res_FW_cma,
    res_RV_cma,
) = skewd(
    x,
    y,
    cma_iter=5000,
    n_cv_splits_boptim=8,
    ecm_max_iter=mi,
    ecm_max_iter_rho=100,
    n_calls_boptim=60,
    n_inits_boptim=40,
)
ground_truth = 1

correct_IT = indep_dir == ground_truth
correct_LL = lik_dir == ground_truth

full_results = {
    "correct_IT": correct_IT,
    "correct_LL": correct_LL,
    "indep_score": indep_score,
    "lik_score": lik_score,
    "res_FW": res_FW,
    "res_RV": res_RV,
    "ground_truth": ground_truth,
    "res_FW_cma": res_FW_cma,
    "res_RV_cma": res_RV_cma,
}

save_file = (
    Path("experiments") / "results" / "Tuebingen_results" / f"result_{pid}.pkl"
)
save_file.parent.mkdir(parents=True, exist_ok=True)
with open(save_file, "wb") as file:
    pickle.dump(full_results, file)

print("Done!")
