"""Run skew data skewd experiments."""

import os
from src.datasets import skewANsLSs
from sklearn.preprocessing import StandardScaler
from src.skewd import skewd
from pathlib import Path
import pickle

dataset_types = ["LSs", "ANs"]
distributions = ["AGN", "SN"]
skew_levels = {"AGN": ["med", "strong", "extreme"], "SN": ["med", "strong"]}
pair_ids = list(range(1, 101))  # 1-100

grid = []
for dt in dataset_types:
    for dist in distributions:
        for level in skew_levels[dist]:
            for pid in pair_ids:
                grid.append((dt, dist, level, pid))

# variable on SLURM 1-1000
task_id = int(os.getenv("PBS_ARRAYID", "unknown")) - 1

dt, dist, level, pid = grid[task_id]

dataset = skewANsLSs(
    dataset_type=dt,
    distribution=dist,
    skewness_level=level,
    pair_id=pid,
    preprocessor=StandardScaler(),
    double=True,
)

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
    ecm_max_iter=5000,
    ecm_max_iter_rho=100,
    n_calls_boptim=60,
    n_inits_boptim=40,
)
ground_truth = 1  # for all cases

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

folder_parts = []
folder_parts.append("pure")
folder_parts.append(dt)
folder_parts.append(dist)
folder_parts.append(level)
inner_folder_name = "_".join(folder_parts)

save_file = (
    Path("experiments")
    / "results"
    / "skew_results"
    / inner_folder_name
    / f"result_{pid}.pkl"
)
save_file.parent.mkdir(parents=True, exist_ok=True)
with open(save_file, "wb") as file:
    pickle.dump(full_results, file)

print("Done!")
