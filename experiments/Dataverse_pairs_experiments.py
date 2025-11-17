"""Run Dataverse pairs skewd experiments."""

import os
from src.datasets import Cha, Multi, Net
from sklearn.preprocessing import StandardScaler
from src.skewd import skewd
from pathlib import Path
import pickle


dataset_types = ["CE-Cha", "CE-Multi", "CE-Net"]
pair_ids = list(range(1, 301))  # 1-300

grid = []
for dt in dataset_types:
    for pid in pair_ids:
        grid.append((dt, pid))

task_id = int(os.getenv("PBS_ARRAYID", "unknown")) - 1  # variable on SLURM 1-900

dt, pid = grid[task_id]

if dt == "CE-Cha":
    dataset = Cha(pid, preprocessor=StandardScaler(), double=True)
elif dt == "CE-Multi":
    dataset = Multi(pid, preprocessor=StandardScaler(), double=True)
elif dt == "CE-Net":
    dataset = Net(pid, preprocessor=StandardScaler(), double=True)
else:
    raise ValueError("dt needs to be a Dataverse pairs folder name")

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
    Path("experiments")
    / "results"
    / "Dataverse_pairs_results"
    / f"{dt}"
    / f"result_{pid}.pkl"
)
save_file.parent.mkdir(parents=True, exist_ok=True)
with open(save_file, "wb") as file:
    pickle.dump(full_results, file)

print("Done!")
