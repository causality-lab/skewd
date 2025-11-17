"""Exemplary usage of skewci."""

from src.skewci import skewci
from src.datasets import skewANsLSs
from sklearn.preprocessing import StandardScaler

# load pair from novel skew-noise dataset with dataset_type from ["ANs", "LSs"]
# set distribution to "SN" for skew-normal or "AGN" for (asymmetric) generalized normal
# set skewness level to "med", "strong" or "extreme" for skewness of -0.455, 0.985 or 1.750
# select pair_id from 1 to 100 
dataset = skewANsLSs(
    dataset_type="LSs",
    distribution="SN",
    skewness_level="strong",
    pair_id=19,
    preprocessor=StandardScaler(),
    double=True,
)
x, y = dataset.cause.flatten().numpy().reshape(
    (-1, 1)
), dataset.effect.flatten().numpy().reshape((-1, 1))


indep_score, indep_dir, lik_score, lik_dir, res_FW, res_RV, _, _ = skewci(
    x, y, cma_iter=500, n_cv_splits_boptim=8, ecm_max_iter=200, ecm_max_iter_rho=50
)

print(lik_dir)
# 1 means x -> y is inferred, 0 means y -> x
