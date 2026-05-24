"""Some utils functions."""
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, KFold

def skewCoeff1D(x):
    """Calculate the empirical skewness of x."""
    n = len(x)
    mu = np.mean(x)
    return (sum((x - mu) ** 3) / n) / (sum((x - mu) ** 2) / n) ** (3 / 2)


def safe_solve(matrix, vector, epsilon=1e-10):
    """Solve system of linear equations, approximate in case of failure."""
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return np.linalg.solve(
            matrix + epsilon * np.eye(matrix.shape[0]), vector
        )

def estimate_KDE(x, n_cv_splits=5, kernel='gaussian', tune_bandwidths=None):
    """Estimate log-density of x via KDE."""
    if tune_bandwidths is None:
        tune_bandwidths = np.logspace(-2, 1, 50)
    # Tune bandwidth:
    cv = KFold(n_splits=n_cv_splits, shuffle=True, random_state=910)
    grid = GridSearchCV(KernelDensity(kernel=kernel),
                        {'bandwidth': tune_bandwidths},
                        cv=cv)
    grid.fit(x)
    kde = grid.best_estimator_
    log_dens = kde.score_samples(x)
    return np.sum(log_dens)
