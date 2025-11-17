"""Some utils functions."""
import numpy as np


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
