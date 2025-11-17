"""Functions for (penalized) (neg. log-) skew-normal likelihood."""
import numpy as np
import sys
from scipy.stats import norm


def m(rho, Z):
    """Compute variance term of heteroscedastic model."""
    return np.exp(Z @ rho)


def sn_lik(y, N, Z, f, rho, lambda_hat):
    """Compute likelihood of heteroscedastic skew-normal model.

    Parameters
    ----------
    - y: outcome (n x 1)
    - N: design matrix for f (n x p_f)
    - Z: design matrix for rho (n x p_rho)
    - f: coefficient vector (p_f x 1)
    - rho: coefficient vector (p_rho x 1)
    - lambda_hat: skewness coefficient (scalar)

    Returns
    -------
    - likelihood (scalar)
    """
    fx = N @ f
    hz = m(rho, Z)
    gx = np.sqrt(hz)
    part1 = 2 / gx
    part2 = norm.pdf(np.maximum((y - fx) / gx, -35))
    part3 = norm.cdf(np.maximum(lambda_hat * (y - fx) / gx, -35))
    vero = part1 * part2 * part3
    return np.maximum(vero, sys.float_info.min)


def sn_nloglik(y, N, Z, params=None, f=None, rho=None, lambda_hat=None):
    """Compute neg. log-likelihood of heteroscedastic skew-normal model.

    Parameters
    ----------
    - y: outcome (n x 1)
    - N: design matrix for f (n x p_f)
    - Z: design matrix for rho (n x p_rho)
    - params: vector of parameters (p_f + p_rho + 1, optional)
    - f: coefficient vector (p_f x 1, optional if params not given)
    - rho: coefficient vector (p_rho x 1, optional if params not given)
    - lambda_hat: skewness coefficient (scalar, optional if params not given)

    Returns
    -------
    - negative log-likelihood (scalar)
    """
    if params is not None:
        f = params[0 : N.shape[1]].reshape(-1, 1)
        rho = params[(N.shape[1]) : -1].reshape(-1, 1)
        lambda_hat = params[-1]
    else:
        if f is None or rho is None or lambda_hat is None:
            raise ValueError(
                "If params is None, f, rho, lambda_hat must be supplied."
            )
    return -np.sum(
        np.log(sn_lik(y=y, N=N, Z=Z, f=f, rho=rho, lambda_hat=lambda_hat))
    )


def sn_nloglikP(
    y, N, Z, K, alpha, M, kappa, params=None, f=None, rho=None, lambda_hat=None
):
    """Compute pen. neg. log-likelihood of heteroscedastic skew-normal model.

    Parameters
    ----------
    - y: outcome (n x 1)
    - N: design matrix for f (n x p_f)
    - Z: design matrix for rho (n x p_rho)
    - K: penalty matrix for f (p_f x p_f)
    - alpha: penalty coefficient for f (scalar)
    - M: penalty matrix for rho (p_rho x p_rho)
    - kappa: penalty coefficient for rho (scalar)
    - params: vector of parameters (p_f + p_rho + 1, optional)
    - f: coefficient vector (p_f x 1, optional if params not given)
    - rho: coefficient vector (p_rho x 1, optional if params not given)
    - lambda_hat: skewness coefficient (scalar, optional if params not given)

    Returns
    -------
    - penalized negative log-likelihood (scalar)
    """
    if params is not None:
        f = params[0 : N.shape[1]].reshape(-1, 1)
        rho = params[(N.shape[1]) : -1].reshape(-1, 1)
        lambda_hat = params[-1]
    else:
        if f is None or rho is None or lambda_hat is None:
            raise ValueError(
                "If params is None, f, rho, lambda_hat must be supplied."
            )
    penalty = alpha / 2 * f.T @ K @ f + kappa / 2 * rho.T @ M @ rho
    return sn_nloglik(y, N, Z, f=f, rho=rho, lambda_hat=lambda_hat) + penalty
