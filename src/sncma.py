"""Estimate het. skew-normal model using cmaes."""

import numpy as np
from src.utils import skewCoeff1D
from src.snlik import sn_lik, sn_nloglikP
from cmaes import CMA
import statsmodels.api as sm
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.space import Real


def optimSNCMA(
    alpha,
    kappa,
    x,
    y,
    N,
    Z,
    K,
    M,
    cma_sigma=1,
    cma_lr_adapt=True,
    cma_population_size=100,
    cma_iter=5000,
    cma_crit=10**-6,
    cma_n_crit=200,
    bounds_rho=None,
    bounds_lambda=None,
    random_state=50,
    inits=None,
):
    """Optimize pen. log-lik. of het. skew-normal model via cmaes.

    Parameters
    ----------
    - alpha: penalty coefficient for f (scalar)
    - kappa: penalty coefficient for rho (scalar)
    - x: covariate (n x 1)
    - y: outcome (n x 1)
    - N: design matrix for f (n x p_f)
    - Z: design matrix for rho (n x p_rho)
    - K: penalty matrix for f (p_f x p_f)
    - M: penalty matrix for rho (p_rho x p_rho)
    - cma_sigma: sigma in CMA (positive scalar, default=1)
    - cma_lr_adapt: adaptive learning rate in CMA (boolean, default=True)
    - cma_population_size: CMA population size (integer, default=100)
    - cma_iter: number of CMA iterations (integer, default=5000)
    - cma_crit: likelihood stopping crit in CMA (pos. scalar, default=10**-6)
    - cma_n_crit: minimum number of iterations in CMA (integer, default=200)
    - bounds_rho: bounds for rho parameter (NumPy array of shape (p_rho, 2))
    - bounds_lambda: bounds for lambda parameter (NumPy array of shape (1, 2))
    - random_state: determines RNG (integer)
    - inits: optional initial parameter vector (flattened + concatenated)


    Returns
    -------
    - dictionary with estimated and fixed model parameters and
      optimization statistics
    """
    if bounds_rho is None:
        bounds_rho = np.array([[-10, 5]] * Z.shape[1])

    if bounds_lambda is None:
        bounds_lambda = np.array([[-25, 25]])

    if inits is None:
        initial_f = sm.OLS(y, N).fit().params.reshape(-1, 1)
        initial_rho = np.zeros(Z.shape[1]).reshape(-1, 1)
        initial_lambda_hat = skewCoeff1D(y)
        initial_params = np.concatenate(
            [
                initial_f.flatten(),
                initial_rho.flatten(),
                initial_lambda_hat.flatten(),
            ]
        )
    else:
        initial_params = inits

    optimal_params = initial_params
    best_fitness = sn_nloglikP(
        y=y,
        N=N,
        Z=Z,
        K=K,
        alpha=alpha,
        M=M,
        kappa=kappa,
        params=optimal_params,
    )
    bounds = np.array([[-np.inf, np.inf]] * len(optimal_params))
    bounds[-1] = bounds_lambda
    for k in range(Z.shape[1]):  # Set bounds for rho
        idx = k + N.shape[1]
        bounds[idx] = bounds_rho[k]
        
    # make sure that the initial lambda value is inside the bounds:
    optimal_params[-1] = np.clip(optimal_params[-1], bounds_lambda[0, 0], bounds_lambda[0, 1]) 

    optimizer = CMA(
        mean=optimal_params,
        sigma=cma_sigma,
        lr_adapt=cma_lr_adapt,
        population_size=cma_population_size,
        bounds=bounds,
        seed=random_state,
    )    
    count = 0
    nlogliks = [
        sn_nloglikP(
            params=optimal_params,
            y=y,
            N=N,
            Z=Z,
            K=K,
            alpha=alpha,
            M=M,
            kappa=kappa,
        )
    ]
    for i in range(cma_iter):
        count = count + 1
        try:
            solutions = []
            for _ in range(optimizer.population_size):
                child = optimizer.ask()
                value = sn_nloglikP(
                    params=child,
                    y=y,
                    N=N,
                    Z=Z,
                    K=K,
                    alpha=alpha,
                    M=M,
                    kappa=kappa,
                )
                solutions.append((child, value))
                if value >= 0 and value < best_fitness:
                    best_fitness = value
                    optimal_params = child
            optimizer.tell(solutions)
            if optimizer.should_stop():
                break
        except Exception:
            break
        nlogliks.append(best_fitness)
        if i > cma_n_crit:
            if abs(nlogliks[i] - nlogliks[i - 200]) < cma_crit:
                break
    return {
        "f": optimal_params[: N.shape[1]].reshape(-1, 1),
        "rho": optimal_params[N.shape[1] : -1].reshape(-1, 1),
        "lambda_hat": optimal_params[-1],
        "x": x,
        "y": y,
        "N": N,
        "Z": Z,
        "K": K,
        "M": M,
        "nlogliks": nlogliks,
        "finalNLL": best_fitness,
        "n_iter": count,
        "alpha": alpha,
        "kappa": kappa,
    }


def boptimSNCMA(
    x,
    y,
    N_knots=12,
    Z_knots=5,
    N_deg=3,
    Z_deg=3,
    cma_sigma=1,
    cma_lr_adapt=True,
    cma_population_size=100,
    cma_iter=5000,
    cma_crit=10**-6,
    cma_n_crit=200,
    bounds_rho=None,
    bounds_lambda=None,
    n_calls_boptim=30,
    n_inits_boptim=20,
    log_bounds_alpha=None,
    log_bounds_kappa=None,
    n_cv_splits_boptim=5,
    random_state=54,
):
    """Find optimal alpha, kappa + model via CMA Bayesian Optimization.

    Parameters
    ----------
    - x: covariate (n x 1)
    - y: outcome (n x 1)
    - N_knots: number of knots for spline matrix N (scalar, default=12)
    - Z_knots: number of knots for spline matrix Z (scalar, default=5)
    - N_deg: degree for spline matrix N (scalar, default=3)
    - Z_deg: degree for spline matrix N (scalar, default=3)
    - cma_sigma: sigma in CMA (positive scalar, default=1)
    - cma_lr_adapt: adaptive learning rate in CMA (boolean, default=True)
    - cma_population_size: CMA population size (integer, default=100)
    - cma_iter: number of CMA iterations (integer, default=5000)
    - cma_crit: likelihood stopping crit in CMA (pos. scalar, default=10**-6)
    - cma_n_crit: minimum number of iterations in CMA (integer, default=200)
    - bounds_rho: bounds for rho parameter (NumPy array of shape (p_rho, 2))
    - bounds_lambda: bounds for lambda parameter (NumPy array of shape (1, 2))
    - n_calls_boptim: n_calls in skopt.gp_minimize (default=30)
    - n_inits_boptim: n_inits in skopt.gp_minimize (default=20)
    - log_bounds_alpha: list [lower bound, upper bound] for alpha
    - log_bounds_kappa: list [lower bound, upper bound] for alpha
    - n_cv_splits: number of CV splits in Bayesian Optimization (default=5)
    - random_state: determines RNG (integer)

    Returns
    -------
    - dictionary with estimated and fixed model parameters and
      optimization statistics
    """
    if bounds_rho is None:
        n_col_Z = Z_knots + Z_deg - 1
        bounds_rho = np.array([[-10, 5]] * n_col_Z)
    if log_bounds_alpha is None:
        log_bounds_alpha = [-4, 4]
    if log_bounds_kappa is None:
        log_bounds_kappa = [-4, 4]

    rng = np.random.default_rng(random_state)

    # Define helper functions:
    def build_NZKM(x_build, N_knots, Z_knots, N_deg, Z_deg):
        # Build spline matrices N and Z
        s = SplineTransformer(
            n_knots=N_knots, degree=N_deg, extrapolation="linear"
        )
        N = s.fit_transform(np.array(x_build).reshape(-1, 1))
        s2 = SplineTransformer(
            n_knots=Z_knots, degree=Z_deg, extrapolation="linear"
        )
        Z = s2.fit_transform(np.array(x_build).reshape(-1, 1))
        D = np.eye(N.shape[1])
        D = np.diff(D, n=2, axis=0)
        K = D.T @ D
        D2 = np.eye(Z.shape[1])
        D2 = np.diff(D2, n=2, axis=0)
        M = D2.T @ D2
        return N, Z, K, M

    N, Z, K, M = build_NZKM(x, N_knots, Z_knots, N_deg, Z_deg)
    saved_cv_results = {}

    def cv_objective(logvalues):
        # Calculate CV neg. log-likelihood given log_alpha, log_kappa.
        log_alpha, log_kappa = logvalues
        alpha = np.exp(log_alpha)
        kappa = np.exp(log_kappa)
        nlls = []
        kf = KFold(
            n_splits=n_cv_splits_boptim,
            shuffle=True,
            random_state=rng.integers(0, 2**32 - 1),
        )
        saved_cv_results[(alpha, kappa)] = []
        for train_idx, val_idx in kf.split(x):
            # Split data
            x_train, _ = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            # Build basis matrices
            N_train = N[train_idx, :]
            Z_train = Z[train_idx, :]
            N_val = N[val_idx, :]
            Z_val = Z[val_idx, :]
            # Fit model using training fold
            result = optimSNCMA(
                alpha,
                kappa,
                x_train,
                y_train,
                N_train,
                Z_train,
                K,
                M,
                cma_sigma=cma_sigma,
                cma_lr_adapt=cma_lr_adapt,
                cma_population_size=cma_population_size,
                cma_iter=cma_iter,
                cma_crit=cma_crit,
                cma_n_crit=cma_n_crit,
                bounds_rho=bounds_rho,
                bounds_lambda=bounds_lambda,
                random_state=rng.integers(0, 2**32 - 1),
                inits=None,
            )
            saved_cv_results[(alpha, kappa)].append(result)
            # Get estimated parameters
            f_hat = result["f"]
            rho_hat = result["rho"]
            lambda_hat = result["lambda_hat"]
            # Compute log-likelihood on validation set
            loglik_val = np.sum(
                np.log(
                    sn_lik(
                        y=y_val,
                        N=N_val,
                        Z=Z_val,
                        f=f_hat,
                        rho=rho_hat,
                        lambda_hat=lambda_hat,
                    )
                )
            )
            nlls.append(-loglik_val)
        return np.mean(nlls)

    # Search space for the parameters on log scale
    space = [
        Real(log_bounds_alpha[0], log_bounds_alpha[1], name="log_alpha"),
        Real(log_bounds_kappa[0], log_bounds_kappa[1], name="log_kappa"),
    ]
    bo_result = gp_minimize(
        func=cv_objective,
        dimensions=space,
        n_calls=n_calls_boptim,
        n_initial_points=n_inits_boptim,
        random_state=rng.integers(0, 2**32 - 1),
        acq_func="EI",
        initial_point_generator="lhs",
    )
    log_alpha_opt, log_kappa_opt = bo_result.x
    alpha_opt = np.exp(log_alpha_opt)
    kappa_opt = np.exp(log_kappa_opt)

    def paramconcat(results):
        return np.concatenate(
            [
                results["f"].flatten(),
                results["rho"].flatten(),
                results["lambda_hat"].flatten(),
            ]
        )

    inits = [None]
    for scr in saved_cv_results[(alpha_opt, kappa_opt)]:
        inits.append(paramconcat(scr))

    results = []
    for i in inits:
        res = optimSNCMA(
            alpha_opt,
            kappa_opt,
            x,
            y,
            N,
            Z,
            K,
            M,
            cma_sigma=cma_sigma,
            cma_lr_adapt=cma_lr_adapt,
            cma_population_size=cma_population_size,
            cma_iter=cma_iter,
            cma_crit=cma_crit,
            cma_n_crit=cma_n_crit,
            bounds_rho=bounds_rho,
            random_state=rng.integers(0, 2**32 - 1),
            inits=i,
            bounds_lambda=bounds_lambda,
        )
        results.append(res)

    best_result = min(results, key=lambda d: d["finalNLL"])
    return best_result
