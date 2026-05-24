"""ECM algorithm for heteroscedastic skew-normal model."""

import numpy as np
from src.snlik import m, sn_nloglikP, sn_nloglikP_MV
from src.utils import safe_solve
from scipy.stats import norm
from cmaes import CMA
from scipy.linalg import block_diag


def SNECM(
    x,
    y,
    initial_f,
    initial_rho,
    initial_lambda_hat,
    N,
    Z,
    K,
    M,
    alpha,
    kappa,
    tol=10**-6,
    ecm_n_crit=25,
    max_iter=5000,
    max_iter_rho=100,
    bounds_rho=None,
    track_history=False,
    random_state=54,
):
    """Perform ECM algorithm for heteroscedastic skew-normal model.

    Parameters
    ----------
    - x: covariate(s) (n x d, d=1 bivariate)
    - y: outcome (n x 1)
    - N: design matrix for f (n x p_f)
    - Z: design matrix for rho (n x p_rho)
    - K: penalty matrix for f (p_f x p_f)
    - M: penalty matrix for rho (p_rho x p_rho)
    - alpha: penalty coefficient for f (scalar)
    - kappa: penalty coefficient for rho (scalar)
    - tol: stopping crit. tolerance (positive scalar, default=10**-6)
    - ecm_n_crit: minimum number of iterations before
      stopping criterion is checked (integer, default=25),
    - max_iter: maximum number of iterations (integer, default=5000)
    - max_iter_rho: maximum number of iterations in optimizing rho
      (integer, default=100)
    - bounds_rho: bounds for rho parameter
      (NumPy array of shape (p_rho, 2))
    - track_history: whether to track parameter updates
      (boolean, default=False)
    - random_state: determines RNG (integer)

    Returns
    -------
    - dictionary with estimated and fixed model parameters and
      optimization statistics
    """
    # Define helper functions:
    def w_phi(u):
        u = np.maximum(u, -35)
        return norm.pdf(u) / norm.cdf(u)

    def Qfunc(rho, f, lambda_hat, w_i, w_i2, y, Z, N, M, kappa):
        n = len(y)
        rho = rho.reshape((Z.shape[1], 1))
        m_i = np.maximum(10**-5, m(rho, Z))
        H = np.diag(1 / m_i.flatten())
        Q1 = -np.sum(np.log(m_i))
        auxQ = y - N @ f
        Q2 = -(1 + lambda_hat**2) / 2 * auxQ.T @ H @ auxQ
        Q3 = -1 / 2 * w_i2.T @ H @ np.ones((n, 1))
        Q4 = lambda_hat * auxQ.T @ H @ w_i
        Q5 = -kappa / 2 * rho.T @ M @ rho
        return -1 * (Q1 + Q2 + Q3 + Q4 + Q5)

    f = initial_f
    rho = initial_rho
    lambda_hat = initial_lambda_hat
    p_star = rho.shape[0]
    if bounds_rho is None:
        bounds_rho = np.array([[-10, 5]] * p_star)
    crit = np.inf
    count = 0
    parameter_history = {"f": [f], "lambda_hat": [lambda_hat], "rho": [rho]}
    initial_parameters = parameter_history
    nlogliksP = [
        sn_nloglikP(
            y=y,
            N=N,
            Z=Z,
            K=K,
            alpha=alpha,
            M=M,
            kappa=kappa,
            f=f,
            rho=rho,
            lambda_hat=lambda_hat,
        )
    ]

    rng = np.random.default_rng(random_state)

    while count < max_iter:
        # 1) E-Step
        e_i = y - N @ f
        sigma2_i = m(rho, Z)
        sigma_i = np.sqrt(sigma2_i)
        aux = w_phi(lambda_hat * e_i / sigma_i)
        w_i = lambda_hat * e_i + sigma_i * aux
        w_i2 = (
            (lambda_hat * e_i) ** 2
            + sigma2_i
            + lambda_hat * sigma_i * e_i * aux
        )

        # 2) CM - step 1
        m_i = np.maximum(10**-5, m(rho, Z))
        H = np.diag(1 / m_i.flatten())
        mu = N @ f
        S = (y - mu).T @ H @ (y - mu)
        f = safe_solve(
            N.T @ H @ N + alpha / (1 + (lambda_hat) ** 2) * K,
            N.T @ H @ (y - lambda_hat / (1 + lambda_hat**2) * w_i),
        )
        lambda_hat = w_i.T @ H @ (y - mu) / S

        # 3) CM - step 2
        # cmaes
        cma_seed = rng.integers(0, 2**32 - 1)
        best_fitness = Qfunc(
            rho=rho,
            f=f,
            lambda_hat=lambda_hat,
            w_i=w_i,
            w_i2=w_i2,
            y=y,
            Z=Z,
            N=N,
            M=M,
            kappa=kappa,
        )
        optimizer = CMA(
            mean=rho.flatten(),
            sigma=1,
            lr_adapt=True,
            bounds=bounds_rho,
            seed=cma_seed,
        )
        for _ in range(max_iter_rho):
            solutions = []
            for _ in range(optimizer.population_size):
                child = optimizer.ask()
                value = Qfunc(
                    rho=child.reshape((p_star, 1)),
                    f=f,
                    lambda_hat=lambda_hat,
                    w_i=w_i,
                    w_i2=w_i2,
                    y=y,
                    Z=Z,
                    N=N,
                    M=M,
                    kappa=kappa,
                )
                solutions.append((child, value))
                if value < best_fitness:
                    best_fitness = value
                    rho = child.reshape((p_star, 1))
            optimizer.tell(solutions)
            if optimizer.should_stop():
                break

        parameter_history["f"].append(f)
        parameter_history["lambda_hat"].append(lambda_hat)
        parameter_history["rho"].append(rho)
        nlogliksP.append(
            sn_nloglikP(
                y=y,
                N=N,
                Z=Z,
                K=K,
                alpha=alpha,
                M=M,
                kappa=kappa,
                f=f,
                rho=rho,
                lambda_hat=lambda_hat,
            )
        )

        count += 1
        if count > ecm_n_crit:
            sqsum = 0
            for param in ["f", "lambda_hat", "rho"]:
                s1 = parameter_history[param][count]
                s2 = parameter_history[param][count - ecm_n_crit]
                diff = s1 - s2
                sqsum += np.sum(np.abs(diff))
            crit = np.sqrt(sqsum)
            if crit < tol:
                break
    if not track_history:
        parameter_history = None

    results = {
        "f": f,
        "lambda_hat": lambda_hat,
        "rho": rho,
        "alpha": alpha,
        "history": parameter_history,
        "x": x,
        "y": y,
        "N": N,
        "Z": Z,
        "K": K,
        "M": M,
        "kappa": kappa,
        "nloglik": nlogliksP[-1],
        "n_iter": count,
        "initial_parameters": initial_parameters,
    }
    return results


def MVSNECM(
    x,
    y,
    initial_f,
    initial_rho,
    initial_lambda_hat,
    N,
    Z,
    K_list,
    M_list,
    alpha_list,
    kappa_list,
    tol=10**-6,
    ecm_n_crit=25,
    max_iter=5000,
    max_iter_rho=100,
    bounds_rho=None,
    track_history=False,
    random_state=54,
):
    """Perform ECM algorithm for heteroscedastic skew-normal model.

    Parameters
    ----------
    - x: covariate(s) (n x d, d=1 bivariate)
    - y: outcome (n x 1)
    - N: design matrix for f (n x p_f)
    - Z: design matrix for rho (n x p_rho)
    - K_list: list of penalty matrices for f, one per covariate (p_f x p_f)
    - M_list: list of penalty matrices for rho, one per covariate (p_rho x p_rho)
    - alpha_list: penalty coefficient(s) for f (list of scalars)
    - kappa_list: penalty coefficient(s) for rho (list of scalars)
    - tol: stopping crit. tolerance (positive scalar, default=10**-6)
    - ecm_n_crit: minimum number of iterations before
      stopping criterion is checked (integer, default=25),
    - max_iter: maximum number of iterations (integer, default=5000)
    - max_iter_rho: maximum number of iterations in optimizing rho
      (integer, default=100)
    - bounds_rho: bounds for rho parameter
      (NumPy array of shape (p_rho, 2))
    - track_history: whether to track parameter updates
      (boolean, default=False)
    - random_state: determines RNG (integer)

    Returns
    -------
    - dictionary with estimated and fixed model parameters and
      optimization statistics
    """
    # Define helper functions:
    def w_phi(u):
        u = np.maximum(u, -35)
        return norm.pdf(u) / norm.cdf(u)

    def Qfunc(rho, f, lambda_hat, w_i, w_i2, y, Z, N, M_kappa):
        n = len(y)
        rho = rho.reshape((Z.shape[1], 1))
        m_i = np.maximum(10**-5, m(rho, Z))
        H = np.diag(1 / m_i.flatten())
        Q1 = -np.sum(np.log(m_i))
        auxQ = y - N @ f
        Q2 = -(1 + lambda_hat**2) / 2 * auxQ.T @ H @ auxQ
        Q3 = -1 / 2 * w_i2.T @ H @ np.ones((n, 1))
        Q4 = lambda_hat * auxQ.T @ H @ w_i
        Q5 = -1 / 2 * rho.T @ M_kappa @ rho
        return -1 * (Q1 + Q2 + Q3 + Q4 + Q5)

    f = initial_f
    rho = initial_rho
    lambda_hat = initial_lambda_hat
    p_star = rho.shape[0]
    if bounds_rho is None:
        bounds_rho = np.array([[-10, 5]] * p_star)
    crit = np.inf
    count = 0
    parameter_history = {"f": [f], "lambda_hat": [lambda_hat], "rho": [rho]}
    initial_parameters = parameter_history
    nlogliksP = [
        sn_nloglikP_MV(
            y=y,
            N=N,
            Z=Z,
            K_list=K_list,
            alpha_list=alpha_list,
            M_list=M_list,
            kappa_list=kappa_list,
            f=f,
            rho=rho,
            lambda_hat=lambda_hat,
        )
    ]

    rng = np.random.default_rng(random_state)

    # Determine K_alpha and M_kappa:
    K_alpha_list = [alpha_list[j] * K_list[j] for j in range(len(K_list))]
    M_kappa_list = [kappa_list[j] * M_list[j] for j in range(len(M_list))]

    K_alpha = block_diag(*K_alpha_list)
    M_kappa = block_diag(*M_kappa_list)

    while count < max_iter:
        # 1) E-Step
        e_i = y - N @ f
        sigma2_i = m(rho, Z)
        sigma_i = np.sqrt(sigma2_i)
        aux = w_phi(lambda_hat * e_i / sigma_i)
        w_i = lambda_hat * e_i + sigma_i * aux
        w_i2 = (
            (lambda_hat * e_i) ** 2
            + sigma2_i
            + lambda_hat * sigma_i * e_i * aux
        )

        # 2) CM - step 1
        m_i = np.maximum(10**-5, m(rho, Z))
        H = np.diag(1 / m_i.flatten())
        mu = N @ f
        S = (y - mu).T @ H @ (y - mu)
        f = safe_solve(
            N.T @ H @ N + 1 / (1 + (lambda_hat) ** 2) * K_alpha,
            N.T @ H @ (y - lambda_hat / (1 + lambda_hat**2) * w_i),
        )
        lambda_hat = w_i.T @ H @ (y - mu) / S

        # 3) CM - step 2
        # cmaes
        cma_seed = rng.integers(0, 2**32 - 1)
        best_fitness = Qfunc(
            rho=rho,
            f=f,
            lambda_hat=lambda_hat,
            w_i=w_i,
            w_i2=w_i2,
            y=y,
            Z=Z,
            N=N,
            M_kappa=M_kappa
        )
        optimizer = CMA(
            mean=rho.flatten(),
            sigma=1,
            lr_adapt=True,
            bounds=bounds_rho,
            seed=cma_seed,
        )
        for _ in range(max_iter_rho):
            solutions = []
            for _ in range(optimizer.population_size):
                child = optimizer.ask()
                value = Qfunc(
                    rho=child.reshape((p_star, 1)),
                    f=f,
                    lambda_hat=lambda_hat,
                    w_i=w_i,
                    w_i2=w_i2,
                    y=y,
                    Z=Z,
                    N=N,
                    M_kappa=M_kappa,
                )
                solutions.append((child, value))
                if value < best_fitness:
                    best_fitness = value
                    rho = child.reshape((p_star, 1))
            optimizer.tell(solutions)
            if optimizer.should_stop():
                break

        parameter_history["f"].append(f)
        parameter_history["lambda_hat"].append(lambda_hat)
        parameter_history["rho"].append(rho)
        nlogliksP.append(
            sn_nloglikP_MV(
                y=y,
                N=N,
                Z=Z,
                K_list=K_list,
                alpha_list=alpha_list,
                M_list=M_list,
                kappa_list=kappa_list,
                f=f,
                rho=rho,
                lambda_hat=lambda_hat,
            )
        )

        count += 1
        if count > ecm_n_crit:
            sqsum = 0
            for param in ["f", "lambda_hat", "rho"]:
                s1 = parameter_history[param][count]
                s2 = parameter_history[param][count - ecm_n_crit]
                diff = s1 - s2
                sqsum += np.sum(np.abs(diff))
            crit = np.sqrt(sqsum)
            if crit < tol:
                break
    if not track_history:
        parameter_history = None

    results = {
        "f": f,
        "lambda_hat": lambda_hat,
        "rho": rho,
        "alpha_list": alpha_list,
        "history": parameter_history,
        "x": x,
        "y": y,
        "N": N,
        "Z": Z,
        "K_list": K_list,
        "M_list": M_list,
        "kappa_list": kappa_list,
        "nloglik": nlogliksP[-1],
        "n_iter": count,
        "initial_parameters": initial_parameters,
    }
    return results