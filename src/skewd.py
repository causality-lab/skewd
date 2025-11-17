"""Skewness-Robust Causal Inference in Het. Skew-Normal Models."""

from src.sncma import boptimSNCMA
from src.snecm import SNECM
from src.hsic import HSIC
import numpy as np
from src.snlik import m, sn_lik


def fithetSN(
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
    ecm_tol=10**-6,
    ecm_n_crit=25,
    ecm_max_iter=5000,
    ecm_max_iter_rho=100,
    ecm_track_history=False,
):
    """Fit heteroscedastic skew-normal model.

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
    - cma_crit: likelihood stopping criterion in CMA
      (positive scalar, default=10**-6)
    - cma_n_crit: minimum number of iterations in CMA
      (integer, default=200)
    - bounds_rho: bounds for rho parameter
      (NumPy array of shape (p_rho, 2))
    - bounds_lambda: bounds for lambda parameter in CMA
      (NumPy array of shape (1,2))
    - n_calls_boptim: n_calls in skopt.gp_minimize (default=30)
    - n_inits_boptim: n_inits in skopt.gp_minimize (default=20)
    - log_bounds_alpha: list [lower bound, upper bound] for alpha
    - log_bounds_kappa: list [lower bound, upper bound] for alpha
    - n_cv_splits_boptim: number of CV splits in Bayesian Optimization
      (default=5)
    - random_state: determines RNG (integer)
    - ecm_tol: ECM stopping criterion tolerance
      (positive scalar, default=10**-6)
    - ecm_n_crit: minimum number of iterations before
      stopping criterion is checked in ECM (integer, default=25),
    - ecm_max_iter: ECM maximum number of iterations
      (integer, default=5000)
    - ecm_max_iter_rho: ECM maximum number of iterations in
      optimizing rho (integer, default=100)
    - ecm_track_history: whether to track parameter updates in ECM
      (boolean, default=False)

    Returns
    -------
    - dictionary with estimated and fixed model parameters and
      optimization statistics from SNECM
    - dictionary with estimated and fixed model parameters and
      optimization statistics from boptimSNCMA
    """
    # Find optimal alpha, kappa via boptim + good initialization
    result_cma = boptimSNCMA(
        x,
        y,
        N_knots=N_knots,
        Z_knots=Z_knots,
        N_deg=N_deg,
        Z_deg=Z_deg,
        cma_sigma=cma_sigma,
        cma_lr_adapt=cma_lr_adapt,
        cma_population_size=cma_population_size,
        cma_iter=cma_iter,
        cma_crit=cma_crit,
        cma_n_crit=cma_n_crit,
        bounds_rho=bounds_rho,
        bounds_lambda=bounds_lambda,
        n_calls_boptim=n_calls_boptim,
        n_inits_boptim=n_inits_boptim,
        log_bounds_alpha=log_bounds_alpha,
        log_bounds_kappa=log_bounds_kappa,
        n_cv_splits_boptim=n_cv_splits_boptim,
        random_state=random_state,
    )
    # Use result from cmaes and improve via ECM algorithm
    result_ECM = SNECM(
        x,
        y,
        initial_f=result_cma["f"],
        initial_rho=result_cma["rho"],
        initial_lambda_hat=result_cma["lambda_hat"],
        N=result_cma["N"],
        Z=result_cma["Z"],
        K=result_cma["K"],
        M=result_cma["M"],
        alpha=result_cma["alpha"],
        kappa=result_cma["kappa"],
        tol=ecm_tol,
        ecm_n_crit=ecm_n_crit,
        max_iter=ecm_max_iter,
        max_iter_rho=ecm_max_iter_rho,
        bounds_rho=bounds_rho,
        track_history=ecm_track_history,
        random_state=random_state,
    )
    # return result
    return result_ECM, result_cma


def skewd(
    x,
    y,
    rng_seed=192,
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
    ecm_tol=10**-6,
    ecm_n_crit=25,
    ecm_max_iter=5000,
    ecm_max_iter_rho=100,
    ecm_track_history=False,
):
    """Run Skewness-Robust Causal Inference.

    Parameters
    ----------
    - x: covariate (n x 1 NumPy array)
    - y: outcome (n x 1 NumPy array)
    - rng_seed: determines RNG (integer)
    - N_knots: number of knots for spline matrix N (scalar, default=12)
    - Z_knots: number of knots for spline matrix Z (scalar, default=5)
    - N_deg: degree for spline matrix N (scalar, default=3)
    - Z_deg: degree for spline matrix N (scalar, default=3)
    - cma_sigma: sigma in CMA (positive scalar, default=1)
    - cma_lr_adapt: adaptive learning rate in CMA (boolean, default=True)
    - cma_population_size: CMA population size (integer, default=100)
    - cma_iter: number of CMA iterations (integer, default=5000)
    - cma_crit: likelihood stopping criterion in CMA
      (positive scalar, default=10**-6)
    - cma_n_crit: minimum number of iterations in CMA
      (integer, default=200)
    - bounds_rho: bounds for rho parameter
      (NumPy array of shape (p_rho, 2))
    - bounds_lambda: bounds for lambda parameter in CMA
      (NumPy array of shape (1,2))
    - n_calls_boptim: n_calls in skopt.gp_minimize (default=30)
    - n_inits_boptim: n_inits in skopt.gp_minimize (default=20)
    - log_bounds_alpha: list [lower bound, upper bound] for alpha
    - log_bounds_kappa: list [lower bound, upper bound] for alpha
    - n_cv_splits_boptim: number of CV splits for training/test in
      Bayesian Optimization (default=5)
    - ecm_tol: ECM stopping criterion tolerance
      (positive scalar, default=10**-6)
    - ecm_n_crit: minimum number of iterations before
      stopping criterion is checked in ECM (integer, default=25),
    - ecm_max_iter: ECM maximum number of iterations
      (integer, default=5000)
    - ecm_max_iter_rho: ECM maximum number of iterations in
      optimizing rho (integer, default=100)
    - ecm_track_history: whether to track parameter updates in ECM
      (boolean, default=False)

    Returns
    -------
    - independence test score
    - independence test direction
    - likelihood score
    - likelihood direction
      (result only valid when covariate is assumed normal)
    - SNECM forward model output
    - SNECM reverse model output
    - boptimSNCMA forward model output
    - boptimSNCMA reverse model output
    """
    # Construct random seeds based on rng_seed
    master_rng = np.random.default_rng(rng_seed)
    rs1 = master_rng.integers(0, 2**32 - 1)
    rs2 = master_rng.integers(0, 2**32 - 1)
    # Call fithetSN forward
    res_FW, res_FW_cma = fithetSN(
        x=x,
        y=y,
        N_knots=N_knots,
        Z_knots=Z_knots,
        N_deg=N_deg,
        Z_deg=Z_deg,
        cma_sigma=cma_sigma,
        cma_lr_adapt=cma_lr_adapt,
        cma_population_size=cma_population_size,
        cma_iter=cma_iter,
        cma_crit=cma_crit,
        cma_n_crit=cma_n_crit,
        bounds_rho=bounds_rho,
        bounds_lambda=bounds_lambda,
        n_calls_boptim=n_calls_boptim,
        n_inits_boptim=n_inits_boptim,
        log_bounds_alpha=log_bounds_alpha,
        log_bounds_kappa=log_bounds_kappa,
        n_cv_splits_boptim=n_cv_splits_boptim,
        random_state=rs1,
        ecm_tol=ecm_tol,
        ecm_n_crit=ecm_n_crit,
        ecm_max_iter=ecm_max_iter,
        ecm_max_iter_rho=ecm_max_iter_rho,
        ecm_track_history=ecm_track_history,
    )
    # Call fithetSN reverse
    res_RV, res_RV_cma = fithetSN(
        x=y,
        y=x,
        N_knots=N_knots,
        Z_knots=Z_knots,
        N_deg=N_deg,
        Z_deg=Z_deg,
        cma_sigma=cma_sigma,
        cma_lr_adapt=cma_lr_adapt,
        cma_population_size=cma_population_size,
        cma_iter=cma_iter,
        cma_crit=cma_crit,
        cma_n_crit=cma_n_crit,
        bounds_rho=bounds_rho,
        bounds_lambda=bounds_lambda,
        n_calls_boptim=n_calls_boptim,
        n_inits_boptim=n_inits_boptim,
        log_bounds_alpha=log_bounds_alpha,
        log_bounds_kappa=log_bounds_kappa,
        n_cv_splits_boptim=n_cv_splits_boptim,
        random_state=rs2,
        ecm_tol=ecm_tol,
        ecm_n_crit=ecm_n_crit,
        ecm_max_iter=ecm_max_iter,
        ecm_max_iter_rho=ecm_max_iter_rho,
        ecm_track_history=ecm_track_history,
    )
    # Calculate HSIC scores to infer causal direction
    my = res_FW["N"] @ res_FW["f"]
    sy = np.sqrt(m(res_FW["rho"], res_FW["Z"]))
    indep_FW = HSIC(x, (y - my) / sy)
    mx = res_RV["N"] @ res_RV["f"]
    sx = np.sqrt(m(res_RV["rho"], res_RV["Z"]))
    indep_RV = HSIC(y, (x - mx) / sx)
    indep_score = indep_RV - indep_FW
    indep_dir = np.where(indep_score > 0, 1, 0)
    # Calculate likelihoods score to infer causal direction
    loglikFW = np.sum(
        np.log(
            sn_lik(
                y=res_FW["y"],
                N=res_FW["N"],
                Z=res_FW["Z"],
                f=res_FW["f"],
                rho=res_FW["rho"],
                lambda_hat=res_FW["lambda_hat"],
            )
        )
    )
    loglikRV = np.sum(
        np.log(
            sn_lik(
                y=res_RV["y"],
                N=res_RV["N"],
                Z=res_RV["Z"],
                f=res_RV["f"],
                rho=res_RV["rho"],
                lambda_hat=res_RV["lambda_hat"],
            )
        )
    )
    lik_score = loglikFW - loglikRV
    lik_dir = np.where(lik_score > 0, 1, 0)
    # return likdir, likscore, indepdir, indepscore,
    # resultFW, resultRV, res_FW_cma, res_RV_cma
    return (
        indep_score,
        indep_dir,
        lik_score,
        lik_dir,
        res_FW,
        res_RV,
        res_FW_cma,
        res_RV_cma,
    )
