"""Multivariate Skewness-Robust Causal Discovery in Skew-Normal LSNMs."""

from causallearn.search.ConstraintBased.PC import pc
from src.graphutils import transform_adj, create_candidates, identify_parents
import numpy as np
from src.utils import estimate_KDE
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from src.sncma import boptimSNCMA
from src.snecm import MVSNECM
from pathos.multiprocessing import ProcessingPool as Pool
from src.snlik import sn_lik

def estimate_MVhetSN(y, 
                     pa_y, # pa_x should be input in matrix form i.e. shape (n, d)
                     N_knots=8, 
                     Z_knots=5, 
                     N_deg=3,
                     Z_deg=3,
                     cma_sigma=1,
                     cma_lr_adapt=True,
                     cma_population_size=100,
                     cma_iter=3000,
                     cma_crit=10**-5,
                     cma_n_crit=200,
                     bounds_rho=None,
                     bounds_lambda=None,
                     n_calls_boptim=20,
                     n_inits_boptim=10,
                     log_bounds_alpha=None,
                     log_bounds_kappa=None,
                     n_cv_splits_boptim=5,
                     random_state=54,
                     ecm_tol=10**-5,
                     ecm_n_crit=25,
                     ecm_max_iter=3000,
                     ecm_max_iter_rho=50,
                     ecm_track_history=False):
    """Estimate conditional loglik in multivariate het. SN LSNM based on additive splines.

    Parameters
    ----------
    - y: outcome (n x 1)
    - pa_y: outcome (n x d)
    - N_knots: number of knots for each variable's spline matrix N (scalar, default=8)
    - Z_knots: number of knots for each variable's spline matrix Z (scalar, default=5)
    - N_deg: degree for spline matrices N (scalar, default=3)
    - Z_deg: degree for spline matrices N (scalar, default=3)
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
    - log_bounds_alpha: list [lower bound, upper bound] for alpha parameters
    - log_bounds_kappa: list [lower bound, upper bound] for kappa parameters
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
    - loglik of skew-normal LSNM pa_y -> y
    """
    # Bayesian optimization
    result_cma = boptimSNCMA(
        x=pa_y,
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
        random_state=random_state,
        multivariate=True
    )
    # ECM algorithm: 
    result_ECM = MVSNECM(
        x=pa_y,
        y=y,
        initial_f=result_cma["f"],
        initial_rho=result_cma["rho"],
        initial_lambda_hat=result_cma["lambda_hat"],
        N=result_cma["N"],
        Z=result_cma["Z"],
        K_list=result_cma["K_list"],
        M_list=result_cma["M_list"],
        alpha_list=result_cma["alpha_list"],
        kappa_list=result_cma["kappa_list"],
        tol=ecm_tol,
        ecm_n_crit=ecm_n_crit,
        max_iter=ecm_max_iter,
        max_iter_rho=ecm_max_iter_rho,
        bounds_rho=bounds_rho,
        track_history=ecm_track_history,
        random_state=random_state,
    )
    # Return only the loglik
    loglik = np.sum(
        np.log(
            sn_lik(
                y=result_ECM["y"],
                N=result_ECM["N"],
                Z=result_ECM["Z"],
                f=result_ECM["f"],
                rho=result_ECM["rho"],
                lambda_hat=result_ECM["lambda_hat"],
            )
        )
    )
    return loglik

def skewdmv(X,
    rng_seed=192,
    N_knots=8,
    Z_knots=5,
    N_deg=3,
    Z_deg=3,
    alpha_pc=0.05,
    indep_test_pc="kci",
    oracle_MEC = None, 
    cma_sigma=1,
    cma_lr_adapt=True,
    cma_population_size=100,
    cma_iter=3000,
    cma_crit=10**-5,
    cma_n_crit=200,
    bounds_rho=None,
    bounds_lambda=None,
    n_calls_boptim=20,
    n_inits_boptim=10,
    log_bounds_alpha=None,
    log_bounds_kappa=None,
    n_cv_splits_boptim=5,
    ecm_tol=10**-5,
    ecm_n_crit=25,
    ecm_max_iter=3000,
    ecm_max_iter_rho=50,
    ecm_track_history=False,
    n_cores_max = 1
    ):
    """Run Multivariate Skewness-Robust Causal Inference.

    Parameters
    ----------
    - X: observed variables (n x d ndarray)
    - rng_seed: determines RNG (integer)
    - N_knots: number of knots for each variable's spline matrix N (scalar, default=8)
    - Z_knots: number of knots for each variable's spline matrix Z (scalar, default=5)
    - N_deg: degree for spline matrices N (scalar, default=3)
    - Z_deg: degree for spline matrices N (scalar, default=3)
    - alpha_pc: significance level for independence test in PC algorithm (default=0.05)
    - indep_test_pc: (default="kci")
    - oracle_MEC: oracle Markov equivalence class adjacency matrix defined as in 
      graphutils.transform_adj. If None the PC algorithm estimates the MEC (default=None)
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
    - log_bounds_alpha: list [lower bound, upper bound] for alpha parameters
    - log_bounds_kappa: list [lower bound, upper bound] for kappa parameters
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
    - n_cores_max: maximum number of cores used in parallelization (default=1)

    Returns
    -------
    - predicted DAG when using KDE
    - predicted DAG under normal assumption 
    - score when using KDE
    - score under normal assumption
    - Markov equivalence class
    """
    # Construct random seed
    master_rng = np.random.default_rng(rng_seed)
    rs1 = master_rng.integers(0, 2**32 - 1)
    # Standardize
    X = StandardScaler().fit_transform(X)

    if oracle_MEC is None:
        # Perform PC algorithm
        result = pc(data=X, alpha=alpha_pc, indep_test=indep_test_pc, stable=True)
        MEC, undirected_pairs = transform_adj(result.G.graph)
    else:
        # Find undirected pairs based on oracle adjacency matrix
        MEC = oracle_MEC
        undirected_pairs = list()
        for i in range(MEC.shape[0]):
            for j in range(i+1, MEC.shape[0]):
                if MEC[i,j] == -1:
                    undirected_pairs.append((i,j)) 

    
    # Compute candidate DAGs
    C = create_candidates(MEC, undirected_pairs)
    # Dictionaries with loglik contributions as values
    # marginals estimated by KDE:
    loglik_dict_KDE = {r: 0 for r in range(len(C))}
    loglik_dict_N = {r: 0 for r in range(len(C))}

    incomplete_nodes = set() # nodes that have an undirected edge
    for i, j in undirected_pairs:
        incomplete_nodes.add(i)
        incomplete_nodes.add(j)

    incomplete_nodes = sorted(incomplete_nodes)

    if n_cores_max == 1:
        for y in incomplete_nodes: 
            # Look up tables for p(x|pa_x) for the nonparametric and normal marginal assumption
            lookup_y_KDE = {} # lookup dictionary with KDE marginals
            lookup_y_N = {} # lookup dictionary with normal marginals
            for r in range(len(C)):
                C_r = C[r]
                pa_y = identify_parents(y, C_r)
                key = tuple(sorted(pa_y)) 
                # Check for contributions that were already calculated
                if key not in lookup_y_KDE.keys(): 
                    if len(pa_y) == 0: # Check whether pa_x is empty
                        # Estimate log marginal through KDE
                        contrib_KDE = estimate_KDE(X[:,y].reshape((-1,1)))
                        # Estimate log marginal under normal assumption, due to standardization N(0,1)
                        contrib_N = np.sum(norm.logpdf(X[:,y].reshape(-1), loc=0, scale=1))
                        lookup_y_KDE[key] = contrib_KDE
                        lookup_y_N[key] = contrib_N
                    else:
                        # Estimate conditional logliks
                        contrib = estimate_MVhetSN(
                        y=X[:,y].reshape((-1,1)), 
                        pa_y=X[:,pa_y],
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
                        ecm_track_history=ecm_track_history                  
                        )
                        # Save newly calculated contribution
                        lookup_y_KDE[key] = contrib
                        lookup_y_N[key] = contrib
                        contrib_KDE = contrib
                        contrib_N = contrib
                else:
                    contrib_KDE = lookup_y_KDE[key]
                    contrib_N = lookup_y_N[key]
                # Add up loglik contribution
                loglik_dict_KDE[r] += contrib_KDE
                loglik_dict_N[r] += contrib_N
    else:
        def process_single_y(y, C, X, N_knots, Z_knots, N_deg, Z_deg, 
            cma_sigma, cma_lr_adapt, cma_population_size,
            cma_iter, cma_crit, cma_n_crit, bounds_rho, bounds_lambda,
            n_calls_boptim, n_inits_boptim, log_bounds_alpha,
            log_bounds_kappa, n_cv_splits_boptim, rs1,
            ecm_tol, ecm_n_crit, ecm_max_iter, ecm_max_iter_rho,
            ecm_track_history):
        
            lookup_y_KDE = {}
            lookup_y_N = {}

            R = len(C)
            contribs_KDE = [0.0] * R
            contribs_N   = [0.0] * R

            for r in range(R):
                C_r = C[r]
                pa_y = identify_parents(y, C_r)
                key = tuple(sorted(pa_y))

                if key not in lookup_y_KDE:
                    if len(pa_y) == 0:
                        # KDE marginal
                        contrib_KDE = estimate_KDE(X[:, y].reshape(-1, 1))
                        contrib_N = np.sum(norm.logpdf(X[:, y], 0, 1))
                    else:
                        # conditional loglik
                        contrib = estimate_MVhetSN(
                            y=X[:,y].reshape((-1,1)), 
                            pa_y=X[:,pa_y],
                            N_knots=N_knots, Z_knots=Z_knots,
                            N_deg=N_deg, Z_deg=Z_deg,
                            cma_sigma=cma_sigma, cma_lr_adapt=cma_lr_adapt,
                            cma_population_size=cma_population_size,
                            cma_iter=cma_iter, cma_crit=cma_crit,
                            cma_n_crit=cma_n_crit,
                            bounds_rho=bounds_rho, bounds_lambda=bounds_lambda,
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
                            ecm_track_history=ecm_track_history
                        )
                        contrib_KDE = contrib
                        contrib_N = contrib
                    lookup_y_KDE[key] = contrib_KDE
                    lookup_y_N[key] = contrib_N
                    

                contribs_KDE[r] = lookup_y_KDE[key]
                contribs_N[r] = lookup_y_N[key]

            return contribs_KDE, contribs_N
    
        tasks = [
            (
                y, C, X, 
                N_knots, Z_knots, N_deg, Z_deg,
                cma_sigma, cma_lr_adapt, cma_population_size,
                cma_iter, cma_crit, cma_n_crit, bounds_rho, bounds_lambda,
                n_calls_boptim, n_inits_boptim, log_bounds_alpha,
                log_bounds_kappa, n_cv_splits_boptim, rs1,
                ecm_tol, ecm_n_crit, ecm_max_iter, ecm_max_iter_rho,
                ecm_track_history,
            )
                for y in incomplete_nodes
            ]

        with Pool() as pool:
            #results = pool.starmap(process_single_y, tasks)  
            results = pool.map(lambda args: process_single_y(*args), tasks)

        loglik_dict_KDE2 = np.zeros(len(C))
        loglik_dict_N2   = np.zeros(len(C))

        for contribs_KDE, contribs_N in results:
            loglik_dict_KDE2 += np.array(contribs_KDE)
            loglik_dict_N2   += np.array(contribs_N)

        loglik_dict_KDE = {i: float(loglik_dict_KDE2[i]) for i in range(len(C))}
        loglik_dict_N   = {i: float(loglik_dict_N2[i])   for i in range(len(C))}


    best_graph_idx_KDE = max(loglik_dict_KDE, key=loglik_dict_KDE.get)
    best_graph_KDE = C[best_graph_idx_KDE]
    score_KDE = loglik_dict_KDE[best_graph_idx_KDE]   

    best_graph_idx_N = max(loglik_dict_N, key=loglik_dict_N.get)
    best_graph_N = C[best_graph_idx_N]   
    score_N = loglik_dict_N[best_graph_idx_N]   

    return best_graph_KDE, score_KDE, best_graph_N, score_N, MEC         
