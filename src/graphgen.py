"""Utility functions for simulating data and evaluating causal discovery algorithms."""

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
import igraph as ig
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from cdt.metrics import SHD
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import skewnorm


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W):
    """Check if W is a DAG."""
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        """Randomly permute the rows and columns of M."""
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        """Randomly orient edges of undirected graph B_und to make it acyclic."""
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        """Convert igraph graph to adjacency matrix."""
        return np.array(G.get_adjacency().data)

    if graph_type == "ER":
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == "BP":
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    elif graph_type == "Fully":
        B = np.triu(np.ones((d, d)), 1)
    else:
        raise ValueError("unknown graph type")
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """

    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]."""
        if sem_type == "gauss":
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == "exp":
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == "gumbel":
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == "uniform":
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == "logistic":
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == "poisson":
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError("unknown sem type")
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError("noise scale must be a scalar or has length d")
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError("W must be a DAG")
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == "gauss":
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError("population risk not available")
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def sample_mean(mech_type, X_parents):
    """Sample mean function for nonlinear SEM."""
    # vllt so ähnlich wie notears, matrixmultiplizieren und einmal sigmoid, einfachere funktion
    def _single_sigmoids(X_pa_j):
        """Sample sigmoidal function for nonlinear SEM."""
        c = random.uniform(-2, 2)
        bern = np.random.binomial(1, 0.5)
        b = bern * random.uniform(0.5, 2) + (1 - bern) * random.uniform(-2, -0.5)
        a = np.random.exponential(scale=1 / 4) + 2.5  # changed from +1 to +2.5
        return a * (b * (X_pa_j + c)) / (1 + abs(b * (X_pa_j + c)))

    if mech_type == "gp":
        gp = GaussianProcessRegressor()  # RBF kernel with length_scale=1
        x_child = gp.sample_y(X_parents, random_state=None).flatten()
    elif mech_type == "sigmoidal":
        sigmoids_list = []
        for j in range(X_parents.shape[1]):
            sigmoids_list.append(_single_sigmoids(X_parents[:, j]))
        x_child = np.sum(sigmoids_list, axis=0)
    else:
        raise ValueError("unknown mech type")
    return x_child


def sample_noise(noise_type, n, x_child=None, skew_noise=False):
    """Sample noise function for nonlinear SEM."""
    if noise_type == "additive":
        if skew_noise:
            noise = 0.2 * skewnorm.rvs(a=20, size=n) 
        else:
            sigma = np.random.uniform(low=1 / 5, high=np.sqrt(2) / 5, size=n)
            noise = np.random.normal(loc=0, scale=sigma, size=n)
    elif noise_type == "LS":
        assert x_child is not None
        if skew_noise:
            noise = 0.2 * skewnorm.rvs(a=20, size=n)
        else:
            sigma = np.random.uniform(low=1 / 5, high=np.sqrt(2) / 5, size=n)
            noise = np.random.normal(loc=0, scale=sigma, size=n)
        noise = (x_child - min(x_child)) * noise
    else:
        raise ValueError("unknown noise type")
    return noise


def select_het_noise_nodes(G, ordered_vertices, het_noise_fraction):
    """Select nodes to be het noise nodes.

    Args:
        G (ig.Graph): graph
        ordered_vertices (list): ordered vertices
        het_noise_fraction (float): fraction of het noise affected non source nodes

    Returns:
       node_het_noise_flag (dict): {node: het_noise_flag}
    """
    non_source_node_flg = [len(G.neighbors(j, mode=ig.IN)) != 0 for j in ordered_vertices]
    nodes_df = pd.DataFrame(
        {
            "ordered_vertices": ordered_vertices,
            "non_source_node_flg": non_source_node_flg,
        }
    )
    nbr_non_source_nodes = nodes_df["non_source_node_flg"].sum()
    nbr_het_noise_nodes = int(np.ceil(het_noise_fraction * nbr_non_source_nodes))

    het_noise_nodes = np.random.choice(
        nodes_df[nodes_df["non_source_node_flg"]]["ordered_vertices"],
        size=nbr_het_noise_nodes,
        replace=False,
    )

    nodes_df["het_noise_node"] = nodes_df["ordered_vertices"].isin(het_noise_nodes)
    return nodes_df.set_index("ordered_vertices")["het_noise_node"].to_dict()


def simulate_single_equation(
    X,
    parents,
    source_node_noise,
    sem_type,
    het_noise_node,
    B_cause_mean=None,
    B_cause_noise=None,
):
    """
    Simulate a single equation of nonlinear SEM.

    Args:
        X (np.array): [n, d]
        parents (list): list of parents
        source_node_noise (str): 'gauss' or 'uniform'
        sem_type (str): One of 'AN', 'AN-s', 'LS', 'LS-s', 'MN', 'LS-p', 'skew-LS-s
        het_noise_node (bool): True if this node is affected by heteroscedastic noise
        B_cause_mean (np.array): [d, d] track which parents cause mean, only relevant for LS-p
        B_cause_noise (np.array): [d, d] track which parent cause noise, only relevant for LS-p
    Returns:
        sim_function (np.array): [n, ] returns mean + noise
        sim_mean_noise (np.array): [n, 2] returns mean, noise
    """
    X_parents = X[:, parents]
    n, pa_size = X_parents.shape

    if pa_size == 0:
        # source node in DAG
        noise = np.zeros(n)
        if source_node_noise == "gauss":
            sigma = np.random.uniform(low=1, high=np.sqrt(2), size=n)
            x_child = np.random.normal(loc=0, scale=sigma, size=n)
        elif source_node_noise == "uniform":
            x_child = np.random.uniform(low=-3.5, high=3.5, size=n)
        else:
            raise ValueError(f"unknown src noise type: {source_node_noise}")
    else:
        # non-source node in DAG
        if sem_type == "AN":
            x_child = sample_mean(mech_type="gp", X_parents=X_parents)
            noise = sample_noise(noise_type="additive", n=n)
        elif sem_type == "AN-s":
            x_child = sample_mean(mech_type="sigmoidal", X_parents=X_parents)
            noise = sample_noise(noise_type="additive", n=n)
        elif sem_type == "LS":
            x_child = sample_mean(mech_type="gp", X_parents=X_parents)
            if het_noise_node:
                noise = sample_noise(noise_type="LS", n=n, x_child=x_child)
            else:
                noise = sample_noise(noise_type="additive", n=n)
        elif sem_type == "LS-s":
            x_child = sample_mean(mech_type="sigmoidal", X_parents=X_parents)
            if het_noise_node:
                noise = sample_noise(noise_type="LS", n=n, x_child=x_child)
            else:
                noise = sample_noise(noise_type="additive", n=n)
        elif sem_type == "MN-s":
            noise_function = sample_mean(mech_type="sigmoidal", X_parents=X_parents)
            noise = noise_function * np.random.uniform(low=0, high=1, size=n)
            x_child = np.zeros(n)
        elif sem_type == "LS-p":
            assert B_cause_mean is not None, "B_cause_mean must be provided for LS-p"
            assert B_cause_noise is not None, "B_cause_noise must be provided for LS-p"
            if pa_size == 1:
                # if only one parent, then this parents influences the mean and noise
                _, sim_mean_noise = simulate_single_equation(
                    X, parents, source_node_noise, sem_type="LS", het_noise_node=1
                )
                x_child = sim_mean_noise[:, 0]
                noise = sim_mean_noise[:, 1]
                B_cause_mean[parents] = 1
                B_cause_noise[parents] = 1
            else:
                # if more than one parent, then select half of the parents
                # that only affect mean and half that only affect noise
                nbr_PA_affect_noise = int(np.ceil(0.5 * pa_size))
                PA_affect_noise = np.random.choice(parents, size=nbr_PA_affect_noise, replace=False)
                PA_affect_mean = np.array(list(set(parents) - set(PA_affect_noise)))
                # print('PA_affect_noise', PA_affect_noise)
                # print('PA_affect_mean', PA_affect_mean)

                B_cause_mean[PA_affect_mean] = 1
                B_cause_noise[PA_affect_noise] = 1

                x_child = sample_mean(mech_type="gp", X_parents=X[:, PA_affect_mean])
                noise_function = sample_mean(mech_type="sigmoidal", X_parents=X[:, PA_affect_noise])
                positive_noise_function = (noise_function - noise_function.min()) * 1.5
                noise = positive_noise_function * sample_noise(noise_type="additive", n=n)
        elif sem_type == "skew-LS-s":
            x_child = sample_mean(mech_type="sigmoidal", X_parents=X_parents)
            if het_noise_node:
                noise = sample_noise(noise_type="LS", n=n, x_child=x_child, skew_noise=True)
            else:
                noise = sample_noise(noise_type="additive", n=n, skew_noise=True)
        else:
            raise ValueError("unknown sem type")

    sim_function = x_child + noise
    sim_mean_noise = np.stack((x_child, noise), axis=1)
    return sim_function, sim_mean_noise

def simulate_nonlinear_sem(B, n, sem_type, source_node_noise="gauss", het_noise_fraction=0):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): One of 'AN', 'AN-s', 'LS', 'LS-s', 'MN', 'LS-p', "skew-LS-s"
        source_node_noise (str): 'gauss' or 'uniform'
        het_noise_fraction (float): fraction of none source nodes that are affected with het-noise
    Returns:
        X (np.ndarray): [n, d] sample matrix
        X_inspect (np.ndarray): [n, d, 2] sampled mean and noise functions separately
    """
    d = B.shape[0]
    X = np.zeros([n, d])
    X_inspect = np.zeros([n, d, 2])  # store mean and noise functions separately
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    het_noise_flag = select_het_noise_nodes(G, ordered_vertices, het_noise_fraction)
    if sem_type == "LS-p":
        B_cause_mean = np.zeros((d, d))  # track which parents cause mean
        B_cause_noise = np.zeros((d, d))  # track which parents cause noise
        for j in ordered_vertices:
            parents = G.neighbors(j, mode=ig.IN)
            X[:, j], X_inspect[:, j] = simulate_single_equation(
                X,
                parents,
                source_node_noise=source_node_noise,
                sem_type=sem_type,
                het_noise_node=het_noise_flag[j],
                B_cause_mean=B_cause_mean[:, j],
                B_cause_noise=B_cause_noise[:, j],
            )
    else:
        B_cause_mean = None
        B_cause_noise = None
        for j in ordered_vertices:
            parents = G.neighbors(j, mode=ig.IN)
            X[:, j], X_inspect[:, j] = simulate_single_equation(
                X,
                parents,
                source_node_noise=source_node_noise,
                sem_type=sem_type,
                het_noise_node=het_noise_flag[j],
            )

    return X, X_inspect, B_cause_mean, B_cause_noise



