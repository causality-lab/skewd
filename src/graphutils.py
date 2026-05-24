"""Graph utils functions."""

import numpy as np
import itertools
import igraph as ig
from graphical_models import DAG
from graphviz import Digraph

def is_dag(W):
    """Check if W is a DAG."""
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def transform_adj(B: np.ndarray):
    """Transform result of causal-learn's PC algorithm into desired adjacency matrix format.
    
    Parameters
    ----------
    B: PC algorithm adjacency matrix (d x d ndarray)

    Returns
    -------
    - desired adjacency matrix
    - undirected pairs: list of tuples (i,j) with i<j
    """
    B = B.copy()  
    d = B.shape[0]
    undirected_pairs = []

    for i in range(d):
        for j in range(i + 1, d):
            Bij = B[i, j]
            Bji = B[j, i]
            # Case 1: Bij=1 and Bji=-1 OR Bij=-1 and Bji=1 
            if Bij != Bji:
                # Replace 1 with 0 and -1 with 1
                if Bij == 1 and Bji == -1:
                    B[i, j] = 0
                    B[j, i] = 1
                elif Bij == -1 and Bji == 1:
                    B[i, j] = 1
                    B[j, i] = 0
                else:
                    raise ValueError(f"Inconsistent values at ({i}, {j}): {Bij}, {Bji}")
            # Case 2: Bij=Bji
            else:
                if Bij == 0:
                    continue
                else:
                    # both are 1 or -1 >> set both to -1
                    B[i, j] = -1
                    B[j, i] = -1
                    undirected_pairs.append((i, j)) # index starts at 0 here.

    return B, undirected_pairs

def build_oracle_MEC(B):
    """Extract Markov equivalence class from given adjacency matrix B.
    
    Input: 
        B: Ground-truth adjacency matrix (dxd ndarray) 

    Returns:
        oracle_MEC: CPDAG, where oracle_MEC[i,j] == -1 indicates undirected edge,
                    same definition as in transform_adj
    """
    arcs = set()
    for i in range(B.shape[0]):
        for j in range(B.shape[0]):
            if i!=j and B[i,j]==1:
                arcs.add((i,j))
    
    nodes = set(j for j in range(B.shape[0]))
    D = DAG(arcs=arcs, nodes=nodes)
    CPDAG_arcs = D.cpdag().arcs

    oracle_MEC = np.zeros(B.shape)
    for i, j in arcs:
        if (i,j) in CPDAG_arcs:
            oracle_MEC[i,j] = 1
        else:
            oracle_MEC[i,j] = -1
            oracle_MEC[j,i] = -1
    return oracle_MEC


def create_candidates(A: np.ndarray, undirected_pairs):
    """
    Given CPDAG A with undirected pairs (i,j) expressed via A[i,j] = A[j,i] = -1,
    generate all possible orientations for these pairs and check for DAG constraint.

    Each undirected pair (i,j) has 2 choices:
        - A[i,j] = 0, A[j,i] = 1
        - A[i,j] = 1, A[j,i] = 0

    Returns:
        candidate_list : list of possible DAGs as numpy arrays
    """
    R1 = len(undirected_pairs)
    candidate_list = []

    # Loop over all 2^R1 binary choices: 
    # bits[0] == 0 <<-->> orient first edge i -> j and vice versa
    for bits in itertools.product([0, 1], repeat=R1):
        A_new = A.copy()

        for (bit, (i, j)) in zip(bits, undirected_pairs):
            if bit == 0:
                A_new[i, j] = 0
                A_new[j, i] = 1
            else:
                A_new[i, j] = 1
                A_new[j, i] = 0

        if is_dag(A_new):
            candidate_list.append(A_new)

    return candidate_list

def identify_parents(y, G):
    """Identify (sorted) parents of y-th node in DAG G. Returns indices."""
    parent_set = []
    for j in range(G.shape[0]):
        if G[j, y] == 1:
            parent_set.append(j)
    return parent_set


def plot_graph(A, node_names=None):
    """Plot a CPDAG (or DAG) given adjacency matrix A."""

    p = A.shape[0]
    if node_names is None:
        node_names = [str(i) for i in range(p)]

    G = Digraph(engine="circo")

    for v in node_names:
        G.node(v)

    # Edges
    for i in range(p):
        for j in range(p):
            if A[i, j] == 1:
                G.edge(node_names[i], node_names[j])

            elif A[i, j] == -1 and i < j:
                G.edge(
                    node_names[i],
                    node_names[j],
                    dir="none",
                    style="dashed",
                    color="gray"
                )
    return G

def undirected_edges(B):
    """Extract undirected edges of CPDAG B."""
    d = B.shape[0]
    edges = []
    for i in range(d):
        for j in range(i+1, d):
            if B[i,j] == -1:
                edges.append((i,j))
    return edges


def correct_edges(edges, GT_graph, pred_graph):
    """For a list of edges, return correctly predicted edges given ground-truth and prediction."""
    correct_edges = []
    for edge in edges:
        i, j = edge[0], edge[1]
        if GT_graph[i,j] == pred_graph[i,j] and GT_graph[j,i] == pred_graph[j,i]:
            correct_edges.append(edge)
    return correct_edges