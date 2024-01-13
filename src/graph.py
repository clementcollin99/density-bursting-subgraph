import numpy as np


def exp_decreasing_distribution(n: int):
    """
    Args:
        n (int): the length of the discrete distribution to be computed

    Returns:
        np.ndarray: an exponentially decreasing discrete distribution
    """
    f = lambda x: np.exp(-x)
    distrib = np.array(list(map(f, np.arange(n))))
    distrib /= np.sum(distrib)
    return distrib


def rand_adjacency_matrix(n_vertices: int, upper_bound: int = 5):
    """
    Randomly generates an adjacency matrix of size (n_vertices, n_vertices).

    Args:
        n_vertices (int): # vertices in the graph
        upper_bound (int, optional): the maximum value for an edge to hold.
                                     Defaults to 5.

    Returns:
        np.ndarray: the adjacency matrix of some undirected sparse graph.
    """
    distrib = exp_decreasing_distribution(upper_bound)
    rand = np.random.choice(upper_bound, (n_vertices, n_vertices), p=distrib)
    affinity_matrix = np.tril(rand) + np.tril(rand, -1).T
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix


class Graph:

    def __init__(self, vertices: list, adj_mat: np.array = None):
        if not adj_mat:
            adj_mat = rand_adjacency_matrix(len(vertices))

        if not isinstance(adj_mat, np.ndarray):
            adj_mat = np.array(adj_mat)

        assert len(adj_mat.shape) == 2
        assert adj_mat.shape[0] == adj_mat.shape[1]
        assert len(vertices) == adj_mat.shape[0]

        self._vertices = vertices
        self._adj_mat = adj_mat

    @property
    def matrix(self):
        return self._adj_mat

    @property
    def vertices(self):
        return self._vertices

    @property
    def edges(self):
        for i, neighbors in enumerate(self._adj_mat):
            for j, v in enumerate(neighbors):
                if v:
                    yield (i, j)

    def __iter__(self):
        self._iter_obj = iter(self._graph_dict)
        return self._iter_obj

    def __next__(self):
        return next(self._iter_obj)

    def __str__(self):
        res = "vertices : "
        for k in self.vertices:
            res += str(k) + " "
        res += "\nedges : "
        for edge in self.edges:
            res += str(edge) + " "
        return res
