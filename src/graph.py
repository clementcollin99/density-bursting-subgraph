import itertools
import numpy as np


def exp_decreasing_distribution(n: int, decreasing_fact: float = 4.):
    """
    Args:
        n (int): length of the discrete distribution to be computed

    Returns:
        np.ndarray: exponentially decreasing discrete distribution
    """
    f = lambda x: np.exp(-decreasing_fact * x)
    distrib = np.array(list(map(f, np.arange(n))))
    distrib /= np.sum(distrib)
    return distrib


def rand_adjacency_matrix(n_vertices: int, upper_bound: int = 5):
    """
    Randomly generates an adjacency matrix of size (n_vertices, n_vertices).

    Args:
        n_vertices (int): # vertices in the graph
        upper_bound (int, optional): maximum value for an edge to hold

    Returns:
        np.ndarray: adjacency matrix of some undirected sparse graph
    """
    distrib = exp_decreasing_distribution(upper_bound)
    rand = np.random.choice(upper_bound, (n_vertices, n_vertices), p=distrib)
    adj_mat = np.tril(rand) + np.tril(rand, -1).T
    np.fill_diagonal(adj_mat, 0)
    return adj_mat


def construct_network(graph: "Graph", guess: float):
    """
    Step 2 of A. V. Goldberg's article

    Args:
        graph (Graph): studied graph
        guess (float): guess for the density of the graph

    Returns:
        Graph: constructed network
    """
    new_vertices = np.append(graph.vertices, ["source", "sink"])
    new_adj_mat = graph.adjacency_matrix

    # add source
    source_connections = np.ones((1, graph.n_vertices)) * graph.total_weight

    def func(vec):
        return np.append(vec, [[0]], axis=1)

    new_adj_mat = np.vstack((new_adj_mat, source_connections))
    new_adj_mat = np.hstack((new_adj_mat, func(source_connections).T))

    # add sink
    sink_connections = func([[
        graph.total_weight + 2 * guess - np.sum(row)
        for row in graph.adjacency_matrix
    ]])

    new_adj_mat = np.vstack((new_adj_mat, sink_connections))
    new_adj_mat = np.hstack((new_adj_mat, func(sink_connections).T))

    # return the network as a graph
    return Graph(new_vertices, new_adj_mat)


def cut_capacity(graph: "Graph", partition: dict, guess: float):
    """
    Computes the capacity of an s-t cut c(S, T) if we build
    the network as described in Goldberg's article.

    Args:
        graph (Graph): studied graph
        partition (dict): partition of the graph into two sets
                          {'S':{'A', 'B'}, 'T':{'C'}}.
                          Implicitly 'source' belongs to S and 'sink' to T.
        guess (float): guess for the density of the graph
    """
    capacity = graph.n_vertices * graph.total_weight
    V_1 = partition["S"]

    if not len(V_1):
        return capacity

    ind_graph = Graph(V_1, graph[V_1])
    capacity += len(V_1) * 2 * (guess - ind_graph.density)

    return capacity


def find_min_cut(graph: "Graph", guess: float):
    """
    Finds the minimal cut of a flow graph.

    Args:
        graph (Graph): studied graph
        guess (float): guess for the density of the graph
    """
    partitions = [(tuple(it), tuple(set(graph.vertices) - set(it)))
                  for r in range(len(graph.vertices))
                  for it in itertools.combinations(graph.vertices, r)]
    partitions = [{
        "S": S,
        "T": T
    } for S, T in partitions[:len(partitions) // 2 + 1]]

    capacities = [cut_capacity(graph, part, guess) for part in partitions]
    idx_min = np.argmin(capacities)

    return partitions[idx_min], capacities[idx_min]


def maximum_density_subgraph(graph: "Graph"):
    """
    Finds the subgraph with maximal density.

    Args:
        graph (Graph): studied graph

    Returns:
        tuple: vertices of the maximum density subgraph and its adjacency matrix
    """
    x = 0
    y = graph.total_weight
    while ((y - x) >= ((graph.n_vertices * (graph.n_vertices - 1)))**(-1)):
        guess = (x + y) / 2
        # net = construct_network(graph, guess)
        partition, _ = find_min_cut(graph, guess)
        if len(partition["S"]) == 0:
            y = guess
        else:
            x = guess
            V_1 = partition["S"]

    return V_1, graph[V_1]


class Graph:
    """_summary_
    """

    def __init__(self, vertices: list, adj_mat: np.array = None):
        if adj_mat is None:
            adj_mat = rand_adjacency_matrix(len(vertices))

        if not isinstance(adj_mat, np.ndarray):
            adj_mat = np.array(adj_mat)

        assert len(adj_mat.shape) == 2 and adj_mat.shape[0] == adj_mat.shape[
            1] and len(vertices) == adj_mat.shape[0]

        self._vertices = vertices
        self._adj_mat = adj_mat

    @property
    def adjacency_matrix(self):
        return self._adj_mat

    @property
    def vertices(self):
        return self._vertices

    @property
    def n_vertices(self):
        return len(self._vertices)

    @property
    def edges(self):
        for i, neighbors in enumerate(self._adj_mat):
            for j, weight in enumerate(neighbors):
                if j >= i and weight:
                    yield ((self._vertices[i], self._vertices[j]), weight)

    @property
    def total_weight(self):
        return np.sum([
            weight for i, neighbors in enumerate(self._adj_mat)
            for j, weight in enumerate(neighbors) if j >= i
        ])

    @property
    def density(self):
        return self.total_weight / self.n_vertices

    @property
    def degrees(self):
        """
        Computes the degrees of the vertices of the graph.
        The degree of a vertex is the sum of the weights of
        the edges that are incident to the vertex.
        
        Returns:
            dict: dictionnary containing the vertices as keys and
                  their degree as values
        """
        return dict((vertex, degree) for vertex, degree in zip(
            self._vertices, np.sum(self._adj_mat, axis=0)))

    # def __iter__(self):
    #     self._iter_obj = iter(self._graph_dict)
    #     return self._iter_obj

    # def __next__(self):
    #     return next(self._iter_obj)

    def __getitem__(self, keys):
        """
        Submatrix induced by a subset of vertices

        Args:
            keys (iterable): list of vertices

        Returns:
            np.ndarray: induced submatrix
        """
        indices = tuple(self._vertices.index(key) for key in keys)
        return self._adj_mat[np.ix_(indices, indices)]

    def __add__(self, object):
        return Graph(self.vertices, self._adj_mat + object.adjacency_matrix)

    def __str__(self):
        res = "vertices : "
        for k in self._vertices:
            res += str(k) + " "
        res += "\nedges : "
        for edge in self.edges:
            res += str(edge) + " "
        return res
